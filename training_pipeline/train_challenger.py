import pathlib
import pickle
import mlflow
import dagshub
from prefect import flow, task
from mlflow import MlflowClient
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
from hyperopt.pyll import scope
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.feature_extraction import DictVectorizer

@task(name = "Train Best Model")
def train_best_model(X_train, X_val, y_train, y_val, dv, best_params) -> None:
    """train a model with best hyperparams and write everything out"""

    with mlflow.start_run(run_name="Best model ever"):
        train = xgb.DMatrix(X_train, label=y_train)
        valid = xgb.DMatrix(X_val, label=y_val)

        mlflow.log_params(best_params)

        # Log a fit model instance
        booster = xgb.train(
            params=best_params,
            dtrain=train,
            num_boost_round=100,
            evals=[(valid, 'validation')],
            early_stopping_rounds=10
        )

        y_pred = booster.predict(valid)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        mlflow.log_metric("rmse", rmse)

        pathlib.Path("models").mkdir(exist_ok=True)
        with open("models/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)

        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")

    return None

@task(name = 'Register Model')
def register_model():
    MLFLOW_TRACKING_URI = mlflow.get_tracking_uri()
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    df = mlflow.search_runs(order_by=['metrics.rmse'])
    run_id = df.loc[df['metrics.rmse'].idxmin()]['run_id']
    run_uri = f"runs:/{run_id}/model"

    result = mlflow.register_model(
        model_uri=run_uri,
        name="nyc-taxi-model-perfect"
    )
    model_name = "nyc-taxi-model-perfect"
    model_version_alias = "champion"

    # create "champion" alias for version 1 of model "nyc-taxi-model"
    client.set_registered_model_alias(
        name=model_name,
        alias=model_version_alias,
        version= '1'
    )


# Reutiliza las funciones y tareas de train.py para el flujo de datos y el entrenamiento

@task(name="Add features")
def add_features(df_train: pd.DataFrame, df_val: pd.DataFrame):
    """Add features to the model"""
    df_train["PU_DO"] = df_train["PULocationID"] + "_" + df_train["DOLocationID"]
    df_val["PU_DO"] = df_val["PULocationID"] + "_" + df_val["DOLocationID"]

    categorical = ["PU_DO"]  # 'PULocationID', 'DOLocationID']
    numerical = ["trip_distance"]

    dv = DictVectorizer()

    train_dicts = df_train[categorical + numerical].to_dict(orient="records")
    X_train = dv.fit_transform(train_dicts)

    val_dicts = df_val[categorical + numerical].to_dict(orient="records")
    X_val = dv.transform(val_dicts)

    y_train = df_train["duration"].values
    y_val = df_val["duration"].values
    return X_train, X_val, y_train, y_val, dv


@task(name="Read Data", retries=4, retry_delay_seconds=[1, 4, 8, 16])
def read_data(file_path: str) -> pd.DataFrame:
    """Read data into DataFrame"""
    df = pd.read_parquet(file_path)

    df.lpep_dropoff_datetime = pd.to_datetime(df.lpep_dropoff_datetime)
    df.lpep_pickup_datetime = pd.to_datetime(df.lpep_pickup_datetime)

    df["duration"] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ["PULocationID", "DOLocationID"]
    df[categorical] = df[categorical].astype(str)

    return df


@task(name="Hyperparameter tuning")
def hyper_parameter_tunning(X_train, X_val, y_train, y_val, dv):
    mlflow.xgboost.autolog()

    training_dataset = mlflow.data.from_numpy(X_train.data, targets=y_train, name="green_tripdata_2024-01")

    validation_dataset = mlflow.data.from_numpy(X_val.data, targets=y_val, name="green_tripdata_2024-02")

    train = xgb.DMatrix(X_train, label=y_train)

    valid = xgb.DMatrix(X_val, label=y_val)

    def objective(params):
        with mlflow.start_run(nested=True):
            # Tag model
            mlflow.set_tag("model_family", "xgboost")

            # Train model
            booster = xgb.train(
                params=params,
                dtrain=train,
                num_boost_round=100,
                evals=[(valid, 'validation')],
                early_stopping_rounds=10
            )

            # Predict in the val dataset
            y_pred = booster.predict(valid)

            # Calculate metric
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))

            # Log performance metric
            mlflow.log_metric("rmse", rmse)

        return {'loss': rmse, 'status': STATUS_OK}

    with mlflow.start_run(run_name="Xgboost Hyper-parameter Optimization", nested=True):
        search_space = {
            'max_depth': scope.int(hp.quniform('max_depth', 4, 100, 1)),
            'learning_rate': hp.loguniform('learning_rate', -3, 0),
            'reg_alpha': hp.loguniform('reg_alpha', -5, -1),
            'reg_lambda': hp.loguniform('reg_lambda', -6, -1),
            'min_child_weight': hp.loguniform('min_child_weight', -1, 3),
            'objective': 'reg:squarederror',
            'seed': 42
        }

        best_params = fmin(
            fn=objective,
            space=search_space,
            algo=tpe.suggest,
            max_evals=10,
            trials=Trials()
        )
        best_params["max_depth"] = int(best_params["max_depth"])
        best_params["seed"] = 42
        best_params["objective"] = "reg:squarederror"

        mlflow.log_params(best_params)

    return best_params



@task(name="Compare Models")
def compare_models(challenger_rmse: float, champion_rmse: float):
    """Compara las métricas RMSE de los modelos y decide cuál será el champion."""
    if challenger_rmse < champion_rmse:
        return "challenger"
    else:
        return "champion"


@task(name="Get Champion RMSE")
def get_champion_rmse():
    """Obtén el RMSE del modelo champion registrado."""
    client = MlflowClient()
    champion_model = client.get_latest_versions(name="nyc-taxi-model-prefect", stages=["Production"])[0]
    run_id = champion_model.run_id
    run = client.get_run(run_id)
    champion_rmse = run.data.metrics["rmse"]
    return champion_rmse


@flow(name="Challenger Flow")
def challenger_flow(year: str, month_train: str, month_val: str):
    # Configurar MLflow
    dagshub.init(url="https://dagshub.com/juanplv04/nyc-taxi-time-prediction", mlflow=True)
    mlflow.set_experiment("nyc-taxi-experiment-prefect")

    # Cargar datos y preparar características
    df_train = read_data(f"../data/green_tripdata_{year}-{month_train}.parquet")
    df_val = read_data(f"../data/green_tripdata_{year}-{month_val}.parquet")
    X_train, X_val, y_train, y_val, dv = add_features(df_train, df_val)

    # Entrenar modelo challenger
    best_params = hyper_parameter_tunning(X_train, X_val, y_train, y_val, dv)
    model = train_best_model(X_train, X_val, y_train, y_val, dv, best_params)

    # Obtener las predicciones
    y_pred = model.predict(X_val)

    # Calcular RMSE del challenger
    challenger_rmse = mean_squared_error(y_val, y_pred, squared=False)

    # Obtener RMSE del champion
    champion_rmse = get_champion_rmse()

    # Comparar modelos y decidir
    winner = compare_models(challenger_rmse, champion_rmse)

    # Registrar el modelo ganador como champion
    if winner == "challenger":
        register_model()

'''
@flow(name="Challenger Flow")
def challenger_flow(year: str, month_train: str, month_val: str):
    # Configurar MLflow
    # MLflow settings
    dagshub.init(url="https://dagshub.com/juanplv04/nyc-taxi-time-prediction", mlflow=True)

    mlflow.set_experiment(experiment_name="nyc-taxi-experiment-prefect")

    # Cargar datos y preparar características
    df_train = read_data(f"../data/green_tripdata_{year}-{month_train}.parquet")
    df_val = read_data(f"../data/green_tripdata_{year}-{month_val}.parquet")
    X_train, X_val, y_train, y_val, dv = add_features(df_train, df_val)

    # Entrenar modelo challenger
    best_params = hyper_parameter_tunning(X_train, X_val, y_train, y_val, dv)
    train_best_model(X_train, X_val, y_train, y_val, dv, best_params)

    # Obtener RMSE del challenger
    challenger_rmse = mean_squared_error(y_val, X_val, squared=False)

    # Obtener RMSE del champion
    champion_rmse = get_champion_rmse()

    # Comparar modelos y decidir
    winner = compare_models(challenger_rmse, champion_rmse)

    # Registrar el modelo ganador como champion
    if winner == "challenger":
        register_model()
'''

challenger_flow(year="2024", month_train="01", month_val="02")
