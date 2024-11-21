import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import accuracy_score
from skopt import BayesSearchCV

PARAM_GRID_RF = {
    'n_estimators': [10, 50, 100],
    'criterion': ['entropy', 'gini'],
    'max_depth': [3, 5, 10, None]
}

PARAM_GRID_GB = {
    'n_estimators': [10, 50, 100],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [10, 20, 30]
}

PARAM_GRID_KNN = {'n_neighbors': [3, 5, 7, 9]}

def model_train_test(df: pd.DataFrame, target: str, test_size=0.3, random_state=42) -> list:
    X, y = df.drop(columns=target), df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state
    )

    return X_train, X_test, y_train, y_test

def mlflow_up(experiment: str, url='http://127.0.0.1:5000'):
    mlflow.set_tracking_uri(url)
    mlflow.set_experiment(experiment)
    mlflow.sklearn.autolog(silent=True)

def rand_search_cv(model, param_grid, X_train, X_test, y_train, y_test, cv=5, n_jobs=-1, verbose=1):
    with mlflow.start_run(run_name=f'RandomSearchCV_{model.__class__.__name__}'):
        rand_search = RandomizedSearchCV(
            model, 
            param_grid, 
            cv=cv, 
            n_jobs=n_jobs,
            verbose=verbose
        )

        rand_search.fit(X_train, y_train)
        
        best_model = rand_search.best_estimator_
        predictions = best_model.predict(X_test)
        accuracy = accuracy_score(predictions, y_test)

        mlflow.log_metric('accuracy', accuracy)
        print(f'Melhores parâmetros: {rand_search.best_params_}')
        print(f'Precisão (acurácia): {accuracy}')

def grid_search_cv(model, param_grid, X_train, X_test, y_train, y_test, cv=5, n_jobs=-1, verbose=1):
    with mlflow.start_run(run_name=f'GridSearchCV{model.__class__.__name__}'):
        grid_search = GridSearchCV(
            model, 
            param_grid, 
            cv=cv, 
            n_jobs=n_jobs,
            verbose=verbose
        )
        
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        predictions = best_model.predict(X_test)
        accuracy = accuracy_score(predictions, y_test)

        mlflow.log_metric('accuracy', accuracy)
        print(f'Melhores parâmetros: {grid_search.best_params_}')
        print(f'Precisão (acurácia): {accuracy}')

def bayesian_search_cv(model, param_grid, X_train, X_test, y_train, y_test, cv=5, n_jobs=-1, verbose=1):
    with mlflow.start_run(run_name=f'BayesSearchCV{model.__class__.__name__}'):
        bayesian_search = BayesSearchCV(
            model, 
            param_grid, 
            cv=cv, 
            n_jobs=n_jobs,
            verbose=verbose
        )
        
        bayesian_search.fit(X_train, y_train)
        
        best_model = bayesian_search.best_estimator_
        predictions = best_model.predict(X_test)
        accuracy = accuracy_score(predictions, y_test)

        mlflow.log_metric('accuracy', accuracy)
        print(f'Melhores parâmetros: {bayesian_search.best_params_}')
        print(f'Precisão (acurácia): {accuracy}')

