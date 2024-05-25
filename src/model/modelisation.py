
import pickle

# Package pour les tableaux et dataframe
import pandas as pd

# Packages scikit-learn pour modélisation
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline





def fit_model (preprocessor, model_name, model, params, X, y) :
    """
    fonction qui définit le pipeline en fonction du 'model'
    charge les hyperparamètres retenus (params)
    entraîne le model avec X et y

    Parameters:
        preprocessor: le pipeline de tranformation.
        model_name : le nom du modèle
        model : l'estimateur
        params : hyperparamètres de l'estimateur
        X : dataframe des features
        y : la target

    Returns:
        pipe : le modèle entraîné
    """
    pipe = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', model)])
    pipe.set_params(**params)
    pipe.fit(X, y)
    #pickle.dump(pipe, open(model_name + ".sav", 'wb'))
    return pipe



def model_evaluation (pipe, model, X_train, y_train, X_test, y_test) :
    """
    fonction qui évalue le modèle entraîné sur les bases train et test
    et retourne les probabilités prédites pour le jeu de test

    Parameters:
        pipe: le modèle entraîné
        X_train, y_train, X_test, y_test

    Returns:
        y_test_proba : liste des proba prédites
    """
    y_train_proba = pipe.predict_proba(X_train )
    y_test_proba = pipe.predict_proba(X_test)
    auc_train = roc_auc_score(y_train, y_train_proba[:,1])
    auc_test = roc_auc_score(y_test, y_test_proba[:,1])
    print(f"Score AUC pour le modèle {model} optimisé : ")
    print(f"{'-'*100}")
    print(f"Train  : {round(auc_train,4)},\n Test  : {round(auc_test,4)}")
    return y_test_proba



def do_randomized_search(preprocessor_, model_name, model_, X, y, seed) :
    """
     Finetuning du modèle 'model', stratégie de randomizedSearch par cross validation n = 5:
     Définition du pipeline
     Cross validation de RandomizedSearchCV sur la base (X, y)
     Affichage du score optimum, de la combinaison d'hyperparamètres correspondante
     Affichage des 8 meilleurs combinaisons en hyperparamètres

    Parameters:
        preprocessor_: pipeline de tranformation des features
        model_name : nom du model
        model_ : estimateur
        X : le dataframe des features
        y : la target
        seed

    Returns:
        df_score_params : dataframe contenant les 8 meilleurs combinaisons d'hyperparametres
        et les scores obtenus.
    """
    pipe_ = Pipeline([
                ('preprocessor', preprocessor_),
                ('classifier',model_["model"])])
    rscv = RandomizedSearchCV(estimator = pipe_, param_distributions= model_["params"], cv = 5,
                               n_jobs = 5,n_iter=40,return_train_score = True,
                               scoring = "roc_auc" ,random_state = seed)
    rscv.fit(X, y)
    df_score_params = pd.DataFrame(rscv.cv_results_).sort_values("mean_test_score",
                                                                 ascending = False)
    print(f"Modèle { model_name} avec RandomizedSearch : ")
    print(f"La meilleure combinaison d'hyperparamètres est :\n{rscv.best_params_}")
    print(f"\n  -> Score AUC optimisé : {round(rscv.best_score_,4)}")
    print(df_score_params[["params","mean_test_score","mean_train_score"]].head(n=8))
    return  df_score_params[["params"]].head(n=8).values.ravel()
