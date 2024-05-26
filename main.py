# Package pour la gestion des objets json
import json

# Package pour les tableaux et dataframe
import pandas as pd

# Package pour analyse statistique
from scipy.stats import loguniform

# Package pour la visualisation graphique
import plotly.graph_objs as go

# Packages scikit-learn pour modélisation
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import  Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from  xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV

# Package SHAP pour interprétabilité
from shap import TreeExplainer

from src.data.importation_data import (
    create_dir_tree,
    move_csv, move_var_json, read_merge_data,
    import_yaml_config
)

from src.data.data_preprocessing import (
    feature_engineering
)

from src.data.data_preprocessing import (
    split_x_y,
    pipeline_trans
)
from src.model.modelisation import (
    fit_model,model_evaluation,
    do_randomized_search
)
from src.model.lift_curve import (
    lift_curve_data,
    set_params_lift_curve
)
from src.model.model_calibration import (
    score_spiegelhalter,
    sklearn_calibration,
    plot_calibration_curve
)
from src.explicatibilite.interpretability import (
    do_permutation_feat_imp,
    barplot_permut_feat_imp,
    plot_shap_beeswarm,
    plot_shap_specific_var
)



pd.options.display.max_columns = None #  pour afficher toutes les colonnes de la base
pd.options.display.max_rows = None  # pour afficher toutes les lignes de la base
pd.set_option('display.max_colwidth', None) # on ne définit pas de largeur max des colonnes


DIR_OUTPUT = "./data/output/"


if __name__ == "__main__":
    config = import_yaml_config()
    INTERM_FILE = config.get("intermediate_data")
    TRAIN_FOLDER = config.get("train_folder")
    EVAL_FOLDER = config.get("eval_folder")
    TRAIN_FILE = config.get("train")
    TEST_FILE = config.get("test")
    EVAL_FILE = config.get("eval")
    OUTPUT_FOLDER = config.get("output_folder")
    OUTPUT_FILE = config.get("output")
    SEED = config.get("seed")



    print("############################################################################\n"
          "######################   IMPORTATION DES DONNEES   #########################\n"
          "###########################################################################\n")
    
    # Création de l'arborescence de fichiers destinée à stocker les différents csv
    [PATH_DATA_DIR, PATH_RAW_DIR, PATH_INTER_DIR, PATH_ML] = create_dir_tree(TRAIN_FOLDER, EVAL_FOLDER)
    move_csv(PATH_RAW_DIR, TRAIN_FOLDER, EVAL_FOLDER)
    move_var_json (PATH_DATA_DIR)

    # Merging des 3 fichiers csv contenu dans le dossier  ./data/raw/train/
    df = read_merge_data(PATH_RAW_DIR + "/" + TRAIN_FOLDER + "/",
                         PATH_INTER_DIR, INTERM_FILE)
    df_eval = read_merge_data(PATH_RAW_DIR + "/" + EVAL_FOLDER + "/",
                              PATH_INTER_DIR, INTERM_FILE)

    # Séparation du df d'entraînement en df_train et df_test
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=SEED)

    # Affichage du nombre d'observations des 2 dataframes obtenus
    print(f"Le set de train contient {df_train.shape[0]} observations.")
    print(f"Le set de test contient {df_test.shape[0]} observations.")

    # Chargement du json 'variable_types.json'
    # regroupant les variables explicatives par type de variable
    with open( PATH_DATA_DIR + "/" + "variable_types.json", "r") as read_file:
        json_ = json.load(read_file)

    # Affichage de l'objet json_
    print(json.dumps(json_, indent=4, ensure_ascii=False))

    # Définition des listes des variables selon leur type
    bool_cols = json_["bool"] # variables booléennes
    cat_cols = json_["cat"]  # variables catégorielles textuelles
    cat_num_cols = json_["cat_num"] # variables catégorielles ordinales
    num_cols = json_["num"] # variable numériques
    all_cols = bool_cols + cat_cols + cat_num_cols + num_cols

    # On applique les discrétisations et features engineering
    df_train = feature_engineering(df_train)
    df_test = feature_engineering(df_test)
    df_eval = feature_engineering(df_eval)

    # On simplifie le dataframe en ne conservant que les features sélectionnées +  la target
    df_train = df_train[all_cols + ["churn_value"]]
    df_test = df_test[all_cols + ["churn_value"]]
    df_eval = df_eval[all_cols ]

    # Export de df_test et df_eval dans un csv
    df_train.to_csv(PATH_ML + '/'+ TRAIN_FOLDER +'/' + TRAIN_FILE, index=False)
    df_test.to_csv(PATH_ML + '/'+ TRAIN_FOLDER +'/' + TEST_FILE, index=False)
    df_eval.to_csv(PATH_ML + '/'+ EVAL_FOLDER +'/' + EVAL_FILE, index=False)

    # Séparation des variables explicatives de la variable cible
    X_train, y_train = split_x_y (df_train)
    X_test, y_test = split_x_y (df_test)

    # Définition du pipeline de tranformation des variables
    preprocessor = pipeline_trans(json_)


    print("############################################################################\n"
          "######################   MODELES CANDIDATS      ############################\n"
          "###########################################################################\n")

    # pour chacun des modèles, nous avons également défini l'espace de recherche
    # des hyperparamètres qui sera utilisé pour le finetuning.
    clf_models  = {
        "Régression Logistique" :  
                {"model": LogisticRegression(max_iter = 1000,penalty = 'l2',random_state= SEED),
                    "params" : {"classifier__solver" : ["lbfgs", "liblinear"],
                    "classifier__C" : loguniform(0.1, 10),
                    "classifier__tol" : loguniform(0.0001,0.3)}
                    },
                "SVC Linéaire" :
                {"model" :  SVC(kernel='linear', random_state=SEED, class_weight="balanced",
                                    probability=True),
                        "params" : { "classifier__C" : loguniform(0.1, 10),
                                    "classifier__tol" : loguniform(0.0001,0.3)}
                    },
                "XGBoost" :
                {"model" :  XGBClassifier(random_state =SEED),
                    "params" : {"classifier__n_estimators" : range(100,600,50),
                                "classifier__max_depth" : range(3,7,1),
                                "classifier__eta" : loguniform(0.01,0.3),
                                "classifier__lambda" : loguniform(0.1,10)}
                                },
                    "RandomForest" :
                    {"model" :  RandomForestClassifier(random_state=SEED),
                    "params" : {"classifier__max_depth" : range(3,7,1),
                    "classifier__n_estimators" : range(100,600,50)}
                    }
                }


    for key, model in clf_models.items() :
        # Définition du pipeline
        pipe = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', model["model"])])

        print(f"\n\n############################ {key} ############################")
        print(pipe)

        # Cross validation du modèle, cv = 5
        # Métrique : roc_auc
        roc_auc = cross_val_score(pipe,X_train, y_train,cv=5, n_jobs=5, scoring ='roc_auc')
        print(f"Résultats pour le modèle {key} non fine-tuné :\n"
            f" AUC moyenne de {round(roc_auc.mean(),4)} ({roc_auc})")


    print("############################################################################\n"
          "######################   HYPERPARAMETRISATION   ############################\n"
          "###########################################################################\n")

    # Hyperparamétrisation des 4 modèles :
    for key, model in clf_models.items():
        print(f"\n\n############################ {key} ############################")
        df_score_params = do_randomized_search(preprocessor,key, model,
                                            X_train, y_train, SEED)
        # on enregistre dans le dictionnaire clf_models df
        # contenant la liste des 8 meilleures combinaisons de paramètres + scores associés
        model["best_params"] = df_score_params


    print("############################################################################\n"
          "##############   EVALUATION DES 4 MODELES FINE-TUNES   ####################\n"
          "###########################################################################\n")

    fig = go.Figure()
    colors = ["#fcb735", "#40b0bf", "#175b91", "#053363"]

    for i, (key, model) in enumerate(clf_models.items()) :
        # On récupère les hyperparamètres retenus
        hyperparams = model["best_params"][1]
        if i == 2 : # pour la random forest on récupère la 1ere combinaison d'hyperparamètres
            hyperparams = model["best_params"][0]
        # entraînement du modele
        pipe = fit_model (preprocessor, key, model["model"],
                        hyperparams, X_train, y_train)
        # Evaluation du modèle
        y_test_proba =  model_evaluation (pipe, key, X_train, y_train, X_test, y_test)
        model["y_test_proba"] = y_test_proba[:,1]
        # Calcul du score de spiegelhalter sur le modèle non calibré
        print(f"Score de Spiegelhalter avant calibration: "
            f"{round(score_spiegelhalter(y_test, y_test_proba[:,1]),2)}")

        # Création de la courbe lift
        lift_df = lift_curve_data(y_test, y_test_proba[:,1])
        fig.add_trace( go.Scatter(x=lift_df.index, y=lift_df["concentration"],
                                mode='lines', name=key, line=dict(width=4),
                                line_color = colors[i]))

    set_params_lift_curve(fig)


    print("############################################################################\n"
          "#########################  XGBoost -   CALIBRATION  ########################\n"
          "###########################################################################\n")

    # Création de la courbe de calibration
    calibration_df_init = sklearn_calibration(y_test, clf_models["XGBoost"]["y_test_proba"], 20)
    #plot_calibration_curve (calibration_df_init)


    # On récupère le modèle XGBoost définit précédemment
    # et les hyperparamètres optimums
    xgb_clf = clf_models["XGBoost"]["model"]
    hyperparams = clf_models["XGBoost"]["best_params"][1]
    xgb_clf.set_params(**hyperparams)

    # Instanciation du modèle de calibration
    calibration_clf = CalibratedClassifierCV(estimator = xgb_clf,
                                            method = "isotonic", cv = 5, n_jobs = 5)
    calibration_xgb =  Pipeline([
                    ('preprocessor', preprocessor),
                    ('classifier', calibration_clf)])

    # Entraînement du modèle de calibration
    calibration_xgb.fit(X_train, y_train)
    print(calibration_xgb)

    # Evaluation du modèle
    y_test_proba_cal =  model_evaluation (calibration_xgb, "XGBoost", X_train, y_train, X_test, y_test)

    # Calcul du score de spiegelhalter sur le modèle  calibré
    print(f"Score de Spiegelhalter : {round(score_spiegelhalter(y_test, y_test_proba_cal[:,1]),2)}")

    # Affichage de la courbe de calibration du modèle avant et après calibration
    calibration_df = sklearn_calibration(y_test, y_test_proba_cal[:,1], 20)
    plot_calibration_curve (calibration_df_init, calibration_df)


    print("############################################################################\n"
          "#########################  PREDICTION - BASE EVAL  ########################\n"
          "###########################################################################\n")
    # Importation des données eval
    df_eval = pd.read_csv(PATH_INTER_DIR + "/" + EVAL_FOLDER + "/" + INTERM_FILE, #"./data/intermediate/eval/df_telco.csv",
                        sep=",",keep_default_na=False,na_values=['NaN'])
    X_eval = pd.read_csv(PATH_ML + "/" + EVAL_FOLDER + "/" + EVAL_FILE,
                        sep=",",keep_default_na=False,na_values=['NaN'])

    # Prédiction de la target
    y_eval_pred_proba =  calibration_xgb.predict_proba(X_eval)
    y_eval_pred =  calibration_xgb.predict(X_eval)

    # Stockage des résultats dans un dataframe
    df_eval_pred = pd.DataFrame({"customer_id" : df_eval["customer_id"],
                                    "y_pred_proba": y_eval_pred_proba[:,1],
                                    "churn_value": y_eval_pred})

    # On trie les clients par ordre décroissant de score:
    # i.e les clients les plus fragiles sont classés en premier dans le dataframe
    df_eval_pred = df_eval_pred.sort_values(by = "y_pred_proba", ascending = False)

    # Affichage des premières lignes
    df_eval_pred.head(n=10)

    # Export dans un csv sous ./data/output/eval_df_telco_churn_status.csv
    df_eval_pred[["customer_id", "churn_value"]].to_csv( OUTPUT_FOLDER + "/" + OUTPUT_FILE, index = False)


    print("############################################################################\n"
          "##########################   INTERPRETABILITE   ############################\n"
          "############################################################################\n")

    # On récupère dans l'ordre le nom des colonnes issues du préprocessing de X_train
    tot_cols = (json_["bool"]
                + calibration_xgb.named_steps.preprocessor.named_transformers_["cat_oh"].steps[-1][1].get_feature_names_out(json_["cat"]).tolist()
                + json_["cat_num"] + json_["num"])


    # Préprocessing de X_test et stockage dans le dataframe X_test_prepro
    X_test_prepro = pd.DataFrame(calibration_xgb.named_steps.preprocessor.transform(X_test),
                                columns=tot_cols)

    print("############################################################################\n"
          "########################## FEATURES' IMPORTANCE ############################\n"
          "############################################################################\n")

    feature_importance = do_permutation_feat_imp( calibration_xgb,
                                                X_test_prepro, y_test,
                                                30, SEED)

    N = 12 # Définition du nombre de variables à représenter
    barplot_permut_feat_imp (feature_importance, N, (12,4))

    # calcul des valeurs de SHAP
    shap_explainer = TreeExplainer(model = xgb_clf)
    shap_values = shap_explainer(X_test_prepro)

    # Représentation graphique résumant la distribution des valeurs SHAP
    # en fonction des valeurs de chaque variable
    plot_shap_beeswarm (shap_values, (12,4))

    variables = {"number_of_referrals_bins" : {"legend" : "Nombre de parrainage",
                                            "bins" : ["0", "1", "2-4", ">=5"]},
                "family_size_bins" : {"legend" : "Taille de la famille",
                                    "bins" : ["1","2","2-4", "> 4"]},
                "age_bins" : {"legend" : "Age (ans)",
                            "bins" : ["< 30","30-40","40-50", "50-65", "> 65"]},
                "tenure_in_months_bins" : {"legend" : "Ancienneté (en mois)", 
                                        "bins" : ['0-6 mois', '7-12 mois', '1-2 ans', '2-3 ans',
                                                    '3-4 ans', '4-5 ans', "> 5ans"]},
                "monthly_charge" : {"legend" : "Facturation mensuelle normalisée centrée"}}

    # Visualisation des valeurs SHAP vs les ["number_of_referrals_bins",
    # "family_size_bins, "age_bins", "tenure_in_months", "monthly_charge"]
    # sous la forme d'un scatterplot
    plot_shap_specific_var ( shap_values, 3 , variables)
