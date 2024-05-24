# Package pour la gestion des objets json
import json

# Package pour les tableaux et dataframe
import numpy as np
import pandas as pd

# Package pour analyse statistique
import scipy.stats as ss

# Packages scikit-learn pour modélisation
from sklearn.model_selection import train_test_split

from importation_data import (
    create_dir_tree,
    move_csv, read_merge_date,
    get_cols_by_type
)

from plot_data_analyses import (
    boxplot_num_cols,
    barplot_cat_bivarie,
    plot_heatmap_cor
)

from data_preprocessing import (
    preprocessing_for_rfe,
    feature_engineering
)

from data_preselection import (
    compute_cramer_v, cramer_v_coeff,
    plot_score_vs_nb_features,
    rfe_evaluate, get_features_names
)


# Définition de la SEED
SEED = 42

# Création de l'arborescence de fichiers destinée à stocker les différents csv
[PATH_DATA_DIR, PATH_RAW_DIR, PATH_INTER_DIR, PATH_ML] = create_dir_tree()
move_csv(PATH_DATA_DIR)

# Merging des 3 fichiers csv contenu dans le dossier  ./data/raw/train/
df = read_merge_date(PATH_RAW_DIR + "/train/", PATH_INTER_DIR)
df_eval = read_merge_date(PATH_RAW_DIR + "/eval/", PATH_INTER_DIR)

# Séparation du df d'entraînement en df_train et df_test
df_train, df_test = train_test_split(df, test_size=0.2, random_state=SEED)

# Affichage du nombre d'observations des 2 dataframes obtenus
print(f"Le set de train contient {df_train.shape[0]} observations.")
print(f"Le set de test contient {df_test.shape[0]} observations.")

# Export de ces 2 dataframes dans des fichiers csv
df_train.to_csv(  PATH_INTER_DIR + '/train/df_train.csv', index = False)
df_test.to_csv( PATH_INTER_DIR + '/train/df_test.csv', index = False)

# Chargement du json contenant la liste des variables triées par thématiques / type
with open("variables.json", "r") as read_file:
    json_ = json.load(read_file)

# Affichage de l'objet json_
print(json.dumps(json_, indent=4, ensure_ascii=False))

# listes des variables booléennes, catégorielles et numériques
variables_type = get_cols_by_type( json_)
bool_cols = variables_type["bool"]
cat_cols = variables_type["cat"]
num_cols = variables_type["num"]
all_cols = bool_cols + cat_cols + num_cols

# Création de la liste des variables désignant les services
# (variables booléennes)
services = json_["additional services"]["bool"]

# Définition des variables numériques à représenter
cols_to_miss = ["churn_value", "total_refunds",
                "total_extra_data_charges",
                "total_long_distance_charges"]
num_cols_ = [elt for elt in num_cols if elt not in cols_to_miss]
boxplot_num_cols (df_train, num_cols_ )
bool_cols.remove("churn_value")
barplot_cat_bivarie(df_train, cat_cols + bool_cols)


df_train = feature_engineering(df_train, services)
# colonnes à supprimer
num_to_del = ["age","number_of_dependents","number_of_referrals",
               "tenure_in_months","total_charges",
               "total_long_distance_charges","total_revenue"]
bool_to_del = ["senior_citizen", "under_30", "dependents", "referred_a_friend"]
for i in num_to_del :
    num_cols.remove(i)
for i in bool_to_del :
    bool_cols.remove(i)


# Nous définissons une liste stockant les variables numériques discrétisées
# Note : ces variables sont des variables catégorielles numériques.
cat_num_cols = ["age_bins", "family_size_bins", "number_of_referrals_bins",
                "total_of_services_add",	"tenure_in_months_bins"]


# Calculons la matrice de Cramer pour les variables catégorielles et booléennes
cramer_matrix = compute_cramer_v(df_train[cat_cols + cat_num_cols + bool_cols + ["churn_value"]])


cramer_test = []
# On évalue de coefficient de Cramer
# et critère de significativité pour chaque couple variable - cible
for i,var in enumerate(cat_cols + cat_num_cols + bool_cols) :
    cramer_results  =  cramer_v_coeff(x=df_train.loc[:,var],
                    y=df_train.loc[:,"churn_value"])
    cramer_test.append(
        {
            'features' : var,
            'cramer_v' : np.round(cramer_results[0],2),
            'p-value' : np.round(cramer_results[1],2)
        })

# Stockage des résultats pour chaque variable dans un dataframe
df_v_cramer = pd.DataFrame(cramer_test)
print(f"CRAMER : {df_v_cramer}")


for var in df_v_cramer.loc[df_v_cramer["cramer_v"]<0.09,"features"].values :
    if var in cat_cols :
        cat_cols.remove(var)
    else :
        bool_cols.remove(var)

# Calcul des corrélations de Pearson
df_pearson = df_train[num_cols + ["churn_value"]].corr()

# Visualisation de la matrice de Pearson
plot_heatmap_cor (df_pearson, size = (5,5))


# Construire un dataframe contenant à la fois les coefficients
# de Pearson ( + leur valeur absolue) et la p-value associés à chaque variable quantitative
pearson_matrix = []
for col in num_cols :
    pearson_values =ss.pearsonr(x=df_train[col], y=df_train["churn_value"])
    pearson_matrix.append(
        {
            'features': col,
            'coef_pearson': round(pearson_values[0],3),
            'abs_coef_pearson': round(abs(pearson_values[0]),3),
            'p_value':  round(pearson_values[1],4)
        }
    )

# Stockage des résultats pour chaque variable dans un dataframe
pearson_matrix = (pd.DataFrame(pearson_matrix)
                 .sort_values(["coef_pearson"])
                 .reset_index(drop=True)
                )

# Supprimons ces 5 variables numériques
for var in pearson_matrix.loc[pearson_matrix["abs_coef_pearson"]<0.09,"features"].values :
    num_cols.remove(var)


# Séparation des features explicatives de la target en X_train et y_train
X_train = df_train[bool_cols + cat_cols + cat_num_cols + num_cols]
y_train = df_train["churn_value"]

print(f"A l'issue du tri univarié, il reste {X_train.shape[1]} variables explicatives :\n"
      f"{X_train.columns.tolist()}")





X_train_dummies = preprocessing_for_rfe (X_train, bool_cols, cat_cols, num_cols)


# Définition du sélecteur RFECV, on définit un minimum de 8 variables à sélectionner
rfe_features, df_rfe = rfe_evaluate (X_train_dummies, y_train, 8,SEED)
plot_score_vs_nb_features (df_rfe, rfe_features)

# Récupération des noms des variables avant encodage :
rfe_features_ = get_features_names(rfe_features)
print(f"Après processing des variables encodées selon un one-hot-encoder, "
      f"{len(rfe_features_)} features sur {X_train.shape[1]} "
      f"sont retenues: \n{rfe_features_}")


# json des variables à exporter
json_var = {
    "bool": ["online_security", "paperless_billing"],
    "cat_num": ["age_bins","family_size_bins","number_of_referrals_bins",
                "tenure_in_months_bins","total_of_services_add"],
    "cat": ["contract", "internet_type", "offer", "payment_method"],
    "num": ["monthly_charge"],
}

# Export du json dans variable_types.json
path_json_output = PATH_DATA_DIR + "/ML_input" + "/variable_types.json"
with open(path_json_output, "w") as fichier:
    json.dump(json_var, fichier, indent=4)

# On simplifie le dataframe en ne conservant que les features sélectionnées +  la target
df_train = df_train[rfe_features_ + ["churn_value"]]
# Export de df_train dans un csv
df_train.to_csv(PATH_DATA_DIR + '/ML_input/train/df_train.csv', index=False)

# On applique les discrétisations et features engineering
df_test = feature_engineering(df_test, services)
df_eval = feature_engineering(df_eval, services)
# On simplifie le dataframe en ne conservant que les features sélectionnées +  la target
df_test = df_test[rfe_features_ + ["churn_value"]]
df_eval = df_eval[rfe_features_ ]
# Export de df_test et df_eval dans un csv
df_test.to_csv(PATH_ML + '/train/df_test.csv', index=False)
df_eval.to_csv(PATH_ML + '/eval/df_eval.csv', index=False)
