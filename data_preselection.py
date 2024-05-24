# Package pour les expressions régulières
import re

# Typing des fonctions
from typing import List

# Package pour les tableaux et dataframe
import numpy as np
import pandas as pd

# Package pour analyse statistique
import scipy.stats as ss

# Package pour la visualisation graphique
import matplotlib.pyplot as plt
import seaborn as sns

# Packages scikit-learn pour modélisation
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier

from plot_data_analyses import def_axe





# calcul V de Cramer pour deux variables
def cramer_v_coeff(x: List, y: List) -> float:
    """Cette fonction permet de calculer le V de
    Cramer entre deux varaibles catégorielles.

    Args:
        x : Le vecteur de variable x.
        y : Le vecteur de variable y.

    Returns:
        float: La valeur V de cramer.
    """
    # Suppression des NAs
    complete_cases = x.isna()*y.isna()
    x = x[~complete_cases]
    y = y[~complete_cases]

    # Calcul du Khi-deux max (dénomimateur du V de Cramer)
    n = len(x)
    khi2_max = n * min(len(x.value_counts()), len(y.value_counts())) - 1

    # Calcul du khi-deux (numérateur du V de Cramer)
    conf_matrix = pd.crosstab(x, y)
    khi2 = ss.chi2_contingency(observed=conf_matrix, correction=True)

    # Calcul V de Cramer et récupération p_value associée
    cramer = round(np.sqrt(khi2[0] / khi2_max), 4)
    p_value = khi2[1]

    return cramer, p_value



# calcul V de Cramer pour un dataframe
def compute_cramer_v(data: pd.DataFrame) -> pd.DataFrame:
    """Calculer le V de cramer pour un dataframe.
    Args:
        data: Jeu de données sur lequel on souhaite
        calculer le V de Cramer.

    Returns:
        DataFrame contenant les différents V de Cramer.
    """
    ncols = data.shape[1]
    cols = data.columns
    cramer_matrix = np.eye(ncols)
    for j in range(ncols - 1):
        for i in range(j + 1, ncols):
            cramer_matrix[[i, j], [j, i]] = cramer_v_coeff(
                x=data.iloc[:, j],
                y=data.iloc[:, i]
            )[0]
    cramer_matrix = pd.DataFrame(cramer_matrix, columns=cols, index=cols)
    return cramer_matrix



def get_features_names (one_hot_encoder_features) :
    """ Permet de récupérer la liste des variables racines dont sont issues les features 
    one_hot_encoder_features
    Args:
        one_hot_encoder_features : liste des features encodées 

    Returns:
        liste des features racines
    """
    features_filter = []
    for var_ in one_hot_encoder_features :
        search_col = re.search("(.+)__", var_)
        if search_col :
            features_filter.append(search_col.group(1))
        else :
            features_filter.append(var_)
    return np.unique(features_filter).tolist()



def rfe_evaluate (X, y, min_features,seed):
    """ fonction qui :
    - entraine RFECV sur X, y
    - récupère la liste des features sélectionnées
    Args:
        X : dataframe des features
        y : dataseries de la target
        min_featuress : le nb de features minimum à sélectionner
        seed

    Returns:
        rfe_features : la liste des features sélectionnées par la RFE
        df_rfe : le dataframe contenant les scores en fonction du 
        nombre de features sélectionnées
    
    """
    # Définition du sélecteur RFECV, on définit un minimum de 8 variables à sélectionner
    rfe_selector = RFECV(RandomForestClassifier(random_state=seed),
                        step=1, min_features_to_select=min_features, cv=5,
                        scoring='roc_auc', verbose=0, n_jobs=-1)

    # entraînement du sélecteur
    rfe_selector.fit(X, y)

    # On récupère les features sélectionnées par la RFE
    rfe_features = X.loc[:,rfe_selector.support_].columns.tolist()
    print(f"RFE : {len(rfe_features)} variables sélectionnées sur {X.shape[1]} : \n"
        f"{rfe_features}")

    # On génère un dataframe contenant les scores en fonction du nombre de features sélectionnées
    df_rfe = pd.DataFrame (data = {
        'nb_features' : range(8,X.shape[1]+1),
        'mean_AUC' : rfe_selector.cv_results_["mean_test_score"]
        })
    return rfe_features, df_rfe



def plot_score_vs_nb_features (df, selected_features) :
    """ fonction qui génère un lineplot de
    l'AUC moyen en fonction du nombre de features

    Args:
        df : dataframe des features
        selected_features : la liste des features sélectionnées par la RFE
        min_featuress : le nb de features minimum à sélectionner

    """
    # Création du graphique de l'AUC moyen en fonction du nombre de features
    fig, ax = plt.subplots(figsize=(12, 4))
    sns.lineplot(data=df, x="nb_features", y="mean_AUC")

    # Ajout de la ligne verticale rouge pour n = 13
    ax.axvline(x=len(selected_features),  color='red', linestyle='--')
    ax.text( len(selected_features) -0.3,  0.99,
            f"n = {len(selected_features)}", color='r', ha='right',
            va='top', rotation=90,
            transform=ax.get_xaxis_transform())
    # Définition des axes
    ax = def_axe(ax, "Nombre de variables", 'AUC moyen')
    ax.set_xlim(8,28)
    xticks = range(8, 29, 2)  # Intervalles de 2, par exemple
    ax.set_xticks(xticks)
    fig.show()
