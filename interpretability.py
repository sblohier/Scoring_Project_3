# Package pour les tableaux et dataframe
import numpy as np
import pandas as pd

# Package pour la visualisation graphique
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.inspection import permutation_importance

# Package SHAP pour interprétabilité
from shap.plots import (scatter, beeswarm)



def do_permutation_feat_imp(estimator, X, y, n_iter, seed) :
    """Calcul de l'importance des features via 
    la Permutation feature importance

    Args:
        estimator : L'estimateur entrainé'.
        X : le dataframe des features préprocessées
        y : la target (dataseries).
        n_iter, seed

    Returns:
        feature_importance -> dataframe.
    """
    # Calcul de l'importance des features via la Permutation feature importance
    model_fi = permutation_importance(estimator.named_steps.classifier,
                                    X, y, scoring="roc_auc",
                                    n_repeats= n_iter, random_state=seed)

    # Création d'un dataframe avec les résultats
    feature_importance= pd.DataFrame({"names" : X.columns.tolist(),
                                    "importance" : np.round(model_fi['importances_mean'],4),
                                    "importance_sstd" : np.round(model_fi['importances_std'],4), })
    # Affichage des résultats
    print(feature_importance.sort_values(by="importance",ascending=False))
    return feature_importance



# Création d'un barplot affichant les 12 plus importantes features selon la méthode
# Permutation Feature Importance
def barplot_permut_feat_imp (df, n, figsize_):
    """Création d'un barplot affichant les 12 plus importantes features
        selon la méthode Permutation Feature Importance

    Args:
        estimator : L'estimateur entrainé'.
        n : les n 1eres features classées par ordre descendant d'importance
        y : la target (dataseries).
        figsize_

    """
    # Initialisation de la figure
    fig, ax = plt.subplots(figsize=figsize_)

    # barplot des 12 1eres features classées selon leur importance
    sns.barplot(x="importance", y="names",
                data=df.sort_values(by = "importance", ascending = False).head(n=n),
                color="b")

    # Gestion des axes
    ax.set_xlim(0, 0.12)
    ax.set_xlabel("Importance des variables", fontsize = 8)
    ax.set_ylabel("")

    # Ajout des lignes horizontales
    ax.grid(visible=True, which='major', axis='x')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)

    # Ajout du titre principal
    fig.suptitle(f'Importance des {n} variables principales.')
    fig.show()



def plot_shap_beeswarm (shap_values, figsize_ = (6,4)) :
    plt.subplots( figsize = figsize_)
    beeswarm(shap_values, plot_size=figsize_)
    plt.show()



def plot_shap_specific_var (shap_values, nbcols, cols_list, figsize_ = (12,6)) :
    """ Visualisation des valeurs SHAP vs les ["number_of_referrals_bins", 
    "family_size_bins, "age_bins", "tenure_in_months", "monthly_charge"]
    sous la forme d'un scatterplot

    Args:
        shap_values : les valeurs SHAP calculées
        nbcols : le nombre d'axe de la figure
        cols_list : liste des variables à représenter
        figsize_

    """
    nbrows = (np.ceil(len(cols_list)/nbcols)).astype(int)
    fig, ax1 = plt.subplots(nbcols, nbrows ,figsize=(figsize_))
    (i,j) = (0,0)
    for var in cols_list :
        if j == 3:
            j = 0
            i += 1
        # Créez la figure et les axes à l'avance
        if var == "monthly_charge" :
            bins = None
        else :
            bins = cols_list[var]["bins"]
        scatter(shap_values[:, var],ax =ax1[i,j], show =False)

        # # Modification des étiquettes des axes
        ax1[i,j].set_xlabel(cols_list[var]["legend"], fontsize = 9)
        ax1[i,j].set_ylabel('Valeur SHAP', fontsize = 9)
        if bin is not None :
            custom_labels = bins
            ax1[i,j].set_xticks(range(len(custom_labels)))
            ax1[i,j].set_xticklabels(custom_labels)
        if var == "tenure_in_months_bins" :
            ax1[i,j].set_xticklabels(custom_labels, rotation=45)
        j += 1
    ax1[i,j].set_axis_off()
    plt.show()
