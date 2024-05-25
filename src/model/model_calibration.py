# Package pour les tableaux et dataframe
import numpy as np
import pandas as pd

# Package pour la visualisation graphique
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

# Packages scikit-learn pour modélisation
from sklearn.calibration import calibration_curve

from src.data.plot_data_analyses import def_axe



def score_spiegelhalter(y_true, y_pred):
    """fonction calculant le score de Spiegelhalter

    Args:
        y_true : liste de la target
        y_pred : liste des proba prédites correspondantes

    Returns:
        float
    """
    numerateur = np.sum(np.multiply(y_true - y_pred, 1 - 2 * y_pred))
    denominateur = np.sqrt(
        np.sum(
            np.multiply(
                np.multiply(np.power(1 - 2 * y_pred, 2), y_pred), 1 - y_pred
            )
        )
    )
    return numerateur / denominateur



def sklearn_calibration(y_true, y_pred, n_bins=20):
    """fonction de calcul de la courbe de calibration via sklearn

    Args:
        y_true : liste de la target
        y_pred : liste des proba prédites correspondantes
        n_bins : nb d'intervalles

    Returns:
        df : dataframe
    """
    prob_true, prob_pred = calibration_curve(
        y_true, y_pred,
        n_bins=n_bins,
        strategy="quantile"
    )
    df = pd.DataFrame({"prob_pred":prob_pred, "prob_true":prob_true})
    return df



def plot_calibration_curve (cal_df_init, cal_df = None) :
    """fonction qui génère la courbe de calibration issue du (des) dataframes 
    cal_df_init (caf_cal)
    """
    # Création de la figure
    fig, axe = plt.subplots(figsize = (5,5), constrained_layout = True)

    # Création du scatterplot
    sns.scatterplot(ax = axe, data = cal_df_init, x = 'prob_pred',
                    y="prob_true", legend = True, color = "#CC226E")
    if cal_df is not None:
        sns.scatterplot(ax = axe, data = cal_df, x = 'prob_pred',
                        y = "prob_true", legend = True, color = "#226ECC")
    # Ajout de la ligne y = x
    sns.lineplot(ax = axe,  x = [0, 1], y=[0,1], color = "grey",
                 linestyle = '--', alpha = 0.5)

    # Gestion des axis et labels
    axe = def_axe(axe , "Probabilité moyenne",
                  "Proportion de la cible")

    # Création de la légende
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Avant calibration',
               markerfacecolor='#CC226E', markersize=7),
        Line2D([0], [0], marker='o', color='w', label='Après calibration',
               markerfacecolor='#226ECC', markersize=7)
    ]
    if cal_df is not None:
        axe.legend(handles=legend_elements, loc='upper left',
                   borderaxespad=0, frameon=False ,bbox_to_anchor= (0.04, 0.9)
        )
    # Ajout du titre principal
    fig.suptitle('Courbe de calibration.')
    fig.show()
