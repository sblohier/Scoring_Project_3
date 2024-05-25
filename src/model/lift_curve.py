# Package pour les tableaux et dataframe
import numpy as np
import pandas as pd





def lift_curve_data(y_true, y_pred):
    """fonction de calcul de la courbe de lift
    """

    # Trier le vecteur des labels (y_true) et celui des
    # proba (y_pred) par ordre croissant de la proba :
    # respectivement y_pred et y_test_sorted.
    sorted_score = np.argsort(y_pred)[::-1]

    # Vérifier que y_true et y_pred sont des arrays
    if not isinstance(y_true, np.ndarray):
        y_true = y_true.to_numpy()
    if not isinstance(y_pred, np.ndarray):
        y_pred = y_pred.to_numpy()

    y_true = y_true[sorted_score]
    y_pred = y_pred[sorted_score]

    # Créer un dataframe contenant les deux vecteurs : lift_df
    # Nommer les colonnes en y_pred et y_true
    lift_df = pd.DataFrame({"y_pred": y_pred, "y_true": y_true})

    # Découper le jeu de données en quantile (centile ici) : perc_bins
    lift_df["perc"] = (lift_df.index / lift_df.shape[0]) * 100
    lift_df["perc_bins"] = pd.cut(lift_df["perc"], np.arange(0, 101, 1))

    # Grouper par intervalle et calculer l'effectif
    # des y = 1 dans chaque intervalle : y_test_sum
    lift_df = (lift_df.groupby(["perc_bins"], as_index=False)
               .agg(y_true_sum=("y_true", "sum"))
              )
    # Calculer l'effectif cumulé des y = 1 : y_true_cusum
    lift_df["y_true_cumsum"] = lift_df["y_true_sum"].cumsum()

    # Calculer la concentration (rapport de y_true_cusum par le total des y = 1) : concentration
    lift_df["concentration"] = (lift_df["y_true_cumsum"] / lift_df["y_true_sum"].sum())*100

    # Rajouter un point de coordonnées (0, 0) sur la première ligne.
    # pour la représentation graphique.
    zeros_df = pd.DataFrame(np.zeros((1, 4)), columns=lift_df.columns)
    lift_df = pd.concat([zeros_df, lift_df]).reset_index(drop=True)

    return lift_df



def set_params_lift_curve (fig, w = 700, h = 400) :
    fig.update_layout(
    xaxis = dict(
        showgrid = True,
        constrain = 'domain',
        color  =  "black",
        tickvals = [0, 20, 40, 60, 80, 100]
    ),
    yaxis = dict(
        showgrid = True,
        scaleanchor = "x",
        scaleratio = 1
    ),
    xaxis_title = '% des clients classés par ordre des scores décroissants',
    yaxis_title = '% de la cible',
    margin = dict(l = 0, r = 0, t = 40, b = 0),
    width = w, height = h
    )
    fig.update_layout(
            title = {"text" :"Courbes de lift",'x': 0.5},

                    legend_title_text = 'Modèles : ')
    fig.update_xaxes(
        range = [0, 100],
        rangemode = "tozero"
    )
    fig.update_yaxes(
        range = [0, 100],
        rangemode = "tozero"
    )
    fig.show()
