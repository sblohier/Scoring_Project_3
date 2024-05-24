# Package pour les tableaux et dataframe
import numpy as np
import pandas as pd

# Package pour la visualisation graphique
import matplotlib.pyplot as plt
import seaborn as sns




def def_axe(axe , xlabel, ylabel) :
    # Gestion des axis et labels
    axe.set_xlabel(xlabel)
    axe.set_ylabel(ylabel)

    # Ajout des lignes horizontales
    axe.grid(visible=True, which='major', axis='y')
    axe.spines['top'].set_visible(False)
    axe.spines['right'].set_visible(False)
    axe.spines['bottom'].set_visible(True)
    axe.spines['left'].set_visible(False)
    return axe



def boxplot_num_cols (df, cols, ncols = 5 ):
    """ Boxplots des variables numériques cols
    Args:
        df : dataframe des features
        cols : les variables numériques
        ncols : pour la figure, nombre de plots par ligne
    """
    ncols = 5
    nrows = (np.ceil(len(cols)/ncols)).astype(int)
    palette = {'1': '#CC226E', '0': '#226ECC'}

    # Création de la figure
    fig, ax = plt.subplots(ncols= ncols, nrows = nrows, figsize=(4*ncols, 4*nrows))

    (i , j) = (0, 0)
    for var in cols :
        if j == ncols :
            i += 1
            j = 0

        # Création du boxplot
        sns.boxplot(ax=ax[i,j], data=df, y=var, x="churn_value",
                width = 0.8, flierprops={"marker": ".", "markersize" : "4"} ,
                palette = palette)
        ax[i,j] = def_axe(ax[i,j] , '', var.replace("_", " ").title())
        j += 1

    # Suppression de la grille pour le dernier ax
    if j<5 :
        ax[i,j].set_axis_off() #

    # Ajout du titre
    fig.suptitle("Distribution des clients"
                "en fonction des variables numériques, "
                "les clients fragiles sont représentés en rouge.",
                fontsize=15,  y=1.01)
    fig.show()


# Pour chaque variable booléenne et catégorielle :
# --> Création de 2 barplots :
#      -> Gauche : pourcentage absolue de chaque catégorie déclinée selon l'indicateur churn_value
#      -> Droite  : Le pourcentage des clients fragiles est normalisé au sein de chaque catégorie
def barplot_cat_bivarie (df, cols):
    """ Visualisation : 
    pour chaque variable de la liste cols : création de 2 barplots: 
         -> Gauche : pourcentage absolue de chaque catégorie 
         déclinée selon l'indicateur churn_value
         -> Droite  : Le pourcentage des clients fragiles est 
         normalisé au sein de chaque catégorie

    Args:
        df : dataframe des features
        cols : les variables catégorielles
    """
    # Création de la figure
    fig, ax = plt.subplots(ncols= 2, nrows = len(cols), figsize=( 5, 3*len(cols)))
    plt.subplots_adjust(hspace=0.6, wspace=0.6) # espace entre les axes

    #Définition de la palette et de la légende
    palette = ['#226ECC', '#CC226E']

    # Pour chaque variable
    for i, var in enumerate(cols) :

        # Création du barplot des poucentages absolus de chaque  couple (catégorie-churn)
        df_plot = (np.round(df.value_counts(subset=[var,"churn_value"],
                                            normalize=True)*100,1)).reset_index()
        sns.barplot(ax=ax[i,0], y=df_plot["proportion"], x=var, hue = "churn_value",
                    data= df_plot , palette = palette)

        # Création du barplot des poucentages normalisés des clients fragiles pour chaque catégorie
        df_plot = np.round((pd.crosstab(index=df[var], columns=df["churn_value"],
                                        normalize="index")*100),2)
        df_plot = df_plot[1].reset_index()
        df_plot.columns = [var, 'percentage_churn']
        sns.barplot(ax=ax[i,1], data = df_plot, x=var, y= 'percentage_churn', color = "#CC226E")

        ax[i, 0].get_legend().remove()
        #  Définition des axis et labels
        for ax1 in ax[i, :]:
            ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, fontsize= 7)
            ax1 = def_axe(ax1, var, "%")
    
    # Ajout de la légende à la figure
    handles = ax[0, 0].get_legend_handles_labels()[0]
    fig.legend(handles, ['No', 'Yes'] , loc='upper right',
            bbox_to_anchor= (1.15, 1), borderaxespad=0,
            frameon=False, title = "Churn",fontsize=9)

    # Ajouter du titre
    fig.tight_layout()
    fig.suptitle("Répartition des clients fragiles"
                " en fonction des variables catégorielles,\n"
                " à droite les pourcentages sont normalisés par modalité.",
                fontsize=9,  y=1.01)
    fig.show()



# fonction pour visualiser une matrice de liaison
def plot_heatmap_cor (cor_matrix : pd.DataFrame, size = (15,15)):
    """ fonction pour visualiser une matrice de liaison
    """
    # Visualisation de la matrice de V de cramer_matrix
    mask = np.triu(np.ones_like(cor_matrix))
    fig, ax = plt.subplots(figsize = size)
    sns.heatmap(cor_matrix, annot=True, fmt=".2f",
                mask= mask, vmax=1, vmin=0, center=0.5)
    ax.set(xlabel="", ylabel="")
    return fig
