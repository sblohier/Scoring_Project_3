# Package pour les tableaux et dataframe
import numpy as np
import pandas as pd

# Package pour la visualisation graphique
import matplotlib.pyplot as plt
import seaborn as sns

# Packages scikit-learn pour modélisation
from sklearn.preprocessing import ( OrdinalEncoder,
                                   OneHotEncoder,
                                    StandardScaler
)
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline

from plot_data_analyses import def_axe





def discretization_variables (df : pd.DataFrame, var,
                              bins) -> pd.DataFrame :
    """Permet la discrétisation d'une variable
    ans une plage d'intervalles

    Args:
        df : dataframe
        var : le nom de la variable à discrétiser
         bins : array des intervalles

    Returns:
        df: le dataframe avec la variable initiale + 
        une nouvelle variable nommée var_bins
    """
    bins_ = [df[var].min()] + bins + [df[var].max()+1]
    df[var + "_bins"]=pd.cut(x = df[var],include_lowest = True,right = False,
                        bins = bins_, labels=False)
    return df



def plot_discretized_variable (df : pd.DataFrame, var, bins) :
    """génère la visualisation de la variable continue (var) 
    sous la forme d'un density plot et 
    la  visualisation du résultat de sa discrétisation 
    selon la plage d'intervalles bin

    Args:
        df : dataframe
        var : le nom de la variable à discrétiser
        bins : array des intervalles
    """
    # Création de la figure
    fig, ax = plt.subplots(figsize=(6,4))

    # Définition des intervalles de discrétisation
    bins_ = [df[var].min()] + bins + [df[var].max()+1]

    # Density plot de la variable continue
    sns.kdeplot(ax = ax,data = df, x=var,  alpha=0.6,
                fill=True,  legend=False, cut= 0,
                color = "#CC226E")

    # Histogramme de la variable discrétisée
    ax2 = ax.twinx()
    sns.histplot(ax = ax2, data=df, x=var, bins=bins_,stat = "frequency", color="#226ECC")

    # Définition des axes
    ax = def_axe(ax,f"{var}", 'Distribution')
    ax2 = def_axe(ax,"", "")

    # Définition du titre du graphique
    ax.set_title(f"Distribution de la variable {var}"
                 " continue et discrétisée - set d'entraînement",
                 fontsize=10)
    fig.show()



def preprocessing_for_rfe (X, bool_cols, cat_cols, num_cols) :
    """préprocessing des featuress pour RFE

    Args:
        X : dataframe des variables explicatives
        bool_cols : le nom desvariables booléennes
        cat_cols : le nom des variables catégorielles
        num_cols : le nom des variables numériques

    Returns:
        X_dummies: le dataframe préprocessé
    """
    # On crée une copie de X_train
    X_c = X.copy()

    # Encodage manuel des variables booléennes
    X_c[bool_cols] = X_c[bool_cols].apply(
        lambda x : x.map({"Yes" : 1, "No" : 0}))

    # Encodage one-hot des variables catégorielles textuelles
    X_dummies = pd.get_dummies(X_c,  prefix_sep='__',  columns=cat_cols)

    # Normalisation des variables numériques
    rs = StandardScaler()
    X_dummies[num_cols] = rs.fit_transform(X_dummies[num_cols])
    return X_dummies



def feature_engineering (df, services_) :
    """fonction qui :
     - discrétise les variables :
    'age',"number_of_dependents","number_of_referrals", "tenure_in_months"
     - crée les variables :
     *total_of_services_add ( = somme des services)
     *family_size et la discrétise.

    Args:
        df : dataframe 
        services_ : la liste des services

    Returns:
        df: le dataframe avec les nouvelles variables
    """
    df_=df.copy()
    df_ = discretization_variables (df_, 'age', [30,40,50,65])
    df_["family_size"] = (df_["number_of_dependents"] + 1
                           + df_["married"].map({"Yes" : 1, "No": 0}))
    df_ = discretization_variables(df_, "family_size", [1.5,2.5,4.5])
    df_ = discretization_variables(df_, "number_of_referrals", [1,2,5])
    df_ = discretization_variables(df_, "tenure_in_months", [6,12,24,36,48,60])
    df_["total_of_services_add"] = (df_[services_]=="Yes").sum(1)
    return df_


def split_x_y (df) :
    y = df["churn_value"]
    X = df.drop(columns="churn_value")
    return X, y


def pipeline_trans (json_) :
    # Initialisation des Imputers
    num_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    cat_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

    # Initialisation des encoders
    ordinal_encoder = OrdinalEncoder(categories= [["No", "Yes"]]*(len(json_["bool"])))
    one_hot_encoder = OneHotEncoder()

    # Initialisation du Standard scaler
    scaler = StandardScaler()

    # Création des pipelines de transformation
    cat_ordinal_transformer = make_pipeline(  ordinal_encoder) # variables booléennes
    # variables catégorielles textuelles
    cat_one_hot_encoder_transformer = make_pipeline( cat_imputer,
                                                    one_hot_encoder)
    cat_transformer = make_pipeline( cat_imputer) # variables catégorielles ordinales
    num_transformer = make_pipeline( num_imputer, scaler ) # variable numérique

    # Définition du préprocesseur
    preprocessor = ColumnTransformer(
            transformers=[
                ("cat_ord", cat_ordinal_transformer, json_["bool"]),
                ("cat_oh", cat_one_hot_encoder_transformer,json_["cat"]),
                ("cat_", cat_transformer, json_["cat_num"]),
                ("numerical", num_transformer, json_["num"])
                ]
    )
    return preprocessor
