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



def feature_engineering (df) :
    """fonction qui :
     - discrétise les variables :
    'age',"number_of_dependents","number_of_referrals", "tenure_in_months"
     - crée les variables :
     *total_of_services_add ( = somme des services)
     *family_size et la discrétise.

    Args:
        df : dataframe 

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
    services =  ["device_protection_plan","phone_service",
                 "multiple_lines", "internet_service",
                 "online_security", "online_backup",
                 "premium_tech_support", "unlimited_data"]

    df_["total_of_services_add"] = (df_[services]=="Yes").sum(1)
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
