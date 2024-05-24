# Package pour les expressions régulières
import re

# Package manipuler les fichiers et répertoires
import os
from pathlib import Path
import shutil

# Package pour les tableaux et dataframe
import numpy as np
import pandas as pd
from pandas.errors import MergeError



def create_dir(path, dirname) :
    """Cette fonction permet de créer un sous-dossier
    s'i n'existe pas

    Args:
        path : le chemin du dossier dans lequel générer un ss-dossier
        dirname : le nom du ss-dossier à générer

    Returns:
        text: Le chemin du sous dossier créé.
    """
    if not os.path.exists(path + '/' + dirname):
        os.mkdir(path + '/' + dirname)
        print(f"FOLDER ./{os.path.relpath(path + '/' + dirname)} CREATED")
    return path + '/' + dirname


def create_dir_tree () :
    """Création d'une arborescence de dossier : 
        ./data/raw/train
                /eval
        ./data/intermediate/train
                        /eval
        ./data/ML_input/train
                    /eval
    Returns:
        Array (text): chemin des 3 sous dossiers de ./data.
    """
    path_datadir_ = create_dir(os.getcwd(),"data")
    paths = []
    paths.append(path_datadir_)
    for name in ["raw", "intermediate", "ML_input"] :
        path_namedir = create_dir(path_datadir_,name)
        create_dir(path_namedir,"train")
        create_dir(path_namedir,"eval")
        paths.append(path_namedir)
    path_namedir = create_dir(path_datadir_,"output")
    return paths


def move_csv (path_datadir_) :
    """ Copie les fichiers *telco_customer*.csv dans :
         ./data/raw/train
                   /eval   
    Args:
        path : le chemin du dossier 'data
    """
    for path in Path('./').rglob('*telco_customer*.csv'):
        rel_path= os.path.relpath(path)
        new_path = os.path.relpath(path_datadir_ + '/raw/eval/' + path.name)
        if rel_path.find("eval") == -1 :
            new_path =   os.path.relpath(path_datadir_ + '/raw/train/' + path.name)
        if (rel_path == new_path) | (Path(new_path).is_file()):
            print(rel_path)
            print("Files already in place")
        else :
            shutil.copy(rel_path,new_path )
            print(f"FILE {path.name} \n---> COPIED TO {new_path}")



def read_merge_date (path_raw_data_, path_inter_data_) :
    """ Récupère le path du dossier contenant les csv,
        Pour chaque fichier :
         * Création d'un dataframe
         * Fusion avec le dataframe prédédent sur la base de "customer_id"
        exporte le dataframe final 
    Args:
        path_raw_data_ : le chemin du dossier où sont contenus les csv
        path_inter_data_ : le chemin du dossier où exporter le résultat 
        de la fusion
    
    Returns:
        df : dataframe des données fusionnées.
    """
    found = False
    has_dup = False
    print(f"Le dossier {path_raw_data_}"
          f" contient {len(os.listdir(path_raw_data_))}"
          f" fichiers à fusionner : ")
    print(f"{'-'*80}")
    for file in os.listdir(path_raw_data_):
        if re.search('.csv', file):
            print(f" * {file}")
            df = pd.read_csv(path_raw_data_+file, sep=',',
                               keep_default_na=False,na_values=['NaN'])
            print(f"    --> contient {df.shape[0]} lignes et {df.shape[1]} variables.")
            if found is True  :
                merge_on_key = df_2.columns.intersection(df.columns)
                try :
                    df = pd.merge(df_2, df, how='inner', on= merge_on_key[0],
                                  validate='one_to_one', sort=False)
                except MergeError :
                    # Le fait de faire un merge outer vient aggreger
                    # tous les id des clients,
                    # s'il y a des doublons au niveau de l'ID alors validate signalera une erreur
                    has_dup = True
                    break
            df_2 = df.copy()
            found = True
    if has_dup == False :
        print(f"Le df issu du merging contient {df.shape[0]} lignes et {df.shape[1]} variables.")
        dirup = os.path.dirname(os.path.dirname(os.path.dirname(path_raw_data_)))
        print(dirup)
        df.to_csv(path_inter_data_+"/" + path_raw_data_.split("/")[-2]+'/df_telco.csv', index=False)
        return df
    else :
        return print("Erreur pendant la fusion des dataframes, présence de doublons.")
    


def get_cols_by_type(json_obj) :
    """ 
    Args:
        json_obj :  json contenant les features

    Returns:
        array (3 arrays) : contenant respectivement les 3 types de variables
        selon l'ordre : "bool", "cat", "num"
    """
    variables_type = {"bool" : [], "cat" : [], "num" : []}
    for key in json_obj.keys():
        for var_ in json_obj[key].keys():
            variables_type[var] += json_obj[key][var]
    return variables_type

