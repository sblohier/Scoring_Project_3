
# Scoring_Project_3

## Description du projet

Ce projet de scoring consiste en la construction d'une modélisation prédictive ayant pour objectif d’identifier les clients fragiles d’une entreprise de télécommunication, à savoir les clients susceptibles de résilier leur contrat prochaînement.

Nous disposons pour ce faire de 2 jeux de données des clients de cette entreprise :
* un jeu de données d’apprentissage ‘train’ (3 fichiers csv) :
  * `df_telco_customer_churn_demographics.csv`,
  * `df_telco_customer_churn_servives.csv`,
  * `df_telco_customer_churn_status.csv`
    
, un jeu de données d’évaluation ‘eval’ (3 fichiers csv)
  * `eval_df_telco_customer_churn_demographics.csv`,
  * `eval_df_telco_customer_churn_servives.csv`,
  * `eval_df_telco_customer_churn_status_no_target.csv`

Ces 2 bases contiennent des indicateurs à la fois socio-démographique, les différents services auxquels les clients ont souscrits, ainsi que des données relatives à leur fidélité et leur facturation. La variable cible, le score de churn nommé '`churn_value`' n’est inclue que dans les données d'entraînement.

Nous procédons à la construction d’un modèle de machine learning permettant de prédire le score de chaque client. Ce modèle est entraîné et calibré par le biais des données “train” et évalué sur la base “eval”. 
La métrique utilisée est l'AUC.

L'analyse préalable concernant la présélection des variables pertinentes est disponible dans ./notebook/Variables_Preselection.ipynb et n'est pas reprise dans `main.py`.
Elle met en évidence qu'il est possible de réduire les variables de 35 à 12 (méthodes utilisées : tri univarié, RFE, Boruta)
La liste des variables retenues est disponible sous /data/variables.json

Les librairies utilisées sont dispo dans `requirements.txt`

## Structure du projet 


```bash
├── data
│   ├── intermediate       <- Data after merging of 3 csv
│   ├── ML_input           <- Data after feature_engineering & discrétisation
│   ├── output             <- Evaluation of database "evl"
│   └── raw            <- ( 2 * 3 csv)
│
├── docs               <- A default mkdocs project; see mkdocs.org for details
│
├── models             <- unused
│
├── notebooks          <- Jupyter notebooks.
|                         Variables_Preselection.ipynb
|                         Modélisation_Evaluation.ipynb
│
├── reports            <- unused
│   └── figures        <- unused
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
└── main                <- Source code for use in this project.
    │
    ├── __init__.py    <- Makes scoring_project a Python module
    │
    ├── data           <- Scripts to generate data and preprocess
    │   └── data_preprocessing.py
    │   └── importation_data.py
    │   └── plot_data_analyses.py
    │
    ├── explicatibilite       <- Scripts for features importance in predictions
    │   └── interpretability.py
    │
    ├── model           <- Scripts to setup, run and evaluation model
    │   └── lift_curve.py
    │   └── model_calibration.py
    │   └── modelisation.py
```


    

