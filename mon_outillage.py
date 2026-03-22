import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.base import clone



def pipeline_nettoyage_modele(df_training):

    # On identifie les colonnes automatiquement
    num_cols = df_training.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = df_training.select_dtypes(include=['object']).columns

    #on gere le nettoyage des chiffres
    num_transformer =Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]) 

    #On gere les textes ou les variables categorielles
    cat_transformer = Pipeline(steps=[
        ('imputer',SimpleImputer(strategy='constant', fill_value='None')),
        ('onehot',OneHotEncoder(handle_unknown='ignore'))
    ])

    #on combine le tout dans un prepocesseur
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_cols),
            ('cat', cat_transformer, cat_cols)
    ])


    #Pipeline nettoyage + modele pour le j'ai choisi Gradient() je le change en fonction du projet
    modele_pipeline = Pipeline(steps=[
        ('preprocessor',preprocessor),
        #('regressor',GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42))
        ('regressor',GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42 ))

    ])

    return modele_pipeline




#Gestion des outliers par la methode IQR

def cleaner_outlier(df, colum):


         # On vérifie donc ici s'il a bien l'attribut 'columns'
    if not hasattr(df, 'columns'):
        return df # Si c'est une Series, on ne fait rien pour éviter le crash
       
    #verifie si la colonne existe dans le dataframe
    if colum not in df.columns:
         print(f"Attention : la colonne {colum} est absente du dataframe")

    Q1 = df[colum].quantile(0.25)
    Q3 = df[colum].quantile(0.75)
    IQR = Q3 - Q1

    #limiter les extremites
    lower_bound = Q1 - 3 * IQR
    upper_bound = Q3 + 3 * IQR

    # On crée un masque : vrai si la valeur est dans les bornes
    masque = (df[colum] >= lower_bound) & (df[colum] <= upper_bound)
    #df_out = df[masque]

    return df[masque].copy()


def compare_modele(X_train,X_test,y_train,y_test, pipeline_base):

    modeles = {
        'ridge':Ridge(),
        'lasso':Lasso(),
        'random forest':RandomForestRegressor(n_estimators=100, random_state=42),
        'gradient boosting':GradientBoostingRegressor(random_state=42)
    }

    #on recupere seulement la partie nettoyage
    preprocessing = pipeline_base.steps[0][1]

    scores = []

    for nom, model_brut in modeles.items():

        #j'utilise le pretraitement que le modele_nettoyage en haut
       
        mon_pipe = make_pipeline(preprocessing, model_brut)

        mon_pipe.fit(X_train, y_train)
        score = mon_pipe.score(X_test, y_test)
        scores.append({'modele':nom, 'Score R2':score})

        
    return pd.DataFrame(scores).sort_values(by="Score R2", ascending=False)
 
