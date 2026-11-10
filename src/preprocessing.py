import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def chargement_donnees():  
    chemin_fichier = "data/student_mental_health.csv"
    try:
        df = pd.read_csv(chemin_fichier)
        return df
    except FileNotFoundError:
        print(f"Le fichier {chemin_fichier} n'a pas été trouvé.")

def analyse_donnees(df):
    print("Aperçu des données :")
    print(df.head())
    print("\nInformations sur les colonnes :")
    print(df.info())
    print("\nStatistiques descriptives :")
    print(df.describe())
    print("\nTypes de données :")
    print(df.dtypes)
    print("\nShape du DataFrame :")
    print(df.shape)

def nettoyage_donnees(df):
    deplication = df.duplicated().sum()
    print(f"Nombre de lignes dupliquées : {deplication}")
    if deplication > 0:
        print(f"Nombre de lignes dupliquées : {deplication}. Suppression des doublons.")
        df = df.drop_duplicates()

    values_manquantes = df.isnull().sum()
    for column, count in values_manquantes.items():
        if count > 0:  
            print(f"Valeurs manquantes dans la colonne '{column}' : {count}")
            if df[column].dtype in ["int64", "float64"]:
                df[column] = df[column].fillna(df[column].mean())
            else:
                df[column] = df[column].fillna(df[column].mode()[0])
    # values_manquantes = df.isnull().sum()
    # print(values_manquantes)



    numerical_cols = df.select_dtypes(
    include=["int64", "float64"]
    ).columns

    plt.figure(figsize=(14, 7))

    sns.boxplot(data=df[numerical_cols])

    plt.title("Détection des valeurs aberrantes")
    plt.xticks(rotation=45)
    plt.ylabel("Valeurs")

    plt.tight_layout()
    plt.show()
    
    return df

if __name__ == "__main__":
    df = chargement_donnees()
    df = nettoyage_donnees(df)

    # analyse_donnees(df)
    someHours = df['Sleep_Hours']*7 + df['Study_Hours_Per_Week'] + df['Social_Media_Hours']*7 + df['Screen_Time_Hours']*7
    print(someHours.sum()/len(df))