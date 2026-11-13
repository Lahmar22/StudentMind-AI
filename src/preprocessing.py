import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    OrdinalEncoder
)

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
    df.drop(columns=['Student_ID'], inplace=True)
    deplication = df.duplicated().sum()
    print(f"Nombre de lignes dupliquées : {deplication}")
    if deplication > 0:
        print(f"Nombre de lignes dupliquées : {deplication}. Suppression des doublons.")
        df = df.drop_duplicates()
    
    # values_manquantes = df.isnull().sum()
    # for column, count in values_manquantes.items():
    #     if count > 0:  
    #         # print(f"Valeurs manquantes dans la colonne '{column}' : {count}")
    #         if df[column].dtype in ["int64", "float64"]:
    #             df[column] = df[column].fillna(df[column].mean())
    #         else:
    #             df[column] = df[column].fillna(df[column].mode()[0])
    # values_manquantes = df.isnull().sum()
    # print(values_manquantes)

    # numerical_cols = df.select_dtypes(
    # include=["int64", "float64"]
    # ).columns

    # plt.figure(figsize=(14, 7))

    # sns.boxplot(data=df[numerical_cols])

    # plt.title("Détection des valeurs aberrantes")
    # plt.xticks(rotation=45)
    # plt.ylabel("Valeurs")

    # plt.tight_layout()
    # plt.show()
    
    return df

def visualisation_donnees(df):
    df['Mental_Health_Risk'].value_counts().plot(
        kind='bar',
        figsize=(8, 5)
    )

    plt.title('Distribution de la variable cible')
    plt.xlabel('Classes')
    plt.ylabel('Nombre d étudiants')
    plt.xticks(rotation=0)
    plt.show()

    # Sleep Hours selon le risque mental
    sns.boxplot(data=df, x='Mental_Health_Risk', y='Sleep_Hours')
    plt.title('Sleep Hours selon le risque mental')
    plt.show()

    correlation = df.select_dtypes(include='number').corr()

    plt.figure(figsize=(12, 8))
    sns.heatmap(
        correlation,
        annot=True,
        cmap='coolwarm',
        fmt='.2f'
    )

    plt.title('Matrice de corrélation')
    plt.show()



def pipelineData():
    df = chargement_donnees()
    df = nettoyage_donnees(df)

    X = df.drop('Mental_Health_Risk', axis=1)
    y = df['Mental_Health_Risk']

    numerical_columns = [
        'Age',
        'Year_of_Study',
        'Sleep_Hours',
        'Study_Hours_Per_Week',
        'GPA',
        'Social_Media_Hours',
        'Screen_Time_Hours',
        'Financial_Stress',
        'Family_Support',
        'Attendance_Rate'
    ]
    gender_columns = ['Gender']

    ordinal_columns = [
        'Physical_Activity',
        'Diet_Quality'
    ]

    binary_columns = ['Previous_Mental_Health_History']

    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    gender_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    ordinal_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder(
            categories=[
                ['Low', 'Moderate', 'High'],
                ['Poor', 'Average', 'Good']
            ],
            handle_unknown='use_encoded_value',
            unknown_value=-1
        ))
    ])

    binary_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder(
            categories=[['No', 'Yes']]
        ))
    ])

    preprocessor = ColumnTransformer([
        ('numerical', numerical_pipeline, numerical_columns),
        ('gender', gender_pipeline, gender_columns),
        ('ordinal', ordinal_pipeline, ordinal_columns),
        ('binary', binary_pipeline, binary_columns)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test, preprocessor


# if __name__ == "__main__":
#     df = chargement_donnees()
#     df = nettoyage_donnees(df)

    

    # categorical_columns = df.select_dtypes(
    #     include=['object', 'category']
    # ).columns

    # print(categorical_columns)
    # print(f"x_train shape: {x_train.shape}")
    # print(f"x_test shape: {x_test.shape}")
    # print(f"y_train shape: {y_train.shape}")
    # print(f"y_test shape: {y_test.shape}")

    # analyse_donnees(df)
    # someHours = df['Sleep_Hours']*7 + df['Study_Hours_Per_Week'] + df['Screen_Time_Hours']*7
    # print(someHours.sum()/len(df))