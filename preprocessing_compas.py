from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def preprocessing_funct_not_enc(df):
    # Elimino colonne inutili
    df = df.drop(columns=[
        'id', 'name', 'first', 'last', 'is_recid', 'compas_screening_date', 'decile_score.1', 
        'c_charge_desc', 'priors_count.1', 'dob', 'age_cat', 'days_b_screening_arrest', 
        'c_jail_in', 'c_jail_out', 'c_case_number', 'c_offense_date', 'c_arrest_date', 
        'c_days_from_compas', 'c_charge_degree', 'r_case_number', 'r_charge_degree', 
        'r_days_from_arrest', 'r_offense_date', 'r_charge_desc', 'r_jail_in', 'r_jail_out', 
        'violent_recid', 'vr_case_number', 'vr_charge_degree', 'vr_offense_date', 
        'vr_charge_desc', 'type_of_assessment', 'screening_date', 'v_type_of_assessment', 
        'v_screening_date', 'in_custody', 'out_custody', 'start', 'end', 'event'
    ], errors='ignore')  # `errors='ignore'` evita errori se qualche colonna non esiste

    # Creazione della colonna `juv_tot`
    df["juv_tot"] = df["juv_fel_count"] + df["juv_misd_count"] + df["juv_other_count"]
    df = df.dropna()

    df=df.drop(columns=["juv_fel_count", "juv_misd_count", "juv_other_count"])
    df.rename(columns={
    "sex": "sex",
    "age": "Age",
    "decile_score": "Recidivism_Risk",
    "priors_count": "Prior_Offenses",
    "is_violent_recid": "Violent_Recidivist",
    "score_text": "Risk_Level",
    "v_decile_score": "Violent_Recidivism_Risk",
    "v_score_text": "Violent_Risk_Level",
    "juv_tot": "Juvenile_Offenses"
    }, inplace=True)
    
    # Divisione iniziale: 60% train, 40% temporaneo
    df_train, df_temp = train_test_split(df, test_size=0.4, shuffle=True, random_state=42, stratify=df["Violent_Recidivist"])

    # Divisione del 40% rimanente in 3 parti uguali (~13.33% ciascuna)
    df_test, df_temp = train_test_split(df_temp, test_size=2/3, shuffle=True, random_state=42, stratify=df_temp["Violent_Recidivist"])
    df_holdout, df_val = train_test_split(df_temp, test_size=0.5, shuffle=True, random_state=42, stratify=df_temp["Violent_Recidivist"])

    
    datasets = [df_train, df_test, df_val, df_holdout]
    #PER AGE

    bins_age = [0, 24, 34, 44, 54, 64, 100]
    labels_age = ['17-24', '25-34', '35-44', '45-54', '55-64', '65-100']

    for df in datasets:
        df['age_group'] = pd.cut(df['Age'], bins=bins_age, labels=labels_age, right=False)
        df.drop(columns=['Age'], inplace=True)

    # Raggruppamento per Prior_Offenses
    bins_prior_offenses = [0, 5, 10, 15, 20, 25, 30, 41, 100]
    labels_prior_offenses = ['0-5', '6-10', '11-15', '16-20', '21-25', '26-30', '31-40', '40+']  

    for df in datasets:
        df['Prior_Offensesgroup'] = pd.cut(df['Prior_Offenses'], bins=bins_prior_offenses, labels=labels_prior_offenses, right=False)
        df.drop(columns=['Prior_Offenses'], inplace=True)
    
    
    

    return df_train, df_test, df_holdout, df_val # Restituisce il DataFrame modificato


def encoding_funct(df_train, df_test, df_holdout, df_val):
    df_train_enc = df_train.copy()
    df_test_enc = df_test.copy()
    df_val_enc = df_val.copy()
    df_holdout_enc = df_holdout.copy()
    
    dataframes = [df_train_enc, df_test_enc, df_val_enc, df_holdout_enc]
    
    le = LabelEncoder()
    
    gender = ['Male', 'Female']
    le.fit(gender)
    for df in dataframes:
            df['sex'] = le.transform(df['sex'])
            
            
    race = ['Caucasian', 'African-American', 'Other', 'Hispanic', 'Asian', 'Native American']
    le.fit(race)
    for df in dataframes:
            df['race'] = le.transform(df['race'])
            
    
    risk = ['Low', 'High', 'Medium']
    le.fit(risk)
    for df in dataframes:
            df['Risk_Level'] = le.transform(df['Risk_Level'])
            
            
    Violent_Risk_Level = ['Low', 'High', 'Medium']
    le.fit(Violent_Risk_Level)
    for df in dataframes:
            df['Violent_Risk_Level'] = le.transform(df['Violent_Risk_Level'])
            
            
            
    
    age = ['25-34', '65-100', '35-44', '45-54', '55-64', '17-24']
    le.fit(age)
    for df in dataframes:
            df['age_group'] = le.transform(df['age_group'])

    
    
    Prior_Offensesgroup = ['0-5', '6-10', '16-20', '11-15', '31-40', '21-25', '26-30', '40+']
    le.fit(Prior_Offensesgroup )
    for df in dataframes:
            df['Prior_Offensesgroup'] = le.transform(df['Prior_Offensesgroup'])
            
    
    return(df_train_enc, df_test_enc, df_holdout_enc, df_val_enc)



def preprocessing_funct_not_enc_SMOTE(df):

    
    #spleet 60 20 20 20 
    df_train, df_val = train_test_split(df, test_size=0.4, shuffle=True, random_state=42, stratify=df["Violent_Recidivist"])
    df_val, df_test = train_test_split(df_val, test_size=0.5, shuffle=True, random_state=42, stratify=df_val["Violent_Recidivist"])

    
    datasets = [df_train, df_test, df_val]
    #PER AGE

    datasets = [df_train, df_test, df_val]
    #PER AGE

    bins_age = [0, 24, 34, 44, 54, 64, 100]
    labels_age = ['17-24', '25-34', '35-44', '45-54', '55-64', '65-100']

    for df in datasets:
        df['age_group'] = pd.cut(df['Age'], bins=bins_age, labels=labels_age, right=False)
        df.drop(columns=['Age'], inplace=True)

    # Raggruppamento per Prior_Offenses
    bins_prior_offenses = [0, 5, 10, 15, 20, 25, 30, 41, 100]
    labels_prior_offenses = ['0-5', '6-10', '11-15', '16-20', '21-25', '26-30', '31-40', '40+']  

    for df in datasets:
        df['Prior_Offensesgroup'] = pd.cut(df['Prior_Offenses'], bins=bins_prior_offenses, labels=labels_prior_offenses, right=False)
        df.drop(columns=['Prior_Offenses'], inplace=True)

    return df_train, df_test,  df_val # Restituisce il DataFrame modificato


def encoding_funct_SMOTE(df_train, df_test, df_val):
    df_train_enc = df_train.copy()
    df_test_enc = df_test.copy()
    df_val_enc = df_val.copy()
   
    
    dataframes = [df_train_enc, df_test_enc]#, df_val_enc]
    
    le = LabelEncoder()
    
    gender = ['Male', 'Female']
    le.fit(gender)
    for df in dataframes:
            df['sex'] = le.transform(df['sex'])
            
            
    race = ['Caucasian', 'African-American', 'Other', 'Hispanic', 'Asian', 'Native American']
    le.fit(race)
    for df in dataframes:
            df['race'] = le.transform(df['race'])
            
    
    risk = ['Low', 'High', 'Medium']
    le.fit(risk)
    for df in dataframes:
            df['Risk_Level'] = le.transform(df['Risk_Level'])
            
            
    Violent_Risk_Level = ['Low', 'High', 'Medium']
    le.fit(Violent_Risk_Level)
    for df in dataframes:
            df['Violent_Risk_Level'] = le.transform(df['Violent_Risk_Level'])
            
            
            
    
    age = ['25-34', '65-100', '35-44', '45-54', '55-64', '17-24']
    le.fit(age)
    for df in dataframes:
            df['age_group'] = le.transform(df['age_group'])

    
    
    Prior_Offensesgroup = ['0-5', '6-10', '16-20', '11-15', '31-40', '21-25', '26-30', '40+']
    le.fit(Prior_Offensesgroup )
    for df in dataframes:
            df['Prior_Offensesgroup'] = le.transform(df['Prior_Offensesgroup'])
            
    
    return(df_train_enc, df_test_enc, df_val_enc)




def metrics_to_compare(y_true, y_pred):
    # Calcolo della matrice di confusione
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Calcolo delle metriche
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Ritorna tutte le metriche e i conteggi di FP e FN
    return accuracy, f1_score, fpr, fnr, fp, fn



def K_subgroups_dataset_and_or(df_pruned, df_holdout, K):
    # Prendi i primi K itemset dal dataframe df_pruned
    itemsets = df_pruned['itemset'].head(K)
    
    # Crea una copia di df_holdout per non modificarlo direttamente
    df_holdout_copy = df_holdout.copy()

    # DataFrame vuoto per mantenere le righe che matchano
    righe_mantenute = pd.DataFrame()

    # Itera su ogni itemset e filtra df_holdout in base alle coppie feature=valore
    for itemset in itemsets:
        if itemset:  # Verifica che l'itemset non sia vuoto
            # Estrai le coppie feature=valore direttamente dall'oggetto frozenset
            condizioni = {feature_val.split('=')[0]: feature_val.split('=')[1] for feature_val in itemset}
            
            # Inizia con l'intero df_holdout_copy e applica il filtro per ogni coppia feature=valore
            df_filtrato = df_holdout_copy
            for feature, valore in condizioni.items():
                df_filtrato = df_filtrato[df_filtrato[feature].astype(str).str.strip() == valore.strip()]

            # Aggiungi le righe trovate al DataFrame righe_mantenute (logica OR tra itemset)
            righe_mantenute = pd.concat([righe_mantenute, df_filtrato])

    # Rimuovi eventuali duplicati se ci sono
    righe_mantenute = righe_mantenute.drop_duplicates()

    return righe_mantenute