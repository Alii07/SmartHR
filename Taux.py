
import streamlit as st
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np
from tensorflow.keras.models import load_model as keras_load_model  # Renommer l'import
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import io
import sklearn
import joblib
import pickle
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_array
from sklearn.preprocessing import LabelEncoder

class Taux:
    def __init__(self, models_info, versement_mobilite):
        """
        Initialise les informations sur les modèles et les règles de versement mobilité.
        """
        self.models_info = models_info
        self.versement_mobilite = versement_mobilite
        self.loaded_models = {}

    def apply_versement_conditions(self, df):
        required_columns = ['Versement Mobilite Taux', 'Effectif', 'Code Insee', 'Versement mobilite Base']

        # Ajouter les colonnes manquantes avec des valeurs par défaut
        for col in required_columns:
            if col not in df.columns:
                df[col] = 0

        df['Versement Mobilite Taux Avant'] = df['Versement Mobilite Taux']
        df['Versement Mobilite Taux'] = df.apply(
            lambda row: 0 if row['Effectif'] < 11 else self.versement_mobilite.get(row['Code Insee'], row['Versement Mobilite Taux']),
            axis=1
        )
        df['Versement Mobilite Montant Pat. Calcule'] = (df['Versement mobilite Base'] * df['Versement Mobilite Taux'] / 100).round(2)
        df['Mobilite_Montant_Anomalie'] = df['Versement Mobilite Montant Pat.'] != df['Versement Mobilite Montant Pat. Calcule']
        df['Mobilite_Anomalie'] = (df['Versement Mobilite Taux Avant'] != df['Versement Mobilite Taux']) | df['Mobilite_Montant_Anomalie']
        df['mobilite Fraud'] = df['Mobilite_Anomalie'].astype(int)

        return df


    def encode_categorical_columns(self, df, categorical_cols):
        if not categorical_cols:
            return df  # Si aucune colonne catégorielle, retourner le DataFrame original

        existing_cols = [col for col in categorical_cols if col in df.columns]
        if not existing_cols:
            return df  # Si aucune des colonnes spécifiées n'existe dans le DataFrame, retourner le DataFrame original

        encoder = OneHotEncoder(drop='first', sparse_output=False)
        encoded_cols = encoder.fit_transform(df[existing_cols])
        encoded_df = pd.DataFrame(encoded_cols, columns=encoder.get_feature_names_out(existing_cols), index=df.index)

        df = df.drop(columns=existing_cols)
        df = pd.concat([df, encoded_df], axis=1)
        return df



    def load_model(self, model_info):
        model_path = model_info.get('model')
        
        if not model_path:
            raise ValueError(f"Le chemin du modèle est vide pour le modèle {model_info}")

        try:
            model = joblib.load(model_path)
            print(f"Modèle chargé depuis {model_path}: {type(model)}")
            
            # Vérifiez que l'objet chargé a une méthode `predict`
            if hasattr(model, 'predict'):
                return model
            else:
                return model                
        except FileNotFoundError:
            raise ValueError(f"Le fichier de modèle spécifié à {model_path} est introuvable.")
        except Exception as e:
            raise ValueError(f"Erreur lors du chargement du modèle depuis {model_path}: {e}")



    def process_7001(self, df, model_name, info, anomalies_report, model_anomalies):
        """
        Traite les anomalies pour le modèle 7001 en utilisant IsolationForest et StandardScaler,
        en respectant la logique de `test_model_7001`.
        """
        try:
            # Charger le modèle
            model_dict = self.load_model(info)
            if model_dict is None:
                print(f"Erreur : Le modèle {model_name} n'a pas été chargé correctement.")
                return
            df['7001Taux 2'] = df['7001Taux 2'].fillna(0)

            # Vérifier les composants du modèle
            iso_forest = model_dict.get('iso_forest', None)
            scaler = model_dict.get('scaler', None)

            if iso_forest is None or scaler is None:
                print(f"Erreur : Le modèle {model_name} est incomplet (iso_forest ou scaler manquant).")
                return

            # Colonne cible à traiter
            colonne = '7001Taux 2'
            if colonne not in df.columns:
                print(f"Erreur : La colonne {colonne} est absente des données.")
                return

            # Créer une colonne pour signaler les NaN sans remplacer les valeurs
            df['is_nan'] = df[colonne].isna().astype(int)

            # Filtrer les lignes où la colonne n'est pas NaN
            df_non_nan = df[df[colonne].notna()].copy()

            if df_non_nan.empty:
                print(f"Aucune donnée valide à traiter pour le modèle {model_name}.")
                return

            # Vérifier les colonnes que le scaler a vues lors de l'entraînement
            scaler_features = getattr(scaler, 'feature_names_in_', [])
            for feature in scaler_features:
                if feature not in df_non_nan.columns:
                    df_non_nan[feature] = 0  # Ajouter des colonnes manquantes avec des valeurs par défaut

            # Filtrer les colonnes pour correspondre exactement aux colonnes utilisées lors de l'entraînement
            df_scaled = df_non_nan[scaler_features]

            # Appliquer le scaler pour transformer les données
            X_scaled_test = scaler.transform(df_scaled)

            # Prédire les anomalies avec IsolationForest
            df_non_nan['Anomaly_IF'] = iso_forest.predict(X_scaled_test)
            df_non_nan['Anomaly_IF'] = df_non_nan['Anomaly_IF'].map({1: 0, -1: 1})  # -1 = anomalie, 1 = normal

            # Réintégrer les résultats dans le DataFrame original
            df['Anomaly_IF'] = np.nan  # Initialiser avec NaN
            df.loc[df_non_nan.index, 'Anomaly_IF'] = df_non_nan['Anomaly_IF']

            # Compter les anomalies détectées
            num_anomalies = df_non_nan['Anomaly_IF'].sum()
            print(f"Nombre d'anomalies détectées pour {model_name} : {num_anomalies}")
            model_anomalies[model_name] = num_anomalies

            # Ajouter les anomalies au rapport
            anomalies_report.update(
                {index: {model_name: "Anomalie détectée"} for index in df_non_nan[df_non_nan['Anomaly_IF'] == 1].index}
            )

        except Exception as e:
            print(f"Erreur lors du traitement du modèle {model_name} : {e}")




    def process_model_with_average(self, df, model_name, info, anomalies_report, model_anomalies):
        model_and_means = joblib.load(info['model'])
        df['7045Taux 2']= df['7045Taux 2'].fillna(0)
        df['7050Taux 2']= df['7050Taux 2'].fillna(0)
        model = model_and_means.get('model')
        average_predicted_taux_by_establishment = model_and_means.get('average_predicted_taux_by_establishment')

        if model is None or average_predicted_taux_by_establishment is None:
            st.error(f"Erreur : Modèle ou moyennes manquantes pour {model_name}.")
            return

        required_columns = ['Etablissement'] + info['numeric_cols']
        for col in required_columns:
            if col not in df.columns:
                df.loc[:, col] = 0  # Ajouter des colonnes manquantes avec des valeurs par défaut

        taux_col = info['numeric_cols'][1]

        df.loc[:, 'rule_based_taux'] = df.apply(lambda row: 0 if row['Effectif'] < 11 else row[taux_col], axis=1)
        df.loc[:, 'taux_per_effectif'] = df[taux_col] / df['Effectif'].replace({0: np.nan})  # Éviter la division par zéro
        df = df.fillna(0)

        X_new = df[['Etablissement', 'Effectif', 'rule_based_taux', 'taux_per_effectif']]

        try:
            df.loc[:, 'predicted_taux'] = model.predict(X_new)

            # Comparaison pour détecter les anomalies
            threshold = 0.01
            df.loc[:, 'Anomalie'] = abs(df[taux_col] - df['predicted_taux']) > threshold

            for index, row in df.iterrows():
                if row['Anomalie']:
                    # Vérifiez si l'index existe dans anomalies_report et assurez-vous qu'il s'agit d'un dict
                    if index not in anomalies_report:
                        anomalies_report[index] = {}  # Initialiser un dictionnaire pour cet index
                    elif isinstance(anomalies_report[index], set):
                        # Si c'est un set, convertir en dict
                        anomalies_report[index] = {key: None for key in anomalies_report[index]}

                    # Ajouter l'anomalie pour ce modèle
                    anomalies_report[index][model_name] = f"Établissement {row['Etablissement']}: Anomalie détectée"

                    # Incrémenter le compteur d'anomalies
                    model_anomalies[model_name] = model_anomalies.get(model_name, 0) + 1

        except ValueError as e:
            st.error(f"Erreur lors de la prédiction avec le modèle {model_name} : {e}")
    
    def process_model_7025(self, df, model_name, info, anomalies_report, model_anomalies):
        """
        Traite les anomalies pour le modèle 7025 en appliquant la logique de test_model_7002.
        """
        try:
            # Charger le modèle depuis le fichier spécifié dans `info`
            model_file = info['model']
            model_and_means = joblib.load(model_file)
            case_1_1 = model_and_means.get('case_1_1', {})
            case_1_2 = model_and_means.get('case_1_2', {})
            case_2 = model_and_means.get('case_2', {})
            print(f"Modèle {model_name} chargé avec succès depuis {model_file}.")
        except Exception as e:
            print(f"Erreur lors du chargement du modèle {model_name} : {e}")
            return

        # Colonnes spécifiques au modèle
        plafond_column = 'PLAFOND CUM'
        taux_column = '7025Taux 2'

        # Ajouter les colonnes calculées nécessaires
        try:
            df['SMIC M CUM'] = df['SMIC M CUM M-1'].fillna(0) + df['SMIC M'].fillna(0)
            df['ASSIETTE CUM'] = df["ASSIETTE CU M-1"].fillna(0) + df["Assiette Mois M (/102)"].fillna(0)
            df[plafond_column] = 3.5 * df['SMIC M CUM'].fillna(0)  # Exemple avec 3.5, ajustez selon votre logique
        except KeyError as e:
            print(f"Erreur : Les colonnes nécessaires pour les calculs sont manquantes pour {model_name} : {e}")
            return

        # Fonction pour traiter chaque cas
        def predict_case(df_case, model_data, numeric_cols, taux_column):
            if df_case.empty:
                print(f"Aucune donnée disponible pour le cas {model_data.get('case_name', 'inconnu')}.")
                return pd.DataFrame()  # Retourner un DataFrame vide

            if 'taux_per_statut' not in df_case.columns:
                df_case['taux_per_statut'] = df_case[taux_column]

            if 'taux_per_statut' not in numeric_cols:
                numeric_cols.append('taux_per_statut')

            df_case[numeric_cols] = df_case[numeric_cols].fillna(0)
            X_case = df_case[numeric_cols]

            df_case['predicted_taux'] = model_data['model'].predict(X_case)
            df_case['taux_diff'] = abs(df_case[taux_column] - model_data['average_predicted_taux'])
            threshold = 0.01
            df_case['Anomalie'] = df_case['taux_diff'].apply(lambda x: 'Oui' if x > threshold else 'Non')

            return df_case[['Matricule'] + numeric_cols + ['Anomalie']]

        # Gestion des cas spécifiques
        try:
            df_case_1_1 = df[(df['ASSIETTE CUM'] < df[plafond_column]) & (df['Statut de salariés'] == 'Stagiaire')].copy()
            df_case_1_2 = df[(df['ASSIETTE CUM'] < df[plafond_column]) & (df['Statut de salariés'] != 'Stagiaire')].copy()
            df_case_2 = df[df['ASSIETTE CUM'] > df[plafond_column]].copy()

            result_case_1_1 = predict_case(df_case_1_1, case_1_1, case_1_1.get('numeric_cols', []), taux_column)
            result_case_1_2 = predict_case(df_case_1_2, case_1_2, case_1_2.get('numeric_cols', []), taux_column)
            result_case_2 = predict_case(df_case_2, case_2, case_2.get('numeric_cols', []), taux_column)

            # Fusionner les résultats
            final_results = pd.concat([result_case_1_1, result_case_1_2, result_case_2])

            # Mettre à jour le rapport et les anomalies
            anomalies = final_results[final_results['Anomalie'] == 'Oui']
            anomalies_report.update(anomalies.to_dict(orient='index'))
            model_anomalies[model_name] = len(anomalies)

            print(f"Nombre d'anomalies détectées pour {model_name} : {len(anomalies)}")
        except Exception as e:
            print(f"Erreur lors du traitement des données pour {model_name} : {e}")
            return




    def verify_montant_conditions(self, df, model_name, anomalies_report, model_anomalies):
        """
        Vérifie les montants patronaux et salariaux pour un modèle donné et ajoute des anomalies si nécessaire.
        
        Parameters:
        df (pd.DataFrame): Le DataFrame contenant les données.
        model_name (str): Le nom du modèle (par exemple, '7001', '7020', etc.).
        anomalies_report (dict): Dictionnaire contenant les anomalies détectées.
        model_anomalies (dict): Dictionnaire pour compter le nombre d'anomalies par modèle.
        """
        
        montant_pat_col = f"{model_name}Montant Pat."
        taux_2_col = f"{model_name}Taux 2"
        montant_sal_col = f"{model_name}Montant Sal."
        taux_col = f"{model_name}Taux"
        rub_col = f"{model_name}Base"

        tolérance = 0.11  # La marge de différence tolérable exacte

        # Parcourir les lignes du DataFrame pour appliquer les vérifications
        for index, row in df.iterrows():
            # Initialiser les détails de l'anomalie pour cette ligne si ce n'est pas encore un dictionnaire
            if index not in anomalies_report:
                anomalies_report[index] = {}

            anomaly_detected = False  # Flag pour savoir si une anomalie a été détectée

            # Vérification pour Montant Pat.
            if montant_pat_col in df.columns and taux_2_col in df.columns and rub_col in df.columns:
                if not pd.isna(row[montant_pat_col]) and row[montant_pat_col] != 0:
                    montant_pat_calcule = round(row[taux_2_col] * row[rub_col] / 100, 2)
                    montant_pat_calcule2 = round(row[taux_2_col] * row[rub_col] / 100 * -1, 2)
                    montant_pat_reel = round(row[montant_pat_col], 2)
                    
                    if not np.isclose(montant_pat_reel, montant_pat_calcule, atol=tolérance) and not np.isclose(montant_pat_reel, montant_pat_calcule2, atol=tolérance):
                        # Ajouter l'anomalie si le montant calculé ne correspond pas
                        anomalies_report[index][model_name] = f"Anomalie détectée ({model_name})"
                        anomaly_detected = True  # Flag anomaly detected

            # Vérification pour Montant Sal.
            if montant_sal_col in df.columns and taux_col in df.columns and rub_col in df.columns:
                if not pd.isna(row[montant_sal_col]) and row[montant_sal_col] != 0:
                    montant_sal_calcule = round(row[taux_col] * row[rub_col] / 100 * -1, 2)
                    montant_sal_calcule2 = round(row[taux_col] * row[rub_col] / 100, 2)
                    montant_sal_reel = round(row[montant_sal_col], 2)
                    
                    if not np.isclose(montant_sal_reel, montant_sal_calcule, atol=tolérance) and not np.isclose(montant_sal_reel, montant_sal_calcule2, atol=tolérance):
                        # Ajouter l'anomalie si le montant calculé ne correspond pas
                        anomalies_report[index][model_name] = f"Anomalie détectée ({model_name})"
                        anomaly_detected = True  # Flag anomaly detected

            # Si une anomalie est détectée pour le modèle, ne comptez qu'une seule anomalie pour ce modèle et cette ligne
            if anomaly_detected:
                model_anomalies[model_name] = model_anomalies.get(model_name, 0) + 1

    def process_model_generic(self, df, model_name, info, anomalies_report, model_anomalies):
        colonnes_a_supprimer = ["CUMUL B MAL M-1"]
        df = df.drop(columns=colonnes_a_supprimer)
        # Prétraitement identique au code de référence
        df['PLAFOND CUM M-1'] = df['PLAFOND CUM M-1'].fillna(0)
        df['SMIC M CUM M-1'] = df['SMIC M CUM M-1'].fillna(0)
        
        df['CUMUL SMIC ( Mois courant inclus)'] = df['SMIC M']+df['SMIC M CUM M-1']
        df['PLAFOND CUM'] = df['PLAFOND M']+df['PLAFOND CUM M-1']
        
        df['Cumul Plafonds Maladie'] = 2.5 * df['CUMUL SMIC ( Mois courant inclus)']
        df['Cumul Plafonds Allocation'] = 3.5 * df['CUMUL SMIC ( Mois courant inclus)']
        df = df.rename(columns={"BASE B 7002":"CUMUL B MAL M-1"})

        df['7002Taux 2'] = df['7002Taux 2'].fillna(0)
        df.loc[:, 'taux_per_statut'] = df['7002Taux 2']

        # Charger les modèles et moyennes
        models_and_means = joblib.load(info['model'])
        df["Cumul d'assiette ( Mois courant inclus) (/102)"] = df["ASSIETTE CU M-1"].fillna(0) + df["Assiette Mois M (/102)"].fillna(0)
        # Cas spécifiques à traiter
        df_new_case_1_1 = df[(df["Cumul d'assiette ( Mois courant inclus) (/102)"] < df['Cumul Plafonds Maladie']) & (df['MALADIE CUM M-1'] == 0)]
        df_new_case_1_2_1 = df[(df["Cumul d'assiette ( Mois courant inclus) (/102)"] < df['Cumul Plafonds Maladie']) & (df['MALADIE CUM M-1'] != 0) & (df['CUMUL B MAL M-1'] == 0)]
        df_new_case_1_2_2 = df[(df["Cumul d'assiette ( Mois courant inclus) (/102)"] < df['Cumul Plafonds Maladie']) & (df['MALADIE CUM M-1'] != 0) & (df['CUMUL B MAL M-1'] != 0)]
        df_new_case_2_1 = df[(df["Cumul d'assiette ( Mois courant inclus) (/102)"] > df['Cumul Plafonds Maladie']) & (df['MALADIE CUM M-1'] == 0)]
        df_new_case_2_2 = df[(df["Cumul d'assiette ( Mois courant inclus) (/102)"] > df['Cumul Plafonds Maladie']) & (df['MALADIE CUM M-1'] != 0)]

        # Fonction pour appliquer les modèles et générer les prédictions pour chaque cas
        def predict_case(df_case, model_data):
            if df_case.empty:
                print(f"Aucune donnée disponible pour le cas {model_data['case_name']}.")
                return pd.DataFrame()

            # Charger le pipeline et la moyenne
            pipeline_case = model_data['model']
            average_predicted_taux = model_data['average_predicted_taux']

            # Ajout des colonnes nécessaires
            df_case.loc[:, 'taux_per_statut'] = df_case['7002Taux 2'].fillna(0)

            # Prédictions
            X_case = df_case[['taux_per_statut']].fillna(0)
            df_case.loc[:, 'predicted_taux'] = pipeline_case.predict(X_case)
            df_case.loc[:, 'taux_diff'] = abs(df_case['7002Taux 2'] - average_predicted_taux)
            df_case.loc[:, 'Anomalie'] = df_case['taux_diff'].apply(lambda x: '1' if x > 0.01 else '0')

            return df_case[['Matricule', '7002Taux 2', 'predicted_taux', '7002Base',
                            "Cumul d'assiette ( Mois courant inclus) (/102)",
                            'Cumul Plafonds Maladie', 'MALADIE CUM M-1', 'CUMUL B MAL M-1',
                            'Statut de salariés', 'Anomalie']]

        # Appliquer les prédictions pour chaque cas
        result_case_1_1 = predict_case(df_new_case_1_1, models_and_means['case_1_1'])
        result_case_1_2_1 = predict_case(df_new_case_1_2_1, models_and_means['case_1_2_1'])
        result_case_1_2_2 = predict_case(df_new_case_1_2_2, models_and_means['case_1_2_2'])
        result_case_2_1 = predict_case(df_new_case_2_1, models_and_means['case_2_1'])
        result_case_2_2 = predict_case(df_new_case_2_2, models_and_means['case_2_2'])

        # Fusionner les résultats
        final_results = pd.concat([result_case_1_1, result_case_1_2_1, result_case_1_2_2,
                                result_case_2_1, result_case_2_2], ignore_index=True)

        # Calculer les anomalies
        anomalies = final_results[final_results['Anomalie'] == '1']
        print(f"Nombre d'anomalies détectées pour {model_name}: {len(anomalies)}")

        # Mettre à jour le rapport et les anomalies
        anomalies_report.update(anomalies.to_dict(orient='index'))
        model_anomalies[model_name] = len(anomalies)

        
    def process_model(self, df, model_name, info, anomalies_report, model_anomalies):

        df_filtered = df
        
        if df_filtered.empty:
            st.write(f"Aucune donnée à traiter pour le modèle {model_name}.")
            return

        required_columns = info['numeric_cols'] + info['categorical_cols']
        missing_columns = [col for col in required_columns if col not in df_filtered.columns]

        if missing_columns:
            return

        # Filtrer uniquement les colonnes nécessaires
        df_inputs = df_filtered[required_columns].copy()

        # Spécifique aux modèles : Remplacer les NaN par 0 pour certaines colonnes
        if model_name == '6000':
            df_inputs['Rub 6000'] = df_inputs['Rub 6000'].fillna(0)


        # Remplir les valeurs manquantes pour les autres colonnes numériques
        df_inputs[info['numeric_cols']] = df_inputs[info['numeric_cols']].fillna(df_inputs[info['numeric_cols']].mean())

        # Encodage des colonnes catégorielles
        if model_name in ['6082', '6084']:
            label_encoder_statut = LabelEncoder()
            label_encoder_frontalier = LabelEncoder()

            # Encodage des colonnes catégorielles
            if 'Statut de salariés' in df_inputs.columns:
                df_inputs['Statut de salariés'] = label_encoder_statut.fit_transform(df_inputs['Statut de salariés'])

            if 'Frontalier' in df_inputs.columns:
                df_inputs['Frontalier'] = label_encoder_frontalier.fit_transform(df_inputs['Frontalier'])

        elif info['categorical_cols']:
            # Utilisation de OneHotEncoder pour les autres modèles
            encoder = OneHotEncoder(drop='first', sparse_output=False)
            encoded_categorical = encoder.fit_transform(df_inputs[info['categorical_cols']])
            df_encoded = pd.DataFrame(encoded_categorical, index=df_inputs.index, columns=encoder.get_feature_names_out(info['categorical_cols']))
            df_inputs = df_inputs.drop(columns=info['categorical_cols'])
            df_inputs = pd.concat([df_inputs, df_encoded], axis=1)

        # Charger le modèle
        model = self.load_model(info)

        # Vérification de l'alignement des colonnes avec celles vues pendant l'entraînement
        if hasattr(model, 'feature_names_in_'):
            # Réordonner les colonnes en fonction de celles du modèle
            df_inputs = df_inputs.reindex(columns=model.feature_names_in_, fill_value=0)

        try:
            y_pred = model.predict(df_inputs)
        except ValueError as e:
            st.error(f"Erreur avec le modèle {model_name} (joblib) : {str(e)}")
            return

        if y_pred is not None:
            df.loc[df_filtered.index, f'{model_name}_Anomalie_Pred'] = y_pred
            num_anomalies = np.sum(y_pred)
            model_anomalies[model_name] = num_anomalies

            for index in df_filtered.index:
                if y_pred[df_filtered.index.get_loc(index)] == 1:
                    anomalies_report.setdefault(index, set()).add(model_name)

    
    def detect_anomaly_for_row(self, row, model_name, model_info):
        # Exemple de méthode pour traiter une seule ligne
        try:
            # Logique de détection d'anomalie pour une ligne
            if row['value'] > model_info['threshold']:  # Exemple
                return True
            return False
        except Exception as e:
            raise ValueError(f"Erreur dans la détection d'anomalie : {e}")
        
    def detect_anomalies(self, df, model_name=None):
        """
        Détecte les anomalies pour les modèles définis.

        Args:
            df (pd.DataFrame): Les données à analyser.
            model_name (str, optional): Si spécifié, ne traite que ce modèle. Sinon, traite tous les modèles.

        Returns:
            dict, dict: Rapport des anomalies et anomalies par modèle.
        """
        anomalies_report = {}
        model_anomalies = {}

        # Prétraitement des colonnes nécessaires
        # Utiliser .loc pour éviter les SettingWithCopyWarning
        df.loc[:, 'SMIC M CUM'] = df['SMIC M CUM M-1'].fillna(0) + df['SMIC M'].fillna(0)
        df.loc[:, 'Assiette cum'] = df["ASSIETTE CU M-1"].fillna(0) + df["Assiette Mois M (/102)"].fillna(0)
        df.loc[:, 'ASSIETTE CUM'] = df["Assiette cum"]
        df.loc[:, 'Assiette Cum'] = df["Assiette cum"]

        # Si un modèle spécifique est fourni, limiter le traitement à ce modèle
        models_to_process = [model_name] if model_name else self.models_info.keys()

        for model_name in models_to_process:
            if model_name not in self.models_info:
                print(f"Le modèle {model_name} n'est pas configuré. Ignoré.")
                continue

            info = self.models_info[model_name]

            # Appeler la méthode spécifique en fonction du type de modèle
            try:
                if model_name in ['7045', '7050']:
                    # Modèles avec moyenne prédite (traitement spécial)
                    self.process_model_with_average(df, model_name, info, anomalies_report, model_anomalies)
                elif model_name == '7001':
                    # Modèle spécifique 7001
                    self.process_7001(df, model_name, info, anomalies_report, model_anomalies)
                elif model_name in ['6082', '6084']:
                    # Modèles spécifiques 6082 et 6084
                    self.process_model_6082_6084(df, model_name, info, anomalies_report, model_anomalies)
                elif model_name == '6002':
                    # Modèle spécifique 6002
                    self.process_model_6002(df, model_name, info, anomalies_report, model_anomalies)
                elif model_name in ['7002']:
                    # Modèles génériques 7002 et 7025
                    self.process_model_generic(df, model_name, info, anomalies_report, model_anomalies)
                elif model_name == '7025' :
                    self.process_model_7025(df,model_name,info,anomalies_report, model_anomalies)
                else:
                    # Autres modèles génériques
                    self.process_model(df, model_name, info, anomalies_report, model_anomalies)

                # Vérification des montants après le traitement
                self.verify_montant_conditions(df, model_name, anomalies_report, model_anomalies)
            except Exception as e:
                print(f"Erreur lors du traitement du modèle {model_name} : {e}")

        return anomalies_report, model_anomalies




    
    def generate_report(self, anomalies_report, model_anomalies, df):
        """
        Génère un rapport formaté des anomalies détectées.

        Args:
            anomalies_report (dict): Détails des anomalies détectées.
            model_anomalies (dict): Compte des anomalies par modèle.
            df (pd.DataFrame): Les données source.

        Returns:
            list: Rapport des anomalies sous forme de lignes de texte.
        """
        report_lines = []

        # Ajouter un résumé des anomalies détectées par modèle
        report_lines.append("=== Résumé des anomalies par modèle ===\n")
        for model_name, count in model_anomalies.items():
            report_lines.append(f"Modèle {model_name} : {count} anomalies détectées.\n")

        # Ajouter les détails des anomalies ligne par ligne
        report_lines.append("\n=== Détails des anomalies ===\n")
        for index, details in anomalies_report.items():
            matricule = df.loc[index, 'Matricule'] if 'Matricule' in df.columns else "Inconnu"
            anomalies_details = ", ".join([f"{model}: {desc}" for model, desc in details.items()])
            report_lines.append(f"Matricule {matricule} (Index {index}) : {anomalies_details}\n")

        # Résumé global
        total_anomalies = sum(model_anomalies.values())
        report_lines.append(f"\n=== Total d'anomalies détectées : {total_anomalies} ===\n")

        return report_lines

    def combine_reports(self, reports):
        """
        Combine les rapports d'anomalies en un seul DataFrame.

        Args:
            reports (dict): Dictionnaire contenant les rapports d'anomalies par modèle.

        Returns:
            pd.DataFrame: Rapport combiné.
        """
        combined = []

        for model_name, report in reports.items():
            if isinstance(report, tuple):
                print(f"Attention : le rapport du modèle {model_name} est un tuple. Vérifiez la fonction associée.")
                continue
            if isinstance(report, pd.DataFrame):
                combined.append(report)
            else:
                print(f"Le rapport du modèle {model_name} n'est ni un DataFrame ni un tuple. Ignoré.")

        if not combined:
            raise ValueError("Aucun rapport valide n'a été trouvé pour la combinaison.")

        return pd.concat(combined, ignore_index=True)

    
    def charger_dictionnaire(fichier):
        dictionnaire = {}
        try:
            with open(fichier, 'r', encoding='utf-8') as f:
                for ligne in f:
                    code, description = ligne.strip().split(' : ', 1)
                    dictionnaire[code] = description
        except FileNotFoundError:
            st.error(f"Le fichier {fichier} n'a pas été trouvé.")
        except Exception as e:
            st.error(f"Erreur lors du chargement du dictionnaire : {e}")
        return dictionnaire

    def process_model_6082_6084(self, df, model_name, info, anomalies_report, model_anomalies):
        # Charger le modèle
        rf_classifier_loaded = joblib.load(info['model'])
        df['6082Taux'] = df['6082Taux'].fillna(0)
        df['6084Taux'] = df['6084Taux'].fillna(0)

        # Charger les encodeurs spécifiques
        label_encoder_statut = joblib.load('./Modèles/Taux/label_encoder_statut.pkl')
        label_encoder_frontalier = joblib.load('./Modèles/Taux/label_encoder_frontalier.pkl')

        # Nettoyer les noms de colonnes pour éviter les espaces accidentels
        df.columns = df.columns.str.strip()

        # Vérifier que les colonnes nécessaires sont présentes
        try:
            features_new = df[info['numeric_cols'] + info['categorical_cols']].copy()
        except KeyError as e:
            st.error(f"Erreur : Colonnes manquantes pour {model_name} : {e}")
            return

        # Remplir les valeurs manquantes dans les colonnes catégorielles
        features_new[info['categorical_cols']] = features_new[info['categorical_cols']].fillna('Unknown')

        # Encodage avec gestion des valeurs inconnues
        def handle_unknown_labels(encoder, series):
            series = series.apply(lambda x: x if x in encoder.classes_ else 'Unknown')
            if 'Unknown' not in encoder.classes_:
                encoder.classes_ = np.append(encoder.classes_, 'Unknown')
            return encoder.transform(series)

        try:
            features_new['Statut de salariés'] = handle_unknown_labels(label_encoder_statut, features_new['Statut de salariés'])
            features_new['Frontalier'] = handle_unknown_labels(label_encoder_frontalier, features_new['Frontalier'])
        except ValueError as e:
            st.error(f"Erreur d'encodage pour {model_name} : {e}")
            return

        # Réorganiser les colonnes pour correspondre au modèle
        try:
            training_columns = rf_classifier_loaded.feature_names_in_
            missing_cols = set(training_columns) - set(features_new.columns)
            for col in missing_cols:
                features_new[col] = 0
            features_new = features_new.reindex(columns=training_columns)
        except AttributeError as e:
            st.error(f"Erreur : {e}")
            return

        # Effectuer les prédictions
        try:
            predictions = rf_classifier_loaded.predict(features_new)
            if len(predictions) != len(df):
                st.error(f"Erreur : Désalignement entre les prédictions et le DataFrame pour {model_name}.")
                return

            df.loc[:, f'{model_name}_Anomalie_Pred'] = predictions
            num_anomalies = np.sum(predictions)
            model_anomalies[model_name] = num_anomalies

            for index, pred in enumerate(predictions):
                if pred == 1:
                    anomalies_report.setdefault(df.index[index], {}).update({model_name: "Anomalie détectée"})
        except ValueError as e:
            st.error(f"Erreur lors de la prédiction pour {model_name} : {e}")


    def process_model_6002(self, df, model_name, info, anomalies_report, model_anomalies):
        """
        Processus spécifique pour le modèle 6002 avec une gestion de l'encodage de 'Region'.
        """

        # Charger le modèle
        model = self.load_model(info)
        
        # Préparation des colonnes nécessaires
        df_filtered = df.copy()

        # Vérifier que les colonnes requises sont présentes
        required_columns = info['numeric_cols'] + info['categorical_cols']
        missing_columns = [col for col in required_columns if col not in df_filtered.columns]

        if missing_columns:
            return

        # Sélectionner les colonnes spécifiques au modèle
        df_inputs = df_filtered[required_columns].copy()

        # Encodage des variables catégorielles
        df_inputs_encoded = pd.get_dummies(df_inputs, columns=info['categorical_cols'], drop_first=True)

        # Vérifier que le nombre de colonnes encodées correspond aux colonnes utilisées pour l'entraînement
        model_features = model.feature_names_in_
        missing_cols = set(model_features) - set(df_inputs_encoded.columns)
        
        # Ajouter les colonnes manquantes avec des valeurs par défaut (0)
        for col in missing_cols:
            df_inputs_encoded[col] = 0
        
        # Réordonner les colonnes dans l'ordre attendu par le modèle
        df_inputs_encoded = df_inputs_encoded.reindex(columns=model_features, fill_value=0)

        try:
            # Faire les prédictions
            predictions = model.predict(df_inputs_encoded)
            df_filtered[f'{model_name}_Anomalie_Pred'] = predictions
            
            # Compter les anomalies détectées
            num_anomalies = np.sum(predictions)
            model_anomalies[model_name] = num_anomalies
            
            # Enregistrer les anomalies détectées dans le rapport
            for index in df_filtered.index:
                if predictions[df_filtered.index.get_loc(index)] == 1:
                    anomalies_report.setdefault(index, set()).add(model_name)

        except ValueError as e:
            st.error(f"Erreur lors de la prédiction avec le modèle {model_name} : {e}")
