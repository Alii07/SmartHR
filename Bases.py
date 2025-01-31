# Importation des bibliothèques nécessaires
import pandas as pd
import numpy as np
from joblib import load
from sklearn.metrics import mean_squared_error, r2_score, classification_report, confusion_matrix
import os
from sklearn.exceptions import InconsistentVersionWarning
import warnings
import pickle


warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# Classe principale
class AnomalyDetection:
    def __init__(self, model_configs):
        self.model_configs = model_configs
        self.models = {}
    
    

    def load_models(self):
        """
        Charger tous les modèles définis dans la configuration.
        """
        for model_name, config in self.model_configs.items():
            print(f"Chargement des modèles pour : {model_name}")
            self.models[model_name] = {}

            # Charger le modèle de classification (si défini)
            if "classification_model" in config:
                try:
                    self.models[model_name]["classification_model"] = load(config["classification_model"])
                    print(f"  Modèle de classification chargé pour {model_name}")
                except Exception as e:
                    print(f"  Erreur lors du chargement du modèle de classification pour {model_name} : {e}")

            # Charger les modèles de régression
            if "regression_models" in config:
                self.models[model_name]["regression_models"] = {}
                for key, path in config["regression_models"].items():
                    try:
                        self.models[model_name]["regression_models"][key] = load(path)
                        print(f"  Modèle de régression chargé pour {model_name}, sous-ensemble {key}")
                    except Exception as e:
                        print(f"  Erreur lors du chargement du modèle de régression pour {model_name}, sous-ensemble {key} : {e}")

            # Charger les modèles combinés
            if "path" in config:
                if model_name == '7002' :
                    try:
                        combined_model = load(config["path"])
                        self.models[model_name]["combined_model"] = combined_model
                        print(f"  Modèle combiné chargé pour {model_name}")
                    except Exception as e:
                        print(f"  Erreur lors du chargement du modèle combiné pour {model_name} : {e}")
                        
                    
                        
                    


    def preprocess_data(self, df, model_name):
        """
        Prétraitement amélioré des données spécifique à chaque modèle
        """
        config = self.model_configs[model_name]
        df["BASE B V"] = df["BASE B V"].fillna(0)
        # Fonction pour créer les labels (clusters) en fonction des règles métier
        df['PLAFOND CUM M-1'] = df['PLAFOND CUM M-1'].fillna(0)
        df['SMIC M CUM M-1'] =  df['SMIC M CUM M-1'].fillna(0)

        # Création des colonnes nécessaires
        df['PLAFOND CUM'] = df['PLAFOND CUM M-1'].fillna(0) + df["PLAFOND M"].fillna(0)
        df['Plafond CUM'] = df['PLAFOND CUM']
        df['TRANCHE C PRE'] = df['ASSIETTE CU M-1'] - 4 * df['PLAFOND CUM M-1']
        df['4*PLAFOND CUM'] = 4 * df['Plafond CUM']
        df['SMIC CUM'] = df['SMIC M CUM M-1'] + df['SMIC M']
        df['Assiette cum'] = df["ASSIETTE CU M-1"] + df["Assiette Mois M (/102)"]
        df["Cumul d'assiette ( Mois courant inclus) (/102)"] = df['Assiette cum']
        df['Brut CUM'] = df["Cumul d'assiette ( Mois courant inclus) (/102)"]
        # Copie pour éviter les modifications accidentelles
        df = df.copy()
        
        # Remplacement des valeurs manquantes par 0
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
        df[numeric_columns] = df[numeric_columns].fillna(0)
        
        # Vérification des colonnes essentielles
        essential_cols = [
            'PLAFOND CUM M-1', 'PLAFOND M', 
            'SMIC M CUM M-1', 'SMIC M',
            'ASSIETTE CU M-1', 'Assiette Mois M (/102)',
            f'{model_name}Base'
        ]
        
        missing_cols = [col for col in essential_cols if col not in df.columns]
        if missing_cols:
            print(f"Colonnes manquantes pour {model_name}: {missing_cols}")
            # Ajouter les colonnes manquantes avec des valeurs 0
            for col in missing_cols:
                df[col] = 0
        
        # Calculs des colonnes dérivées
        df['PLAFOND CUM'] = df['PLAFOND CUM M-1'] + df['PLAFOND M']
        df['SMIC M CUM'] = df['SMIC M CUM M-1'] + df['SMIC M']
        df['Assiette cum'] = df['ASSIETTE CU M-1'] + df['Assiette Mois M (/102)']
        df['4*PLAFOND CUM'] = 4 * df['PLAFOND CUM']
        
        # Prétraitement spécifique au modèle
        if model_name in self.model_configs:
            config = self.model_configs[model_name]
            
            # Ajout des colonnes spécifiques au modèle
            if 'numeric_cols' in config:
                for col in config['numeric_cols']:
                    if col not in df.columns:
                        df[col] = 0
                        
            if 'categorical_cols' in config:
                for col in config['categorical_cols']:
                    if col not in df.columns:
                        df[col] = 'Unknown'
        
        return df








    def detect_anomalies(self, df, model_name):
        """
        Détecter les anomalies pour un modèle donné.
        
        :param df: DataFrame préparé pour le modèle.
        :param model_name: Nom du modèle à traiter.
        :return: DataFrame avec les anomalies détectées.
        """
        

        config = self.model_configs[model_name]

        if model_name == "7015":
            df['PLAFOND CUM M-1'] = df['PLAFOND CUM M-1'].fillna(0)
            df['SMIC M CUM M-1'] =  df['SMIC M CUM M-1'].fillna(0)
            df["BASE B V"] = df["BASE B V"].fillna(0)
            print(f"Détection des anomalies pour {model_name}...")
            
            df["ASSIETTE CU M-1"] = df["ASSIETTE CU M-1"].fillna(0)
            df['Assiette cum'] = df["ASSIETTE CU M-1"] + df["Assiette Mois M (/102)"]
            
            # Étape 1 : Vérifier que les modèles de régression sont chargés
            if "regression_models" not in self.models[model_name]:
                raise ValueError(f"Aucun modèle de régression défini pour {model_name}.")

            regression_models = self.models[model_name]["regression_models"]

            # Étape 2 : Créer la colonne 'CUM Base 7015 precedente' si elle n'existe pas
            if "CUM Base 7015 precedente" not in df.columns:
                print("Création de la colonne 'CUM Base 7015 precedente'...")
                df["CUM Base 7015 precedente"] = df["BASE B V"]

            # Étape 3 : Vérifier et générer la colonne 'label_predit'
            if "classification_model" in self.models[model_name]:
                classification_model = self.models[model_name]["classification_model"]

                # Vérifier les colonnes nécessaires
                if "Assiette cum" in df.columns and "PLAFOND CUM" in df.columns:
                    X_classification = df[["Assiette cum", "PLAFOND CUM"]]

                    # Gérer les valeurs manquantes
                    if X_classification.isnull().any().any():
                        print("Des valeurs manquantes détectées pour la classification. Remplissage en cours...")
                        X_classification = X_classification.fillna(X_classification.mean())

                    # Faire la prédiction
                    print("Génération de la colonne 'label_predit'...")
                    df["label_predit"] = classification_model.predict(X_classification)
                else:
                    raise ValueError(f"Colonnes manquantes pour la classification : 'Assiette cum' et/ou 'PLAFOND CUM'")
            else:
                raise ValueError(f"Modèle de classification manquant pour {model_name}")

            # Étape 4 : Appliquer les modèles de régression pour chaque label
            for key, regression_model in regression_models.items():
                print(f"Traitement pour le label {key}...")

                # Filtrer les données pour le sous-ensemble
                subset = df[df["label_predit"] == key]

                if subset.empty:
                    print(f"Aucune donnée pour le label {key}.")
                    continue

                # **Ajouter condition pour 'Statut de Salariés'**
                subset["7015Base_predicted"] = 0  # Valeur par défaut
                subset.loc[subset["Statut de salariés"] == 0, "7015Base_predicted"] = 0

                # Définir les caractéristiques en fonction du label
                if key == 0:
                    features = ["PLAFOND CUM", "CUM Base 7015 precedente"]
                elif key == 1:
                    features = ["Assiette cum", "CUM Base 7015 precedente"]

                # Appliquer la régression uniquement si 'Statut de Salariés' != 0
                subset_to_regress = subset[subset["Statut de salariés"] != 0]

                # Vérifier la présence des colonnes nécessaires
                missing_columns = [col for col in features if col not in subset_to_regress.columns]
                if missing_columns:
                    raise ValueError(f"Colonnes manquantes pour {model_name}, label {key} : {missing_columns}")

                # Préparer les données pour la régression
                if not subset_to_regress.empty:
                    X = subset_to_regress[features]
                    y_true = subset_to_regress["7015Base"]

                    # Gérer les valeurs manquantes
                    if X.isnull().any().any():
                        print(f"Des valeurs manquantes détectées pour le label {key}. Remplissage en cours...")
                        X = X.fillna(X.mean())

                    # Faire les prédictions
                    try:
                        y_pred = regression_model.predict(X)
                        subset_to_regress["7015Base_predicted"] = y_pred

                        # Détecter les anomalies
                        subset_to_regress["anomalie"] = (
                            abs(subset_to_regress["7015Base"] - subset_to_regress["7015Base_predicted"]) > 0.01
                        ).astype(int)
                        print(f"Anomalies détectées pour le label {key} : {subset_to_regress['anomalie'].sum()}")

                        # Ajouter les résultats au DataFrame principal
                        df.loc[subset_to_regress.index, "7015Base_predicted"] = subset_to_regress["7015Base_predicted"]
                        df.loc[subset_to_regress.index, "anomalie"] = subset_to_regress["anomalie"]
                    except Exception as e:
                        print(f"Erreur lors de la régression pour le label {key} : {e}")

        if model_name == "7002":
            print(f"Détection des anomalies pour {model_name}...")
            
            colonnes_a_supprimer = ["CUMUL B MAL M-1"]
            df = df.drop(columns=colonnes_a_supprimer)
            df = df.rename(columns={"BASE B 7002":"CUMUL B MAL M-1"})
            


            # Charger les modèles combinés pour 7002
            regression_models = self.models[model_name].get("combined_model", {})
            if not regression_models:
                raise ValueError(f"Les modèles combinés pour {model_name} n'ont pas été chargés correctement.")

            # Préparer les colonnes nécessaires
            if "BASE B 7002" in df.columns:
                df = df.rename(columns={"BASE B 7002": "CUMUL B MAL M-1"})
            else:
                print("Attention : 'BASE B 7002' est absente du fichier et ne sera pas renommée.")
            df['PLAFOND CUM M-1'] = df['PLAFOND CUM M-1'].fillna(0)
            df['SMIC M CUM M-1'] =  df['SMIC M CUM M-1'].fillna(0)
            df['CUMUL SMIC ( Mois courant inclus)'] = df['SMIC M'] + df['SMIC M CUM M-1']
            df['PLAFOND CUM'] = df['PLAFOND M']+df['PLAFOND CUM M-1']

            df['Cumul Plafonds Maladie'] = 2.5 * df['CUMUL SMIC ( Mois courant inclus)']
            df['Cumul Plafonds Allocation'] = 3.5 * df['CUMUL SMIC ( Mois courant inclus)']

            # Ajouter les colonnes de logique métier
            df['is_below_plafond'] = (df["Cumul d'assiette ( Mois courant inclus) (/102)"] < df["Cumul Plafonds Maladie"]).astype(int)
            df['has_previous_base'] = (df["MALADIE CUM M-1"] != 0).astype(int)

            df['SMIC M CUM'] = df['SMIC M CUM M-1'] + df['SMIC M']
            df.fillna(0, inplace=True)
            df['7002Base_predicted'] = np.nan

            # Définir les groupes et les features (identiques à ceux utilisés pour l'entraînement)
            groups = {
                (0, 0): ["Cumul d'assiette ( Mois courant inclus) (/102)"],
                (0, 1): ["Cumul d'assiette ( Mois courant inclus) (/102)", "CUMUL B MAL M-1"],
                (1, 1): ["CUMUL B MAL M-1"],
            }

            # Appliquer les prédictions pour chaque groupe
            for key, model in regression_models.items():
                subset = df[(df['is_below_plafond'] == key[0]) & (df['has_previous_base'] == key[1])]
                if not subset.empty:
                    X = subset[groups[key]].fillna(0)  # Remplacer les valeurs manquantes par 0
                    predictions = model.predict(X)
                    df.loc[subset.index, '7002Base_predicted'] = np.round(predictions, 2)

            # Appliquer les règles métier
            apprenti_status = ['Apprenti (B.C)', 'Apprenti (W.C)']
            df.loc[df['Statut de salariés'].isin(apprenti_status), '7002Base_predicted'] = np.nan
            df.loc[(df['is_below_plafond'] == 1) & (df['has_previous_base'] == 0), '7002Base_predicted'] = 0

            # Calculer les anomalies
            df['7002Base_predicted'] = df['7002Base_predicted'].fillna(0)
            df['anomalie'] = (df['7002Base'] != df['7002Base_predicted']).astype(int)

            print(f"Anomalies détectées : {df['anomalie'].sum()} lignes marquées comme anomalies.")




        if model_name in ['6081', '6085']:
            if 'BASE CSG' in df.columns:
                df = df.rename(columns={'BASE CSG': 'Tranche C pre'})
            if 'BASE CSG' in df.columns:
                print("La colonne 'BASE CSG' est présente dans la base de données.")

            if "Assiette cum" in df :
                df = df.rename(columns={ "Assiette cum":"Cumul d'assiette ( Mois courant inclus) (/102)"})
            df = df.rename(columns={'BASE CSG': 'Tranche C pre'})
            
            if 'Tranche C pre' in df.columns:
                print("La colonne 'Tranche C pre' est présente dans la base de données.")
            else:
                raise ValueError("La colonne 'Tranche C pre' est absente après la modification.")
            # Charger les modèles sauvegardés pour le modèle actuel
            classification_model = load(f"Modèles/Bases/classification_model_{model_name}_new.pkl")
            regression_models = {
                (0, 0): load(f"Modèles/Bases/regression_model_{model_name}_(0, 0).pkl"),
                (0, 1): load(f"Modèles/Bases/regression_model_{model_name}_(0, 1).pkl"),
                (1, 1): load(f"Modèles/Bases/regression_model_{model_name}_(1, 1).pkl")
            }

            # Étape 1 : Utiliser le modèle de classification pour ajouter la colonne 'is_below_plafond'
            X_classification = df[["Cumul d'assiette ( Mois courant inclus) (/102)", '4*PLAFOND CUM']]
            df['is_below_plafond'] = classification_model.predict(X_classification)

            # Étape 2 : Ajouter la colonne 'has_previous_base'
            df['has_previous_base'] = (df['Tranche C pre'] > 0).astype(int)

            # Étape 3 : Déterminer le sous-ensemble pour chaque ligne
            df['sous_ensemble'] = list(zip(df['is_below_plafond'], df['has_previous_base']))

            # Étape 4 : Calculer les prédictions pour les non-apprentis
            for key, model in regression_models.items():
                # Filtrer les indices pour le sous-ensemble actuel
                subset_indices = df[df['sous_ensemble'] == key].index

                if not subset_indices.empty:
                    # Sélectionner les caractéristiques en fonction du sous-ensemble
                    if key in [(0, 0), (0, 1)]:
                        features = ['4*PLAFOND', 'Assiette Mois M (/102)']
                    elif key == (1, 1):
                        features = ['Tranche C pre']

                    # Vérifier si les colonnes nécessaires sont présentes
                    if all(feature in df.columns for feature in features):
                        X = df.loc[subset_indices, features]

                        # Appliquer le modèle et enregistrer les prédictions dans les lignes originales
                        df.loc[subset_indices, f'{model_name}Base_predicted'] = model.predict(X).round(2)

            # Étape 5 : Gérer les apprentis (B.C et W.C)
            # Considérer qu'une colonne appelée 'Statut de salariés' contient le type de salarié
            apprenti_indices = df[
                df['Statut de salariés'].isin(['Apprenti (B.C)', 'Apprenti (W.C)'])
            ].index
            df.loc[apprenti_indices, f'{model_name}Base_predicted'] = np.nan

            # Étape 6 : Détecter les anomalies
            # Comparer la valeur prédite avec la valeur existante uniquement pour les non-apprentis
            df['anomalie'] = np.where(
                (df['Statut de salariés'].isin(['Apprenti (B.C)', 'Apprenti (W.C)']) == False) &  # Si ce n'est pas un apprenti
                (abs(df[f'{model_name}Base'] - df[f'{model_name}Base_predicted']) > 0.01),  # Comparer les valeurs (tolérance de 0.01)
                1,
                0
            )

            # Afficher uniquement les lignes avec des anomalies
            anomalies_data = df[df['anomalie'] == 1]
            print("\n--- Anomalies détectées ---")
            print(anomalies_data[["Cumul d'assiette ( Mois courant inclus) (/102)", 'anomalie']])

            # Retourner les données mises à jour
            return df

                    
        if model_name in ["6082", "6084"]:
            # Charger les modèles sauvegardés
            model_path = config["path"]
            with open(model_path, "rb") as f:
                saved_data = pickle.load(f)
                model_case1 = saved_data["model_case1"]
                model_case2 = saved_data["model_case2"]

            # Ajuster la marge pour la détection des anomalies
            margin = config.get("threshold_margin", 0.01)

            # Préparation des colonnes nécessaires
            df[f'{model_name}Base'] = df[f'{model_name}Base'].fillna(0)
            df['Total Brut'] = df['Total Brut'].replace({'\.': '', ',': '.'}, regex=True).astype(float)
            df['4PlafondCum'] = 4 * df['PLAFOND M']
            if 'anomalie' not in df.columns:
                df['anomalie'] = 0  # Initialisation avec 0 (pas d'anomalie)

            # Ajout et remplissage de `1001Montant Sal.`
            if '6S89Base' in df.columns:
                df['6S89Base'] = df['6S89Base'].fillna(0)
                df['1001Montant Sal.'] = df['Assiette Mois M (/102)'] - df['6S89Base']
            else:
                df['1001Montant Sal.'] = df['Assiette Mois M (/102)']

            # Préparation des listes pour stocker les labels et anomalies
            actual_labels = []
            predicted_labels = []
            misclassified_rows = []

            # Cas 1 : Prédictions sur `Plafond CUM` vs `6082Base` ou `6084Base`
            df_case1 = df[df['Case'] == 'Cas 1']
            if not df_case1.empty:
                x_new_case1 = df_case1[['PLAFOND M']].values
                y_new_case1 = df_case1[f'{model_name}Base'].values
                actual_labels_case1 = df_case1['anomalie'].values

                # Prédictions
                y_pred_new_case1 = model_case1.predict(x_new_case1)

                # Appliquer la condition pour les apprentis
                is_apprenti_case1 = df_case1['Statut de salariés'].str.contains('apprenti', case=False, na=False).values
                y_pred_new_case1[is_apprenti_case1] = 0

                # Calcul des erreurs relatives
                errors_case1 = abs((y_pred_new_case1 - y_new_case1) / np.maximum(y_new_case1, 1e-10))

                # Détection des anomalies
                predicted_labels_case1 = np.where(errors_case1 > margin, 1, 0)
                actual_labels.extend(actual_labels_case1)
                predicted_labels.extend(predicted_labels_case1)

                # Mettre à jour les colonnes dans le DataFrame principal
                df.loc[df_case1.index, f'Predicted {model_name}Base'] = y_pred_new_case1
                df.loc[df_case1.index, 'anomalie'] = np.maximum(df.loc[df_case1.index, 'anomalie'], predicted_labels_case1)

                # Lignes mal classifiées
                misclassified_case1 = df_case1[actual_labels_case1 != predicted_labels_case1]
                misclassified_case1[f'Predicted {model_name}Base'] = y_pred_new_case1[actual_labels_case1 != predicted_labels_case1]
                misclassified_rows.append(misclassified_case1[['Matricule', f'{model_name}Base', f'Predicted {model_name}Base', 
                                                                'Assiette Mois M (/102)', 'Case']])

            # Cas 2 : Prédictions sur `1001Montant Sal.` vs `6082Base` ou `6084Base`
            df_case2 = df[df['Case'] == 'Cas 2']
            if not df_case2.empty:
                x_new_case2 = df_case2[['1001Montant Sal.']].values
                y_new_case2 = df_case2[f'{model_name}Base'].values
                actual_labels_case2 = df_case2['anomalie'].values

                # Prédictions
                y_pred_new_case2 = model_case2.predict(x_new_case2)

                # Appliquer la condition pour les apprentis
                is_apprenti_case2 = df_case2['Statut de salariés'].str.contains('apprenti', case=False, na=False).values
                y_pred_new_case2[is_apprenti_case2] = 0

                # Calcul des erreurs relatives
                errors_case2 = abs((y_pred_new_case2 - y_new_case2) / np.maximum(y_new_case2, 1e-10))

                # Détection des anomalies
                predicted_labels_case2 = np.where(errors_case2 > margin, 1, 0)
                actual_labels.extend(actual_labels_case2)
                predicted_labels.extend(predicted_labels_case2)

                # Mettre à jour les colonnes dans le DataFrame principal
                df.loc[df_case2.index, f'Predicted {model_name}Base'] = y_pred_new_case2
                df.loc[df_case2.index, 'anomalie'] = np.maximum(df.loc[df_case2.index, 'anomalie'], predicted_labels_case2)

                # Lignes mal classifiées
                misclassified_case2 = df_case2[actual_labels_case2 != predicted_labels_case2]
                misclassified_case2[f'Predicted {model_name}Base'] = y_pred_new_case2[actual_labels_case2 != predicted_labels_case2]
                misclassified_rows.append(misclassified_case2[['Matricule', f'{model_name}Base', f'Predicted {model_name}Base', 
                                                                'Assiette Mois M (/102)', '4PlafondCum', 'Case', 
                                                                'PLAFOND M', "Cumul d'assiette ( Mois courant inclus) (/102)"]])

            # Convertir les listes en tableaux NumPy pour les métriques
            actual_labels = np.array(actual_labels)
            predicted_labels = np.array(predicted_labels)

            # Lignes mal classifiées (concaténation pour affichage)
            if misclassified_rows:
                misclassified_all = pd.concat(misclassified_rows, ignore_index=True)
                print("\n--- Lignes mal classifiées ---")
                print(misclassified_all)



            return df
        
        if model_name == '7025':
            
            
            # Charger les modèles sauvegardés pour 7025
            with open(self.model_configs['7025']['path'], 'rb') as f:
                modeles_lineaires = pickle.load(f)

            # Dictionnaire pour stocker les colonnes utilisées pour chaque cluster
            features_par_cluster = {
                cluster: self.model_configs['7025']['numeric_cols']
                for cluster in modeles_lineaires.keys()
            }

            # Appliquer les modèles par cluster
            for cluster, model_info in modeles_lineaires.items():
                # Filtrer les données pour le cluster courant
                subset_indices = df[df['Cluster'] == cluster].index
                subset = df.loc[subset_indices].copy()

                # Vérifier si le sous-ensemble est vide
                if subset.empty:
                    print(f"Cluster {cluster} est vide. Passer au suivant.")
                    continue

                try:
                    if len(model_info) == 2:  # Régression linéaire avec scaler
                        model, scaler = model_info

                        # Vérifier si les colonnes nécessaires existent
                        features = features_par_cluster.get(cluster, [])
                        if all(feature in subset.columns for feature in features):
                            # Préparer les données
                            X_test = subset[features].values

                            # Normaliser les données de test
                            X_test_scaled = scaler.transform(X_test)

                            # Prédire avec le modèle
                            y_pred = model.predict(X_test_scaled)
                            subset['y_pred'] = y_pred

                            # Calculer les résidus
                            subset['residuals'] = np.abs(subset['7025Base'] - y_pred)

                            # Détecter les anomalies
                            subset['anomalie'] = (subset['residuals'] > self.model_configs['7025']['threshold_margin']).astype(int)

                            # Mettre à jour les colonnes dans le DataFrame principal
                            df.loc[subset_indices, 'anomalie'] = subset['anomalie']

                        else:
                            print(f"Colonnes manquantes pour le cluster {cluster}: {features}")

                    else:  # Isolation Forest
                        isolation_forest_model = model_info
                        if hasattr(isolation_forest_model, 'fit_predict'):
                            # Appliquer Isolation Forest pour détecter les anomalies
                            subset['anomalie'] = isolation_forest_model.fit_predict(subset[['7025Base']])
                            subset['anomalie'] = (subset['anomalie'] == -1).astype(int)

                            # Mettre à jour les colonnes dans le DataFrame principal
                            df.loc[subset_indices, 'anomalie'] = subset['anomalie']

                except Exception as e:
                    print(f"Erreur lors du traitement du cluster {cluster}: {e}")

            # Vérification finale pour les anomalies détectées
            anomalies = df[df['anomalie'] == 1]
            print(f"Nombre total d'anomalies détectées pour 7025 : {len(anomalies)}")

            # Retourner le DataFrame avec les anomalies détectées
            return df



                





        if model_name not in ["7002","7015",'6081','6085','7025'] :
            if "path" in config:
                combined_model = self.models[model_name].get("combined_model", {})
                for case_name, case_config in config["cases"].items():
                    # Extraire les caractéristiques et la cible
                    feature_col = case_config["feature_col"]
                    target_col = case_config["target_col"]

                    # Vérifier la présence des colonnes nécessaires
                    if feature_col not in df.columns or target_col not in df.columns:
                        raise ValueError(f"Colonnes manquantes pour {model_name}, cas {case_name}")

                    # Filtrer les données pour ce cas
                    subset = df.copy()

                    # Extraire les modèles de régression et scalers
                    if case_name in combined_model:
                        model_info = combined_model[case_name]
                        model = model_info["model"]
                        scaler = model_info["scaler"]

                        # Préparer les données pour la régression
                        X = subset[[feature_col]].values
                        X_scaled = scaler.transform(X)

                        # Faire les prédictions
                        subset[f"{target_col}_predicted"] = model.predict(X_scaled).round(2)

                        # Calculer les résidus et détecter les anomalies
                        residuals = abs(subset[target_col] - subset[f"{target_col}_predicted"])
                        subset["anomalie"] = (residuals > config["threshold_margin"]).astype(int)

                        # Ajouter les résultats dans le DataFrame principal
                        df.loc[subset.index, f"{target_col}_predicted"] = subset[f"{target_col}_predicted"]
                        df.loc[subset.index, "anomalie"] = subset["anomalie"]

            # Cas 2 : Modèles classiques (classification + régression_models)
            else:
                if "classification_model" in self.models[model_name]:
                    classification_model = self.models[model_name]["classification_model"]
                    classification_features = config["classification_features"]
                    
                    # Vérification des colonnes pour la classification
                    missing_classification_features = [col for col in classification_features if col not in df.columns]
                    if missing_classification_features:
                        raise ValueError(f"Colonnes manquantes pour la classification : {missing_classification_features}")
                    
                    # Prédire les labels
                    df['label_predit'] = classification_model.predict(df[classification_features])
                
                # Étape 2 : Régression
                if "regression_models" in self.models[model_name]:
                    regression_models = self.models[model_name]["regression_models"]
                    target_col = config["target_col"]
                    predicted_col = f"{target_col}_predicted"

                    for label, model in regression_models.items():
                        # Filtrer les données pour le label actuel
                        subset = df[df['label_predit'] == label]
                        if subset.empty:
                            print(f"Aucune donnée pour le label {label} dans le modèle {model_name}.")
                            continue

                        regression_features = config["regression_features"][label]

                        # Vérification des colonnes pour la régression
                        missing_regression_features = [col for col in regression_features if col not in subset.columns]
                        if missing_regression_features:
                            print(f"Colonnes manquantes pour la régression (label {label}) : {missing_regression_features}")
                            continue

                        # Prédire les valeurs cibles
                        X_regression = subset[regression_features]
                        try:
                            df.loc[subset.index, predicted_col] = model.predict(X_regression)
                        except Exception as e:
                            print(f"Erreur lors de la régression pour le label {label} : {e}")
                            continue

                    # Vérification finale : la colonne doit être créée
                    if predicted_col not in df.columns or df[predicted_col].isna().all():
                        raise ValueError(f"La colonne {predicted_col} n'a pas été générée pour le modèle {model_name}.")

            # Étape 3 : Détecter les anomalies
            df['anomalie'] = (
                abs(df[target_col] - df[predicted_col]) > config["threshold_margin"]
            ).astype(int)

            print(f"Anomalies détectées pour {model_name} : {df['anomalie'].sum()} anomalies.")



            # Étape 3 : Gestion des cas spécifiques (ex. apprentis)
            if "apprenti_status" in config:
                apprenti_indices = df[df["Statut de salariés"].isin(config["apprenti_status"])].index
                df.loc[apprenti_indices, f"{config['target_col']}_predicted"] = np.nan

            # Étape 4 : Détecter les anomalies
            df['anomalie'] = (abs(df[target_col] - df[predicted_col]) > config["threshold_margin"]).astype(int)

        # Retourner le DataFrame annoté
        return df
    
    


    def detect_anomalies_simple_comparison(self, df, model_name):

        # Vérifiez la présence des colonnes nécessaires
        required_columns = [f"{model_name}Base", "Total Brut"]
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            print(f"Colonnes manquantes pour le modèle {model_name} : {missing_columns}")
            return None

        # Nettoyer et convertir uniquement la colonne 'Total Brut'
        df["Total Brut"] = (
            df["Total Brut"]
            .astype(str)  # Convertir en chaîne de caractères pour appliquer les remplacements
            .str.replace(".", "", regex=False)  # Supprimer les séparateurs de milliers
            .str.replace(",", ".", regex=False)  # Remplacer les virgules par des points
            .astype(float)  # Convertir en float
        )

        # Comparer les colonnes pour détecter les anomalies
        # NE PAS MODIFIER {model_name}Base
        df["anomalie"] = (abs(df[f"{model_name}Base"] - df["Total Brut"]) > 0.01).astype(int)

        # Ajoutez la colonne Valeur prédite avec les valeurs de Total Brut
        df[f"Predicted {model_name}Base"] = df["Total Brut"]

        # Affichez les résultats
        anomalies_detected = df["anomalie"].sum()
        print(f"Nombre d'anomalies détectées pour {model_name} : {anomalies_detected}")

        # Retourner le DataFrame mis à jour
        return df


    def combine_reports(self, reports):
        """
        Combiner les rapports de détection d'anomalies pour tous les modèles.

        :param reports: Dictionnaire contenant les DataFrames d'anomalies pour chaque modèle.
        :return: DataFrame combiné avec toutes les anomalies détectées.
        """
        combined_data = []

        for model_name, report in reports.items():
            # Vérifiez si des anomalies sont présentes dans le rapport
            if "anomalie" in report.columns:
                anomalies = report[report["anomalie"] == 1].copy()
                anomalies["nom_du_modèle"] = model_name

                # Ajouter des colonnes supplémentaires pour enrichir le rapport
                predicted_col = f"Predicted {model_name}Base"
                target_col = f"{model_name}Base"

                # Assurez-vous que les colonnes sont cohérentes
                anomalies["Valeur prédite"] = anomalies.get(predicted_col, anomalies["Total Brut"])
                anomalies["Valeur réelle"] = anomalies.get(target_col, np.nan)

                combined_data.append(anomalies)
            else:
                print(f"Attention : La colonne 'anomalie' est absente pour le modèle {model_name}. Ignoré.")

        if combined_data:
            combined_report = pd.concat(combined_data, ignore_index=True)
        else:
            # Si aucun rapport n'a d'anomalies, retourner un DataFrame vide
            combined_report = pd.DataFrame(columns=["Matricule", "nom_du_modèle", "Valeur réelle", "Valeur prédite", "anomalie"])

        return combined_report
    
    def generate_final_report(self, combined_report):
        """
        Générer un rapport final avec toutes les anomalies détectées.

        :param combined_report: DataFrame combiné contenant les anomalies détectées.
        :return: Liste de phrases décrivant les anomalies détectées.
        """
        # Vérifier si le DataFrame est vide
        if combined_report.empty:
            return ["Aucune anomalie détectée."]

        # Identifier les modèles à comparaison simple
        simple_comparison_models = combined_report["nom_du_modèle"].unique()

        # Nettoyer et convertir uniquement les prédictions des modèles simples
        def clean_and_convert_predicted(value, model):
            if model in simple_comparison_models and isinstance(value, str):
                return float(
                    value.replace(".", "").replace(",", ".")
                )  # Nettoyage pour les formats spécifiques
            return value  # Retourner tel quel si déjà un float ou non applicable

        try:
            combined_report["Valeur prédite"] = combined_report.apply(
                lambda row: clean_and_convert_predicted(row["Valeur prédite"], row["nom_du_modèle"]),
                axis=1
            )
        except ValueError as e:
            print(f"Erreur lors de la conversion de la colonne 'Valeur prédite' : {e}")
            return ["Erreur dans le format des valeurs de prédiction."]

        # Grouper les anomalies par matricule
        grouped_anomalies = combined_report.groupby("Matricule").apply(
            lambda group: {
                "Matricule": group.name,
                "Détails": group[
                    ["nom_du_modèle", "Valeur réelle", "Valeur prédite"]
                ].to_dict(orient="records"),
            }
        ).tolist()

        # Générer une liste de phrases pour chaque matricule
        final_report = []
        for anomaly in grouped_anomalies:
            matricule = anomaly["Matricule"]
            details = anomaly["Détails"]

            # Générer des détails par modèle
            model_details = []
            for detail in details:
                model = detail["nom_du_modèle"]
                valeur_reelle = detail["Valeur réelle"]
                valeur_predite = detail["Valeur prédite"]

                # Ne garder que les anomalies significatives
                if abs(valeur_reelle - valeur_predite) > 0.01:
                    model_details.append(
                        f"- Modèle {model}: Valeur réelle = {valeur_reelle:.2f}, Valeur prédite = {valeur_predite:.2f}"
                    )

            if model_details:
                phrase = f"Pour la matricule {matricule}, des anomalies ont été détectées :\n" + "\n".join(model_details)
                final_report.append(phrase)

        return final_report if final_report else ["Aucune anomalie détectée."]
