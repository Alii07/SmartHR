import joblib
from config import load_and_preprocess_data, MODEL_REGISTRY
from config_taux import TAUX_MODEL_REGISTRY
from collections import defaultdict
import warnings
import streamlit as st
import pandas as pd
import io
warnings.filterwarnings('ignore', category=UserWarning)

def _tuple_to_condition(tup):
    """Convertit un tuple en condition SQL pour filtrer les sous-ensembles."""
    return " and ".join([f"`C{i}` == {value}" for i, value in enumerate(tup)])

def check_missing_features(data, features):
    """Vérifie les features manquantes dans le DataFrame."""
    missing_features = [feat for feat in features if feat not in data.columns]
    present_features = [feat for feat in features if feat in data.columns]
    return missing_features, present_features

def detect_anomalies(y_true, predictions, threshold=0.01):
    """
    Détecte les anomalies entre les valeurs prédites et réelles.
    Returns: liste de tuples (index, valeur réelle, valeur prédite, différence)
    """
    anomalies = []
    for idx, (true_val, pred_val) in enumerate(zip(y_true, predictions)):
        diff = abs(true_val - pred_val)
        if diff > threshold:
            anomalies.append((idx, true_val, pred_val, diff))
    return anomalies

def generate_matricule_report(data, anomalies_by_model, output_file="rapport_anomalies.txt"):
    """Génère un rapport des anomalies par matricule et le sauvegarde dans un fichier."""
    matricule_anomalies = defaultdict(list)
    
    for model_name, anomalies_info in anomalies_by_model.items():
        base_code = model_name.split(" - ")[0]
        for idx, true_val, pred_val, _ in anomalies_info:
            matricule = data.iloc[idx]["Matricule"]
            anomaly_info = {
                'base': base_code,
                'true_val': true_val,
                'pred_val': pred_val
            }
            matricule_anomalies[matricule].append(anomaly_info)
    
    # Création du rapport
    report_lines = []
    report_lines.append("=== Rapport des anomalies par matricule ===")
    report_lines.append("-" * 50)
    
    for matricule, anomalies in sorted(matricule_anomalies.items()):
        for anomaly in anomalies:
            line = (f"Matricule {matricule} : {anomaly['base']} : "
                   f"Valeur réelle = {anomaly['true_val']:.2f}, "
                   f"Valeur prédite = {anomaly['pred_val']:.2f}")
            report_lines.append(line)
            print(line)
    
    # Sauvegarder dans un fichier
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"\nRapport sauvegardé dans : {output_file}")

def predict_for_base(data, base_code, prediction_type='both', seuil=1e-10):
    """Fait des prédictions pour une base spécifique"""
    data = load_and_preprocess_data(data.copy())
    model_config = MODEL_REGISTRY.get_config(base_code)
    
    if not model_config:
        return [], {}
    
    all_predictions = []
    anomalies_by_model = {}
    
    # Pour le modèle 7P10, ajouter des logs détaillés
    if base_code == '7P10':
        print("\n=== Analyse détaillée du modèle 7P10 ===")
        print("1. Vérification des classifications...")

    # Générer les colonnes de classification
    for clf_id, clf_config in model_config.classifiers.items():
        column_name = f"C{clf_id[-1]}"
        try:
            data[column_name] = data.eval(clf_config.condition).astype(int)
            if base_code == '7P10':
                print(f"   - Classification {clf_id} appliquée")
        except Exception as e:
            if base_code == '7P10':
                print(f"   - Erreur classification {clf_id}: {str(e)}")
            return [], {}

    # Prédictions de régression
    if prediction_type in ['regression', 'both']:
        if base_code == '7P10':
            print("\n2. Application des régresseurs...")
            
        for reg_config in model_config.regressors.values():
            try:
                subset_df = data.query(_tuple_to_condition(reg_config.condition_tuple))
                if not subset_df.empty:
                    if base_code == '7P10':
                        print(f"\n   - Régresseur {reg_config.name}:")
                        print(f"     Nombre d'enregistrements à traiter: {len(subset_df)}")
                    
                    model = joblib.load(f"Models/Bases/{reg_config.name}.joblib")
                    predictions = model.predict(subset_df[reg_config.features])
                    predictions = [0 if abs(pred) < seuil else pred for pred in predictions]
                    
                    data.loc[subset_df.index, f"{base_code}_Pred_reg_{reg_config.condition_tuple}"] = predictions
                    y_true = subset_df[reg_config.target]
                    
                    if base_code == '7P10':
                        for idx, (true_val, pred_val) in enumerate(zip(y_true, predictions)):
                            if abs(true_val - pred_val) > 0.01:  # Seuil d'anomalie
                                print(f"     Ligne {subset_df.index[idx]}:")
                                print(f"     → Valeur réelle: {true_val:.2f}")
                                print(f"     → Valeur prédite: {pred_val:.2f}")
                                print(f"     → Différence: {abs(true_val - pred_val):.2f}")
                    
                    anomalies = detect_anomalies(y_true, predictions)
                    if anomalies:
                        anomalies_by_model[f"{base_code} - {reg_config.name}"] = anomalies
            
            except Exception as e:
                if base_code == '7P10':
                    print(f"   - Erreur dans le régresseur: {str(e)}")
                continue

    if base_code == '7P10':
        print("\n=== Fin de l'analyse 7P10 ===")
        # Vérifier les colonnes nécessaires pour 7P10
        required_columns = {
            "Base": "Base 7P10",
            "Cumul Base": "Cumul Base 7P10",
            "Cum Assiette": "Cum Assiette 7P10",
            "Employee type": "Employee type",
            "Cum Plafond": "Cum Plafond",
        }
        print("\nVérification des colonnes nécessaires pour 7P10:")
        missing_cols = []
        for desc, col in required_columns.items():
            if col not in data.columns:
                missing_cols.append(f"- {desc}: {col}")
            else:
                print(f"✓ {desc}: {col} présente")
                if desc == "Base":
                    print("\nComparaison des valeurs pour 7P10:")
                    print("-" * 50)
                    pred_col = f"7P10_Pred_reg_{next(iter(model_config.regressors.values())).condition_tuple}"
                    # Debug prints
                    print(f"Colonne de prédiction recherchée: {pred_col}")
                    print(f"Colonnes disponibles: {data.columns.tolist()}")
                    
                    real_vals = data[col]
                    print(f"\nValeurs réelles disponibles: {len(real_vals)}")
                    
                    if pred_col in data.columns:
                        pred_vals = data[pred_col]
                        print("\nValeurs prédites trouvées!")
                        print("Index | Valeur réelle | Valeur prédite | Différence")
                        print("-" * 60)
                        for idx, (real, pred) in enumerate(zip(real_vals, pred_vals)):
                            diff = abs(real - pred)
                            print(f"{idx:5d} | {real:12.2f} | {pred:13.2f} | {diff:10.2f}")
                    else:
                        print(f"\nColonne de prédiction '{pred_col}' non trouvée dans le DataFrame")
                                        
        if missing_cols:
            print("\n⚠️ Colonnes manquantes:")
            for col in missing_cols:
                print(col)
        print("\n=== Fin de la vérification ===\n")
        
    return all_predictions, anomalies_by_model

def predict_for_model(data, base_code, prediction_type='both', seuil=1e-10):
    """Fait des prédictions pour une base et son taux spécifique"""
    # Prétraitement obligatoire des données
    data = load_and_preprocess_data(data)  # Appliquer le prétraitement général
    from config_taux import preprocess_taux_data
    data = preprocess_taux_data(data)  # Appliquer le prétraitement des taux
    
    base_anomalies = predict_for_base(data, base_code, prediction_type, seuil)
    taux_anomalies = predict_for_taux(data, base_code)
    
    # Combiner les anomalies
    combined_anomalies = []
    if base_anomalies[0] or taux_anomalies[0]:  # Si l'une des listes contient des anomalies
        combined_anomalies = list(set(base_anomalies[0] + taux_anomalies[0]))
    
    # Combiner les détails des anomalies
    combined_details = {
        **base_anomalies[1],  # Détails des anomalies de base
        **taux_anomalies[1]   # Détails des anomalies de taux
    }
    
    return combined_anomalies, combined_details

def predict_for_taux(data, base_code):
    """Fait des prédictions pour un taux spécifique"""
    model_config = TAUX_MODEL_REGISTRY.get_config(base_code)
    if not model_config:
        return [], {}
    
    all_predictions = []
    anomalies_by_model = {}
    
    # Vérifier les features manquantes
    missing_features, present_features = check_missing_features(data, model_config.get_all_features())
    if missing_features:
        print(f"\nERREUR - Features manquantes pour le taux {base_code}:")
        for feat in missing_features:
            print(f"- {feat}")
        return [], {}
    
    # Prédictions de classification pour les taux
    for clf_id, clf_config in model_config.classifiers.items():
        try:
            print(f"\nPrédictions taux {base_code} - classification {clf_id}...")
            model = joblib.load(f"Models/Taux/{clf_config.name}.joblib")  # Déjà correct
            predictions = model.predict(data[clf_config.features])
            
            # Stocker les prédictions
            column_name = f"{base_code}_Taux_Pred_{clf_id}"
            data[column_name] = predictions
            
            # Vérifier les anomalies
            y_true = data[f"{base_code}Taux"].values  # Supposons que la colonne du taux réel suit ce format
            anomalies = [(idx, true_val, pred_val, abs(true_val - pred_val)) 
                        for idx, (true_val, pred_val) in enumerate(zip(y_true, predictions)) 
                        if abs(true_val - pred_val) > 0.01]
            
            if anomalies:
                anomalies_by_model[f"{base_code}_Taux_{clf_id}"] = anomalies
            
        except Exception as e:
            print(f"Erreur {base_code} - Taux {clf_id}: {str(e)}")
    
    return all_predictions, anomalies_by_model

def print_predictions(y_true, predictions, indices, is_regression=False, model_name=""):
    print("\nComparaisons:")
    print("Index  |  Réel  |  Prédit")
    print("-" * 35)
    fmt = ".2f" if is_regression else "d"
    
    if is_regression:
        # Créer un DataFrame pour l'affichage Streamlit
        comparison_data = []
        for idx in indices:
            comparison_data.append({
                "Index": idx,
                "Valeur Réelle": round(y_true[idx], 2),
                "Valeur Prédite": round(predictions[idx], 2),
                "Différence": round(abs(y_true[idx] - predictions[idx]), 2)
            })
        df_comparison = pd.DataFrame(comparison_data)
        
        
        anomalies = detect_anomalies(y_true, predictions)
        if anomalies:
            base_code = model_name.split(" - ")[0]
            st.write(f"\nPour la rubrique {base_code} le système a détecté : {len(anomalies)} anomalies")
            
            # Afficher les 5 premières anomalies dans un tableau
            anomalies_data = []
            for idx, true_val, pred_val, diff in anomalies[:5]:
                anomalies_data.append({
                    "Index": idx,
                    "Valeur Réelle": round(true_val, 2),
                    "Valeur Prédite": round(pred_val, 2),
                    "Différence": round(diff, 2)
                })
        return len(anomalies)
    
    # Pour la classification, gardez l'affichage console existant
    for idx in indices[:5]:
        print(f"{idx:5d}  |  {y_true[idx]:{fmt}}  |  {predictions[idx]:{fmt}}")
    return 0

def streamlit_anomaly_detection(col2):
    """Interface de détection avancée des anomalies"""
    # Partie upload dans col2
    with col2:
        st.markdown("""
            <div style='background-color: white; padding: 15px; border-radius: 5px; text-align: left;'>
                <h4 style='color: #386161;'>📄 Fichier à analyser</h4>
            </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Chargement fichier",
            type=["csv"],
            help="Chargez votre bulletin de paie extrait sous format CSV pour l'analyse",
            key="analyze_file_autres"
        )

    # Si un fichier est uploadé, le reste du processus prend toute la largeur
    if uploaded_file is not None:
        try:
            # Charger les données brutes
            data = pd.read_csv(uploaded_file)
            
            # Prétraiter les données
            data = load_and_preprocess_data(data.copy())
            st.success("Fichier chargé et prétraité avec succès.")
            
            # Aperçu des données sur toute la largeur
            st.markdown("""
                <div style='background-color: white; padding: 15px; border-radius: 5px; margin-top: 20px;'>
                    <h4 style='color: #386161;'>📋 Aperçu des données</h4>
                </div>
            """, unsafe_allow_html=True)
            st.dataframe(data.head(), height=200, use_container_width=True)

            # Le reste du code sur toute la largeur...
            if st.button("🔍 Lancer la détection des anomalies", use_container_width=True):
                with st.spinner("Analyse en cours..."):
                    all_anomalies = {}
                    progress_bar = st.progress(0)
                    
                    # Pour chaque base de données
                    total_bases = len(MODEL_REGISTRY.keys())
                    for idx, base_code in enumerate(MODEL_REGISTRY.keys()):
                        st.write(f"Analyse de la base {base_code}...")
                        
                        try:
                            predictions, anomalies = predict_for_model(data, base_code)
                            if anomalies:  # Vérifier si anomalies n'est pas vide
                                all_anomalies.update(anomalies)
                        except Exception as e:
                            st.warning(f"Erreur lors de l'analyse de {base_code}: {str(e)}")
                            continue  # Passer à la base suivante en cas d'erreur
                        
                        # Mettre à jour la barre de progression
                        progress_bar.progress((idx + 1) / total_bases)
                    
                    # Générer et afficher le rapport
                    if all_anomalies:
                        st.subheader("Rapport des anomalies")
                        
                        # Générer le rapport formaté
                        report_text = "=== Rapport des anomalies par matricule ===\n"
                        report_text += "-" * 50 + "\n"
                        
                        # Créer le dictionnaire des anomalies par matricule
                        matricule_anomalies = defaultdict(list)
                        for model_name, anomalies_info in all_anomalies.items():
                            base_code = model_name.split(" - ")[0]
                            for idx, true_val, pred_val, _ in anomalies_info:
                                matricule = data.iloc[idx]["Matricule"]
                                anomaly_info = {
                                    'base': base_code,
                                    'true_val': true_val,
                                    'pred_val': pred_val
                                }
                                matricule_anomalies[matricule].append(anomaly_info)
                        
                        # Ajouter les lignes au rapport
                        for matricule, anomalies in sorted(matricule_anomalies.items()):
                            for anomaly in anomalies:
                                report_text += (f"Matricule {matricule} : {anomaly['base']} : "
                              f"Valeur réelle = {anomaly['true_val']:.2f}, "
                              f"Valeur prédite = {anomaly['pred_val']:.2f}\n")
                        
                        # Afficher et permettre le téléchargement du rapport
                        st.text_area("Rapport des anomalies", report_text, height=300)
                        st.download_button(
                            label="Télécharger le rapport",
                            data=report_text,
                            file_name="rapport_anomalies.txt",
                            mime="text/plain"
                        )
                        
                        # Créer un DataFrame pour afficher les anomalies
                        anomalies_list = []
                        for model_name, model_anomalies in all_anomalies.items():
                            for idx, true_val, pred_val, diff in model_anomalies:
                                anomalies_list.append({
                                    "Matricule": data.iloc[idx]["Matricule"],
                                    "Cotisation": model_name.split(" - ")[0],
                                    "Valeur réelle": true_val,
                                    "Valeur prédite": pred_val,
                                    "Différence": diff
                                })
                        
                        if anomalies_list:
                            df_anomalies = pd.DataFrame(anomalies_list)
                            st.dataframe(df_anomalies)
                            
                            # Bouton de téléchargement du rapport
                            csv = df_anomalies.to_csv(index=False)
                            st.download_button(
                                label="Télécharger le rapport détaillé (CSV)",
                                data=csv,
                                file_name="rapport_anomalies_detaille.csv",
                                mime="text/csv"
                            )
                    else:
                        st.success("🎉 Aucune anomalie détectée.")
                            
                progress_bar.empty()
                st.success("Analyse terminée!")
                
        except Exception as e:
            st.error(f"Erreur lors de l'analyse : {str(e)}")

def predict_main(col2):
    """Point d'entrée principal pour la détection d'anomalies"""
    streamlit_anomaly_detection(col2)

if __name__ == "__main__":
    predict_main()