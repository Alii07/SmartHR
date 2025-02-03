import streamlit as st
import pandas as pd
from collections import defaultdict
from predict import predict_for_base, load_and_preprocess_data
from config import MODEL_REGISTRY

def streamlit_modification_interface(uploaded_file=None):
    if uploaded_file is not None:
        try:
            # Chargement et prétraitement des données
            print("\nChargement du fichier CSV...")
            df = pd.read_csv(uploaded_file, encoding='utf-8')
            print("\nDébut du prétraitement...")
            df = load_and_preprocess_data(df)
            print("Colonnes après prétraitement:", df.columns.tolist())
            
            st.session_state.df = df
            st.success("Données chargées avec succès.")
            
            # Aperçu des données
            st.markdown("""
                <div style='background-color: white; padding: 15px; border-radius: 5px; border: 1px solid #8ac447; margin-top: 20px;'>
                    <h4 style='color: #386161;'>📋 Aperçu des données</h4>
                </div>
            """, unsafe_allow_html=True)
            st.dataframe(st.session_state.df.head(), height=200)

            # Interface de sélection
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                    <div style='background-color: white; padding: 15px; border-radius: 5px; border: 1px solid #8ac447;'>
                        <h4 style='color: #386161;'>👤 Sélection de l'employé</h4>
                    </div>
                """, unsafe_allow_html=True)
                try:
                    employe_id = st.selectbox(
                        "",
                        df['Matricule'].unique(),
                        key="employe_select"
                    )
                except KeyError:
                    st.error("❌ La colonne 'Matricule' est manquante.")
                    return None

            with col2:
                st.markdown("""
                    <div style='background-color: white; padding: 15px; border-radius: 5px; border: 1px solid #8ac447;'>
                        <h4 style='color: #386161;'>📑 Choix de la cotisation</h4>
                    </div>
                """, unsafe_allow_html=True)
                rubriques = [col for col in df.columns if col.startswith("Rub")]
                cotisation = st.selectbox("", rubriques)

            # Configuration des colonnes de cotisation
            cotisation_code = cotisation.split(" ")[1]
            colonnes_cotisation = [
                f"{cotisation_code}Base",
                f"{cotisation_code}Taux",
                f"{cotisation_code}Montant Sal.",
                f"{cotisation_code}Taux 2",
                f"{cotisation_code}Montant Pat."
            ]
            colonnes_cotisation = [col for col in colonnes_cotisation if col in df.columns]

            # Interface de modification avec mise à jour instantanée
            st.markdown("### ✏️ Modification des valeurs")
            
            # Ajout du style CSS pour la mise en page
            st.markdown("""
                <style>

                .label-box {
                    background-color: #386161;
                    padding: 10px;
                    border-radius: 5px;
                    color: white;
                    width: 100%;
                    display: flex;
                    align-items: center;
                }
                .value-box {
                    width: 100%;
                }
                .stNumberInput {
                    width: 100%;
                }
                </style>
            """, unsafe_allow_html=True)
            
            for colonne in colonnes_cotisation:
                valeur_actuelle = df.loc[df['Matricule'] == employe_id, colonne].iloc[0]
                col1, col2 = st.columns([0.3, 0.7])
                with col1 :# Création d'un conteneur pour chaque ligne
                    st.markdown(
                        f"""
                        <div class="modification-container">
                            <div class="label-box">
                                <strong>{colonne}</strong>
                            </div>
                            <div class="value-box">
                        """, 
                        unsafe_allow_html=True
                    )
                with col2: 
                    # Input numérique
                    nouvelle_valeur = st.number_input(
                        label="",  # Label vide car déjà affiché dans le div
                        value=float(valeur_actuelle),
                        format="%.2f",
                        key=f"input_{colonne}_{employe_id}",
                        label_visibility="collapsed"  # Cache complètement le label
                    )
                
                    st.markdown("</div></div>", unsafe_allow_html=True)

                    # Mise à jour instantanée de la valeur dans le DataFrame
                    if nouvelle_valeur != valeur_actuelle:
                        df.loc[df['Matricule'] == employe_id, colonne] = nouvelle_valeur
                        if 'historique_modifications' not in st.session_state:
                            st.session_state.historique_modifications = []
                        st.session_state.historique_modifications.append({
                            "Matricule": employe_id,
                            "Rubrique modifiée": colonne,
                            "Ancienne Valeur": valeur_actuelle,
                            "Nouvelle Valeur": nouvelle_valeur
                        })

            # Affichage de la ligne modifiée
            st.markdown("### 📋 Ligne courante")
            st.dataframe(df[df['Matricule'] == employe_id])

            # Bouton de détection d'anomalies
            if st.button("🔍 Détecter les anomalies"):
                with st.spinner("Analyse en cours..."):
                    try:
                        data = load_and_preprocess_data(df.copy())
                        all_anomalies = {}
                        progress_bar = st.progress(0)
                        
                        for idx, base_code in enumerate(MODEL_REGISTRY.keys()):
                            predictions, anomalies = predict_for_base(data, base_code)
                            if anomalies:
                                all_anomalies.update(anomalies)
                            progress_bar.progress((idx + 1) / len(MODEL_REGISTRY.keys()))

                        if all_anomalies:
                            report_text = generate_anomaly_report(all_anomalies, data)
                            st.text_area("📄 Rapport d'anomalies", report_text, height=200)
                        else:
                            st.success("✅ Aucune anomalie détectée.")

                        progress_bar.empty()
                    except Exception as e:
                        st.error(f"❌ Erreur lors de la détection: {e}")

        except Exception as e:
            st.error(f"Erreur: {e}")

def generate_anomaly_report(all_anomalies, data):
    report_text = "=== Rapport des anomalies ===\n\n"
    matricule_anomalies = defaultdict(list)
    
    for model_name, anomalies_info in all_anomalies.items():
        base_code = model_name.split(" - ")[0]
        for idx, _, _, _ in anomalies_info:
            matricule = data.iloc[idx]["Matricule"]
            if base_code not in matricule_anomalies[matricule]:
                matricule_anomalies[matricule].append(base_code)
    
    for matricule, bases in sorted(matricule_anomalies.items()):
        for base in bases:
            report_text += f"Anomalie détectée - Matricule {matricule} : {base}\n"
    
    return report_text

def modification_main(uploaded_file=None):
    streamlit_modification_interface(uploaded_file)

if __name__ == "__main__":
    modification_main()
