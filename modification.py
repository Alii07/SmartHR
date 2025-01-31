import streamlit as st
import pandas as pd
import io
from collections import defaultdict
from predict import predict_for_base, load_and_preprocess_data
from config import MODEL_REGISTRY

def streamlit_modification_interface(uploaded_file=None):  # Modifier pour accepter un fichier en paramètre

    if uploaded_file is not None:
        try:
            # Charger les données brutes
            print("\nChargement du fichier CSV...")
            df = pd.read_csv(uploaded_file, encoding='utf-8')
            print("Colonnes avant prétraitement:", df.columns.tolist())
            
            # Prétraiter explicitement les données
            print("\nDébut du prétraitement...")
            df = load_and_preprocess_data(df.copy())
            print("Colonnes après prétraitement:", df.columns.tolist())
            
            st.session_state.df = df
            st.success("Fichier chargé et prétraité avec succès.")
            
            # Section : Aperçu des données
            st.markdown("""
                <div style='background-color: white; padding: 15px; border-radius: 5px; border: 1px solid #8ac447; margin-top: 20px;'>
                    <h4 style='color: #386161;'>📋 Aperçu des données</h4>
                </div>
            """, unsafe_allow_html=True)
            st.dataframe(st.session_state.df.head(), height=200)

            # Sélection employé et rubrique sur la même ligne
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
                    st.error("❌ La colonne 'Matricule' est manquante dans le fichier.")
                    return None

            with col2:
                st.markdown("""
                    <div style='background-color: white; padding: 15px; border-radius: 5px; border: 1px solid #8ac447;'>
                        <h4 style='color: #386161;'>📑 Choix de la cotisation</h4>
                    </div>
                """, unsafe_allow_html=True)
                rubriques = [col for col in df.columns if col.startswith("Rub")]
                cotisation = st.selectbox("", rubriques)

            # Extraire le code de cotisation et préparer les colonnes associées
            cotisation_code = cotisation.split(" ")[1]
            colonnes_cotisation = [
                f"{cotisation_code}Base",
                f"{cotisation_code}Taux",
                f"{cotisation_code}Montant Sal.",
                f"{cotisation_code}Taux 2",
                f"{cotisation_code}Montant Pat."
            ]
            colonnes_cotisation = [col for col in colonnes_cotisation if col in df.columns]

            # Section : Modification des valeurs avec style
            st.markdown("---")
            st.markdown("### ✏️ **Modifier les Valeurs de la Cotisation**")
            st.markdown(
                """
                <style>
                .value-box {
                    border-radius: 5px;
                    padding: 10px;
                    margin-bottom: 10px;
                    background-color : #386161;
                    color: white;
                }
                </style>
                """,
                unsafe_allow_html=True
            )

            valeurs_modifiees = {}
            ligne_originale = df[df['Matricule'] == employe_id].copy()
            ligne_modifiee = ligne_originale.copy()

            # Interface de modification des valeurs
            for colonne in colonnes_cotisation:
                valeur_actuelle = ligne_modifiee[colonne].values[0]
                st.markdown(
                    f"""
                    <div class="value-box">
                        <strong>{colonne}</strong><br>
                        Valeur actuelle : {valeur_actuelle}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                nouvelle_valeur = st.text_input(
                    f"Modifier la valeur pour {colonne}",
                    value=str(valeur_actuelle),
                    help=f"Entrez une nouvelle valeur pour {colonne}."
                )
                valeurs_modifiees[colonne] = nouvelle_valeur
                ligne_modifiee[colonne] = nouvelle_valeur

            # Afficher la ligne modifiée
            st.markdown("---")
            st.markdown("### 📋 **Ligne Modifiée**")
            st.dataframe(ligne_modifiee)

            # Section : Actions
            st.markdown("---")
            st.markdown("### 🎯 **Actions**")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Confirmer les modifications"):
                    st.session_state.df = ligne_modifiee.copy()
                    csv_buffer = st.session_state.df.to_csv(index=False).encode('utf-8')
                    st.session_state.uploaded_file = csv_buffer

                    # Mise à jour de l'historique
                    if "historique_modifications" not in st.session_state:
                        st.session_state.historique_modifications = []
                    for colonne, nouvelle_valeur in valeurs_modifiees.items():
                        if str(ligne_originale[colonne].values[0]) != str(nouvelle_valeur):
                            st.session_state.historique_modifications.append({
                                "Matricule": employe_id,
                                "Rubrique modifiée": colonne,
                                "Ancienne Valeur": ligne_originale[colonne].values[0],
                                "Nouvelle Valeur": nouvelle_valeur
                            })

                    st.success("✅ Modifications confirmées.")
                    
                    # Bouton de téléchargement de la ligne modifiée
                    st.download_button(
                        label="📥 Télécharger la ligne modifiée",
                        data=st.session_state.df.to_csv(index=False).encode('utf-8'),
                        file_name="ligne_modifiee.csv",
                        mime="text/csv"
                    )
            with col2 :
                # Bouton de détection des anomalies
                if st.button("🔍 Lancer la détection des anomalies"):
                    if "uploaded_file" in st.session_state:
                        with st.spinner("Analyse en cours..."):
                            try:
                                print("\nChargement des données pour la détection d'anomalies...")
                                data = pd.read_csv(io.StringIO(st.session_state.uploaded_file.decode('utf-8')))
                                print("Colonnes avant prétraitement:", data.columns.tolist())
                                
                                print("\nPrétraitement des données pour la détection...")
                                data = load_and_preprocess_data(data.copy())
                                print("Colonnes après prétraitement:", data.columns.tolist())
                                
                                # Détecter les anomalies
                                all_anomalies = {}
                                progress_bar = st.progress(0)
                                
                                for idx, base_code in enumerate(MODEL_REGISTRY.keys()):
                                    predictions, anomalies = predict_for_base(data, base_code)
                                    if anomalies:
                                        all_anomalies.update(anomalies)
                                    progress_bar.progress((idx + 1) / len(MODEL_REGISTRY.keys()))

                                # Générer et afficher le rapport
                                if all_anomalies:
                                    report_text = "=== Rapport des anomalies par matricule ===\n"
                                    report_text += "-" * 50 + "\n"
                                    
                                    matricule_anomalies = defaultdict(list)
                                    for model_name, anomalies_info in all_anomalies.items():
                                        base_code = model_name.split(" - ")[0]
                                        for idx, _, _, _ in anomalies_info:
                                            matricule = data.iloc[idx]["Matricule"]
                                            if base_code not in matricule_anomalies[matricule]:
                                                matricule_anomalies[matricule].append(base_code)
                                    
                                    for matricule, bases in sorted(matricule_anomalies.items()):
                                        for base in bases:
                                            report_text += f"On a détecté une anomalie dans le Matricule {matricule} : {base}\n"
                                    
                                    st.text_area("📄 Rapport d'anomalies", report_text, height=200)
                                    st.download_button(
                                        label="📥 Télécharger le rapport",
                                        data=report_text,
                                        file_name="rapport_anomalies.txt",
                                        mime="text/plain"
                                    )
                                else:
                                    st.success("🎉 Aucune anomalie détectée.")

                                progress_bar.empty()
                                
                            except Exception as e:
                                st.error(f"❌ Erreur lors de la détection des anomalies : {e}")
                else:
                    st.warning("⚠️ Aucun fichier disponible pour la détection.")

            # Historique des modifications
            st.markdown("---")
            st.markdown("### 🕒 **Historique des Modifications**")
            with st.expander("Afficher l'historique des modifications"):
                if "historique_modifications" in st.session_state and st.session_state.historique_modifications:
                    df_historique = pd.DataFrame(st.session_state.historique_modifications)
                    st.dataframe(df_historique)
                    
                    csv_buffer = df_historique.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Télécharger l'historique des modifications",
                        data=csv_buffer,
                        file_name="historique_modifications.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("ℹ️ Aucune modification enregistrée pour le moment.")

        except Exception as e:
            st.error(f"Erreur lors du traitement : {e}")

def modification_main(uploaded_file=None):  # Modifier pour accepter un fichier en paramètre
    streamlit_modification_interface(uploaded_file)

if __name__ == "__main__":
    modification_main()
