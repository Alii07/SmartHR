import camelot
import pandas as pd
import streamlit as st
import tempfile
import concurrent.futures
from io import StringIO
from PyPDF2 import PdfReader
import csv
import re
import random
from extraction_process import *
import io

def extraction_main():
    st.markdown(
        """
        <style>
            /* Personnalisation de l'en-tête */
            .header-title {
                font-size: 36px;
                font-weight: bold;

                padding-top: 25px; 
                padding-bottom: 25px;
            }

            /* Style pour les boutons */
            .stButton > button {
                color: black;

            }
            .stButton > button:hover {
                background-color: white;
                color: #8ac447;
            }

            /* Style des zones d'entrée */
            .stTextInput, .stFileUploader, .stSelectbox {
                background-color: white;
                color: #8ac447;
                border-radius: 5px;
                padding-top: 25px; 
                padding-bottom: 25px;
            }

            /* Style des messages d'erreur */
            .stAlert {
                background-color: #ffcccc;
                color: red;
                border-radius: 5px;
            }

            /* Alignement du logo */
            .logo-container {
                text-align: right;
                padding-right: 20px;
            }
            .hero-subtitle {
                font-size: 1.3em !important;
                margin-bottom: 2rem;
                font-style: italic;
            }

            /* Assure que le footer reste au fond de la page */
            .main {
                display: flex;
                flex-direction: column;
                min-height: 100vh;
            }
            .footer {
                margin-top: auto;
            }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    st.title('Extraction de bulletins de paie')
    
    st.markdown("<p class='hero-subtitle'>Ici vous pourrez extraire et restructurer vos bulletins de paie pour toute analyse ultérieure.</p>", unsafe_allow_html=True)

    # Section 1: Sélection du type de cotisation
    st.markdown("""
        <div style='background-color: white; padding-top: 25px; padding-bottom: 25px; border-radius: 10px;'>
            <h3 style='color: #386161;'>1️⃣ Type de cotisation</h3>
            <p style='color: #666666;'>
                Sélectionnez les types de cotisations que vous souhaitez extraire des bulletins de paie.
                Cette sélection déterminera les colonnes requises dans les fichiers complémentaires.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Remplacer le selectbox simple par une sélection multiple
    report_types = st.multiselect(
        'Types de cotisations à extraire',
        ['URSSAF', 'Pole emploi', 'Retraite complémentaire', 'Prevoyance', 'APEC', 'CEG', 'CET'],
        default=['URSSAF'],
        help="""
        Sélectionnez un ou plusieurs types de cotisations :
        - URSSAF : Cotisations de sécurité sociale
        - Pole emploi : Cotisations chômage
        - Retraite complémentaire : AGIRC-ARRCO
        - Prevoyance : Cotisations prévoyance
        - APEC : Cotisations cadres
        - CEG : Contribution d'équilibre général
        - CET : Contribution d'équilibre technique
        
        Les colonnes requises seront combinées pour tous les types sélectionnés.
        """
    )

    # Section 2: Chargement des données
    st.markdown("""
        <div style='background-color: white; padding-top: 25px; padding-bottom: 25px; border-radius: 10px;'>
            <h3 style='color: #386161;'>2️⃣ Chargement des fichiers</h3>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div style='background-color: #f0f2f6; padding: 15px; border-radius: 5px;'>
            <h4 style='color: #8ac447;'>📄 Bulletins de paie</h4>
            <p>Chargez le fichier PDF contenant les bulletins de paie à analyser.</p>
        </div>
    """, unsafe_allow_html=True)
    uploaded_pdf = st.file_uploader(
        "Bulletins de paie (PDF)",
        type=["pdf"],
        help="""
        Le fichier PDF doit contenir les bulletins de paie avec:
        - Les informations d'identification (matricule, nom)
        - Le détail des cotisations
        - Les bases, taux et montants
        - Les totaux (brut, net)
        """
    )

    st.markdown("""
        <div style='background-color: #f0f2f6; padding: 15px; border-radius: 5px; text-align: left;'>
            <h4 style='color: #8ac447;'>📊 Données complémentaires</h4>
            <p>Chargez les fichiers Excel contenant les données additionnelles.</p>
            <p>
            Pour mieux comprendre les fichiers complémentaires, veuillez vous référer à la 
            <a href="#documentation-section" style='color: #8ac447;'>documentation</a>.
        </p>
        </div>
    """, unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader(
        "Fichiers complémentaires (Excel)",
        type=['xlsx', 'xls'],
        accept_multiple_files=True,
        help="Pour connaître la structure et le contenu requis des fichiers complémentaires, consultez la documentation ci-dessous."
    )



    st.markdown("""
        <div style='background-color: white; padding-top: 25px; border-radius: 10px;' id="documentation-section">
            <h3 style='color: #386161;'>3️⃣ Documentation et Template</h3>
            <p style='color: #666666;'>
                Visualisez et téléchargez le template des données complémentaires requis selon vos sélections.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # st.markdown("""
    #     <style>
    #         /* Style pour centrer le texte du bouton */
    #         .stButton>button {
    #             display: flex !important;
    #             align-items: center !important;
    #             justify-content: center !important;
    #             text-align: center !important;
    #         }
    #     </style>
    # """, unsafe_allow_html=True)

    if st.button("📋 Afficher le template des données requises", use_container_width=True):
        template_df = generate_template_dataframe(report_types)
        
        st.markdown("""
            <div style='background-color: white; padding: 15px; border-radius: 5px; margin-top: 20px;'>
                <h4 style='color: #386161;'>Structure du template</h4>
                <p style='color: #666666;'>Les colonnes suivantes sont requises pour les types de cotisations sélectionnés :</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Afficher seulement l'en-tête du template
        st.dataframe(template_df, height=100, use_container_width=True)
        
        # Convertir le DataFrame en fichier Excel en mémoire
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            template_df.to_excel(writer, index=False, sheet_name='Template')
        
        st.markdown("""
            <div style='background-color: #f0f2f6; padding: 15px; border-radius: 5px; margin-top: 20px;'>
                <p style='color: #386161; text-align: center;'>
                    Pour télécharger ce template au format Excel, cliquez ci-dessous 👇
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        st.download_button(
            label="📥 Télécharger le template Excel",
            data=output.getvalue(),
            file_name="template_donnees_complementaires.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    

    try:
        with open("Livrable_Template.pdf", "rb") as livrable_file:
            livrable_content = livrable_file.read()
            st.download_button(
                label="📕 Guide d'utilisation du template",
                data=livrable_content,
                file_name="Livrable_Template.pdf",
                mime="application/pdf",
                help="Guide détaillé expliquant la structure des fichiers et leur utilisation",
                use_container_width=True
            )
        st.markdown("""<div class="espace"></div>""", unsafe_allow_html=True)
        st.markdown("""<style>.espace{margin-bottom: 400px;}</style>""", unsafe_allow_html=True)

    except FileNotFoundError:
        st.error("Guide PDF non trouvé")

    # Section 4: Résultats et téléchargement
    if uploaded_pdf is not None and uploaded_files:
        st.markdown("""
            <div style='background-color: white; padding-top: 25px; padding-bottom: 25px; border-radius: 10px;'>
                <h3 style='color: #386161;'>4️⃣ Traitement et résultats</h3>
            </div>
        """, unsafe_allow_html=True)

    # Mise à jour de la vérification des colonnes requises pour gérer la sélection multiple
    if uploaded_files:
        # Initialiser l'ensemble des colonnes requises
        required_columns = set()
        
        # Ajouter les colonnes requises pour chaque type sélectionné
        for report_type in report_types:
            columns_mapping = {
                'URSSAF': URSSAF_COLUMNS,
                'Pole emploi': POLE_EMPLOI_COLUMNS,
                'Retraite complémentaire': RETRAITE_COMPLEMENTAIRE_COLUMNS,
                'Prevoyance': PREVOYANCE_COLUMNS,
                'APEC': APEC_COLUMNS,
                'CEG': CEG_COLUMNS,
                'CET': CET_COLUMNS
            }
            required_columns.update(columns_mapping[report_type])

        # Vérifier les colonnes présentes dans les fichiers uploadés
        uploaded_columns = set()
        for file in uploaded_files:
            df = pd.read_excel(file)
            uploaded_columns.update(df.columns)

        missing_columns = required_columns - uploaded_columns
        
        if missing_columns:
            st.error("⚠️ Colonnes manquantes pour les types sélectionnés:")
            for col in sorted(missing_columns):
                st.warning(f"- {col}")
            st.info("Veuillez ajouter ces colonnes dans vos fichiers Excel avant de continuer.")
            return

    template = pd.read_excel("Template.xlsx")
    template_columns = []

    uploaded_columns = set()

    for file in uploaded_files:
        df = pd.read_excel(file)
        uploaded_columns.update(df.columns)

    missing_columns = [col for col in template_columns if col not in uploaded_columns]

    if uploaded_pdf is not None and uploaded_files:
        if missing_columns:
            for col in missing_columns:
                st.error(f"La colonne '{col}' manque dans les données téléchargées.")
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
                temp_pdf.write(uploaded_pdf.read())  
                temp_pdf_path = temp_pdf.name
            
            csv_files = {}

            reader = PdfReader(temp_pdf_path)
            total_pages = len(reader.pages)

            current_page_count = 0
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            max_workers = 4

            st.write(f"Extraction des tableaux pour toutes les {total_pages} pages...")

            results_list = []

            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                other_page_futures = {
                    executor.submit(process_pages, temp_pdf_path, 300, 3, str(page)): page for page in range(1, total_pages + 1)
                }
                
                for future in concurrent.futures.as_completed(other_page_futures):
                    page = other_page_futures[future]
                    try:
                        results = future.result()
                        for page_number, df_stream in results:
                            csv_content = save_table_to_memory_csv(df_stream)
                            csv_files[page_number] = csv_content

                        current_page_count += 1
                        progress_value = current_page_count / total_pages
                        progress_bar.progress(progress_value)
                        status_text.text(f"Traitement : {min(current_page_count, total_pages)}/{total_pages} pages traitées")
                        
                    except Exception as e:
                        st.write(f"Erreur lors du traitement des pages {page}: {e}")

            sorted_csv_files = dict(sorted(csv_files.items()))

            results_list.sort(key=lambda x: x[0])

            final_df = pd.DataFrame()
            for page_number, df_stream in results_list:
                final_df = pd.concat([final_df, df_stream], ignore_index=True)

            output_buffer = StringIO()
            final_df.to_csv(output_buffer, index=False)
            final_csv_content = output_buffer.getvalue()

            st.write("Fusion des fichiers Excel et CSV terminée.")

            required_elements = ['CodeLibellé', 'Base', 'Taux', 'Montant Sal.', 'Taux', 'Montant Pat.']
            required_elements2 = ['Code','Libellé', 'Base', 'Taux', 'Montant Sal.', 'Taux', 'Montant Pat.']

            filtered_files = []

            for filename, csv_content in sorted_csv_files.items():
                if check_second_line(csv_content, required_elements) or check_second_line(csv_content, required_elements2):
                    filtered_files.append((filename, csv_content))

            st.write("Les fichiers CSV filtrés sont prêts à être utilisés.")

            st.write(len(filtered_files))

            reader = PdfReader(uploaded_pdf)
            all_matricules = set()

            for page in reader.pages:
                text = page.extract_text()
                if check_for_mat(text):
                    all_matricules.update(extract_matricules(text))

            matricules_buffer = StringIO()
            csv_writer = csv.writer(matricules_buffer)
            csv_writer.writerow(["Matricule"])

            for matricule in sorted(all_matricules):
                csv_writer.writerow([matricule])

            matricules_buffer.seek(0)

            st.write(pd.read_csv(matricules_buffer))

            st.write("Extraction des matricules terminée.")

            required_elements_new = ['CodeLibellé', 'Base', 'Taux', 'Montant Sal.', 'Taux 2', 'Montant Pat.']
            required_elements2_new = ['Code','Libellé', 'Base', 'Taux', 'Montant Sal.', 'Taux 2', 'Montant Pat.']

            filtered_files = {}

            for filename, csv_content in csv_files.items():
                if check_second_line(csv_content, required_elements) or check_second_line(csv_content, required_elements2):
                    csv_content = rename_second_taux(csv_content)
                    filtered_files[filename] = csv_content

            clean_csv_files = {}
            rest_csv_files = {}

            for filename, csv_content in filtered_files.items():
                file_like_object = StringIO(csv_content)
                reader = csv.reader(file_like_object)
                
                header = next(reader)
                second_line = next(reader)

                required_indices, other_indices = split_columns(header, second_line, required_elements_new + required_elements2_new)

                clean_buffer = StringIO()
                rest_buffer = StringIO()

                clean_writer = csv.writer(clean_buffer)
                rest_writer = csv.writer(rest_buffer)

                clean_writer.writerow([header[i] for i in required_indices])
                clean_writer.writerow([second_line[i] for i in required_indices])
                for row in reader:
                    clean_writer.writerow([row[i] for i in required_indices])

                file_like_object.seek(0)
                reader = csv.reader(file_like_object)
                header = next(reader)
                second_line = next(reader)

                rest_writer.writerow([header[i] for i in other_indices])
                rest_writer.writerow([second_line[i] for i in other_indices])
                for row in reader:
                    rest_writer.writerow([row[i] for i in other_indices])

                clean_csv_files[filename] = clean_buffer.getvalue()
                rest_csv_files[filename] = rest_buffer.getvalue()

            st.write("Fichiers CSV divisés.")

            restructured_files = {}

            for filename, csv_content in clean_csv_files.items():
                try:
                    headers_row, values_row = transform_to_two_lines(csv_content, required_elements_new, required_elements2_new)

                    restructured_buffer = StringIO()
                    writer = csv.writer(restructured_buffer)
                    writer.writerow(headers_row)
                    writer.writerow(values_row)
                    
                    restructured_files[filename] = restructured_buffer.getvalue()

                except ValueError as e:
                    st.write(f"Erreur lors du traitement de {filename}: {e}")

            if restructured_files:
                st.write(f"{len(restructured_files)} fichiers restructurés prêts à être téléchargés.")
            else:
                st.write("Aucun fichier restructuré disponible.")

            processed_csv_files = process_csv_in_memory(restructured_files)
            combined_headers, combined_data = merge_csv_in_memory(processed_csv_files)
            combined_headers, combined_data = add_taux_2_columns(combined_headers, combined_data)
            combined_csv_content = write_combined_csv_to_memory(combined_headers, combined_data)

            st.write("Traitement des fichiers CSV terminé.")

            updated_csv_files = {}

            for filename, csv_content in rest_csv_files.items():
                updated_csv_files[filename] = update_headers(csv_content)

            absence_report_csv = generate_absences_report(updated_csv_files)

            matricules = sorted(list(all_matricules))
            
            # Extraire les totaux bruts du PDF
            brut_totals = extract_brut_totals(temp_pdf_path)
            
            # Modifier l'appel pour inclure brut_totals
            merged_csv_content = merge_bulletins_with_matricules(matricules, combined_csv_content, brut_totals)

            if len(uploaded_files) >= 1:
                # Lire le premier fichier pour les données de cumul
                cumul_data = pd.read_excel(uploaded_files[0])
                cumul_data['Matricule'] = cumul_data['Matricule'].astype(str)

                # Initialiser le DataFrame pour les informations des salariés
                info_salaries_data = pd.DataFrame()
                info_salaries_data['Matricule'] = pd.NA

                # Traiter tous les fichiers additionnels (à partir du deuxième)
                if len(uploaded_files) >= 2:
                    for additional_file in uploaded_files[1:]:
                        additional_data = pd.read_excel(additional_file)
                        additional_data['Matricule'] = additional_data['Matricule'].astype(str)
                        
                        if info_salaries_data.empty or 'Matricule' not in info_salaries_data.columns:
                            info_salaries_data = additional_data
                        else:
                            # Supprimer les colonnes en double sauf 'Matricule'
                            additional_data = additional_data.loc[:, ~additional_data.columns.duplicated()]
                            overlapping_columns = set(info_salaries_data.columns).intersection(additional_data.columns) - {'Matricule'}
                            additional_data = additional_data.drop(columns=overlapping_columns, errors='ignore')
                            
                            # Fusionner avec les données existantes
                            info_salaries_data = info_salaries_data.merge(
                                additional_data, on='Matricule', how='left'
                            )

                main_table = pd.read_csv(StringIO(merged_csv_content))
                for col in main_table.columns:
                    if main_table[col].dtype == 'object':
                        main_table[col] = main_table[col].apply(convert_to_float)

                main_table['Matricule'] = main_table['Matricule'].astype(str)
                cumul_data['Matricule'] = cumul_data['Matricule'].astype(str)
                info_salaries_data['Matricule'] = info_salaries_data['Matricule'].astype(str)

                # Ajout du calcul de Cum Plafond pour ALL et Pole emploi
                if report_type in ['Toutes', 'Pole emploi']:
                    if 'PLAFOND CUM M-1' in cumul_data.columns and 'PLAFOND M' in cumul_data.columns:
                        cumul_data['Cum Plafond'] = cumul_data['PLAFOND CUM M-1'] + cumul_data['PLAFOND M']

                merged_df = main_table.merge(cumul_data, on='Matricule', how='left')
                final_df = merged_df.merge(info_salaries_data, on='Matricule', how='left')

                cols = list(final_df.columns)
                if 'Nom Prénom' in cols:
                    cols.insert(1, cols.pop(cols.index('Nom Prénom')))
                final_df = final_df[cols]

                # Afficher l'aperçu des données extraites
                st.markdown("""
                    <div style='background-color: white; padding: 15px;  border-radius: 5px; margin-top: 20px;'>
                        <h4 style='color: #386161;'>📋 Aperçu des données extraites</h4>
                    </div>
                """, unsafe_allow_html=True)
                
                st.dataframe(final_df.head(), height=200, use_container_width=True)

                output_buffer = StringIO()
                final_df.to_csv(output_buffer, index=False)
                final_csv_content = output_buffer.getvalue()

                st.markdown("""
                    <div style='background-color: #f0f2f6; padding: 15px; padding-bottom:0px !important; margin-bottom:10px !important; border-radius: 5px; margin-top: 20px;'>
                        <p style='color: #386161; text-align: center;'>
                            Pour télécharger l'extraction complète, appuyez sur ce bouton 👇
                        </p>
                    </div>
                """, unsafe_allow_html=True)

                st.download_button(
                    label="📥 Télécharger le fichier complet",
                    data=final_csv_content,
                    file_name="extraction_complete.csv",
                    mime="text/csv",
                    use_container_width=True
                )

def generate_template_dataframe(selected_types):
    """Génère un DataFrame template basé sur les types de cotisations sélectionnés."""
    required_columns = {'Matricule'}  # Commence avec Matricule comme colonne obligatoire
    
    # Mapping des colonnes requises par type de cotisation
    columns_mapping = {
        'URSSAF': URSSAF_COLUMNS,
        'Pole emploi': POLE_EMPLOI_COLUMNS,
        'Retraite complémentaire': RETRAITE_COMPLEMENTAIRE_COLUMNS,
        'Prevoyance': PREVOYANCE_COLUMNS,
        'APEC': APEC_COLUMNS,
        'CEG': CEG_COLUMNS,
        'CET': CET_COLUMNS
    }
    
    # Ajouter les colonnes requises pour chaque type sélectionné
    for report_type in selected_types:
        if report_type in columns_mapping:
            required_columns.update(columns_mapping[report_type])
    
    # Créer un DataFrame vide avec les colonnes requises
    template_df = pd.DataFrame(columns=sorted(list(required_columns)))

    
    return template_df.head(0)


    
if __name__ == "__main__":
    extraction_main()
