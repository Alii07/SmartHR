import camelot
import pandas as pd
import streamlit as st
import tempfile
import concurrent.futures
from io import StringIO
from PyPDF2 import PdfReader
import csv
import re


def extract_table_from_pdf(pdf_file_path, edge_tol, row_tol, pages):
    try:
        st.write(f"Extraction des tableaux de la page {pages} avec edge_tol={edge_tol} et row_tol={row_tol}")
        tables_stream = camelot.read_pdf(
            pdf_file_path,
            flavor='stream',
            pages=pages,
            strip_text='\n',
            edge_tol=edge_tol,
            row_tol=row_tol
        )
        return tables_stream

    except Exception as e:
        return None


@st.cache_data
def save_table_to_memory_csv(df):
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    return csv_buffer.getvalue()

def check_second_line(file_content, required_elements):
    file_like_object = StringIO(file_content)
    reader = csv.reader(file_like_object)
    next(reader)  # Ignorer la première ligne (header)
    second_line = next(reader, None)  # Lire la deuxième ligne
    if second_line and all(elem in second_line for elem in required_elements):
        return True
    return False

def split_columns(header, second_line, required_elements):
    required_indices = [i for i, col in enumerate(second_line) if col in required_elements]
    other_indices = [i for i, col in enumerate(second_line) if col not in required_elements]
    return required_indices, other_indices

# Fonction pour traiter les pages du PDF
def process_pages(pdf_file_path, edge_tol, row_tol, page):
    tables_stream = extract_table_from_pdf(pdf_file_path, edge_tol, row_tol, pages=page)
    results = []
    if tables_stream is not None and len(tables_stream) > 0:
        largest_table = max(tables_stream, key=lambda t: t.df.shape[0] * t.df.shape[1])
        df_stream = largest_table.df
        df_stream.replace('\n', '', regex=True, inplace=True)
        df_stream.fillna('', inplace=True)
        page_number = largest_table.parsing_report['page']

        if 'Montant Sal.Taux' in df_stream.iloc[0].values:
            refined_tables = extract_table_from_pdf(pdf_file_path, edge_tol=500, row_tol=5, pages=str(page_number))
            if refined_tables is not None and len(refined_tables) > 0:
                largest_table = max(refined_tables, key=lambda t: t.df.shape[0] * t.df.shape[1])
                df_stream = largest_table.df
                df_stream.replace('\n', '', regex=True, inplace=True)
                df_stream.fillna('', inplace=True)
        
        results.append((page_number, df_stream))
    
    return results


def convert_to_float(value):
    if type(value) != float:
        try:
            value = value.replace('.', '').replace(',', '.')
            return float(value)
        except ValueError:
            return None
    return None


def process_csv_in_memory(csv_contents):
    csv_files = {}
    
    # Trier les fichiers CSV par ordre alphabétique des noms
    for filename in sorted(csv_contents.keys()):  # Trier les fichiers par ordre alphabétique
        csv_content = csv_contents[filename]
        file_like_object = StringIO(csv_content)
        reader = csv.DictReader(file_like_object)
        data = list(reader)
        headers = reader.fieldnames

        for row in data:
            for header in headers:
                if 'Base' in header:
                    rubrique_base = header
                    rubrique_montant_pat = header.replace('Base', 'Montant Pat.')
                    rubrique_taux_2 = header.replace('Base', 'Taux 2')

                    if rubrique_montant_pat in headers and rubrique_taux_2 in headers:
                        base = row.get(rubrique_base)
                        if base == 'NET A PAYER AVANT IMPÔT SUR LE REVENU':
                            base = ''
                        montant_pat = row.get(rubrique_montant_pat)
                        base_float = convert_to_float(base)
                        montant_pat_float = convert_to_float(montant_pat)
                        if base_float and montant_pat_float and not row.get(rubrique_taux_2):
                            row[rubrique_taux_2] = round((montant_pat_float / base_float) * 100, 3)

        # Enregistrer le contenu traité dans un buffer mémoire
        output_buffer = StringIO()
        writer = csv.DictWriter(output_buffer, fieldnames=headers)
        writer.writeheader()
        writer.writerows(data)
        csv_files[filename] = output_buffer.getvalue()

    return csv_files

# Fonction pour fusionner les fichiers CSV en mémoire
def merge_csv_in_memory(csv_files):
    combined_headers = []
    combined_data = []

    # Trier les fichiers CSV par ordre alphabétique des noms
    for i, (filename, csv_content) in enumerate(sorted(csv_files.items())):  # Trier par ordre alphabétique
        file_like_object = StringIO(csv_content)
        reader = csv.reader(file_like_object)
        headers = next(reader)
        rows = list(reader)

        if i == 0:
            combined_headers = headers
            combined_data = rows
        else:
            for header in headers:
                if header not in combined_headers:
                    combined_headers.append(header)

            for row in rows:
                new_row = []
                for header in combined_headers:
                    if header in headers:
                        new_row.append(row[headers.index(header)])
                    else:
                        new_row.append('')

                combined_data.append(new_row)

    return combined_headers, combined_data

# Fonction pour ajouter la colonne "Taux 2" si elle n'existe pas déjà
def add_taux_2_columns(combined_headers, combined_data):
    code_pattern = re.compile(r'^(\d{4})\s')

    for i, header in enumerate(combined_headers):
        match = code_pattern.match(header)
        if match:
            code = match.group(1)
            base_column = f'{code}Base'
            montant_pat_column = f'{code}Montant Pat.'
            taux_2_column = f'{code}Taux 2'

            if base_column in combined_headers and montant_pat_column in combined_headers:
                base_idx = combined_headers.index(base_column)
                montant_pat_idx = combined_headers.index(montant_pat_column)

                if taux_2_column not in combined_headers:
                    combined_headers.append(taux_2_column)

                for row in combined_data:
                    try:
                        base_value = float(row[base_idx])
                        montant_pat_value = float(row[montant_pat_idx])
                        taux_2_value = base_value / montant_pat_value if montant_pat_value != 0 else ''

                        if taux_2_value != '':
                            taux_2_value = format(round(taux_2_value, 3), '.3f')
                    except (ValueError, TypeError):
                        taux_2_value = ''

                    if len(row) < len(combined_headers):
                        row.append(taux_2_value)
                    else:
                        row[combined_headers.index(taux_2_column)] = taux_2_value

    return combined_headers, combined_data

# Fonction pour écrire le CSV fusionné et retourné sous forme de StringIO
def write_combined_csv_to_memory(combined_headers, combined_data):
    output_buffer = StringIO()
    writer = csv.writer(output_buffer)
    writer.writerow(combined_headers)
    writer.writerows(combined_data)
    return output_buffer.getvalue()

# Traiter et fusionner les fichiers CSV


st.write("Traitement des fichiers CSV terminé.")


# Fonction pour mettre à jour les en-têtes des fichiers CSV en mémoire
def update_headers(csv_content):
    df = pd.read_csv(StringIO(csv_content), header=None)

    new_headers = []

    for column in df:
        column_values = df[column].astype(str)
        if 'Abs.' in column_values.values:
            new_headers.append('Abs.')
        elif 'Date Equipe Hor.Abs.' in column_values.values:
            new_headers.append('Date Equipe Hor.Abs.')
        else:
            new_headers.append(df.columns[column])
    df.columns = new_headers

    output_buffer = StringIO()
    df.to_csv(output_buffer, index=False)
    return output_buffer.getvalue()
# Fonction pour renommer la deuxième occurrence de "Taux" en "Taux 2"
def rename_second_taux(csv_content):
    file_like_object = StringIO(csv_content)
    reader = csv.reader(file_like_object)
    lines = list(reader)
    
    if len(lines) > 1:
        second_line = lines[1]
        taux_indices = [i for i, col in enumerate(second_line) if col == 'Taux']
        if len(taux_indices) > 1:
            second_line[taux_indices[1]] = 'Taux 2'
    
    output_buffer = StringIO()
    writer = csv.writer(output_buffer)
    writer.writerows(lines)
    return output_buffer.getvalue()

# Fonction pour vérifier la deuxième ligne du fichier CSV
def check_second_line(csv_content, required_elements):
    file_like_object = StringIO(csv_content)
    reader = csv.reader(file_like_object)
    next(reader)  # Ignorer la première ligne (header)
    second_line = next(reader, None)  # Lire la deuxième ligne
    if second_line and all(elem in second_line for elem in required_elements):
        return True
    return False

# Fonction pour séparer les colonnes requises et non requises
def split_columns(header, second_line, required_elements):
    required_indices = [i for i, col in enumerate(second_line) if col in required_elements]
    other_indices = [i for i, col in enumerate(second_line) if col not in required_elements]
    return required_indices, other_indices

# Fonction pour générer un taux aléatoire autour du taux initial
def generer_taux_2(taux_initial):
    return round(random.uniform(taux_initial - 0.2, taux_initial + 0.2), 2)

# Fonction pour convertir une valeur en float en toute sécurité
def safe_float_conversion(value):
    try:
        return float(value)
    except ValueError:
        return None

def transform_to_two_lines(csv_content, required_elements_new, required_elements2_new):
    headers_row = []
    values_row = []

    # Créer un objet similaire à un fichier à partir de csv_content
    file_like_object = StringIO(csv_content)
    reader = csv.reader(file_like_object)

    # Lire l'en-tête et la deuxième ligne
    header = next(reader, None)
    second_line = next(reader, None)

    # Validation des colonnes: Assouplissement de la condition
    if not any(all(elem in second_line for elem in req_set) for req_set in (required_elements_new, required_elements2_new)):
        raise ValueError("Les colonnes requises ne sont pas présentes dans le fichier CSV.")

    # Trouver les index des colonnes
    code_index = second_line.index('Code') if 'Code' in second_line else None
    libelle_index = second_line.index('Libellé') if 'Libellé' in second_line else None
    codelibelle_index = second_line.index('CodeLibellé') if 'CodeLibellé' in second_line else None

    # Initialiser la liste des rubriques
    rubriques = []

    # Parcourir le fichier CSV pour construire les en-têtes
    if codelibelle_index is not None:
        for row in reader:
            rubrique = row[codelibelle_index]
            if rubrique and rubrique[0].isdigit():
                rubrique = rubrique[:4]  # Utiliser les 4 premiers caractères pour la rubrique
                if rubrique not in rubriques:
                    rubriques.append(rubrique)
                    headers_row.extend([
                        f"{rubrique}Base",
                        f"{rubrique}Taux",
                        f"{rubrique}Montant Sal.",
                        f"{rubrique}Taux 2",  
                        f"{rubrique}Montant Pat."
                    ])
    elif code_index is not None and libelle_index is not None:
        for row in reader:
            rubrique = f"{row[code_index]}{row[libelle_index]}"
            if rubrique and rubrique[0].isdigit():
                rubrique = rubrique[:4]  # Utiliser les 4 premiers caractères pour la rubrique
                if rubrique not in rubriques:
                    rubriques.append(rubrique)
                    headers_row.extend([
                        f"{rubrique}Base",
                        f"{rubrique}Taux",
                        f"{rubrique}Montant Sal.",
                        f"{rubrique}Taux 2",  
                        f"{rubrique}Montant Pat."
                    ])
    
    # Réinitialiser le lecteur pour lire à nouveau les lignes
    file_like_object.seek(0)
    reader = csv.reader(file_like_object)
    next(reader)  # Ignorer la première ligne (header)
    next(reader)  # Ignorer la deuxième ligne (second_line)

    values_row = ['' for _ in range(len(headers_row))]

    # Remplir les valeurs pour chaque rubrique
    for row in reader:
        if codelibelle_index is not None:
            code_libelle = row[codelibelle_index]
            if code_libelle and code_libelle[0].isdigit():
                code_libelle = code_libelle[:4]  # Utiliser les 4 premiers caractères
        elif code_index is not None and libelle_index is not None:
            code_libelle = f"{row[code_index]}{row[libelle_index]}"
            if code_libelle and code_libelle[0].isdigit():
                code_libelle = code_libelle[:4]  # Utiliser les 4 premiers caractères
        else:
            raise ValueError("Les colonnes 'Code' et 'Libellé' ou 'CodeLibellé' sont manquantes")

        # Gestion de l'index pour la rubrique
        try:
            rubrique_index = headers_row.index(f"{code_libelle}Base") // 5 * 5
            for i, col in enumerate(['Base', 'Taux', 'Montant Sal.', 'Taux 2', 'Montant Pat.']):
                if col in second_line:
                    col_index = second_line.index(col)
                    values_row[rubrique_index + i] = row[col_index]
        except ValueError:
            continue

    return headers_row, values_row

# Fonction pour vérifier si "Mat:" est présent dans le texte d'une page
def check_for_mat(text):
    return 'Mat:' in text

# Fonction pour extraire les matricules du texte d'une page
def extract_matricules(text):
    matricules = set()
    for line in text.split('\n'):
        if 'Mat:' in line:
            start = line.find('Mat:') + len('Mat:')
            end = line.find('/ Gest:', start)
            if end == -1:
                end = len(line)
            matricule = line[start:end].strip()
            matricules.add(matricule)
    return matricules

# Fonction pour traiter un DataFrame et calculer les absences
def process_dataframe(df):
    absences_par_jour = 0
    absences_par_heure = 0.0

    columns_to_check = [col for col in df.columns if 'Abs.' in col or 'Date Equipe Hor.Abs.' in col]

    for col in columns_to_check:
        for cell in df[col].astype(str):
            cell = cell.strip()
            if 'AB' in cell:
                absences_par_jour += 1
                match = re.search(r'(\d+(?:\.\d+)?)AB$', cell)
                if match:
                    absences_par_heure += float(match.group(1))

    return absences_par_jour, absences_par_heure

# Fonction pour traiter les fichiers CSV en mémoire et générer un rapport d'absences
def generate_absences_report(csv_files):
    report_data = [['Nom du Fichier', 'Absences par Jour', 'Absences par Heure']]

    for filename, csv_content in csv_files.items():
        df = pd.read_csv(StringIO(csv_content))
        absences_par_jour, absences_par_heure = process_dataframe(df)
        report_data.append([filename, absences_par_jour, absences_par_heure])

    output_buffer = StringIO()
    writer = csv.writer(output_buffer)
    writer.writerows(report_data)
    return output_buffer.getvalue()

# Fonction pour fusionner les bulletins avec les matricules en mémoire
def merge_bulletins_with_matricules(matricules, combined_csv_content):
    combined_data = []
    reader = csv.reader(StringIO(combined_csv_content))
    combined_data = [row for row in reader]

    if len(matricules) > len(combined_data):
        raise ValueError("Le fichier de matricules contient plus de lignes que le fichier combined_output.")

    merged_data = []
    merged_data.append(["Matricule"] + combined_data[0])

    for i, row in enumerate(combined_data[1:], start=1):
        if i <= len(matricules):
            matricule = matricules[i - 1]
            merged_data.append([matricule] + row)
        else:
            break

    output_buffer = StringIO()
    writer = csv.writer(output_buffer)
    writer.writerows(merged_data)
    return output_buffer.getvalue()







        
    # Conversion des types et fusion avec les autres fichiers CSV
    def convert_to_float(value):
        if pd.notnull(value):
            try:
                value_str = str(value).strip()
                if value_str.endswith('-'):
                    value_str = '-' + value_str[:-1]
                value_str = value_str.replace('.', '').replace(',', '.')
                return pd.to_numeric(value_str, errors='coerce')
            except ValueError:
                return value
        return value

    main_table = pd.read_csv(StringIO(merged_csv_content))
    for col in main_table.columns:
        if main_table[col].dtype == 'object':
            main_table[col] = main_table[col].apply(convert_to_float)

    main_table['Matricule'] = main_table['Matricule'].astype(str)
    cumul_data['Matricule'] = cumul_data['Matricule'].astype(str)
    info_salaries_data['Matricule'] = info_salaries_data['Matricule'].astype(str)

    merged_df = main_table.merge(cumul_data, on='Matricule', how='left')
    final_df = merged_df.merge(info_salaries_data, on='Matricule', how='left')

    # Réarranger les colonnes si nécessaire
    cols = list(final_df.columns)
    if 'Nom Prénom' in cols:
        cols.insert(1, cols.pop(cols.index('Nom Prénom')))
    final_df = final_df[cols]

    # Sauvegarder le résultat final dans un buffer CSV
    output_buffer = StringIO()
    final_df.to_csv(output_buffer, index=False)
    final_csv_content = output_buffer.getvalue()

    # Proposer le fichier final en téléchargement
    st.download_button(
        label="Télécharger le fichier fusionné",
        data=final_csv_content,
        file_name="fichier_fusionné.csv",
        mime="text/csv"
    )

    st.write("Fusion des fichiers Excel et CSV terminée.")
# Titre de l'application Streamlit
st.header("Extraction de bulletins de paie à partir de PDF")

uploaded_pdf = st.file_uploader("Téléverser un fichier PDF", type=["pdf"])

num_files = st.number_input("How many Excel files do you want to upload?", min_value=1, max_value=5, step=1)

uploaded_files = []
for i in range(1, num_files + 1):  # Adjust the loop based on the chosen number
    file = st.file_uploader(f"Upload Excel file {i}", type=['xlsx', 'xls'], key=f"file_{i}")
    if file:
        uploaded_files.append(file)

if len(uploaded_files) >= 1:  # At least one file is uploaded
    # Process the first file as `cumul_data`
    cumul_data = pd.read_excel(uploaded_files[0])
    cumul_data['Matricule'] = cumul_data['Matricule'].astype(str)  # Ensure consistent data type

    # Initialize `info_salaries_data` with the second file or None
    if len(uploaded_files) >= 2:
        info_salaries_data = pd.read_excel(uploaded_files[1])
        info_salaries_data['Matricule'] = info_salaries_data['Matricule'].astype(str)  # Ensure consistent data type
        
        # Remove duplicate columns in info_salaries_data
        info_salaries_data = info_salaries_data.loc[:, ~info_salaries_data.columns.duplicated()]

        # Merge additional files into `info_salaries_data`
        for additional_file in uploaded_files[2:]:
            additional_data = pd.read_excel(additional_file)
            additional_data['Matricule'] = additional_data['Matricule'].astype(str)  # Ensure consistent data type
            
            # Remove duplicate columns in the additional data
            additional_data = additional_data.loc[:, ~additional_data.columns.duplicated()]
            
            # Align columns to avoid conflicts
            overlapping_columns = set(info_salaries_data.columns).intersection(additional_data.columns) - {'Matricule'}
            additional_data = additional_data.drop(columns=overlapping_columns, errors='ignore')
            
            # Merge on Matricule
            info_salaries_data = info_salaries_data.merge(
                additional_data, on='Matricule', how='left'
            )
    else:
        info_salaries_data = pd.DataFrame()
        info_salaries_data['Matricule'] = pd.NA
try:
    with open("Template.xlsx", "rb") as template_file:
        template_content = template_file.read()

    with open("Livrable_Template.pdf", "rb") as livrable_file:
        livrable_content = livrable_file.read()

    # Ajouter les liens de téléchargement
    st.markdown("### Livrables")
    st.download_button(
        label="Template des colonnes requises pour l'extraction",
        data=template_content,
        file_name="Template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    st.download_button(
        label="Livrable d'explication du contenu du template",
        data=livrable_content,
        file_name="Livrable_Template.pdf",
        mime="application/pdf"
    )
except FileNotFoundError as e:
    st.error(f"Fichier manquant : {e}")


template = pd.read_excel("Template.xlsx")
template_columns = []

# Initialiser un ensemble pour stocker toutes les colonnes des fichiers téléchargés
uploaded_columns = set()

# Traiter les fichiers téléchargés
for file in uploaded_files:
    df = pd.read_excel(file)
    uploaded_columns.update(df.columns)  # Ajouter les colonnes au set

# Vérifier si toutes les colonnes du template sont présentes dans l'union des fichiers téléchargés
missing_columns = [col for col in template_columns if col not in uploaded_columns]

if uploaded_pdf is not None and len(uploaded_files) == num_files:
    if missing_columns:
        for col in missing_columns:
            st.error(f"La colonne '{col}' manque dans les données téléchargées.")
    else :
        # Créer un fichier temporaire pour le PDF téléchargé
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(uploaded_pdf.read())  
            temp_pdf_path = temp_pdf.name
        
        # Dictionnaire pour stocker les fichiers CSV en mémoire
        csv_files = {}

        reader = PdfReader(temp_pdf_path)
        total_pages = len(reader.pages)

        current_page_count = 0
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        max_workers = 4  # Limite du nombre de threads/process

        st.write(f"Extraction des tableaux pour toutes les {total_pages} pages...")

        results_list = []  # Liste pour stocker les résultats

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            other_page_futures = {
                executor.submit(process_pages, temp_pdf_path, 300, 3, str(page)): page for page in range(1, total_pages + 1)
            }
            
            for future in concurrent.futures.as_completed(other_page_futures):
                page = other_page_futures[future]
                try:
                    results = future.result()
                    for page_number, df_stream in results:
                        # Sauvegarder le contenu CSV en mémoire avec le numéro de page comme clé
                        csv_content = save_table_to_memory_csv(df_stream)
                        csv_files[page_number] = csv_content  # Utiliser le numéro de page comme clé pour trier facilement

                    # Mettre à jour la barre de progression et le texte de statut
                    current_page_count += 1
                    progress_value = current_page_count / total_pages
                    progress_bar.progress(progress_value)
                    status_text.text(f"Traitement : {min(current_page_count, total_pages)}/{total_pages} pages traitées")
                    
                except Exception as e:
                    st.write(f"Erreur lors du traitement des pages {page}: {e}")

        # Trier les fichiers CSV par ordre des pages
        sorted_csv_files = dict(sorted(csv_files.items()))


        # Trier les résultats en fonction de l'ordre des pages
        results_list.sort(key=lambda x: x[0])  # Trier par numéro de page (x[0] représente le numéro de page)

        # Créer un DataFrame pour fusionner toutes les données dans l'ordre
        final_df = pd.DataFrame()
        for page_number, df_stream in results_list:
            # Concaténer les données dans l'ordre des pages
            final_df = pd.concat([final_df, df_stream], ignore_index=True)

        # Sauvegarder le résultat final dans un buffer CSV
        output_buffer = StringIO()
        final_df.to_csv(output_buffer, index=False)
        final_csv_content = output_buffer.getvalue()

        st.write("Fusion des fichiers Excel et CSV terminée.")


        # Liste des éléments requis
        required_elements = ['CodeLibellé', 'Base', 'Taux', 'Montant Sal.', 'Taux', 'Montant Pat.']
        required_elements2 = ['Code','Libellé', 'Base', 'Taux', 'Montant Sal.', 'Taux', 'Montant Pat.']

        filtered_files = []

        # Parcourir les fichiers CSV en mémoire et filtrer selon `check_second_line`
        for filename, csv_content in sorted_csv_files.items():
            if check_second_line(csv_content, required_elements) or check_second_line(csv_content, required_elements2):
                filtered_files.append((filename, csv_content))  # Ajouter le nom et le contenu du fichier filtré
        
        st.write("Les fichiers CSV filtrés sont prêts à être utilisés.")

        # Exemple d'affichage des fichiers filtrés
        st.write(len(filtered_files))

        

        # Lire le fichier PDF téléchargé
        reader = PdfReader(uploaded_pdf)
        all_matricules = set()

        # Parcourir chaque page du PDF et extraire les matricules
        for page in reader.pages:
            text = page.extract_text()
            if check_for_mat(text):
                all_matricules.update(extract_matricules(text))
                
        # Simuler les matricules et le fichier combiné
        matricules = sorted(list(all_matricules))

        # Créer un buffer CSV en mémoire pour les matricules
        matricules_buffer = StringIO()
        csv_writer = csv.writer(matricules_buffer)
        csv_writer.writerow(["Matricule"])  # Écrire l'en-tête
        
        

        # Écrire chaque matricule dans le buffer
        for matricule in sorted(all_matricules):
            csv_writer.writerow([matricule])

        # Remettre le curseur au début du buffer
        matricules_buffer.seek(0)

        st.write(pd.read_csv(matricules_buffer))

        # Affichage du message de succès
        st.write("Extraction des matricules terminée.")

        

        # Filtrage des fichiers en mémoire et renommage de la deuxième colonne "Taux"
        required_elements = ['CodeLibellé', 'Base', 'Taux', 'Montant Sal.', 'Taux', 'Montant Pat.']
        required_elements2 = ['Code', 'Libellé', 'Base', 'Taux', 'Montant Sal.', 'Taux', 'Montant Pat.']

        filtered_files = {}

        for filename, csv_content in csv_files.items():
            if check_second_line(csv_content, required_elements) or check_second_line(csv_content, required_elements2):
                # Renommer le deuxième "Taux"
                csv_content = rename_second_taux(csv_content)
                filtered_files[filename] = csv_content

        # Diviser les fichiers en deux : colonnes requises et autres
        clean_csv_files = {}
        rest_csv_files = {}

        for filename, csv_content in filtered_files.items():
            file_like_object = StringIO(csv_content)
            reader = csv.reader(file_like_object)
            
            # Lire l'en-tête et la deuxième ligne
            header = next(reader)
            second_line = next(reader)

            required_elements_new = ['CodeLibellé', 'Base', 'Taux', 'Montant Sal.', 'Taux 2', 'Montant Pat.']
            required_elements2_new = ['Code','Libellé', 'Base', 'Taux', 'Montant Sal.', 'Taux 2', 'Montant Pat.']
            
            # Obtenir les indices des colonnes requises et autres
            required_indices, other_indices = split_columns(header, second_line, required_elements_new + required_elements2_new)

            # Créer les buffers pour les fichiers CSV divisés
            clean_buffer = StringIO()
            rest_buffer = StringIO()

            clean_writer = csv.writer(clean_buffer)
            rest_writer = csv.writer(rest_buffer)

            # Écrire les colonnes requises dans clean_buffer
            clean_writer.writerow([header[i] for i in required_indices])
            clean_writer.writerow([second_line[i] for i in required_indices])
            for row in reader:
                clean_writer.writerow([row[i] for i in required_indices])

            # Réinitialiser le lecteur pour les autres colonnes
            file_like_object.seek(0)
            reader = csv.reader(file_like_object)
            header = next(reader)
            second_line = next(reader)

            # Écrire les autres colonnes dans rest_buffer
            rest_writer.writerow([header[i] for i in other_indices])
            rest_writer.writerow([second_line[i] for i in other_indices])
            for row in reader:
                rest_writer.writerow([row[i] for i in other_indices])

            # Stocker les résultats dans les dictionnaires
            clean_csv_files[filename] = clean_buffer.getvalue()
            rest_csv_files[filename] = rest_buffer.getvalue()

        # Afficher des boutons pour télécharger les fichiers CSV traités
        st.write("Fichiers CSV divisés.")
        
        restructured_files = {}


        # Traiter et restructurer les fichiers
        for filename, csv_content in clean_csv_files.items():

            try:
                headers_row, values_row = transform_to_two_lines(csv_content, required_elements_new, required_elements2_new)

                # Créer un buffer CSV pour le fichier restructuré
                restructured_buffer = StringIO()
                writer = csv.writer(restructured_buffer)
                writer.writerow(headers_row)
                writer.writerow(values_row)
                
                # Stocker le fichier restructuré en mémoire
                restructured_files[filename] = restructured_buffer.getvalue()

            except ValueError as e:
                st.write(f"Erreur lors du traitement de {filename}: {e}")

        # Vérifier si les fichiers ont bien été ajoutés
        if restructured_files:
            st.write(f"{len(restructured_files)} fichiers restructurés prêts à être téléchargés.")
        else:
            st.write("Aucun fichier restructuré disponible.")

        processed_csv_files = process_csv_in_memory(restructured_files)
        combined_headers, combined_data = merge_csv_in_memory(processed_csv_files)
        combined_headers, combined_data = add_taux_2_columns(combined_headers, combined_data)
        combined_csv_content = write_combined_csv_to_memory(combined_headers, combined_data)
        
        # Fusionner les bulletins avec les matricules
        merged_csv_content = merge_bulletins_with_matricules(matricules, combined_csv_content)


        updated_csv_files = {}

        for filename, csv_content in rest_csv_files.items():
            updated_csv_files[filename] = update_headers(csv_content)
            
        # Générer un rapport d'absences à partir des fichiers CSV en mémoire
        absence_report_csv = generate_absences_report(updated_csv_files)

        