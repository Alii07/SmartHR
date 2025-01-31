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

URSSAF_COLUMNS = [
    'Matricule',
    'PLAFOND CUM M-1',
    'ASSIETTE CU M-1',
    'SMIC M CUM M-1',
    'Statut de salariés',
    'Assiette Mois M (/102)',
    'SMIC M',
    'Brut CUM',
    'CUMUL B MAL M-1',
    'BASE B V',
    'BASE B 7025',
    'Frontalier',
    'Effectif'
]

POLE_EMPLOI_COLUMNS = [
    'Type de contrat',
    'PLAFOND CUM M-1',
    'PLAFOND M',
    'ASS POLE EM',
    'Cumul Base 7C00',
    'Cumul Base 7C10'
]

RETRAITE_COMPLEMENTAIRE_COLUMNS = [
    'Statut de salariés',
    'RER. T1',
    'RER. T2',
    'ASS RET.'
]

PREVOYANCE_COLUMNS = [
    'Statut de salariés',
    'PRÉV. TA',
    'PRÉV. TB',
    'PRÉV. TC',
    'ASS PRÉ NC',
    'ASS PRÉ CAD'
]

APEC_COLUMNS = [
    'Statut de salariés',
    'APEC T1',
    'APEC T2',
    'ASS RET.'
]

CEG_COLUMNS = [
    'Statut de salariés',
    'CEG T1',
    'CEG T2',
    'ASS RET.'
]

CET_COLUMNS = [
    'Statut de salariés',
    'CET T1',
    'CET T2',
    'ASS RET.'
]

ALL_COLUMNS = list(set(
    URSSAF_COLUMNS + 
    POLE_EMPLOI_COLUMNS + 
    RETRAITE_COMPLEMENTAIRE_COLUMNS + 
    PREVOYANCE_COLUMNS + 
    APEC_COLUMNS + 
    CEG_COLUMNS + 
    CET_COLUMNS
))

def extract_table_from_pdf(pdf_file_path, edge_tol, row_tol, pages):
    try:
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

def save_table_to_memory_csv(df):
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    return csv_buffer.getvalue()

def check_required_columns(csv_content, report_type):
    """
    Vérifie si les colonnes requises sont présentes selon le type de rapport.
    """
    required_columns = {
        'URSSAF': URSSAF_COLUMNS,
        'POLE EMPLOI': POLE_EMPLOI_COLUMNS,
        'ALL': ALL_COLUMNS
    }.get(report_type, [])

    df = pd.read_csv(StringIO(csv_content))
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Colonnes manquantes pour le rapport {report_type}: {', '.join(missing_columns)}")
    
    return True

def check_second_line(file_content, required_elements):
    file_like_object = StringIO(file_content)
    reader = csv.reader(file_like_object)
    next(reader)  # Ignorer la première ligne (header)
    second_line = next(reader, None)  # Lire la deuxième ligne
    
    if second_line and any(elem in second_line for elem in required_elements):
        return True
    return False

def split_columns(header, second_line, required_elements):
    required_indices = [i for i, col in enumerate(second_line) if col in required_elements]
    other_indices = [i for i, col in enumerate(second_line) if col not in required_elements]
    return required_indices, other_indices

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

def check_for_mat(text):
    return 'Mat:' in text

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

def transform_to_two_lines(csv_content, required_elements_new, required_elements2_new):
    headers_row = []
    values_row = []

    file_like_object = StringIO(csv_content)
    reader = csv.reader(file_like_object)

    header = next(reader, None)
    second_line = next(reader, None)

    if not any(all(elem in second_line for elem in req_set) for req_set in (required_elements_new, required_elements2_new)):
        raise ValueError("Les colonnes requises ne sont pas présentes dans le fichier CSV.")

    code_index = second_line.index('Code') if 'Code' in second_line else None
    libelle_index = second_line.index('Libellé') if 'Libellé' in second_line else None
    codelibelle_index = second_line.index('CodeLibellé') if 'CodeLibellé' in second_line else None

    rubriques = []

    if codelibelle_index is not None:
        for row in reader:
            rubrique = row[codelibelle_index]
            if rubrique and rubrique[0].isdigit():
                rubrique = rubrique[:4]
                if rubrique not in rubriques:
                    rubriques.append(rubrique)
                    headers_row.extend([
                        f"Rub {rubrique}",  # Ajout de la nouvelle colonne
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
                rubrique = rubrique[:4]
                if rubrique not in rubriques:
                    rubriques.append(rubrique)
                    headers_row.extend([
                        f"Rub {rubrique}",  # Ajout de la nouvelle colonne
                        f"{rubrique}Base",
                        f"{rubrique}Taux",
                        f"{rubrique}Montant Sal.",
                        f"{rubrique}Taux 2",
                        f"{rubrique}Montant Pat."
                    ])
    
    file_like_object.seek(0)
    reader = csv.reader(file_like_object)
    next(reader)
    next(reader)

    values_row = ['' for _ in range(len(headers_row))]

    for row in reader:
        if codelibelle_index is not None:
            code_libelle = row[codelibelle_index]
            if code_libelle and code_libelle[0].isdigit():
                code_libelle = code_libelle[:4]
        elif code_index is not None and libelle_index is not None:
            code_libelle = f"{row[code_index]}{row[libelle_index]}"
            if code_libelle and code_libelle[0].isdigit():
                code_libelle = code_libelle[:4]
        else:
            raise ValueError("Les colonnes 'Code' et 'Libellé' ou 'CodeLibellé' sont manquantes")

        try:
            rubrique_index = headers_row.index(f"Rub {code_libelle}")  # Mise à jour de l'index
            # Remplir la valeur de Rub {code}
            values_row[rubrique_index] = code_libelle
            # Remplir les autres valeurs
            for i, col in enumerate(['Base', 'Taux', 'Montant Sal.', 'Taux 2', 'Montant Pat.']):
                if col in second_line:
                    col_index = second_line.index(col)
                    values_row[rubrique_index + i + 1] = row[col_index]  # +1 pour tenir compte de la nouvelle colonne
        except ValueError:
            continue

    return headers_row, values_row

def convert_to_float(value):
    if value:
        try:
            value = value.replace('.', '').replace(',', '.')
            return float(value)
        except ValueError:
            return None
    return None

def process_csv_in_memory(csv_contents):
    csv_files = {}
    
    for filename in sorted(csv_contents.keys()):
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

        output_buffer = StringIO()
        writer = csv.DictWriter(output_buffer, fieldnames=headers)
        writer.writeheader()
        writer.writerows(data)
        csv_files[filename] = output_buffer.getvalue()

    return csv_files

def merge_csv_in_memory(csv_files):
    combined_headers = []
    combined_data = []

    for i, (filename, csv_content) in enumerate(sorted(csv_files.items())):
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

def write_combined_csv_to_memory(combined_headers, combined_data):
    output_buffer = StringIO()
    writer = csv.writer(output_buffer)
    writer.writerow(combined_headers)
    writer.writerows(combined_data)
    return output_buffer.getvalue()

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

def extract_brut_totals(pdf_file):
    brut_totals = []
    with open(pdf_file, 'rb') as file:
        reader = PdfReader(file)
        num_pages = len(reader.pages)
        for i in range(num_pages):
            page = reader.pages[i]
            text = page.extract_text()
            totals = re.findall(r"TOTAL BRUT\s+([\d,.]+)", text)
            if totals:
                for total in totals:
                    brut_totals.append({"Page": i+1, "Total Brut": total})
    return brut_totals

def merge_bulletins_with_matricules(matricules, combined_csv_content, brut_totals):
    combined_data = []
    reader = csv.reader(StringIO(combined_csv_content))
    combined_data = [row for row in reader]

    if len(matricules) > len(combined_data):
        raise ValueError("Le fichier de matricules contient plus de lignes que le fichier combined_output.")

    merged_data = []
    merged_data.append(["Matricule", "Total Brut"] + combined_data[0])

    for i, row in enumerate(combined_data[1:], start=1):
        if i <= len(matricules):
            matricule = matricules[i - 1]
            total_brut = ""
            # Chercher le total brut correspondant à la page
            for bt in brut_totals:
                if bt["Page"] == i:
                    total_brut = bt["Total Brut"]
                    break
            merged_data.append([matricule, total_brut] + row)
        else:
            break

    output_buffer = StringIO()
    writer = csv.writer(output_buffer)
    writer.writerows(merged_data)
    return output_buffer.getvalue()

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
