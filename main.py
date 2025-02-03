# Configuration de la page - DOIT ÊTRE LA PREMIÈRE COMMANDE STREAMLIT
import streamlit as st
import hashlib
import sqlite3

st.set_page_config(
    page_title = "SMARTHR",
    page_icon = "./logo.svg",
    initial_sidebar_state = "expanded"
)

hide_streamlit_style = """
    <style>
    #appCreatorAvatar {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    footer {display: none;}
    .stDeployButton {display: none;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Autres imports
from streamlit_extras.bottom_container import bottom
import pandas as pd
import tempfile
import os
from Bases import AnomalyDetection
from Taux import Taux
from datetime import datetime
import io
from extracteur import *
from extraction_interface import *
from extraction_process import *
from predict import *
from modification import *
from interface_intro import *

historique_modifications = []

versement_mobilite = { '87005' : 1.80 }
models_info = {
    '6000': {
        'type' : 'joblib',
        'model': './Modèles/Taux/6000.pkl',
        'numeric_cols': ['Rub 6000',  '6000Taux'],
        'categorical_cols': ['Frontalier'],
        'target_col': 'anomalie_frontalier'
    },
    '6002': {
        'type' : 'joblib',
        'model': './Modèles/Taux/6002.pkl',
        'numeric_cols': ['Rub 6002',  '6002Taux'],
        'categorical_cols': ['Region'],
        'target_col': 'anomalie_alsace_moselle'
    },
    '6082': {
        'type': 'joblib',
        'model': './Modèles/Taux/6082.pkl',
        'numeric_cols': ['Rub 6082', '6082Taux'],
        'categorical_cols': ['Statut de salariés', 'Frontalier'],
        'target_col': 'anomalie_csg'
    },
    '6084': {
        'type' : 'joblib',
        'model': './Modèles/Taux/6084.pkl',
        'numeric_cols': ['Rub 6084', '6084Taux'],
        'categorical_cols': ['Statut de salariés', 'Frontalier'],
        'target_col': 'anomalie_crds'
    },
    '7001': {
            'type' : 'joblib',
            'model': './Modèles/Taux/7001.pkl',
            'numeric_cols': ['Matricule', 'Absences par Jour', 'Absences par Heure', 'PLAFOND CUM', 'ASSIETTE CUM', 'MALADIE CUM', '7001Base', '7001Taux 2', '7001Montant Pat.'],
            'categorical_cols': ['Statut de salariés'],
            'target_col': 'anomalie_maladie_reduite'
        },
    '7002': {
        'type' : 'joblib',
        'model': './Modèles/Taux/7002_cases.pkl',  # Utilisez le chemin vers votre modèle
        'numeric_cols': ['SMIC M CUM', '7002Taux 2', 'ASSIETTE CUM'],
        'categorical_cols': ['Statut de salariés'],
        'target_col': 'anomalie_maladie_diff'
    },

    '7010': {
        'type' : 'joblib',
        'model': './Modèles/Taux/7010.pkl',
        'numeric_cols': ['Rub 7010',  '7010Taux 2','7010Taux' ,'Effectif'],
        'categorical_cols': ['Statut de salariés'],
        'target_col': 'Anomalie_7010'
    },

    '7015': {
        'type' : 'joblib',
        'model': './Modèles/Taux/7015.pkl',
        'numeric_cols': ['Rub 7015','7015Taux', '7015Taux 2' ,'Effectif'],
        'categorical_cols': ['Statut de salariés'],
        'target_col': 'Anomalie_7015'
    },

    '7020': {
        'type' : 'joblib',
        'model': './Modèles/Taux/7020.pkl',
        'numeric_cols': ['Rub 7020',  '7020Taux 2' ,'Effectif'],
        'categorical_cols': ['Statut de salariés'],
        'target_col': 'anomalie_fnal'
    },

    '7025': {
            'type': 'joblib',
            'model': './Modèles/Taux/7025.pkl',
            'numeric_cols': ['7025Taux 2', 'ASSIETTE CUM', 'PLAFOND CUM'],
            'categorical_cols': ['Statut de salariés'],
            'target_col': '7025Taux 2'
        },
    '7030': {
        'type' : 'joblib',
        'model': './Modèles/Taux/7030.pkl',
        'numeric_cols': ['PLAFOND CUM', 'ASSIETTE CUM','7030Taux 2', 'Rub 7030'],
        'categorical_cols': [],
        'target_col': 'anomalie_allocation_reduite'
    },
    '7035': {
        'type' : 'joblib',
        'model': './Modèles/Taux/7035.pkl',
        'numeric_cols': ['Rub 7035','7035Taux 2'],
        'categorical_cols': ['Statut de salariés'],
        'target_col': '7035 Fraud'
    },
    '7040': {
        'type' : 'joblib',
        'model': './Modèles/Taux/7040.pkl',
        'numeric_cols': ['Effectif', '7040Taux 2' ,'Rub 7040'],
        'categorical_cols': ['Statut de salariés'],
        'target_col': 'anomalie_7040taux 2'
    },
    '7045': {
        'type' : 'joblib',
        'model': './Modèles/Taux/7045.pkl',
        'numeric_cols': ['Effectif', '7045Taux 2'],
        'categorical_cols': ['Etablissement'],
        'target_col': 'anomalie_transport'
    },
    '7050': {
        'type' : 'joblib',
        'model': './Modèles/Taux/7050.pkl',
        'numeric_cols': ['Effectif', '7050Taux 2'],
        'categorical_cols': ['Etablissement'],
        'target_col': 'anomalie_cotisation_accident'
    }
}

model_configs = {
    "6081": {
        "classification_model": "Modèles/Bases/classification_model_6081_new.pkl",
        "regression_models": {
            (0, 0): "Modèles/Bases/regression_model_6081_(0, 0).pkl",
            (0, 1): "Modèles/Bases/regression_model_6081_(0, 1).pkl",
            (1, 1): "Modèles/Bases/regression_model_6081_(1, 1).pkl",
        },
        "numeric_cols": {
            (0, 0): ["4*PLAFOND CUM", "Cumul d'assiette ( Mois courant inclus) (/102)"],
            (0, 1): ["4*PLAFOND CUM", "Cumul d'assiette ( Mois courant inclus) (/102)"],
            (1, 1): ["Tranche C pre"],
        },
        "target_col": "6081Base",
        "apprenti_status": ["Apprenti (B.C)", "Apprenti (W.C)"],
        "threshold_margin": 0.01,
    },
    "6085": {
        "classification_model": "Modèles/Bases/classification_model_6085_new.pkl",
        "regression_models": {
            (0, 0): "Modèles/Bases/regression_model_6085_(0, 0).pkl",
            (0, 1): "Modèles/Bases/regression_model_6085_(0, 1).pkl",
            (1, 1): "Modèles/Bases/regression_model_6085_(1, 1).pkl",
        },
        "numeric_cols": {
            (0, 0): ["4*PLAFOND CUM", "Cumul d'assiette ( Mois courant inclus) (/102)"],
            (0, 1): ["4*PLAFOND CUM", "Cumul d'assiette ( Mois courant inclus) (/102)"],
            (1, 1): ["Tranche C pre"],
        },
        "target_col": "6085Base",
        "apprenti_status": ["Apprenti (B.C)", "Apprenti (W.C)"],
        "threshold_margin": 0.01,
    },
    "6082": {
        "path": "Modèles/Bases/6082.pkl",
        "cases": {
            "Cas 1": {"feature_col": "Plafond CUM", "target_col": "6082Base"},
            "Cas 2": {"feature_col": "1001Montant Sal.", "target_col": "6082Base"},
        },
        "threshold_margin": 0.01,
    },
    "6084": {
        "path": "Modèles/Bases/6084.pkl",
        "cases": {
            "Cas 1": {"feature_col": "Plafond CUM", "target_col": "6084Base"},
            "Cas 2": {"feature_col": "1001Montant Sal.", "target_col": "6084Base"},
        },
        "threshold_margin": 0.01,
    },
    
    "7002": {
        "path": "Modèles/Bases/7002.pkl",
        "target_col": "7002Base",
        "threshold_margin": 0.01,
    },
    
    "7015": {
        "classification_model": "Modèles/Bases/classification_assiette_plafond.joblib",
        "regression_models": {
            0: "Modèles/Bases/model_label_0.joblib",
            1: "Modèles/Bases/model_label_1.joblib",
        },
        "classification_features": ["Assiette cum", "PLAFOND CUM"],
        "regression_features": {
            0: ["PLAFOND CUM", "CUM Base 7015 precedente"],
            1: ["Assiette cum", "CUM Base 7015 precedente"],
        },
        "required_columns": ["Assiette cum", "PLAFOND CUM", "CUM Base precedente", "7015Base"],
        "target_col": "7015Base",
        "threshold_margin": 0.01,
    },
    "7025": {
        "path": "Modèles/Bases/7025.pkl",
        "numeric_cols": ["7025Base", "Base CUM M-1", "Brut CUM", "Plafond CUM", "Total Brut"],
        "categorical_cols": ["Cluster"],
        "threshold_margin": 0.01,
    },
}



simple_models = ["7001", "7020", "7030", "7035", "7040", "7045", "7050"]
montant_models = ["7001", "7020", "7030", "7035", "7040", "7045", "7050", "7002", "7015", "6082", "6084", "6081", "6085", "7025"]


def detect_montant_anomalies_streamlit(df, montant_models):
    """
    Génère un rapport des anomalies détectées pour les montants, adapté à Streamlit.
    """
    error_log = []  # Stockage des erreurs pour un éventuel débogage
    report_lines = []  # Lignes du rapport des montants

    try:
        for model_name in montant_models:
            try:
                if model_name in ["6081", "6085", "6082", "6084"]:
                    if all(col in df.columns for col in [f'{model_name}Base', f'{model_name}Taux', f'{model_name}Montant Sal.']):
                        # Vérification pour les modèles spécifiques
                        df_filtered = df[(df[f'{model_name}Taux'] != 0)]  # Filtrer les lignes avec un taux non nul
                        df[f'{model_name}Base'] = df[f'{model_name}Base'].fillna(0)
                        df_filtered[f'{model_name}Taux'] = df_filtered[f'{model_name}Taux'].fillna(0)
                        df_filtered[f'{model_name}Montant Sal.'] = df_filtered[f'{model_name}Montant Sal.'].fillna(0)
                        if df_filtered[f'{model_name}Taux'] != 0 :
                            df[f'{model_name}Anomalie'] = abs(
                                (df_filtered[f'{model_name}Base'] / df_filtered[f'{model_name}Taux']) - 
                                df_filtered[f'{model_name}Montant Sal.']
                            ) > 0.01  # Vérifier la marge de 0.01
                        else:
                            df[f'{model_name}Anomalie'] = df_filtered[f'{model_name}Montant Sal.'] == 0
                    else:
                        error_log.append(f"Colonnes manquantes pour le modèle {model_name} dans les montants.")

                else:
                    if all(col in df.columns for col in [f'{model_name}Base', f'{model_name}Taux 2', f'{model_name}Montant Pat.']):
                        # Vérification pour les autres modèles
                        df_filtered = df[(df[f'{model_name}Taux 2'] != 0)]  # Filtrer les lignes avec un taux non nul
                        df[f'{model_name}Anomalie'] = abs(
                            (df_filtered[f'{model_name}Base'] / df_filtered[f'{model_name}Taux 2']) - 
                            df_filtered[f'{model_name}Montant Pat.']
                        ) > 0.01  # Vérifier la marge de 0.01
                    else:
                        error_log.append(f"Colonnes manquantes pour le modèle {model_name} dans les montants.")

                # Filtrer les anomalies et ajouter au rapport
                anomalies = df[df[f'{model_name}Anomalie']]
                for _, row in anomalies.iterrows():
                    matricule = row['Matricule'] if 'Matricule' in row else f"Ligne {row.name}"
                    report_lines.append(
                        f"Nous avons détecté pour le Matricule {matricule} une anomalie dans le montant {abs(
                            (df_filtered[f'{model_name}Base'] / df_filtered[f'{model_name}Taux']) - 
                            df_filtered[f'{model_name}Montant Sal.']
                        )} : {model_name}\n"
                    )
            except Exception as e:
                error_log.append(f"Erreur pour le modèle {model_name} dans les montants : {e}")
    except Exception as e:
        error_log.append(f"Erreur lors de la détection des anomalies de montants : {e}")

    return report_lines, error_log



def generate_base_anomalies_report_streamlit(df, simple_models, model_configs):
    """
    Génère un rapport des anomalies détectées pour les bases, adapté à Streamlit.
    """
    base_anomaly_detector = AnomalyDetection(model_configs)
    base_anomaly_detector.load_models()

    # Conteneur pour les rapports d'anomalies
    base_reports = {}
    error_log = []  # Stockage des erreurs pour un éventuel débogage

    # Détection des anomalies simples
    for model_name in simple_models:
        try:
            report = base_anomaly_detector.detect_anomalies_simple_comparison(df.copy(), model_name)
            if report is not None:
                base_reports[model_name] = report
        except Exception as e:
            error_log.append(f"Erreur pour le modèle simple {model_name}: {e}")

    # Détection des anomalies avancées
    for model_name in model_configs.keys():
        try:
            df_preprocessed = base_anomaly_detector.preprocess_data(df.copy(), model_name)
            
            if isinstance(model_configs[model_name].get('numeric_cols'), dict):
                # Gestion spécifique pour les modèles avancés (ex : 6081, 6085)
                numeric_cols = []
                for subset_cols in model_configs[model_name]['numeric_cols'].values():
                    numeric_cols.extend(subset_cols)  # Ajouter toutes les colonnes pour chaque sous-ensemble
                required_columns = list(set(numeric_cols)) + model_configs[model_name].get('categorical_cols', [])
            else:
                # Gestion standard (ex : modèles simples)
                required_columns = model_configs[model_name].get('numeric_cols', []) + model_configs[model_name].get('categorical_cols', [])

            # Vérifier les colonnes manquantes
            missing_columns = [col for col in required_columns if col not in df_preprocessed.columns]
            if missing_columns:
                error_log.append(f"Le modèle {model_name} manque des colonnes : {', '.join(missing_columns)}")
                continue

            # Détection des anomalies
            report = base_anomaly_detector.detect_anomalies(df_preprocessed, model_name)
            if report is not None:
                base_reports[model_name] = report
        except Exception as e:
            error_log.append(f"Erreur pour le modèle avancé {model_name}: {e}")

    # Combiner les rapports
    try:
        base_combined_report = base_anomaly_detector.combine_reports(base_reports)
    except Exception as e:
        error_log.append(f"Erreur lors de la combinaison des rapports : {e}")
        return None, error_log

    # Générer les lignes du rapport des bases
    report_lines = []
    for index, row in base_combined_report.iterrows():
        matricule = row['Matricule'] if 'Matricule' in row else f"Ligne {index}"
        model_name = row['nom_du_modèle'] if 'nom_du_modèle' in row else "Modèle inconnu"
        report_lines.append(f"Nous avons détecté pour le Matricule {matricule} une anomalie dans la cotisation : {model_name}\n")

    return report_lines, error_log



def detect_taux_anomalies_streamlit(df):
    """
    Génère un rapport des anomalies détectées pour les taux, adapté à Streamlit.
    """
    taux_anomaly_detector = Taux(models_info, versement_mobilite)
    error_log = []  # Stockage des erreurs pour un éventuel débogage

    try:
        anomalies, _ = taux_anomaly_detector.detect_anomalies(df)
    except Exception as e:
        error_log.append(f"Erreur lors de la détection des anomalies des taux: {e}")
        return None, error_log

    # Générer les lignes du rapport des taux
    report_lines = []
    for index, details in anomalies.items():
        matricule = df.loc[index, 'Matricule'] if 'Matricule' in df.columns else f"Ligne {index}"
        filtered_models = [model for model in details if model in models_info.keys()]
        if filtered_models:
            report_lines.append(f"Nous avons détecté pour le Matricule {matricule} une anomalie dans la cotisation : {', '.join(filtered_models)}\n")

    return report_lines, error_log


def merge_anomalies_reports_streamlit(base_lines, taux_lines, montant_lines):
    """
    Combine les rapports des anomalies des bases, des taux, et des montants en triant les matricules par ordre croissant.
    """
    anomalies_by_matricule = {}

    def extract_anomalies(lines):
        for line in lines:
            if "une anomalie dans la cotisation :" in line:
                # Vérifier que la ligne contient le texte attendu
                try:
                    parts = line.split("une anomalie dans la cotisation :")
                    if len(parts) < 2:
                        continue  # Ignorer les lignes mal formatées

                    matricule = parts[0].split("Matricule")[-1].strip()
                    models = [model.strip() for model in parts[1].split(",")]

                    if matricule not in anomalies_by_matricule:
                        anomalies_by_matricule[matricule] = set()

                    anomalies_by_matricule[matricule].update(models)
                except Exception as e:
                    # Ajouter un message d'erreur pour une ligne problématique
                    print(f"Erreur lors du traitement de la ligne : {line}. Erreur : {e}")

    # Extraire les anomalies des trois rapports
    extract_anomalies(base_lines)
    extract_anomalies(taux_lines)
    extract_anomalies(montant_lines)

    # Trier les anomalies par matricule
    sorted_anomalies = dict(sorted(anomalies_by_matricule.items(), key=lambda x: x[0]))

    # Générer le rapport fusionné
    combined_lines = ["=== Rapport combiné des anomalies ===\n\n"]
    for matricule, models in sorted_anomalies.items():
        models_list = ", ".join(sorted(models))  # Trier les modèles pour un format cohérent
        combined_lines.append(f"Nous avons détecté pour le Matricule {matricule} une anomalie dans la cotisation : {models_list}\n")

    return "".join(combined_lines)

with bottom():
    col1, col2 = st.columns(2)
    with col1 :
        st.markdown("<span style='font-size: 0.75em; color:white;'>SmartHR</span>", unsafe_allow_html=True)
    with col2 : 
        st.markdown("<span style='font-size: 0.75em; color:white;'>All rights reserverd to HCM CP 2025</span>", unsafe_allow_html=True)
    
    st.html(
        """
        <style>
            # div[data-testid="stColumn"]:nth-of-type(1)
            # {
            #     border:1px solid red;
            # }
    
             div[data-testid="stColumn"]:nth-of-type(2)
             {
                 # border:1px solid blue;
                 text-align: end;
                 size: 0.1em !important;
                 text-color : white;
             }
            [data-testid="stBottomBlockContainer"] {
                background-color: #8ac447;
                width : 100% !important;
                max-width: none;
                padding: 0.5rem 1rem 0.5rem 1rem;
            }
        </style>
        """
    )
    


def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (username TEXT PRIMARY KEY, password TEXT)''')
    
    # Ajout d'un utilisateur par défaut
    default_username = "HCM-admin"
    default_password = "HCM2025"
    hashed_password = hash_password(default_password)
    
    try:
        c.execute("INSERT OR IGNORE INTO users VALUES (?, ?)", 
                 (default_username, hashed_password))
        conn.commit()
    except sqlite3.IntegrityError:
        pass
    finally:
        conn.close()

def hash_password(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def add_user(username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    hashed_password = hash_password(password)
    try:
        c.execute("INSERT INTO users VALUES (?, ?)", (username, hashed_password))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def login_user(username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    hashed_password = hash_password(password)
    c.execute("SELECT * FROM users WHERE username=? AND password=?", (username, hashed_password))
    result = c.fetchone()
    conn.close()
    return result is not None

def login_interface():
    st.markdown(
        """
        <style>
        .auth-container {
            background-color: white;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin: 2rem auto;
            max-width: 400px;
        }
        .auth-title {
            color: #386161;
            text-align: center;
            margin-bottom: 2rem;
        }
        .auth-subtitle {
            color: #666666;
            text-align: center;
            font-style: italic;
            margin-bottom: 2rem;
        }
        .auth-input {
            margin-bottom: 1rem;
        }
        .auth-button {
            background-color: #8ac447;
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 5px;
            cursor: pointer;
            width: 100%;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<h1 class='auth-title'>SMARTHR - Authentification</h1>", unsafe_allow_html=True)
    st.markdown("<p class='auth-subtitle'>Veuillez vous connecter pour accéder à l'application</p>", unsafe_allow_html=True)

    with st.container():
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            with st.form("login_form"):
                username = st.text_input("Nom d'utilisateur")
                password = st.text_input("Mot de passe", type="password")
                submit = st.form_submit_button("Se connecter")

                if submit:
                    if login_user(username, password):
                        st.session_state['authenticated'] = True
                        st.session_state['username'] = username
                        st.success("Connexion réussie!")
                        st.rerun()
                    else:
                        st.error("Nom d'utilisateur ou mot de passe incorrect")


def main():
    # Initialisation de la base de données
    init_db()

    # Vérification de l'authentification
    if 'authenticated' not in st.session_state:
        st.session_state['authenticated'] = False

    if not st.session_state['authenticated']:
        login_interface()
        return

    # Le reste du code main existant
    if "menu" not in st.session_state:
        st.session_state["menu"] = "intro"

    st.logo("./Logo.png", icon_image="./Logo.png")
    st.sidebar.markdown(
        """
        <style>
        img[data-testid="stLogo"] {
            height: 9rem;
            width: 9rem;
        }       
        [data-testid="stSidebar"] {
            background: linear-gradient(to top, #8ac447, #FFFFFF); 
            border-right : 0.1px solid #8ac447;
        }
        .stButton Button {
            width : 100%;
            display : flex;
            justify-content : left;
            align-items : left;
            padding : 10px;
        }
        .stButton Button:hover {
            background-color: white !important;
            color : #8ac447 !important;
            border : 0.1px solod #8ac447 !important;
        }
        .stButton Button:clicked, .stButton Button:active {
            background-color: #8ac447s !important;
            color : white !important;
        }
        .title {
            color: white !important; 
            font-size: 2em !important;
            font-weight: bold !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Navigation avec des clés uniques - Déplacé avant la condition

    
    if st.sidebar.button("Page d'accueil", key="sidebar_home"):
        st.session_state["menu"] = "intro"
        st.rerun()

    if st.sidebar.button("Extraction des données", key="sidebar_extraction"):
        st.session_state["menu"] = "Extraction des données"
        st.rerun()

    if st.sidebar.button("Modification employé", key="sidebar_modification"):
        st.session_state["menu"] = "Modification cotisation employée"
        st.rerun()

    if st.sidebar.button("Détection d'anomalies", key="sidebar_detection"):
        st.session_state["menu"] = "Détection d'anomalies"
        st.rerun()

    if st.sidebar.button("Mode d'emploi", key="sidebar_manuel"):
        st.session_state["menu"] = "Manuel"
        st.rerun()

    # Ajouter le JavaScript pour écouter les messages
    st.markdown("""
        <script>
        window.addEventListener('message', (event) => {
            if (event.data.type === 'setMenu') {
                const menu = event.data.menu;
                const menuOptions = {
                    'Extraction des données': 'Extraction des données',
                    'Modification cotisation employée': 'Modification cotisation employée',
                    'Détection d\'anomalies': 'Détection d\'anomalies'
                };
                const selectedMenu = menuOptions[menu];
                if (selectedMenu) {
                    window.parent.postMessage({ type: 'setSessionState', key: 'menu', value: selectedMenu }, '*');
                }
            }
        });
        </script>
        """, unsafe_allow_html=True)

    # Affichage du contenu principal selon le menu sélectionné
    menu = st.session_state["menu"]
    if menu == "intro":
        intro_interface()
    else:
        import base64
        def get_image_as_base64(image_path):
            with open(image_path, "rb") as img_file:
                return base64.b64encode(img_file.read()).decode()

        #logo_base64 = get_image_as_base64("./logo.svg")
        
        #st.markdown(
        #    f"""
        #    <div style="text-align: right;">
        #        <img src="data:image/svg+xml;base64,{logo_base64}" width="150">
        #    </div>
        #    """,
        #    unsafe_allow_html=True
        #)
        #st.title("SMARTHR : Détection d'anomalies")

    # Initialiser l'état des données et l'historique si non défini
    if "df" not in st.session_state:
        st.session_state.df = None
    if "historique_modifications" not in st.session_state:
        st.session_state.historique_modifications = []

    # Gestion des différentes pages
    menu = st.session_state["menu"]
    
    if menu == "Détection d'anomalies":
        st.title("Détection des Anomalies")

        st.markdown("""
            <style>
                .hero-subtitle {
                    font-size: 1.3em !important;
                    margin-bottom: 2rem;
                    font-style: italic;
                }
            </style>
            <p class='hero-subtitle'>Ici vous pourrez analyser vos données pour détecter les anomalies dans les bases et les taux.</p>
        """, unsafe_allow_html=True)


        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
                <div style='background-color: white; padding: 15px; border-radius: 5px;'>
                    <h4 style='color: #386161;'>📊 Type de cotisations</h4>
                </div>
            """, unsafe_allow_html=True)
            analyse_type = st.selectbox(
                "Type de cotisations à analyser",
                ["URSSAF", "Autres"],
                help="Sélectionnez le type de cotisations à analyser pour la détection des anomalies",
                key="analyse_type",               
            )
        if analyse_type == "URSSAF":
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
                    key="analyze_file"
                )
                if uploaded_file is not None:
                    try:
                        if analyse_type == "URSSAF":
                            df = pd.read_csv(uploaded_file)
                            
                            # Aperçu des données
                            st.markdown("""
                                <div style='background-color: white; padding: 15px; border-radius: 5px; margin-top: 20px;'>
                                    <h4 style='color: #386161;'>📋 Aperçu des données</h4>
                                </div>
                            """, unsafe_allow_html=True)
                            st.dataframe(df.head(), height=200)

                            # Options de filtrage
                        
                            # col1, col2 = st.columns(2)
                            # with col1:
                            #     st.markdown("""
                            #         <div style='background-color: white; padding: 15px; border-radius: 5px;'>
                            #             <h4 style='color: #386161;'>👤 Filtrer par employé</h4>
                            #         </div>
                            #     """, unsafe_allow_html=True)
                            #     employes = ['Tous les employés'] + list(df['Matricule'].unique())
                            #     employe_filtre = st.selectbox(
                            #         "Sélection employé",
                            #         employes,
                            #         label_visibility="collapsed"
                            #     )

                            # with col2:
                            #     st.markdown("""
                            #         <div style='background-color: white; padding: 15px; border-radius: 5px;'>
                            #             <h4 style='color: #386161;'>📑 Filtrer par cotisation</h4>
                            #         </div>
                            #     """, unsafe_allow_html=True)
                            #     rubriques = ['Toutes les rubriques'] + [col for col in df.columns if col.startswith("Rub")]
                            #     rubrique_filtre = st.selectbox(
                            #         "Sélection rubrique",
                            #         rubriques,
                            #         label_visibility="collapsed"
                            #     )

                            # Appliquer les filtres
                            df_analyse = df.copy()
                            employe_filtre = 'Tous les employés'
                            rubrique_filtre = 'Toutes les rubriques'
                            if employe_filtre != 'Tous les employés':
                                df_analyse = df_analyse[df_analyse['Matricule'] == employe_filtre]
                            if rubrique_filtre != 'Toutes les rubriques':
                                code_rubrique = rubrique_filtre.split(" ")[1]
                                colonnes_filtrees = [col for col in df_analyse.columns if code_rubrique in col]
                                df_analyse = df_analyse[['Matricule'] + colonnes_filtrees]


                        # Bouton d'analyse
                        if st.button("🔍 Lancer la détection des anomalies", use_container_width=True):
                            with st.spinner("Analyse en cours..."):
                                if analyse_type == "URSSAF":
                                    try:
                                        base_lines, base_errors = generate_base_anomalies_report_streamlit(df_analyse, simple_models, model_configs)
                                        taux_lines, taux_errors = detect_taux_anomalies_streamlit(df_analyse)
                                        montant_lines, montant_errors = detect_montant_anomalies_streamlit(df_analyse, montant_models)

                                        if any([base_lines, taux_lines, montant_lines]):
                                            st.markdown("""
                                                <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin-top: 20px;'>
                                                    <h3 style='color: #386161;'>📊 Résultats de l'analyse</h3>
                                                </div>
                                            """, unsafe_allow_html=True)
                                            
                                            combined_report = merge_anomalies_reports_streamlit(base_lines, taux_lines, montant_lines)
                                            st.text_area("Rapport détaillé", combined_report, height=300)
                                            
                                            col1, col2 = st.columns(2)
                                            with col1:
                                                st.download_button(
                                                    "📥 Télécharger le rapport (TXT)",
                                                    combined_report,
                                                    file_name="rapport_anomalies.txt",
                                                    mime="text/plain"
                                                )
                                        else:
                                            st.success("✅ Aucune anomalie détectée!")
                                            
                                    except Exception as e:
                                        st.error(f"❌ Erreur lors de l'analyse : {e}")
                            
                                    
                    except Exception as e:
                        st.error(f"❌ Erreur lors du chargement du fichier : {e}")
        else:  # si analyse_type == "Autres"
            predict_main(col2)  # Le col2 n'est utilisé que pour l'upload

        

    elif menu == "Modification cotisation employée":
        st.title("Modification des Cotisations")

        st.markdown("""
            <style>
                .hero-subtitle {
                    font-size: 1.3em !important;
                    margin-bottom: 2rem;
                    font-style: italic;
                }
            </style>
            <p class='hero-subtitle'>Ici vous pourrez modifier les données des cotisations des employés et suivez les changements.</p>
        """, unsafe_allow_html=True)

        col1, uploaded_col = st.columns(2)
        
        with col1:
            st.markdown("""
                <div style='background-color: white; padding: 5px; border-radius: 5px;'>
                    <h4 style='color: #386161;'>📊 Type de cotisations</h4>
                </div>
            """, unsafe_allow_html=True)
            modification_type = st.selectbox(
                "Type de cotisations",
                ["URSSAF", "Autres"],
                help="Sélectionnez le type de cotisations à modifier pour la détection des anomalies",
                key="modification_type"
            )
        
        with uploaded_col:
            st.markdown("""
                <div style='background-color: white; padding: 5px; border-radius: 5px; text-align: left;'>
                    <h4 style='color: #386161;'>📄 Fichier à modifier</h4>
                </div>
            """, unsafe_allow_html=True)
            uploaded_file = st.file_uploader(
                "Fichier que vous souhaitez modifier",
                type=["csv"],
                help="Chargez votre bulletin de paie extrait sous format CSV pour la modification et les tests",
                key="modification_file"
            )
            
        if uploaded_file is not None:
            if modification_type == "URSSAF":
                try:
                    df = pd.read_csv(uploaded_file, encoding='utf-8')
                    st.session_state.df = df
                    
                    # Aperçu des données
                    st.markdown("""
                        <div style='background-color: white; padding: 15px; border-radius: 5px; margin-top: 20px;'>
                            <h4 style='color: #386161;'>📋 Aperçu des données</h4>
                        </div>
                    """, unsafe_allow_html=True)
                    st.dataframe(df.head(), height=200)

                    # Sélection employé et rubrique
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("""
                            <div style='background-color: white; padding: 15px; border-radius: 5px;'>
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
                            <div style='background-color: white; padding: 15px; border-radius: 5px;'>
                                <h4 style='color: #386161;'>📑 Choix de la cotisation</h4>
                            </div>
                        """, unsafe_allow_html=True)
                        rubriques = [col for col in df.columns if col.startswith("Rub")]
                        if not rubriques:
                            st.error("❌ Aucune rubrique trouvée dans le fichier.")
                            return
                        cotisation = st.selectbox(
                            "",
                            rubriques,
                            key="rubrique_select"
                        )

                    # Extraction du code de cotisation et colonnes associées
                    try:
                        cotisation_code = cotisation.split(" ")[1]
                        colonnes_cotisation = [
                            f"{cotisation_code}Base",
                            f"{cotisation_code}Taux",
                            f"{cotisation_code}Montant Sal.",
                            f"{cotisation_code}Taux 2",
                            f"{cotisation_code}Montant Pat."
                        ]
                        colonnes_cotisation = [col for col in colonnes_cotisation if col in df.columns]
                        
                        if not colonnes_cotisation:
                            st.warning("⚠️ Aucune colonne associée trouvée pour cette cotisation.")
                            return

                        # Interface de modification
                        st.markdown("""
                            <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin-top: 20px;'>
                                <h3 style='color: #386161;'>💡 Modification des valeurs</h3>
                            </div>
                        """, unsafe_allow_html=True)

                        ligne_originale = df[df['Matricule'] == employe_id].copy()
                        if ligne_originale.empty:
                            st.error("❌ Employé non trouvé dans les données.")
                            return

                        ligne_modifiee = ligne_originale.copy()
                        valeurs_modifiees = {}

                        # Création des champs de modification
                        for colonne in colonnes_cotisation:
                            try:
                                valeur_actuelle = ligne_modifiee[colonne].iloc[0]
                                st.markdown(
                                    f"""
                                    <div style="background-color: #386161; padding: 10px; border-radius: 5px; margin-bottom: 10px; color: white;">
                                        <strong>{colonne}</strong><br>
                                        Valeur actuelle : {valeur_actuelle}
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )
                                nouvelle_valeur = st.text_input(
                                    f"Nouvelle valeur pour {colonne}",
                                    value=str(valeur_actuelle),
                                    key=f"input_{colonne}"
                                )
                                try:
                                    nouvelle_valeur = float(nouvelle_valeur)
                                    valeurs_modifiees[colonne] = nouvelle_valeur
                                    ligne_modifiee[colonne] = nouvelle_valeur
                                except ValueError:
                                    st.error(f"❌ La valeur entrée pour {colonne} n'est pas un nombre valide.")
                                    continue

                            except Exception as e:
                                st.error(f"❌ Erreur lors de la modification de {colonne}: {str(e)}")
                                continue

                        # Boutons d'action
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("✅ Confirmer les modifications"):
                                try:
                                    # Mise à jour du DataFrame
                                    df.loc[df['Matricule'] == employe_id, colonnes_cotisation] = ligne_modifiee[colonnes_cotisation].values
                                    st.session_state.df = df
                                    
                                    # Mise à jour de l'historique
                                    for colonne, nouvelle_valeur in valeurs_modifiees.items():
                                        if str(ligne_originale[colonne].iloc[0]) != str(nouvelle_valeur):
                                            st.session_state.historique_modifications.append({
                                                "Matricule": employe_id,
                                                "Rubrique modifiée": colonne,
                                                "Ancienne Valeur": ligne_originale[colonne].iloc[0],
                                                "Nouvelle Valeur": nouvelle_valeur
                                            })
                                    
                                    st.success("✅ Modifications enregistrées avec succès!")
                                    
                                    # Bouton de téléchargement
                                    csv_buffer = df.to_csv(index=False).encode('utf-8')
                                    st.download_button(
                                        label="📥 Télécharger le fichier modifié",
                                        data=csv_buffer,
                                        file_name="donnees_modifiees.csv",
                                        mime="text/csv"
                                    )
                                except Exception as e:
                                    st.error(f"❌ Erreur lors de la sauvegarde des modifications: {str(e)}")
                        
                        with col2:
                            if st.button("🔍 Détecter les anomalies"):
                                try:
                                    with st.spinner("Analyse des anomalies en cours..."):
                                        base_lines, base_errors = generate_base_anomalies_report_streamlit(df, simple_models, model_configs)
                                        taux_lines, taux_errors = detect_taux_anomalies_streamlit(df)
                                        montant_lines, montant_errors = detect_montant_anomalies_streamlit(df, montant_models)
                                        
                                        rapport = merge_anomalies_reports_streamlit(base_lines, taux_lines, montant_lines)
                                        if rapport:
                                            st.text_area("📊 Rapport d'anomalies", rapport, height=200)
                                        else:
                                            st.success("✅ Aucune anomalie détectée!")
                                except Exception as e:
                                    st.error(f"❌ Erreur lors de la détection des anomalies: {str(e)}")

                        # Affichage de l'historique
                        if st.session_state.historique_modifications:
                            st.markdown("### 📜 Historique des modifications")
                            df_historique = pd.DataFrame(st.session_state.historique_modifications)
                            st.dataframe(df_historique)

                    except Exception as e:
                        st.error(f"❌ Erreur lors du traitement de la cotisation: {str(e)}")

                except Exception as e:
                    st.error(f"❌ Erreur lors du chargement du fichier: {str(e)}")

            else:  # modification_type == "Autres"
                modification_main(uploaded_file)  # Utiliser le même fichier pour les deux types

    elif menu == "Extraction des données":
        extraction_main()

    elif menu == "Manuel":
        
        st.title("Mode d'emploi SMARTHR")

        st.markdown("""
            <style>
                .hero-subtitle {
                    font-size: 1.3em !important;
                    margin-bottom: 2rem;
                    font-style: italic;
                }
            </style>
            <p class='hero-subtitle'>Guide d'utilisation détaillé de l'application.</p>
        """, unsafe_allow_html=True)
        st.markdown("""
            <div style='background-color: white; padding: 20px; border-radius: 10px; margin: 20px 0;'>
                <h2 style='color: #386161; margin-bottom: 15px;'>Introduction</h2>
                <p style='color: #666666; line-height: 1.6;'>
                    SMARTHR MARTHR est une solution innovant combine expertise RH et 
                    intelligence artificielle pour optimiser la gestion des bulletins de paie grâce à l'IA.
                </p>
                <p style='color: #666666; line-height: 1.6;'>
                    L'application s'articule autour de trois fonctionnalités principales qui permettent de :
                    <ul>
                        <li>Extraire automatiquement les données des bulletins de paie PDF</li>
                        <li>Modifier et ajuster les cotisations de manière précise</li>
                        <li>Détecter les anomalies grâce à des modèles d'IA spécialisés</li>
                    </ul>
                </p>
            </div>

        """, unsafe_allow_html=True)

        # Continue with the existing sections...
        st.markdown("""
            <div style='background-color: white; padding: 20px; border-radius: 10px; margin: 20px 0;'>
                <h2 style='color: #386161; margin-bottom: 15px;'>1. Extraction des données 📤</h5>
                <h5>La première étape de votre périple pour détecter les anomalies dans votre bulletin de paie.</h5>
                <h5>Il est, clairement, très important d'avoir les données nécessaires, et il est tout autant important de les avoir dans les bonnes conditions, c'est pour cela qu'il va falloir nécessairement passer par l'extraction et la restructuration de vos données. </h5>
                <p style='color: #666666; margin-bottom: 15px;'>
                    <strong>Objectif :</strong> Extraire et structurer les données des bulletins de paie
                </p>
                <h4 style='color: #8ac447;'>Documents requis :</h4>
                <ul style='color: #666666;'>
                    <li>📄 Bulletins de paie (format PDF)</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

        # Display the first image
        st.image("images/Bulletin.png", caption="Exemple de bulletin de paie", use_container_width=True)

        st.markdown("""
            <div style='background-color: white; padding: 20px; border-radius: 10px; margin: 20px 0;'>
                <ul style='color: #666666;'>
                    <li>📊 Fichiers Excel avec informations complémentaires</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
        # Créer le DataFrame des colonnes requises
        colonnes_requises = {
            'Colonne': [
                'Matricule',
                'PLAFOND CUM',
                'ASSIETTE CUM',
                'MALADIE CUM',
                'Statut de salariés',
                'Region',
                'Frontalier',
                'Effectif',
                'Etablissement',
                'SMIC M CUM',
                'CUM Base precedente',
                'Base CUM M-1',
                'Brut CUM',
                'Total Brut',
                'Cluster',
                "Cumul d'assiette ( Mois courant inclus) (/102)",
                'Tranche C pre'
            ],
            'Description': [
                'Identifiant unique du salarié',
                'Cumul des plafonds de sécurité sociale',
                'Cumul des assiettes de cotisations',
                'Cumul des indemnités maladie',
                'Catégorie professionnelle du salarié',
                'Région d\'emploi du salarié',
                'Statut frontalier du salarié',
                'Nombre de salariés dans l\'établissement',
                'Code de l\'établissement',
                'Cumul du SMIC mensuel',
                'Cumul de la base précédente',
                'Base de cotisation du mois précédent',
                'Cumul du salaire brut',
                'Montant total du salaire brut',
                'Groupe de classification',
                'Cumul d\'assiette incluant le mois en cours',
                'Tranche C précédente'
            ]
        }
        
        df_colonnes = pd.DataFrame(colonnes_requises)
        # Supprimer les colonnes vides (si elles existent)
        df_colonnes = df_colonnes.loc[:, df_colonnes.columns.notnull()]


        # Ajuster la largeur d'affichage des colonnes
        pd.set_option('display.max_colwidth', None)  # Désactiver la limite de largeur (ou définir une valeur spécifique)

        
        # Styliser et afficher le DataFrame
        st.markdown("""
            <style>
            [data-testid="stDataFrame"] {
                width: 100% ;
            }
            
            [data-testid="stDataFrame"] > div {
                width: 100% !important;
            }
            
            /* Style du DataFrame lui-même */
            .dataframe {
                width: 100% !important;
                margin: auto !important;
            }
            
            /* Style des cellules */
            .dataframe td, .dataframe th {
                width: 50% !important;  /* Chaque colonne prend exactement 50% de la largeur */
                padding: 8px !important;
                text-align: left !important;
                white-space: normal !important;  /* Permet le retour à la ligne du texte */
            }
            
            /* Style des en-têtes */
            .dataframe th {
                background-color: #8ac447 !important;
                color: white !important;
                font-weight: bold !important;
            }
            
            /* Empêcher l'apparition de colonnes vides */
            .dataframe colgroup {
                width: 50% !important;
            }
            </style>
        """, unsafe_allow_html=True)
        
        st.dataframe(
            df_colonnes,
            height=200,
            use_container_width=True
        )




        st.markdown("""
            <div style='background-color: white; padding: 20px; border-radius: 10px; margin: 20px 0;'>
                <h4 style='color: #8ac447;'>Processus :</h4>
                <ol style='color: #666666;'>
                    <li>Importer les bulletins de paie PDF</li>
                    <li>Ajouter les fichiers Excel complémentaires</li>
                    <li>Lancer l'extraction</li>
                    <li>Télécharger le fichier CSV structuré</li>
                </ol>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("""
            <div style='background-color: white; padding: 20px; border-radius: 10px; margin: 20px 0;'>
                <h4 style='color: #8ac447;'>Exemple de données extraites</h4>
            </div>
        """, unsafe_allow_html=True)

        # Charger et afficher l'exemple de données extraites
        try:
            exemple_df = pd.read_csv("livrables/Exemple.csv")
            # Afficher seulement les 5 premières lignes avec la même mise en forme que précédemment
            st.markdown("""
                <style>
                [data-testid="stDataFrame"] {
                    width: 100% ;
                }
                
                [data-testid="stDataFrame"] > div {
                    width: 100% !important;
                }
                
                .dataframe {
                    width: 100% !important;
                    margin: auto !important;
                }
                
                .dataframe td, .dataframe th {
                    padding: 8px !important;
                    text-align: left !important;
                    white-space: normal !important;
                }
                
                .dataframe th {
                    background-color: #8ac447 !important;
                    color: white !important;
                    font-weight: bold !important;
                }
                </style>
            """, unsafe_allow_html=True)
            
            st.dataframe(
                exemple_df.head(),
                height=200,
                use_container_width=True
            )
            st.caption("Exemple d'un extrait de données structurées après extraction")
            
        except FileNotFoundError:
            st.error("❌ Fichier d'exemple non trouvé")
            st.caption("L'exemple des données extraites n'est pas disponible")

        # Section Modification
        st.markdown("""
            <div style='background-color: white; padding: 20px; border-radius: 10px; margin: 20px 0;'>
                <h2 style='color: #386161; margin-bottom: 15px;'>2. Modification des cotisations 📝</h2>
                <h5>Si vous êtes arrivés jusqu'ici c'est que vos données sont extraits correctement. Félicitations ! Faîtes un petit détour par cette rubrique vous permettant faire des tests unitaires des données de vos employés en modifiant la rubrique de votre choix pour l'employé de votre choix. Une méthode plus intuitive et pratique de vérifier vos données.</h5>
                <p style='color: #666666; margin-bottom: 15px;'>
                    <strong>Objectif :</strong> Modifier et tester les valeurs des cotisations par employé
                </p>
                <h4 style='color: #8ac447;'>Fonctionnalités :</h4>
                <ul style='color: #666666;'>
                    <li>🔍 Sélection précise de l'employé</li>
                    <li>📊 Modification des rubriques de cotisation</li>
                    <li>✅ Validation des modifications</li>
                    <li>📥 Export des données modifiées</li>
                </ul>
                <div style='background-color: #f8f9fa; padding: 10px; border-radius: 5px; margin-top: 10px;'>
                    <p style='color: #386161; margin: 0;'>💡 <em>Utilisez le fichier CSV extrait à l'étape précédente</em></p>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # Section Détection
        st.markdown("""
            <div style='background-color: white; padding: 20px; border-radius: 10px; margin: 20px 0;'>
                <h2 style='color: #386161; margin-bottom: 15px;'>3. Détection d'anomalies 🔍</h2>
                <h5>Vous y êtes enfin ! La dernière étape de votre parcours qui vous permet de détecter les anomalies dans vos données. Grâce à des modèles d'IA spécialisés, vous pouvez identifier toutes les incohérences et les erreurs dans les cotisations de vos employés. Une solution complète pour garantir la fiabilité de vos bulletins de paie.</h5>
                <p style='color: #666666; margin-bottom: 15px;'>
                    <strong>Objectif :</strong> Analyser et identifier les anomalies dans les bulletins de paie
                </p>
                <h4 style='color: #8ac447;'>Caractéristiques :</h4>
                <ul style='color: #666666;'>
                    <li>🔍 Analyse complète des cotisations</li>
                    <li>📊 Visualisation des anomalies détectées</li>
                    <li>📝 Rapport détaillé téléchargeable</li>
                    <li>🔄 Traitement par lots</li>
                </ul>
                <div style='background-color: #f8f9fa; padding: 10px; border-radius: 5px; margin-top: 10px;'>
                    <p style='color: #386161; margin: 0;'>💡 <em>Le rapport peut être consulté en ligne ou téléchargé au format TXT</em></p>
                </div>
            </div>
        """, unsafe_allow_html=True)

    # Ajout du bouton de déconnexion
    if st.sidebar.button("📤 Déconnexion"):
        st.session_state['authenticated'] = False
        st.session_state['username'] = None
        st.rerun()

if __name__ == "__main__":
    main()