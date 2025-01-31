import pandas as pd
from model_config import BaseModelConfig, ModelRegistry
import streamlit as st

def load_and_preprocess_data(data):
    """Charge et prétraite les données."""
    if isinstance(data, str):
        data = pd.read_csv(data)

    # Employee type mapping
    if 'Statut de salariés' in data.columns:
        statut_mapping = {
            "Senior executive": "Cadre",
            "Executive": "Cadre",
            "Cadre Interne": "Cadre",
            "Cadre": "Cadre",
            "Collaborateur 4/4bis": "Non cadre",
            "AM 4/4bis": "Non cadre",
            "Admi / Tech art 36": "Non cadre",
            "AM art 36": "Non cadre",
            "Ouvrier art 36": "Non cadre",
            "Admi / Tech": "Non cadre",
            "Agent de maitrise": "Non cadre",
            "Employé": "Non cadre",
            "Technicien": "Non cadre",
            "Ouvrier": "Non cadre",
            "Apprenti (W.C)": "Cadre",
            "Apprenti (B.C)": "Non cadre",
            "Contrat pro (Cadre)": "Cadre",
            "Contrat pro (NC-W.C)": "Cadre",
            "Contrat pro (NC-B.C)": "Non cadre",
            "Stage": "Non cadre"
        }
        data['Employee type'] = data['Statut de salariés'].map(statut_mapping)
    
    # Création des colonnes de cumul
    if all(col in data.columns for col in ["PLAFOND CUM M-1", "PLAFOND M"]):
        data['Cum Plafond'] = data["PLAFOND CUM M-1"] + data["PLAFOND M"]
        data["4* Cum Plafond"] = 4 * data["Cum Plafond"]
        data["8* Cum Plafond"] = 8 * data["Cum Plafond"]
    
    # Mapping des colonnes de base
    base_mapping = {
        '7C10': '7C10Base',
        '7C00': '7C00Base',
        '7R53': '7R53Base',
        '7R54': '7R54Base',
        '7P20': '7P20Base',
        '7P25': '7P25Base',
        '7P26': '7P26Base',
        '7R44': '7R44Base',
        '7R45': '7R45Base',
        '7R63': '7R63Base',
        #'7R71': '7R64Base',
        '7R70': '7R70Base',
        #'7R71': '7R71Base',
        '7P10': '7P10Base',
    }
    
    # Création des colonnes Base et Cumul Base
    for model_code, base_col in base_mapping.items():
        if base_col in data.columns:
            data[f"Base {model_code}"] = data[base_col].copy()

    # Mapping des colonnes de cumul pour chaque cotisation
    cumul_mapping = {
        '7R53': 'RER. T1',
        '7R54': 'RER. T2',
        '7P20': 'PRÉV. TA',
        '7P25': 'PRÉV. TB',
        '7P26': 'PRÉV. TC',
        '7R44': 'APEC T1',
        '7R45': 'APEC T2',
        '7R70': 'CET T1',
        #'7R71': 'CET T2',
        '7R63': 'CEG T1',
        #'7R71': 'CEG T2',
        '7P10': 'PRÉV. TA',
    }

    # Création des colonnes Cumul Base selon le mapping
    for code, cumul_col in cumul_mapping.items():
        if cumul_col in data.columns:
            data[f"Cumul Base {code}"] = data[cumul_col].copy()

    # Mapping des colonnes d'assiette
    assiette_mapping = {
        '7C00': 'ASS POLE EM',
        '7C10': 'ASS POLE EM',
        '7R53': 'ASS RET.',
        '7R54': 'ASS RET.',
        '7P20': 'ASS PRÉ CAD',
        '7P25': 'ASS PRÉ CAD',
        '7P26': 'ASS PRÉ CAD',
        '7R44': 'ASS RET.',
        '7R45': 'ASS RET.',
        '7R70': 'ASS RET.',
        #'7R71': 'ASS RET.',
        '7R63': 'ASS RET.',
        #'7R71': 'ASS RET.',
        '7P10': 'ASS PRÉ NC',
    }

    # Création des colonnes Cum Assiette
    for model_code, assiette_col in assiette_mapping.items():
        if assiette_col in data.columns:
            data[f'Cum Assiette {model_code}'] = data[assiette_col].copy()
    
    # Création des colonnes calculées supplémentaires
    data["4*Cum Plafond"] = data["4* Cum Plafond"] 
    data["4 * Cum Plafond"] = data["4* Cum Plafond"] 
    
    
    return data

def create_model_configs() -> ModelRegistry:
    registry = ModelRegistry()    
    # Configuration 7R71 (nouvelle)
    config_7r71 = BaseModelConfig("7R71")
    
    # Ajout des classifieurs pour 7R71
    config_7r71.add_classifier(
        "C0",
        "7R71_Base_classification_model_C0",
        ["Type de contrat"],
        "`Type de contrat` not in ['Stagiaire', 'Apprenti']"
    )
    config_7r71.add_classifier(
        "C1",
        "7R71_Base_classification_model_C1",
        ["Cum Plafond", "Cum Assiette"],
        "`Cum Plafond` > `Cum Assiette`"
    )
    
    # Ajout des régresseurs pour 7R71
    config_7r71.add_regressor(
        (1, 0),
        "7R71_Base_regression_model_1_0",
        ["Cum Assiette", "Cum Plafond"],
        "Base 7R71"
    )
    config_7r71.add_regressor(
        (1, 1),
        "7R71_Base_regression_model_1_1",
        ["Cumul Base 7R71"],
        "Base 7R71"
    )
    
    registry.register_model(config_7r71)
    
    # Configuration 7R70 (nouvelle)
    config_7r70 = BaseModelConfig("7R70")
    
    # Ajout des classifieurs pour 7R70
    config_7r70.add_classifier(
        "C0",
        "7R70_Base_classification_model_C0",
        ["Type de contrat"],
        "`Type de contrat` not in ['Stagiaire', 'Apprenti']"
    )
    config_7r70.add_classifier(
        "C1",
        "7R70_Base_classification_model_C1",
        ["Cum Plafond", "Cum Assiette"],
        "`Cum Plafond` > `Cum Assiette`"
    )
    
    # Ajout des régresseurs pour 7R70
    config_7r70.add_regressor(
        (1, 0),
        "7R70_Base_regression_model_1_0",
        ["Cum Plafond", "Cumul Base 7R70"],
        "Base 7R70"
    )
    config_7r70.add_regressor(
        (1, 1),
        "7R70_Base_regression_model_1_1",
        ["Cumul Base 7R70"],
        "Base 7R70"
    )
    
    registry.register_model(config_7r70)
    
    # Configuration 7R64 (nouvelle)
    config_7r64 = BaseModelConfig("7R64")
    
    # Ajout des classifieurs pour 7R64
    config_7r64.add_classifier(
        "C0",
        "7R64_Base_classification_model_C0",
        ["Statut de salariés"],
        "`Statut de salariés` not in ['Apprenti (B.C)','Stagiaire','Apprenti (W.C)']"
    )
    config_7r64.add_classifier(
        "C1",
        "7R64_Base_classification_model_C1",
        ["Cum Plafond", "Cum Assiette"],
        "4 * `Cum Plafond` > `Cum Assiette`"
    )
    
    # Ajout des régresseurs pour 7R64
    config_7r64.add_regressor(
        (1, 0),
        "7R64_Base_regression_model_1_0",
        ["Cum Assiette", "Cumul Base 7R63", "Cumul Base 7R64", "Base 7R63"],
        "Base 7R64"
    )
    config_7r64.add_regressor(
        (1, 1),
        "7R64_Base_regression_model_1_1",
        ["Cumul Base 7R64"],
        "Base 7R64"
    )
    
    registry.register_model(config_7r64)
    
    # Configuration 7R44 (nouvelle)
    config_7r44 = BaseModelConfig("7R44")
    
    # Ajout des classifieurs pour 7R44
    config_7r44.add_classifier(
        "C0",
        "7R44_Base_classification_model_C0",
        ["Employee type"],
        "`Employee type` not in ['Cadre']"
    )
    config_7r44.add_classifier(
        "C1",
        "7R44_Base_classification_model_C1",
        ["Cum Plafond", "Cum Assiette"],
        "`Cum Plafond` > `Cum Assiette`"
    )
    
    # Ajout des régresseurs pour 7R44
    config_7r44.add_regressor(
        (0, 0),
        "7R44_Base_regression_model_0_0",
        ["Cum Plafond", "Cumul Base 7R44"],
        "Base 7R44"
    )
    config_7r44.add_regressor(
        (0, 1),
        "7R44_Base_regression_model_0_1",
        ["Cum Assiette", "Cumul Base 7R44"],
        "Base 7R44"
    )
    
    registry.register_model(config_7r44)
    
    # Configuration 7R45 (nouvelle)
    config_7r45 = BaseModelConfig("7R45")
    
    # Ajout des classifieurs pour 7R45
    config_7r45.add_classifier(
        "C0",
        "7R45_Base_classification_model_C0",
        ["Employee type"],
        "`Employee type` not in ['Cadre']"
    )
    config_7r45.add_classifier(
        "C1",
        "7R45_Base_classification_model_C1",
        ["Cum Plafond", "Cum Assiette"],
        "`Cum Plafond` > `Cum Assiette`"
    )
    config_7r45.add_classifier(
        "C2",
        "7R45_Base_classification_model_C2",
        ["Cum Plafond", "Cum Assiette"],
        "4*`Cum Plafond` > `Cum Assiette`"
    )
    
    # Ajout des régresseurs pour 7R45
    config_7r45.add_regressor(
        (0, 0, 0),
        "7R45_Base_regression_model_0_0_0",
        ["Cum Plafond","Cumul Base 7R44", "Cumul Base 7R45","Base 7R44"],
        "Base 7R45"
    )
    config_7r45.add_regressor(
        (0, 0, 1),
        "7R45_Base_regression_model_0_0_1",
        ["Cum Assiette", "Cumul Base 7R45", "Cumul Base 7R44","Base 7R44"],
        "Base 7R45"
    )
    
    config_7r45.add_regressor(
        (0, 1, 1),
        "7R45_Base_regression_model_0_1_1",
        ["Cumul Base 7R45"],
        "Base 7R45"
    )
    
    registry.register_model(config_7r45)
    
    # Configuration 7P26 (nouvelle)
    config_7p26 = BaseModelConfig("7P26")
    
    # Ajout des classifieurs pour 7P26
    config_7p26.add_classifier(
        "C0",
        "7P26_Base_classification_model_C0",
        ["Employee type"],
        "`Employee type` not in ['Cadre']"
    )
    config_7p26.add_classifier(
        "C1",
        "7P26_Base_classification_model_C1",
        ["Cum Plafond", "Cum Assiette"],
        "`Cum Plafond` > `Cum Assiette`"
    )
    config_7p26.add_classifier(
        "C2",
        "7P26_Base_classification_model_C2",
        ["Cum Plafond", "Cum Assiette"],
        "8*`Cum Plafond` > `Cum Assiette`"
    )
    
    # Ajout des régresseurs pour 7P26
    config_7p26.add_regressor(
        (0, 0, 0),
        "7P26_Base_regression_model_0_0_0",
        ["Cum Plafond", "Cumul Base 7P20", "Cumul Base 7P25", "Cumul Base 7P26", "Base 7P20", "Base 7P25"],
        "Base 7P26"
    )
    config_7p26.add_regressor(
        (0, 0, 1),
        "7P26_Base_regression_model_0_0_1",
        ["Cum Assiette", "Cumul Base 7P26", "Cumul Base 7P25", "Cumul Base 7P20", "Base 7P20", "Base 7P25"],
        "Base 7P26"
    )
    
    registry.register_model(config_7p26)
    
    # Configuration 7P25 (nouvelle)
    config_7p25 = BaseModelConfig("7P25")
    
    # Ajout des classifieurs pour 7P25
    config_7p25.add_classifier(
        "C0",
        "7P25_Base_classification_model_C0",
        ["Employee type"],
        "`Employee type` not in ['Cadre']"
    )
    config_7p25.add_classifier(
        "C1",
        "7P25_Base_classification_model_C1",
        ["Cum Plafond", "Cum Assiette"],
        "`Cum Plafond` > `Cum Assiette`"
    )
    config_7p25.add_classifier(
        "C2",
        "7P25_Base_classification_model_C2",
        ["Cum Plafond", "Cum Assiette"],
        "4*`Cum Plafond` > `Cum Assiette`"
    )
    
    # Ajout des régresseurs pour 7P25
    config_7p25.add_regressor(
        (0, 0, 0),
        "7P25_Base_regression_model_0_0_0",
        ["Cum Plafond", "Cumul Base 7P20", "Cumul Base 7P25", "Base 7P20"],
        "Base 7P25"
    )
    config_7p25.add_regressor(
        (0, 0, 1),
        "7P25_Base_regression_model_0_0_1",
        ["Cum Assiette", "Cumul Base 7P25", "Cumul Base 7P20", "Base 7P20"],
        "Base 7P25"
    )
    config_7p25.add_regressor(
        (0, 1, 1),
        "7P25_Base_regression_model_0_1_1",
        ["Cumul Base 7P25"],
        "Base 7P25"
    )
    
    registry.register_model(config_7p25)
    
    # Configuration 7P20 (nouvelle)
    config_7p20 = BaseModelConfig("7P20")
    
    # Ajout des classifieurs pour 7P20
    config_7p20.add_classifier(
        "C0",
        "7P20_Base_classification_model_C0",
        ["Employee type"],
        "`Employee type` not in ['Cadre']"
    )
    config_7p20.add_classifier(
        "C1",
        "7P20_Base_classification_model_C1",
        ["Cum Plafond", "Cum Assiette"],
        "`Cum Plafond` > `Cum Assiette`"
    )
    
    # Ajout des régresseurs pour 7P20
    config_7p20.add_regressor(
        (0, 0),
        "7P20_Base_regression_model_0_0",
        ["Cum Plafond", "Cumul Base 7P20"],
        "Base 7P20"
    )
    config_7p20.add_regressor(
        (0, 1),
        "7P20_Base_regression_model_0_1",
        ["Cum Assiette", "Cumul Base 7P20"],
        "Base 7P20"
    )
    
    registry.register_model(config_7p20)
    
    # Configuration 7P10 (nouvelle)
    config_7p10 = BaseModelConfig("7P10")
    
    # Ajout des classifieurs pour 7P20
    config_7p10.add_classifier(
        "C0",
        "7P10_Base_classification_model_C0",
        ["Employee type"],
        "`Employee type` not in ['Cadre']"
    )
    config_7p10.add_classifier(
        "C1",
        "7P10_Base_classification_model_C1",
        ["Cum Plafond", "Cum Assiette"],
        "`Cum Plafond` > `Cum Assiette`"
    )
    
    # Ajout des régresseurs pour 7P20
    config_7p10.add_regressor(
        (0, 0),
        "7P10_Base_regression_model_0_0",
        ["Cum Plafond", "Cumul Base 7P10"],
        "Base 7P10"
    )
    config_7p10.add_regressor(
        (0, 1),
        "7P10_Base_regression_model_0_1",
        ["Cum Assiette", "Cumul Base 7P10"],
        "Base 7P10"
    )
    
    registry.register_model(config_7p10)
    
    # Configuration 7R54 (nouvelle)
    config_7r54 = BaseModelConfig("7R54")
    
    # Ajout des classifieurs pour 7R54
    config_7r54.add_classifier(
        "C0",
        "7R54_Base_classification_model_C0",
        ["Employee type"],
        "`Employee type` not in ['Cadre']"
    )
    config_7r54.add_classifier(
        "C1",
        "7R54_Base_classification_model_C1",
        ["Cum Plafond", "Cum Assiette"],
        "`Cum Plafond` > `Cum Assiette`"
    )
    
    # Ajout des régresseurs pour 7R54
    config_7r54.add_regressor(
        (0, 0),
        "7R54_Base_regression_model_0_0",
        ["Cum Assiette", "Cumul Base 7R53", "Cumul Base 7R54", "Base 7R53"],
        "Base 7R54"
    )
    config_7r54.add_regressor(
        (0, 1),
        "7R54_Base_regression_model_0_1",
        ["Cumul Base 7R54"],
        "Base 7R54"
    )
    
    registry.register_model(config_7r54)
    
    # Configuration 7R53 (nouvelle)
    config_7r53 = BaseModelConfig("7R53")
    
    # Ajout des classifieurs
    config_7r53.add_classifier(
        "C0",
        "7R53_Base_classification_model_C0",
        ["Employee type"],
        "`Employee type` not in ['Cadre']"
    )
    config_7r53.add_classifier(
        "C1",
        "7R53_Base_classification_model_C1",
        ["Cum Plafond", "Cum Assiette"],
        "`Cum Plafond` > `Cum Assiette`"
    )
    
    # Ajout des régresseurs
    config_7r53.add_regressor(
        (0, 0),
        "7R53_Base_regression_model_0_0",
        ["Cum Plafond", "Cumul Base 7R53"],
        "Base 7R53"
    )
    config_7r53.add_regressor(
        (0, 1),
        "7R53_Base_regression_model_0_1",
        ["Cum Assiette", "Cumul Base 7R53"],
        "Base 7R53"
    )
    
    registry.register_model(config_7r53)
    
    # Configuration 7R63 (nouvelle)
    config_7r63 = BaseModelConfig("7R63")
    
    # Ajout des classifieurs pour 7R63
    config_7r63.add_classifier(
        "C0",
        "7R63_Base_classification_model_C0",
        ["Statut de salariés"],
        "`Statut de salariés` not in ['Apprenti (B.C)','Stagiaire','Apprenti (W.C)']"
    )
    config_7r63.add_classifier(
        "C1",
        "7R63_Base_classification_model_C1",
        ["Cum Plafond", "Cum Assiette"],
        "`Cum Plafond` > `Cum Assiette`"
    )
    
    # Ajout des régresseurs pour 7R63
    config_7r63.add_regressor(
        (1, 0),
        "7R63_Base_regression_model_1_0",
        ["Cum Plafond", "Cumul Base 7R63"],
        "Base 7R63"
    )
    config_7r63.add_regressor(
        (1, 1),
        "7R63_Base_regression_model_1_1",
        ["Cum Assiette", "Cumul Base 7R63"],
        "Base 7R63"
    )
    
    registry.register_model(config_7r63)
    
    # Configuration 7C00
    config_7c00 = BaseModelConfig("7C00")
    
    # Ajout des classifieurs
    config_7c00.add_classifier(
        "C0",
        "7C00_Base_classification_model_C0",
        ["Type de contrat"],
        "`Type de contrat` not in ['Mandat Social', 'Stagiaire', 'Apprenti']"
    )
    config_7c00.add_classifier(
        "C1",
        "7C00_Base_classification_model_C1",
        ["Cum Plafond", "Cum Assiette"],
        "`Cum Plafond` > `Cum Assiette`"
    )
    
    # Ajout des régresseurs
    config_7c00.add_regressor(
        (1, 0),
        "7C00_Base_regression_model_1_0",
        ["Cum Plafond", "Cumul Base 7C00"],
        "Base 7C00"
    )
    
    config_7c00.add_regressor(
        (1,1),
        "7C00_Base_regression_model_1_1",
        ["Cum Assiette", "Cumul Base 7C00", "Cumul Base 7C10"],
        "Base 7C00"
    )
    
    registry.register_model(config_7c00)
    
    # Configuration 7C10 (similaire)
    config_7c10 = BaseModelConfig("7C10")
    
    # Ajout des classifieurs
    config_7c10.add_classifier(
        "C0",
        "Base_7C10_classification_model_C0",
        ["Type de contrat"],
        "`Type de contrat` not in ['Mandat Social', 'Stagiaire', 'Apprenti']"
    )
    config_7c10.add_classifier(
        "C1",
        "Base_7C10_classification_model_C1",
        ["Cum Plafond", "Cum Assiette"],
        "`Cum Plafond` > `Cum Assiette`"
    )
    config_7c10.add_classifier(
        "C2",
        "Base_7C10_classification_model_C2",
        ["4* Cum Plafond", "Cum Assiette"],
        "`4* Cum Plafond` > `Cum Assiette`"
    )
    
    # Ajout des régresseurs
    config_7c10.add_regressor(
        (1, 0, 0),
        "Base_7C10_regression_model_1_0_0",
        ["4* Cum Plafond", "Cumul Base 7C00", "Cumul Base 7C10", "Base 7C00"],
        "Base 7C10"
    )
    config_7c10.add_regressor(
        (1, 0, 1),
        "Base_7C10_regression_model_1_0_1",
        ["Cum Assiette", "Cumul Base 7C00", "Cumul Base 7C10", "Base 7C00"],
        "Base 7C10"
    )
    config_7c10.add_regressor(
        (1, 1, 1),
        "Base_7C10_regression_model_1_1_1",
        ["Cumul Base 7C10"],
        "Base 7C10"
    )
    
    registry.register_model(config_7c10)
    return registry

# Remplacer MODEL_CONFIGS par:
MODEL_REGISTRY = create_model_configs()


