from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import pandas as pd

@dataclass
class TauxClassifierConfig:
    name: str
    features: List[str]
    condition: str

class TauxModelConfig:
    def __init__(self, code: str):
        self.code = code
        self.classifiers: Dict[str, TauxClassifierConfig] = {}
        
    def add_classifier(self, classifier_id: str, name: str, features: List[str], condition: str):
        self.classifiers[classifier_id] = TauxClassifierConfig(name, features, condition)
        
    def get_all_features(self) -> set:
        features = set()
        for clf in self.classifiers.values():
            features.update(clf.features)
        return features

class TauxModelRegistry:
    def __init__(self):
        self.configs: Dict[str, TauxModelConfig] = {}
        
    def register_model(self, model_config: TauxModelConfig):
        self.configs[model_config.code] = model_config
        
    def get_config(self, code: str) -> Optional[TauxModelConfig]:
        return self.configs.get(code)
        
    def keys(self):
        return self.configs.keys()

# Définition des configurations globales
MODEL_NAMES = {
    "classification": {
        "7C00": {
            "C0": "Taux_7C00_classification_model_C0.joblib"
        },
        "7C10": {
            "C0": "Taux_7C10_classification_model_C0.joblib"
        }
    }
}

# Configuration d'entraînement pour chaque modèle
INPUT_TRAIN = {
    "7C00": {
        "C0": {
            "features": ["7C00 Base", "7C00 Taux"],
            "condition": "True",  # Toujours vrai car on veut toujours classifier
            "target_column": "7C00 Taux"
        }
    },
    "7C10": {
        "C0": {
            "features": ["7C10 Base", "7C10 Taux"],
            "condition": "True",  # Toujours vrai car on veut toujours classifier
            "target_column": "7C10 Taux"
        }
    }
}

def create_taux_model_configs() -> TauxModelRegistry:
    registry = TauxModelRegistry()
    
    for base_code, config in INPUT_TRAIN.items():
        model_config = TauxModelConfig(base_code)
        for classifier_id, classifier_config in config.items():
            model_config.add_classifier(
                classifier_id,
                MODEL_NAMES["classification"][base_code][classifier_id],
                classifier_config["features"],
                classifier_config["condition"]
            )
        registry.register_model(model_config)
    
    return registry

TAUX_MODEL_REGISTRY = create_taux_model_configs()

def preprocess_taux_data(data):
    """Prétraite les données pour les modèles de taux."""
    processed_data = data.copy()
    
    for code, config in INPUT_TRAIN.items():
        try:
            for classifier_id, classifier_config in config.items():
                if all(feature in data.columns for feature in classifier_config["features"]):
                    # Copier les features nécessaires
                    for feature in classifier_config["features"]:
                        processed_data[feature] = data[feature]
                    # Ajouter la colonne de fraude si elle existe
                    if f"Taux {code}_Fraud" in data.columns:
                        processed_data[f"Taux {code}_Fraud"] = data[f"Taux {code}_Fraud"]
        except Exception as e:
            print(f"Skip {code}: {str(e)}")
    
    return processed_data
