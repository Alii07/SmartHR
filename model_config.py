from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import pandas as pd

@dataclass
class ClassifierConfig:
    name: str
    features: List[str]
    condition: str
    
@dataclass
class RegressorConfig:
    name: str
    features: List[str]
    target: str
    condition_tuple: tuple

@dataclass
class TauxClassifierConfig:
    name: str
    features: List[str]
    condition: str

class BaseModelConfig:
    def __init__(self, code: str):
        self.code = code
        self.classifiers: Dict[str, ClassifierConfig] = {}
        self.regressors: Dict[tuple, RegressorConfig] = {}
        
    def add_classifier(self, classifier_id: str, name: str, features: List[str], condition: str):
        self.classifiers[classifier_id] = ClassifierConfig(name, features, condition)
        
    def add_regressor(self, condition_tuple: tuple, name: str, features: List[str], target: str):
        self.regressors[condition_tuple] = RegressorConfig(name, features, target, condition_tuple)
        
    def get_all_features(self) -> set:
        features = set()
        for clf in self.classifiers.values():
            features.update(clf.features)
        for reg in self.regressors.values():
            features.update(reg.features)
        return features

class TauxModelConfig:
    """Configuration spécifique pour les modèles de taux qui n'ont que des classifieurs."""
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

class ModelRegistry:
    def __init__(self):
        self.configs: Dict[str, BaseModelConfig] = {}
        self.taux_configs: Dict[str, TauxModelConfig] = {}  # Ajout du registre pour les taux
        
    def register_model(self, model_config: BaseModelConfig):
        self.configs[model_config.code] = model_config
        
    def register_taux_model(self, taux_config: TauxModelConfig):  # Nouvelle méthode
        self.taux_configs[taux_config.code] = taux_config
        
    def get_config(self, code: str) -> Optional[BaseModelConfig]:
        return self.configs.get(code)
        
    def get_taux_config(self, code: str) -> Optional[TauxModelConfig]:  # Nouvelle méthode
        return self.taux_configs.get(code)
        
    def keys(self):
        """Retourne les codes des modèles enregistrés."""
        return list(self.configs.keys()) + list(self.taux_configs.keys())
