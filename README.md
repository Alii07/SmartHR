# Projet HCM CP : Détection d'Anomalies pour les Bulletins de Paie

Ce répertoire contient les codes pour le projet **HCM CP : Détection d'anomalies pour les bulletins de paie**.

---

## Structure du Répertoire

### **1. Génération de Base de Données Fictive**
Dans ce répertoire, nous avons généré une base de données fictive selon les règles des cotisations URSSAF. 

### **2. Modèles**
Ce répertoire contient la méthodologie utilisée pour entraîner les modèles, divisée en deux catégories :

- **Les taux** :
  - Utilisation de **Isolation Forest** ou **Random Forest** selon le cas.
- **Les bases** :
  - Classification avec **Random Forest**.
  - Prédiction avec **régression linéaire**.

### **3. Fusion URSSAF**
Ce répertoire comprend :

- La fusion des modèles dans un cadre déployable sur **Streamlit**.
- Les classes principales **Bases** et **Taux**, utilisant leurs modèles respectifs simultanément.
- Le fichier `main.py`, qui exécute l'intégralité du processus.
- Une vérification de la cohérence des montants via la multiplication des taux et des bases.

**Ordre d'application des classes :**
1. **Base**
2. **Taux**
3. **Montant**

Si une anomalie est détectée dans une ligne par l'une des classes, cette ligne est marquée comme erronée. Un rapport final téléchargeable est généré, indiquant chaque ligne et les anomalies détectées.

---

## Fonctionnalités Principales

1. **Base de données fictive** : Générée pour simuler des cas réels selon les règles URSSAF.
2. **Entraînement des modèles** :
   - Modèles dédiés aux taux et aux bases.
   - Approches variées adaptées à chaque type de données.
3. **Fusion des résultats** :
   - Intégration des prédictions dans un processus automatisé.
   - Détection et marquage des anomalies avec un rapport détaillé.

---

## Rapport Final

Le rapport final contient :
- Chaque ligne analysée.
- Les anomalies détectées par type.
- Une structure téléchargeable facilement interprétable.

---

## Améliorations Futures

- Intégration d'algorithmes supplémentaires pour une meilleure précision.
- Support multilingue pour les rapports.
- Interface utilisateur améliorée sur **Streamlit**.

---

## Installation

### Prérequis

Assurez-vous d'avoir Python installé et exécutez :
```bash
pip install -r requirements.txt
```

### Exécution

Pour démarrer l'application Streamlit :
```bash
streamlit run main.py
```

---

### Liste des Contributions Traitées

Tableau récapitulatif des différentes contributions traitées et de l'état d'avancement de leur traitement :
| Contribution     | Libelle                   | Implemented     | Changed         | Validated       | Comments                                 |
|:----------------:|:-------------------------:|:---------------:|:---------------:|:---------------:|:-------------------------                |
| - 7C00           | Pôle Emploi               |       ✅        |       ❌        |       ✅        | Valider par la data reelle              |
| - 7C10           | Pôle Emploi               |       ✅        |       ✅        |       ✅        | Valider par la data reelle                      |
| - 7C20           | AGS                       |       ✅        |       ✅        |       ✅        |  Valider par la data reelle         |
| - 7P20           | Prévoyance Tranche A      |       ✅        |       ❌        |       ❌        | Valider par la data            |
| - 7P25           | Prévoyance Tranche B      |       ✅        |       ✅        |       ✅        | Valider par la data             |
| - 7P26           | Prévoyance Tranche C      |       ✅        |       ✅        |       ✅        | Valider par la data                     |
| - 7P10           | Prévoyance Non Cadre TA   |       ✅        |       ✅        |       ✅        | Data manquante     |
| - 7P11           | Prévoyance Non Cadre TB   |       ✅        |       ✅        |       ❌        | Data manquante       |
| - 7R53           | Retraite Tranche A        |       ✅        |       ❌        |       ✅        | Valider par la data reelle              |
| - 7R54           | Retraite Tranche B        |       ✅        |       ✅        |       ✅        | Valider par la data reelle                        |
| - 7R63           | CEG Tranche A             |       ✅        |       ✅        |       ✅        | Valider par la data reelle                        |
| - 7R64           | CEG Tranche B             |       ✅        |       ✅        |       ✅        | Valider par la data reelle                         |
| - 7P70           | CET Tranche A             |       ✅        |       ✅        |       ✅        | Valider par la data reelle                         |
| - 7P71           | CET Tranche B             |       ✅        |       ✅        |       ❌        | Valider par la data                       |
| - Mutuelle       | Mutuelle                  |       ✅        |       ❌        |       ❌        | Data manquante                         |
| - APEC           | APEC                      |       ✅        |       ✅        |       ❌        | Data manquante                        |


---

### Contact
 
Pour toute question ou contribution, contactez l'équipe via [ali.slaoui@hcm-cp.com](mailto:ali.slaoui@hcm-cp.com).
