# Real Estate Price Prediction - Paris

**Authors:** Colin Pietri & Cristian Larrain

## Business Challenge

Predicting real estate prices is a critical challenge for real estate professionals who face:
- Limited information about properties
- Short response times for client inquiries
- Reputation risks from inaccurate estimates

This project develops an AI-powered assistant that accurately predicts apartment prices in Paris based on key property characteristics.

## Dataset

**Source:** DVF (Demandes de Valeurs Foncières) - Public French real estate transaction data

**Coverage:**
- Period: 2021-2024
- Original size: 16,000,000 transactions
- 43 columns
- Focus: Paris apartments only (~150,000 transactions after filtering)

**Key Features:**
- surface_m2: Property surface area
- nombre_pieces_principales: Number of main rooms
- code_postal: Postal code (arrondissements)
- annee: Year of transaction
- mois: Month of transaction
- valeur_fonciere: Property value (target)

## Reproduction Instructions

### Prerequisites
- Python 3.12
- Dataset file: clean_dataset (Parquet format)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/colinpietri20/Supervised_LearningII.git
cd Supervised_LearningII
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Place the dataset in the root directory

### Running the Training Pipeline

Train a new model:
```bash
python main.py --train
```

Evaluate existing model:
```bash
python main.py --evaluate
```

## Baseline Model

**Model:** Linear Regression

**Features:**
- Numeric: surface_m2, nombre_pieces_principales, annee, mois
- Categorical: code_postal (One-Hot Encoded)

**Preprocessing:**
- Standardization: StandardScaler
- Categorical Encoding: One-Hot Encoding

**Results:**

| Metric | Value | Interpretation |
|--------|-------|----------------|
| R² | 0.02 | Model explains only 2% of variance |
| RMSE | 14,989,000€ | Very high prediction error |
| MAPE | 100% | Predictions systematically incorrect |

**Why it failed:**
- Extreme outliers (max surface: 3,160m², max price: 762M€)
- Mixed property types (houses vs apartments)
- Model too simple

## Experiment Tracking

### Experiment 1: Outlier Removal
**Changes:**
- Filtered to apartments only
- Surface: 9-300 m²
- Price: 50K-3M€
- Removed 20,000 outliers

### Experiment 2: Advanced Models
**Tested:** XGBoost, CatBoost, Random Forest, LightGBM

**Results:**

| Model | R² | RMSE | MAE | MAPE |
|-------|-----|------|-----|------|
| **LightGBM** | **0.816** | **175,400€** | **100,400€** | **20%** |
| CatBoost | 0.814 | 176,000€ | 100,900€ | - |
| XGBoost | 0.80 | 179,000€ | 102,000€ | - |

**Improvements:**
- R² improved from 0.02 to 0.816 (40x better)
- RMSE reduced from 14.9M€ to 175K€ (85x better)

### Feature Importance
1. surface_m2 (dominant)
2. code_postal
3. mois
4. nombre_pieces
5. annee

## Contributors

- **Colin Pietri**
- **Cristian Larrain**

---
Albert School - M1 Data Finance
