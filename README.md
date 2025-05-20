# socialiot
AI to Detect Social Engineering Attacks in Social IoT Environments

# SIoT Attack Detection

This repository contains Python scripts for preprocessing and analyzing network traffic data to detect social engineering attacks in Social Internet of Things (SIoT) environments, as outlined in the research paper on leveraging behavioral biometrics and AI.

## Overview

The project processes the CIC IoT-DIAD 2024 dataset and applies machine learning models (Random Forest, XGBoost, LSTM, CNN) to classify network traffic as Benign, BruteForce, DNSSpoofing, or ARPSpoofing. It includes preprocessing steps and generates performance metrics and visualizations as described in the research paper.

## Files

### 1. `IOTM.py`
**Purpose**: Preprocesses the CIC IoT-DIAD 2024 dataset for machine learning tasks.

**Functionality**:
- Loads and labels network traffic data from CSV files (`BenignTraffic`, `DictionaryBruteForce`, `DNS_Spoofing`, `MITM-ArpSpoofing`).
- Maps labels to numerical values: Benign (0), BruteForce (1), DNSSpoofing (2), ARPSpoofing (3).
- Selects relevant features (e.g., `Protocol`, `Flow Duration`, `Packet Length`) and removes non-informative columns (e.g., `Flow ID`, `Src IP`).
- Merges datasets and saves the cleaned output as `cleaned_dataset.csv`.
- Includes commented-out code for merging raw PCAP flow CSVs, if needed.

**Output**: `cleaned_dataset.csv` with selected features and numerical class labels.

### 2. `IOTAttackDetection.py`
**Purpose**: Trains and evaluates machine learning models to detect social engineering attacks in SIoT environments.

**Functionality**:
- Loads the preprocessed dataset (`cleaned_dataset.csv` or `.xlsx`).
- Trains four models: Random Forest, XGBoost, LSTM, and CNN.
- Evaluates models using accuracy, precision, recall, F1-score, and AUC-ROC.
- Generates:
  - **Table 4.1**: Model performance metrics (%).
  - **Table 4.2**: Confusion matrix for XGBoost.
  - **Table 4.3**: Model accuracy by attack type (BruteForce, DNSSpoofing, ARPSpoofing).
  - **Figure 4.1**: ROC curve for XGBoost.
  - **Figure 4.2**: AUC performance comparison of models (including placeholders for Deep Learning and SVM).
- Saves results to CSV files (`model_performance_metrics.csv`, `confusion_matrix_xgboost.csv`, `attack_type_accuracy.csv`) and visualizations as PNG files (`roc_curve_xgboost.png`, `auc_comparison.png`).

## Dependencies
- Python 3.x
- Libraries: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `tensorflow`, `matplotlib`, `seaborn`
- Environment variables are set to mitigate TensorFlow GPU-related errors.

## Usage
1. **Preprocess Data**:
   - Run `IOTM.py` to preprocess raw dataset files into `cleaned_dataset.csv`.
   - Ensure input CSV files (`BenignTraffic.pcap_Flow.csv`, etc.) are available or use the commented-out merging code.

2. **Train and Evaluate Models**:
   - Update the file path in `IOTAttackDetection.py` to point to `cleaned_dataset.csv` (or convert to `.xlsx` if needed).
   - Run `IOTAttackDetection.py` to train models, evaluate performance, and generate results.

3. **Notes**:
   - Verify the dataset file path in `IOTAttackDetection.py`.
   - The dataset should be preprocessed as per the research paper (e.g., Min-Max normalization, missing value imputation).
   - Adjust GPU environment variables if TensorFlow errors occur.

## Installation
```bash
pip install pandas numpy scikit-learn xgboost tensorflow matplotlib seaborn
