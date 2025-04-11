# 💊 Warfarin Dosage Estimation using XGBoost and Explainable AI

> A pharmacogenomic machine learning approach for personalized anticoagulant therapy using XGBoost, evaluated with Explainable AI tools.

## 🧬 Overview

Warfarin is a widely used blood thinner with highly variable dose requirements across individuals. Incorrect dosing can lead to severe side effects like internal bleeding or ineffective treatment. This project proposes a **personalized warfarin dosage prediction model** leveraging:

- **XGBoost Regression**
- **Feature Engineering & Transformation**
- **Pharmacogenomic Insights**
- **Explainable AI (XAI)** using **LIME** and **SHAP**

---

## 📌 Project Title

**Estimation of Warfarin Dosage using a Specialized XGBoost-Based Pharmacogenomic Machine Learning Model and Evaluation using XAI**

---

## 👩‍⚕️ Motivation

- Traditional trial-and-error dosing methods are time-consuming and risky.
- Personalized ML models can minimize hospitalization, reduce risks, and improve clinical outcomes.
- Explaining AI decisions helps ensure model acceptance in healthcare settings.

---

## 📂 Dataset

**Source**: International Warfarin Pharmacogenetics Consortium (IWPC)  
**Data Includes**:
- **Demographics**: Age, Gender, Ethnicity, Height, Weight, BMI
- **Clinical**: Target INR, Smoking, Comorbidities
- **Medications**: Aspirin, Amiodarone, Simvastatin
- **Genetics**: CYP2C9, VKORC1 genotypes

---

## 🧠 Methodology

### 🔧 Preprocessing
- KNN Imputation for missing values
- One-hot encoding for categorical variables
- Feature transformations (e.g., polynomial interactions of Height × Weight)

### ⚙️ Model: XGBoost Regressor
- Handles non-linearity and missing data efficiently
- Incorporates feature interactions for improved accuracy
- Hyperparameter tuning with learning rate, max depth, subsample, etc.

### 📈 Evaluation Metrics
- **R² Score**
- **Mean Absolute Error (MAE)**
- **Individual Prediction Percentage (IPP)**

---

## 📊 Explainable AI

To ensure model transparency in clinical use:

- **LIME**: Explains local predictions using interpretable linear models.
- **SHAP**: Provides a global view of feature impact via Shapley values.

### 🧪 Key Influencing Features
| Feature | Effect on Dose |
|--------|----------------|
| CYP2C9 Genotype (*1/*2) | ↑ |
| VKORC1 Genotype (AA) | ↓ |
| BMI (High) | ↓ |
| Age (Older) | ↓ |
| Height & Weight | ↑ |
| Aspirin Use | ↓ |
| Amiodarone Use | ↓ |
| Target INR | ↑ |

---

## 📎 Key Findings

- XGBoost with feature transformation improves dose prediction accuracy significantly.
- XAI tools provide valuable clinical insights.
- The model can aid clinicians in early, personalized dose recommendations.

---

## 👨‍💻 Team

- **Aaditya Rengarajan (21z202)**
- **Akshay Perison Davis (21z205)**
- **Navaneetha Krishnan K S (21z233)**
- **R Vishal (21z240)**
- **Subhasri Shreya S L (21z260)**  
**Guide**: *L. S. Jayashree*

---

## 📚 References

- International Warfarin Pharmacogenetics Consortium (IWPC)
- LIME: https://github.com/marcotcr/lime
- SHAP: https://github.com/slundberg/shap
- XGBoost: https://xgboost.ai/

---

## 📌 Future Work

- Clinical trials for model validation.
- Integration with electronic health records (EHR).
- Extension to other personalized medicine applications.






## Mobile application

1. Install dependencies

   ```bash
   npm install
   ```

2. Start the app

   ```bash
    npm run start
   ```



   
