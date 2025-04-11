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


---
---


## Mobile application

1. Install dependencies

   ```bash
   npm install
   ```

2. Start the app

   ```bash
    npm run start
   ```
---
---


   # 🏥Server (FastAPI + MongoDB)

> A secure, role-based digital healthcare platform for managing patients, doctors, and reports using FastAPI, MongoDB, and JWT authentication.

---

## 🚀 Features

- 🔐 **JWT-based authentication** for Admins, Doctors, and Patients
- 👩‍⚕️ Role-specific Dashboards:
  - Admin: Manage database items
  - Doctor: Manage patients, assign care, edit dosages
  - Patient: View reports, submit INR, log medication
- 📁 INR Report Uploads (with file storage)
- 📈 Dynamic Charting for INR reports & dosage schedules
- 🧾 Daily report submission (side effects, medication, lifestyle)
- 🧠 Intelligent doctor reassignment and missed dose tracking
- 🧰 Explainable AI utilities integrated for clinical trust (via `utils.py`)

---

## 🛠️ Stack

- **FastAPI** - Web framework
- **MongoDB Atlas** - NoSQL cloud database
- **JWT** - Secure token-based authentication
- **Pydantic** - Data validation and serialization
- **Uvicorn** - ASGI server
- **Jinja2** - HTML templating (optional)
- **Static Files** - For patient reports and uploads

---

## 📁 Project Structure

```
📦app/
 ┣ 📄 main.py               # Main FastAPI server
 ┣ 📄 models.py             # Pydantic models
 ┣ 📄 utils.py              # Utility functions
 ┣ 📂 templates/            # Jinja2 templates (optional UI)
 ┣ 📂 static/patient_docs/  # Uploaded INR documents
 ┣ 📂 static/               # Static files
```

---

## 🔑 Roles & Access

| Role     | Permissions                                 |
|----------|---------------------------------------------|
| Admin    | Full DB access (CRUD on `items`)            |
| Doctor   | Manage assigned patients, update dosages    |
| Patient  | Upload reports, log doses, submit updates   |

---

## 🧪 API Endpoints Overview

| Method | Endpoint                    | Role     | Description                           |
|--------|-----------------------------|----------|---------------------------------------|
| POST   | `/login`                    | All      | Login and get JWT token               |
| GET    | `/admin`                    | Admin    | Admin dashboard                       |
| GET    | `/doctor`                   | Doctor   | Doctor dashboard                      |
| POST   | `/doctor/add-patient`      | Doctor   | Add new patient                       |
| GET    | `/patient`                  | Patient  | Patient home                          |
| POST   | `/patient/update-inr`       | Patient  | Upload INR test report                |
| POST   | `/doctor/edit-dosage/{id}`  | Doctor   | Edit patient's dosage schedule        |
| POST   | `/doctor/reassign/{id}`     | Doctor   | Reassign caretaker/doctor             |

---

## 🔐 Authentication

- JWT generated after login with role-specific payload
- Tokens used via `Authorization: Bearer <token>`
- Middleware ensures endpoint protection by role

---

## 🧰 Utilities

- 🔎 `calculate_monthly_inr_average`
- ⏰ `get_medication_dates`
- 🧾 `find_missed_doses`

Located in `utils.py`

---

## 🧪 Running the Server

```bash
uvicorn main:app --reload --port 4502
```

---

## 📝 Notes

- Replace `SECRET_KEY` with a secure random string
- Use environment variables for credentials and DB URIs
- File uploads stored in `static/patient_docs/`
- Jinja2 templates used for HTML rendering (optional)

---

