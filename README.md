# 👥 Employee Attrition & Workforce Intelligence System

**An AI-Powered HR Analytics Dashboard for Employee Attrition Analysis, Risk Prediction and Workforce Insights**

> IBM PBL Virtual Internship — Data Analytics with AI

---

## 📋 Project Overview

This project is a complete, portfolio-level HR Analytics and AI system built with Python and Streamlit. It ingests the IBM HR Analytics Employee Attrition dataset and provides a multi-page interactive dashboard covering exploratory data analysis, machine learning-based attrition prediction, workforce segmentation, model explainability, and evidence-based workforce insights.

The application demonstrates the complete data analytics and machine learning workflow — from raw data loading through preprocessing, EDA, feature engineering, model training and evaluation, interactive prediction, and actionable workforce intelligence.

---

## ❗ Problem Statement

Employee attrition is one of the most costly challenges for organizations. Replacing a single employee can cost 50–200% of their annual salary when accounting for recruiting, onboarding, training, and lost productivity. Traditional HR reporting identifies attrition after it occurs. This project aims to shift that paradigm — using data analytics and machine learning to identify patterns, predict at-risk employees, and provide actionable insights before attrition happens.

---

## 🎯 Objectives

1. Analyze the IBM HR dataset to identify patterns and factors associated with employee attrition
2. Build and evaluate multiple machine learning classifiers for attrition risk prediction
3. Create an interactive employee attrition risk predictor with probability scores
4. Segment the workforce into natural groups using K-Means clustering
5. Provide data-driven, evidence-based workforce intelligence insights
6. Deliver an interpretable, explainable model using feature importance analysis
7. Build a professional, polished Streamlit dashboard suitable for academic and business presentations

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🏢 Executive Overview | KPI cards + workforce composition charts |
| 👥 Workforce Analytics | Satisfaction, compensation, and demographic breakdowns |
| 📉 Attrition Analysis | Multi-dimensional attrition rate analysis |
| 🔬 Correlation & Features | Heatmap, distribution comparison, descriptive stats |
| 🤖 ML Model Performance | 4 models compared with full metrics, confusion matrices, ROC curves |
| 🎯 Employee Risk Prediction | Interactive prediction form with gauge visualization |
| 🗂️ Workforce Segmentation | K-Means clustering with elbow + silhouette analysis |
| 💡 Workforce Intelligence | Auto-generated, evidence-based insights |
| 🔍 Model Explainability | Feature importance (RF, GB) + LR coefficients |
| 🔽 Interactive Filters | Department, Role, Gender, Travel, OverTime, Level, Age range |

---

## 📊 Dataset Description

| Attribute | Value |
|-----------|-------|
| **Source** | IBM HR Analytics Employee Attrition & Performance (Kaggle) |
| **Link** | https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset?resource=download |
| **Rows** | 1,470 employees |
| **Features** | 35 original columns |
| **Target Variable** | `Attrition` (Yes / No) |
| **Attrition Rate** | ~16.1% (237 Yes / 1,233 No) — imbalanced |
| **Missing Values** | None |

**Removed constant/irrelevant columns:** `EmployeeCount` (all=1), `Over18` (all='Y'), `StandardHours` (all=80)

**Key feature categories:**
- **Demographic:** Age, Gender, MaritalStatus, EducationField, Education
- **Job-related:** JobRole, Department, JobLevel, JobInvolvement, OverTime
- **Compensation:** MonthlyIncome, DailyRate, HourlyRate, MonthlyRate, PercentSalaryHike, StockOptionLevel
- **Satisfaction:** JobSatisfaction, EnvironmentSatisfaction, RelationshipSatisfaction, WorkLifeBalance
- **Tenure:** YearsAtCompany, YearsInCurrentRole, YearsSinceLastPromotion, YearsWithCurrManager, TotalWorkingYears
- **Travel/Distance:** BusinessTravel, DistanceFromHome

---

## 🛠️ Technologies Used

| Technology | Version | Purpose |
|------------|---------|---------|
| Python | 3.9+ | Core programming language |
| Streamlit | ≥1.28 | Interactive web dashboard |
| Pandas | ≥1.5 | Data manipulation and analysis |
| NumPy | ≥1.23 | Numerical operations |
| Plotly | ≥5.15 | Interactive visualizations |
| Scikit-learn | ≥1.2 | ML models, preprocessing, evaluation |

---

## 🏗️ Project Architecture / Workflow

```
Raw CSV Data
     │
     ▼
Data Loading (pandas)
     │
     ▼
Data Preprocessing
  • Drop constant columns (EmployeeCount, Over18, StandardHours)
  • Label encode categorical features
  • Encode target: Attrition → 0/1
     │
     ▼
Exploratory Data Analysis
  • KPI metrics
  • Distribution charts
  • Attrition rate breakdowns
  • Correlation heatmap
     │
     ▼
Feature Engineering
  • Age groups, Income bands, Tenure groups
  • Feature selection for ML (30 features)
     │
     ▼
Machine Learning Pipeline
  • StandardScaler → Classifier
  • class_weight='balanced' for imbalance
  • 80/20 stratified train-test split
     │
     ▼
Model Evaluation
  • Accuracy, Precision, Recall, F1, ROC-AUC
  • Confusion matrix, ROC curves
     │
     ▼
Workforce Segmentation
  • K-Means (k=2..6), Elbow + Silhouette
  • PCA 2D visualization
     │
     ▼
Prediction & Insights
  • Interactive risk prediction UI
  • Feature importance / explainability
  • Auto-generated evidence-based insights
```

---

## 🤖 Machine Learning Models

All models use `class_weight='balanced'` to handle the ~84/16% class imbalance. Training uses a `Pipeline` (StandardScaler → Classifier) to prevent data leakage.

| Model | Notes |
|-------|-------|
| **Logistic Regression** | Interpretable linear baseline; coefficients indicate directional associations |
| **Decision Tree** | Rule-based; max_depth=6 to limit overfitting |
| **Random Forest** | 200 tree ensemble; provides reliable feature importances |
| **Gradient Boosting** | Sequential ensemble; 150 estimators, learning_rate=0.08 |

---

## 📈 Model Evaluation Results (Actual — from dataset)

> Evaluated on stratified 20% hold-out test set (294 samples).

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.7517 | 0.3673 | **0.7660** | **0.4966** | **0.8074** |
| Decision Tree | 0.7585 | 0.3421 | 0.5532 | 0.4228 | 0.6381 |
| Random Forest | 0.8299 | 0.4211 | 0.1702 | 0.2424 | 0.7812 |
| Gradient Boosting | **0.8401** | **0.5000** | 0.1489 | 0.2295 | 0.7980 |

**Selected model: Logistic Regression** — highest F1-Score (0.4966) and ROC-AUC (0.8074).

> For attrition risk identification, Recall (catching actual at-risk employees) and F1-Score are prioritized over raw accuracy, because a model predicting "No attrition" for all employees would achieve ~84% accuracy while being useless for HR risk management.

---

## 🗂️ Workforce Segmentation

K-Means clustering applied to 10 workforce features:
`Age, MonthlyIncome, TotalWorkingYears, YearsAtCompany, JobSatisfaction, WorkLifeBalance, EnvironmentSatisfaction, JobLevel, DistanceFromHome, NumCompaniesWorked`

- Optimal k determined via Silhouette score comparison across k=2..6
- PCA used for 2D visualization only (not for clustering)
- Cluster profiles describe each segment by observed data characteristics

---

## ⚙️ How to Install

```bash
# Clone or download the project
cd "Employee Attrition & Workforce Intelligence System"

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 How to Run

```bash
streamlit run app.py
```

The application will open at `http://localhost:8501` in your browser.

**Requirements:** The dataset file `WA_Fn-UseC_-HR-Employee-Attrition.csv` must be in the same directory as `app.py`.

---

## 📁 Project Structure

```
Employee Attrition & Workforce Intelligence System/
├── app.py                                    # Complete Streamlit application (single file)
├── WA_Fn-UseC_-HR-Employee-Attrition.csv    # IBM HR Analytics dataset
├── requirements.txt                          # Python package dependencies
├── README.md                                 # This file
└── project_report.docx                       # Academic project report
```

---

## 💡 Key Insights (Observed from Dataset)

> All insights are calculated from the actual dataset. These are observed associations — not proven causal relationships.

1. **OverTime** — Employees working overtime show a significantly higher observed attrition rate (~30.5%) compared to those who do not (~10.4%)
2. **Business Travel** — Frequent travelers show higher observed attrition rates than non-travelers
3. **Job Role** — Sales Representative and Laboratory Technician roles show comparatively higher attrition rates in this dataset
4. **Income** — Employees who left had lower average monthly income compared to those who stayed
5. **Tenure** — Employees with shorter company tenure show higher attrition rates; the first few years appear to be the highest-risk period
6. **Satisfaction** — Lower job satisfaction and environment satisfaction scores are associated with higher attrition rates
7. **Stock Options** — Employees with no stock options (level 0) show higher attrition rates
8. **Job Level** — Entry-level (level 1) employees show higher attrition rates than senior employees

---

## ⚠️ Limitations

- The dataset is a synthetic/anonymized IBM dataset created for educational purposes — real-world HR data would have different characteristics
- Dataset size (1,470 records) limits model complexity and generalizability
- Class imbalance (~16% attrition) makes prediction challenging even with balancing techniques
- Models capture associations present in this historical dataset; they cannot predict future behavior with certainty
- Feature importance indicates model associations, not causal drivers of attrition
- The application does not perform fairness/bias auditing across demographic groups

---

## 🔮 Future Enhancements

1. SHAP values for per-prediction explainability
2. Time-series analysis if timestamped data becomes available
3. Hyperparameter optimization (GridSearchCV / Optuna)
4. HR-specific threshold tuning for Precision/Recall trade-off
5. Batch prediction: upload CSV of employees and get risk scores for all
6. Demographic fairness / bias analysis
7. Integration with real HRIS systems via API

---

## ⚖️ Ethical Considerations

- **Predictions are probabilistic, not deterministic** — model outputs represent statistical risk estimates, not certainties
- **Predictions should support HR judgment, not replace it** — human decisions require context the model cannot fully capture
- **Privacy:** Employee data should be handled with appropriate data governance and anonymization
- **Bias:** Models trained on historical data may reflect historical biases; regular bias audits are essential before real-world deployment
- **Fairness:** Predictions should be evaluated for disparate impact across gender, age, ethnicity, and other protected characteristics
- **Transparency:** Employees should be informed if predictive models are used in HR decisions affecting them

---

## 📚 Dataset Source

**IBM HR Analytics Employee Attrition & Performance**
- Kaggle: https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset
- Created by IBM data scientists as a fictional dataset for educational purposes

---

## 🏛️ Internship Context

This project was developed as part of the **IBM PBL Virtual Internship — Data Analytics with AI** program.

---

*Built with Python · Streamlit · Plotly · Scikit-learn*
