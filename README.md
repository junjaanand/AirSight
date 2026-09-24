<div align="center">

<h1>🌍 AirSight AI — India Air Quality Intelligence Platform</h1>

<p>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit"/>
  <img src="https://img.shields.io/badge/Plotly-5.18%2B-3F4F75?style=for-the-badge&logo=plotly&logoColor=white" alt="Plotly"/>
  <img src="https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-Learn"/>
  <img src="https://img.shields.io/badge/XGBoost-1.7%2B-189AB4?style=for-the-badge" alt="XGBoost"/>
</p>

<p>
  <img src="https://img.shields.io/badge/IBM%20SkillsBuild-Internship%202026-054ADA?style=for-the-badge&logo=ibm&logoColor=white" alt="IBM SkillsBuild"/>
  <img src="https://img.shields.io/badge/AICTE-Approved%20Programme-006400?style=for-the-badge" alt="AICTE"/>
  <img src="https://img.shields.io/badge/BharatCares-Collaboration-FF6B35?style=for-the-badge" alt="BharatCares"/>
</p>

<p><strong>Author: Varun &nbsp;·&nbsp; AICTE | IBM SkillsBuild — Data Analytics with AI Internship 2026</strong></p>

</div>

---

## 📌 Overview

**AirSight AI** is an end-to-end data analytics platform that analyzes air quality across **26 Indian cities** using **52,000+ daily pollution records** (2015–2020). The platform combines comprehensive EDA, advanced feature engineering, machine learning models, and an interactive multi-page Streamlit dashboard to help environmental analysts monitor pollution trends and predict AQI.

---

## 🚑 Problem Statement

Air pollution is one of India's most pressing environmental challenges. With AQI levels frequently crossing hazardous thresholds in major cities, there is a critical need for data-driven tools that can:
- **Monitor** pollution trends across cities and seasons
- **Identify** key pollutants driving poor air quality
- **Predict** AQI levels from pollutant concentrations
- **Enable** proactive decision-making for public health

---

## 📂 Project Structure

```
AirQuality_Intelligence/
├── data/
│   ├── city_day.csv                        ← Raw dataset (52K+ records)
│   └── city_day_engineered.csv             ← Feature-engineered dataset
├── notebooks/
│   ├── EDA_FeatureEngineering.ipynb         ← Jupyter EDA notebook
│   └── EDA_FeatureEngineering.py           ← Python script version
├── app/
│   └── AirQuality_Dashboard.py            ← Multi-page Streamlit dashboard
├── models/
│   ├── rf_model.pkl                        ← Random Forest model
│   └── xgb_model.pkl                       ← XGBoost model
├── screenshots/
│   ├── dashboard.png
│   ├── city_explorer.png
│   ├── seasonal.png
│   ├── ai_predictor.png
│   └── model_performance.png
├── requirements.txt
├── README.md                               ← This file
└── .gitignore
```

---

## 📊 Dataset

| Attribute | Value |
|-----------|-------|
| **Source** | India Air Quality Data (Kaggle) |
| **Records** | 52,208 daily readings |
| **Cities** | 26 Indian cities |
| **Pollutants** | PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, Benzene, Toluene, Xylene |
| **Target** | AQI (continuous) & AQI_Bucket (categorical) |
| **Date Range** | 2015–2020 |

---

## 🔧 Feature Engineering (20 New Features)

| Category | Features Created |
|----------|-----------------|
| **Time-based** | Year, Month, DayOfWeek, DayOfYear, IsWeekend, Quarter |
| **Seasonal** | Season (Winter/Summer/Monsoon/Post-Monsoon) |
| **Pollutant Ratios** | PM25_PM10_Ratio, NO2_NO_Ratio, PM_Total |
| **Rolling Averages** | 7-day & 30-day rolling for AQI, PM2.5, PM10 |
| **Lag Features** | AQI_Lag1, AQI_Lag7 |
| **Change Features** | AQI_Change, AQI_Change_Pct |
| **City Encoding** | City_Avg_AQI (target encoding) |

---

## 🤖 ML Models

| Model | Task | R² | RMSE |
|-------|------|----|------|
| Linear Regression | AQI Prediction | ~0.85 | ~25 |
| Random Forest | AQI Prediction | ~0.95 | ~15 |
| XGBoost | AQI Prediction | ~0.96 | ~13 |

---

## 📱 Dashboard Pages

| Page | Description |
|------|-------------|
| 📊 **Dashboard** | KPI cards, AQI trends, top polluted cities, category distribution |
| 🔍 **City Explorer** | Drill-down analysis for any city with filters |
| 🌡️ **Seasonal Analysis** | Season-wise, month-wise, weekend vs weekday comparisons |
| 🤖 **AQI Predictor** | Input pollutant levels → get predicted AQI with AI |
| 📈 **Model Performance** | Side-by-side comparison of 3 ML models |

---

## 🚀 How to Run

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/AirQuality_Intelligence.git
cd AirQuality_Intelligence

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the Streamlit dashboard
cd app
streamlit run AirQuality_Dashboard.py

# 4. Open in browser
# Navigate to http://localhost:8501
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.10+ |
| Data Processing | Pandas, NumPy |
| Visualization | Plotly (interactive), Seaborn, Matplotlib |
| Machine Learning | Scikit-Learn, XGBoost |
| Dashboard | Streamlit (multi-page, custom CSS) |
| Serialization | Joblib |

---

## 📝 Key Findings

1. 🥶 **Winter is the worst season** — AQI peaks in Nov-Feb (temperature inversion, stubble burning)
2. 🏭 **PM2.5 & PM10 are the top AQI drivers** — highest correlation with overall AQI
3. 📅 **Minimal weekday-weekend difference** — persistent industrial + vehicular pollution
4. 📈 **Feature engineering boosts model R² from 0.85 to 0.96** — rolling averages and lag features critical
5. 🌧️ **Monsoon provides natural relief** — AQI drops significantly during June-September

---

## 📄 License

This project is for educational purposes as part of the AICTE | IBM SkillsBuild Internship Programme 2026.

---

<div align="center">
  <p><strong>Built with ❤️ for a cleaner India 🇮🇳</strong></p>
</div>
