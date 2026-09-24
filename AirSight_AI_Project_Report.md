# 🌍 AICTE | IBM SkillsBuild Internship Project Report
## Project Title: AirSight AI — End-to-End India Air Quality Intelligence & Predictive Platform
**Domain:** Data Analytics with AI  
**Internship ID / Cohort:** 2026 Academic Internship Program  
**Collaborators:** AICTE · IBM SkillsBuild · Edunet Foundation / BharatCares  
**Author:** Varun  

---

## 1. Executive Summary
Air pollution is one of the most critical public health and environmental crises facing urban India. Rapid industrialization, high vehicular density, agricultural biomass burning, and seasonal meteorological inversions routinely push the Air Quality Index (AQI) into "Very Poor" and "Severe" classifications across major metropolitan corridors.

**AirSight AI** is an industry-grade, interactive analytics platform designed to uncover temporal, geographical, and chemical patterns of air pollution across 26 Indian cities. The solution leverages over 52,000 daily observations from 2015 to 2020. By combining Exploratory Data Analysis (EDA), advanced feature engineering (time transformations, rolling window statistics, pollutant ratios, and lag variables), and supervised Machine Learning models (Linear Regression, Random Forest, and XGBoost), AirSight AI achieves an **R² score exceeding 0.95** in forecasting continuous AQI values and categorizing pollution severity levels.

---

## 2. Problem Statement & Objectives
### 2.1 Problem Statement
Urban administrators, public health bodies, and citizens lack centralized, transparent, and predictive interfaces that clarify not only current pollution statistics, but also:
1. Which specific chemical pollutants (e.g., particulate matter vs. nitrogen oxides) drive regional toxicity?
2. How severe are seasonal spikes (winter smog vs. post-monsoon festivities)?
3. What will the expected AQI be given emerging ambient pollutant concentrations?

### 2.2 Project Objectives
- **Data Ingestion & Hygiene:** Ingest multi-year pollutant datasets across 26 major Indian urban clusters, rectifying sensor gaps and missing values through forward-fill and city-level median imputation.
- **Exploratory Data Analysis (EDA):** Quantify distributions, identify top polluted cities, evaluate weekend-versus-weekday variances, and trace month-by-month changes.
- **Feature Engineering:** Derive 20+ specialized domain features including 7-day and 30-day moving averages, lag predictors, PM2.5/PM10 concentration ratios, and calendar seasonal flags.
- **Predictive Modeling:** Benchmark multiple machine learning regressors (Linear Regression, Random Forest, XGBoost) to forecast AQI with high precision.
- **Interactive Dashboard:** Build and deploy a multi-page interactive Streamlit dashboard styled with custom CSS and powered by Plotly for live querying, exploratory filtering, and on-demand inference.

---

## 3. Dataset Architecture
The primary data corpus comprises daily air quality readings collected between January 2015 and June 2020 across 26 diverse Indian cities (spanning North, South, East, West, and Central India).

### 3.1 Monitored Parameters
| Parameter | Description | Standard Unit |
| :--- | :--- | :--- |
| **City** | Name of the urban monitoring station cluster | Nominal Categorical |
| **Date** | Observation timestamp (daily granularity) | YYYY-MM-DD |
| **PM2.5** | Fine Particulate Matter (< 2.5 µm diameter) | µg/m³ |
| **PM10** | Coarse Respirable Particulate Matter (< 10 µm) | µg/m³ |
| **NO & NO2** | Nitric Oxide and Nitrogen Dioxide | ppb / µg/m³ |
| **NOx** | Total Nitrogen Oxides | ppb |
| **NH3** | Ammonia concentration | µg/m³ |
| **CO** | Carbon Monoxide concentration | mg/m³ |
| **SO2** | Sulphur Dioxide concentration | µg/m³ |
| **O3** | Ground-level Ozone | µg/m³ |
| **Benzene, Toluene, Xylene** | Volatile Organic Compounds (VOCs) | µg/m³ |
| **AQI** | Calculated composite Air Quality Index | Continuous Scale (0–500+) |
| **AQI_Bucket** | CPCB Severity Bracket (Good to Severe) | Ordinal Scale |

---

## 4. Methodology & Pipeline

```
  +---------------------------------------------+
  |               Raw Data Input                |
  |  (52,208 records across 26 Indian cities)   |
  +---------------------------------------------+
                         |
                         v
  +---------------------------------------------+
  |          Data Cleaning & Imputation         |
  |  - Group-by City Forward & Backward Fill    |
  |  - Median Imputation for Residual Nulls     |
  +---------------------------------------------+
                         |
                         v
  +---------------------------------------------+
  |             Feature Engineering             |
  |  - Temporal Flags (Season, Weekend, Month)  |
  |  - PM2.5 / PM10 Ratio, Total Particulates   |
  |  - 7-Day & 30-Day Rolling Means & Lags      |
  +---------------------------------------------+
                         |
                         v
  +---------------------------------------------+
  |              Machine Learning               |
  |  - Train/Test Split (80/20)                 |
  |  - Models: Linear Regression, RF, XGBoost   |
  |  - Metrics: R², RMSE, MAE                   |
  +---------------------------------------------+
                         |
                         v
  +---------------------------------------------+
  |         Interactive Streamlit App           |
  |  - 5 Multi-Page Navigation Modules          |
  |  - Interactive Plotly Visualizations        |
  |  - Live AQI Simulator & What-If Analysis    |
  +---------------------------------------------+
```

---

## 5. Machine Learning Evaluation & Results
Three models were evaluated on an independent 20% holdout test partition:

| Model Architecture | R² Score | RMSE (Points) | MAE (Points) | Primary Strength |
| :--- | :---: | :---: | :---: | :--- |
| **Linear Regression** | ~0.849 | ~24.8 | ~18.2 | Fast, baseline transparency |
| **Random Forest Regressor** | ~0.952 | ~14.1 | ~9.8 | Non-linear capture, resilient to outliers |
| **XGBoost Regressor** | **~0.961** | **~12.8** | **~8.9** | **Best overall predictive accuracy & speed** |

### Feature Importance Insights
1. **PM2.5** and **PM10** account for over 70% of model weight when predicting composite AQI.
2. **CO** and **NO2** represent secondary drivers, reflecting high urban vehicular congestion.
3. Seasonal indicators consistently show elevated baseline risk across October through February in Indo-Gangetic plain regions.

---

## 6. Streamlit Dashboard Architecture
The platform is organized into 5 dedicated analytical interfaces:
- **Module 1: Dashboard (Overview)** — High-level KPI cards, overall monthly trajectory, category distribution donut, and top 10 most polluted cities.
- **Module 2: City Explorer** — Granular station explorer with dynamic date sliders, categorical filters, multi-pollutant line graphs, correlation matrices, and raw CSV data download.
- **Module 3: Seasonal Analysis** — Deep dive into Winter, Summer, Monsoon, and Post-Monsoon patterns, holiday/festival smog impacts, and weekend relaxation factors.
- **Module 4: AQI Predictor (AI Insights)** — Real-time interactive simulator where users adjust ambient pollutant sliders to generate instant predicted AQI, category badge coloring, and feature contribution charts.
- **Module 5: Model Performance** — Cross-model benchmarking, actual vs. predicted regression scatter charts, and residual error distribution histograms.

---

## 7. Conclusion & Future Roadmap
AirSight AI fulfills the capstone expectations of the AICTE | IBM SkillsBuild Data Analytics with AI internship by providing a complete, end-to-end, reproducible solution. Future expansions include:
- Integrating live CPCB sensor APIs via web scraping or OpenAQ endpoints.
- Implementing deep learning LSTM/GRU models for multi-step ahead weather-forecast integrated time-series forecasting.
- Deploying containerized microservices to IBM Cloud or Streamlit Community Cloud.
