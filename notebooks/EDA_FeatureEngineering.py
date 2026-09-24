# %% [markdown]
# # 🌍 AirSight AI — Exploratory Data Analysis & Feature Engineering
# ### India Air Quality Intelligence Platform
# **AICTE | IBM SkillsBuild — Data Analytics with AI Internship 2026**
# 
# ---
# 
# ## Table of Contents
# 1. Data Loading & Initial Inspection
# 2. Data Cleaning & Missing Value Treatment
# 3. Exploratory Data Analysis (EDA)
# 4. Feature Engineering
# 5. Correlation & Statistical Analysis
# 6. ML-Ready Dataset Preparation

# %% [markdown]
# ## 1. 📦 Data Loading & Initial Inspection

# %%
import os
import sys
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set default renderer to none for non-interactive terminal execution to prevent browser popups
if not hasattr(sys, 'ps1'):
    pio.renderers.default = 'browser'

pd.set_option('display.max_columns', 25)
pd.set_option('display.float_format', '{:.2f}'.format)

print("[OK] Modern Data Analytics & Plotly libraries loaded successfully!")

# %%
# Load the dataset
data_path = '../data/city_day.csv' if os.path.exists('../data/city_day.csv') else 'd:/project/AirQuality_Intelligence/data/city_day.csv'
df = pd.read_csv(data_path)

print(f"[DATA] Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
print(f"[DATA] Cities: {df['City'].nunique()}")
print(f"[DATA] Date Range: {df['Date'].min()} to {df['Date'].max()}")
print("\n" + "="*60)
print("Column Details:")
print("="*60)
print(df.info())

# %%
# First 5 rows
df.head()

# %%
# Statistical Summary
df.describe().round(2)

# %% [markdown]
# ## 2. 🧹 Data Cleaning & Missing Value Treatment

# %%
# Missing value analysis
missing = df.isnull().sum()
missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
missing_df = pd.DataFrame({
    'Column': missing.index,
    'Missing Count': missing.values,
    'Missing %': missing_pct.values
}).sort_values('Missing %', ascending=False)

print("\nMissing Value Summary:")
print("=" * 50)
print(missing_df[missing_df['Missing Count'] > 0].to_string(index=False))

# %%
# Visualize missing values with Plotly
fig_missing = px.bar(
    missing_df[missing_df['Missing Count'] > 0],
    x='Column', y='Missing %',
    text='Missing %',
    title='Missing Values by Column (%)',
    color='Missing %',
    color_continuous_scale='Reds',
    template='plotly_white'
)
fig_missing.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
fig_missing.update_layout(height=450)

# %%
# Parse dates
df['Date'] = pd.to_datetime(df['Date'])

# Handle missing values: Forward fill & backward fill within each city, then median fill
numeric_cols = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 
                'Benzene', 'Toluene', 'Xylene', 'AQI']

df = df.sort_values(['City', 'Date']).reset_index(drop=True)

for col in numeric_cols:
    df[col] = df.groupby('City')[col].transform(lambda x: x.ffill().bfill())
    df[col] = df.groupby('City')[col].transform(lambda x: x.fillna(x.median()))
    df[col] = df[col].fillna(df[col].median())

# Fill AQI_Bucket based on CPCB AQI standards
def get_aqi_bucket(aqi):
    if pd.isna(aqi): return 'Unknown'
    if aqi <= 50: return 'Good'
    elif aqi <= 100: return 'Satisfactory'
    elif aqi <= 200: return 'Moderate'
    elif aqi <= 300: return 'Poor'
    elif aqi <= 400: return 'Very Poor'
    else: return 'Severe'

df['AQI_Bucket'] = df['AQI'].apply(get_aqi_bucket)

print(f"\n[OK] Missing values after treatment:\n{df[numeric_cols].isnull().sum()}")
print(f"\n[OK] Clean dataset ready: {len(df):,} records")

# %% [markdown]
# ## 3. 📊 Exploratory Data Analysis (EDA)

# %% [markdown]
# ### 3.1 AQI Distribution Overview

# %%
# AQI Distribution Histogram with Plotly
fig_dist = px.histogram(
    df, x='AQI', nbins=60,
    title='Overall AQI Distribution Across All Cities',
    color_discrete_sequence=['#1e5799'],
    template='plotly_white',
    marginal='box'
)
fig_dist.add_vline(x=df['AQI'].mean(), line_dash='dash', line_color='red', annotation_text=f"Mean: {df['AQI'].mean():.1f}")
fig_dist.add_vline(x=df['AQI'].median(), line_dash='dot', line_color='orange', annotation_text=f"Median: {df['AQI'].median():.1f}")
fig_dist.update_layout(height=450)

# %%
# AQI Category Distribution Donut Chart
bucket_counts = df['AQI_Bucket'].value_counts().reset_index()
bucket_counts.columns = ['Category', 'Count']
color_map = {
    'Good': '#00B050', 'Satisfactory': '#92D050', 'Moderate': '#FFFF00',
    'Poor': '#FF9900', 'Very Poor': '#FF0000', 'Severe': '#C00000'
}

fig_donut = px.pie(
    bucket_counts, names='Category', values='Count',
    title='Proportion of Days by AQI Category',
    hole=0.45,
    color='Category',
    color_discrete_map=color_map,
    template='plotly_white'
)
fig_donut.update_layout(height=450)

# %% [markdown]
# ### 3.2 City-wise Analysis

# %%
# Top 10 Most Polluted Cities
city_avg = df.groupby('City')['AQI'].mean().sort_values(ascending=False).reset_index()

fig_top_cities = px.bar(
    city_avg.head(10),
    x='AQI', y='City', orientation='h',
    title='Top 10 Most Polluted Cities in India (Average AQI)',
    color='AQI', color_continuous_scale='Reds',
    template='plotly_white', text_auto='.1f'
)
fig_top_cities.update_layout(yaxis={'categoryorder': 'total ascending'}, height=450)

# %%
# City-wise AQI Box Plot
fig_box = px.box(
    df, x='City', y='AQI', color='City',
    title='AQI Distribution Spread Across 26 Indian Cities',
    template='plotly_white'
)
fig_box.update_layout(showlegend=False, height=520, xaxis_tickangle=-45)

# %% [markdown]
# ### 3.3 Temporal Analysis

# %%
# Monthly AQI Trend
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month

monthly_avg = df.groupby(df['Date'].dt.to_period('M'))['AQI'].mean().reset_index()
monthly_avg['Date'] = monthly_avg['Date'].dt.to_timestamp()

fig_trend = px.line(
    monthly_avg, x='Date', y='AQI',
    title='Multi-Year Monthly Average AQI Trend (2015-2020)',
    template='plotly_white', labels={'Date': 'Timeline', 'AQI': 'Mean AQI'}
)
fig_trend.update_traces(line=dict(width=3, color='#e74c3c'))
fig_trend.update_layout(height=420)

# %%
# Monthly Heatmap (City x Month)
city_month_pivot = df.pivot_table(index='City', columns='Month', values='AQI', aggfunc='mean')

fig_heat = px.imshow(
    city_month_pivot,
    title='Seasonal City Pollution Heatmap (Cities x Months)',
    labels=dict(x="Month", y="City", color="AQI"),
    color_continuous_scale='YlOrRd', aspect="auto"
)
fig_heat.update_layout(height=650)

# %% [markdown]
# ### 3.4 Pollutant Analysis & Correlations

# %%
# Top Pollutant Correlation with AQI
pollutant_cols = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 
                  'Benzene', 'Toluene', 'Xylene']
aqi_corr = df[pollutant_cols + ['AQI']].corr()['AQI'].drop('AQI').reset_index()
aqi_corr.columns = ['Pollutant', 'Correlation']
aqi_corr = aqi_corr.sort_values('Correlation', ascending=True)

fig_corr_bar = px.bar(
    aqi_corr, x='Correlation', y='Pollutant', orientation='h',
    title='Pollutant Correlation Strength with Overall AQI',
    color='Correlation', color_continuous_scale='Bluered',
    template='plotly_white', text_auto='.3f'
)
fig_corr_bar.update_layout(height=450)

# %%
# Full Correlation Matrix
corr_mat = df[pollutant_cols + ['AQI']].corr()

fig_mat = px.imshow(
    corr_mat, text_auto='.2f', aspect='auto',
    title='Full Pollutant Cross-Correlation Matrix',
    color_continuous_scale='RdBu_r', zmin=-1, zmax=1
)
fig_mat.update_layout(height=600)

# %% [markdown]
# ## 4. 🔧 Feature Engineering

# %%
print("\n[INFO] Engineering domain features...")

# 4.1 Time features
df['DayOfWeek'] = df['Date'].dt.dayofweek
df['DayOfYear'] = df['Date'].dt.dayofyear
df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
df['Quarter'] = df['Date'].dt.quarter

# 4.2 Season feature
def get_season(month):
    if month in [12, 1, 2]: return 'Winter'
    elif month in [3, 4, 5]: return 'Summer'
    elif month in [6, 7, 8, 9]: return 'Monsoon'
    else: return 'Post-Monsoon'

df['Season'] = df['Month'].apply(get_season)

# 4.3 Pollutant ratios & totals
df['PM25_PM10_Ratio'] = np.where(df['PM10'] > 0, df['PM2.5'] / df['PM10'], 0)
df['NO2_NO_Ratio'] = np.where(df['NO'] > 0, df['NO2'] / df['NO'], 0)
df['PM_Total'] = df['PM2.5'] + df['PM10']

# 4.4 Rolling temporal windows
for col in ['AQI', 'PM2.5', 'PM10']:
    df[f'{col}_Rolling7'] = df.groupby('City')[col].transform(lambda x: x.rolling(7, min_periods=1).mean())
    df[f'{col}_Rolling30'] = df.groupby('City')[col].transform(lambda x: x.rolling(30, min_periods=1).mean())

# 4.5 Lag features & day-over-day changes
df['AQI_Lag1'] = df.groupby('City')['AQI'].shift(1).bfill()
df['AQI_Lag7'] = df.groupby('City')['AQI'].shift(7).bfill()
df['AQI_Change'] = df.groupby('City')['AQI'].diff().fillna(0)

# 4.6 Target Encoding for City Baseline
city_mean_aqi = df.groupby('City')['AQI'].mean()
df['City_Avg_AQI'] = df['City'].map(city_mean_aqi)

# Impute any edge-case boundary NaNs
df = df.ffill().bfill()

print("[OK] Feature Engineering Complete!")
print(f"[DATA] Final Dataset Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

# %%
# Visualizing Seasonal Patterns with Plotly
fig_season = px.box(
    df, x='Season', y='AQI', color='Season',
    title='Seasonal Variations in Air Quality (Winter Smog vs Monsoon Relief)',
    category_orders={'Season': ['Winter', 'Summer', 'Monsoon', 'Post-Monsoon']},
    template='plotly_white',
    color_discrete_sequence=['#1e5799', '#f39c12', '#27ae60', '#e74c3c']
)
fig_season.update_layout(height=450)

# %% [markdown]
# ## 5. 💾 ML-Ready Dataset Export

# %%
output_path = '../data/city_day_engineered.csv' if os.path.exists('../data') else 'd:/project/AirQuality_Intelligence/data/city_day_engineered.csv'
df.to_csv(output_path, index=False)
print(f"[OK] Clean, feature-engineered dataset successfully saved to: {output_path}")
print(f"[INFO] Records: {len(df):,} | Features: {df.shape[1]}")
print("[OK] Ready for Machine Learning Modeling and Dashboard Deployment!")
