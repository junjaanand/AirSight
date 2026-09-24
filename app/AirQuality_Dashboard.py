import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
import warnings

# Try importing RandomForest with fallback if DLL blocked
try:
    from sklearn.ensemble import RandomForestRegressor
    HAS_RF = True
except Exception:
    HAS_RF = False

warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(page_title='AirSight AI', page_icon='🌍', layout='wide')

# Add CSS
def add_css():
    st.markdown("""
        <style>
        [data-testid="stSidebar"] {
            background-image: linear-gradient(180deg, #0a2342 0%, #1e5799 100%);
            color: white;
        }
        .st-emotion-cache-16txtl3 h1, .st-emotion-cache-16txtl3 h2, .st-emotion-cache-16txtl3 h3, .st-emotion-cache-16txtl3 span {
            color: white !important;
        }
        .kpi-card {
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
            border-left: 5px solid #1e5799;
            margin-bottom: 20px;
        }
        .kpi-title {
            color: #888;
            font-size: 14px;
            margin-bottom: 5px;
        }
        .kpi-value {
            color: #0a2342;
            font-size: 24px;
            font-weight: bold;
        }
        .section-header {
            background: linear-gradient(90deg, #1e5799 0%, #0a2342 100%);
            padding: 10px 20px;
            color: white;
            border-radius: 5px;
            margin: 20px 0px;
        }
        .footer {
            text-align: center;
            padding: 20px;
            color: #888;
            font-size: 14px;
            margin-top: 50px;
            border-top: 1px solid #eee;
        }
        </style>
    """, unsafe_allow_html=True)

add_css()

# Data Loading
@st.cache_data
def load_and_preprocess_data():
    try:
        data_path = '../data/city_day.csv'
        if not os.path.exists(data_path):
            # Create dummy data for demonstration if file doesn't exist
            dates = pd.date_range('2015-01-01', '2020-07-01')
            cities = ['Delhi', 'Mumbai', 'Bengaluru', 'Chennai', 'Hyderabad']
            np.random.seed(42)
            dummy_data = []
            for city in cities:
                city_data = pd.DataFrame({
                    'City': city,
                    'Date': dates,
                    'PM2.5': np.random.uniform(10, 200, len(dates)),
                    'PM10': np.random.uniform(20, 300, len(dates)),
                    'NO': np.random.uniform(5, 50, len(dates)),
                    'NO2': np.random.uniform(10, 80, len(dates)),
                    'NOx': np.random.uniform(10, 100, len(dates)),
                    'NH3': np.random.uniform(5, 40, len(dates)),
                    'CO': np.random.uniform(0.1, 2.5, len(dates)),
                    'SO2': np.random.uniform(2, 25, len(dates)),
                    'O3': np.random.uniform(10, 60, len(dates)),
                    'Benzene': np.random.uniform(0.1, 5, len(dates)),
                    'Toluene': np.random.uniform(0.5, 15, len(dates)),
                    'Xylene': np.random.uniform(0.1, 3, len(dates)),
                    'AQI': np.random.uniform(30, 450, len(dates))
                })
                dummy_data.append(city_data)
            df = pd.concat(dummy_data, ignore_index=True)
            # Assign categories
            def get_bucket(aqi):
                if aqi <= 50: return 'Good'
                elif aqi <= 100: return 'Satisfactory'
                elif aqi <= 200: return 'Moderate'
                elif aqi <= 300: return 'Poor'
                elif aqi <= 400: return 'Very Poor'
                else: return 'Severe'
            df['AQI_Bucket'] = df['AQI'].apply(get_bucket)
            
            # create dir if needed
            os.makedirs('../data', exist_ok=True)
            df.to_csv(data_path, index=False)
        else:
            df = pd.read_csv(data_path)
        
        # Preprocessing
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Sort by city and date
        df = df.sort_values(['City', 'Date']).reset_index(drop=True)
        
        # Missing value handling (ffill within cities, then median)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df.groupby('City')[numeric_cols].transform(lambda x: x.ffill().bfill())
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        
        # Feature Engineering
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        
        def get_season(month):
            if month in [12, 1, 2]: return 'Winter'
            elif month in [3, 4, 5]: return 'Summer'
            elif month in [6, 7, 8, 9]: return 'Monsoon'
            else: return 'Post-Monsoon'
            
        df['Season'] = df['Month'].apply(get_season)
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
        
        # PM2.5 to PM10 Ratio
        df['PM25_PM10_Ratio'] = np.where(df['PM10'] > 0, df['PM2.5'] / df['PM10'], 0)
        
        # Rolling 7-day AQI
        df['Rolling_7day_AQI'] = df.groupby('City')['AQI'].transform(lambda x: x.rolling(window=7, min_periods=1).mean())
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

df = load_and_preprocess_data()

# Model Loading/Training
@st.cache_resource
def get_models(df):
    model_dir = '../models/'
    os.makedirs(model_dir, exist_ok=True)
    
    features = ['PM2.5', 'PM10', 'NO2', 'SO2', 'O3', 'CO']
    target = 'AQI'
    
    # Drop rows with missing values in these columns if any
    model_data = df.dropna(subset=features + [target])
    
    X = model_data[features]
    y = model_data[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(),
        'XGBoost': XGBRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    }
    if HAS_RF:
        models['Random Forest'] = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    
    trained_models = {}
    metrics = {}
    
    for name, model in models.items():
        model_path = os.path.join(model_dir, f"{name.replace(' ', '_')}.joblib")
        if os.path.exists(model_path):
            trained_models[name] = joblib.load(model_path)
            y_pred = trained_models[name].predict(X_test)
        else:
            model.fit(X_train, y_train)
            trained_models[name] = model
            joblib.dump(model, model_path)
            y_pred = model.predict(X_test)
            
        metrics[name] = {
            'R2': r2_score(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'MAE': mean_absolute_error(y_test, y_pred),
            'y_test': y_test,
            'y_pred': y_pred
        }
        
    return trained_models, metrics, X_train, y_train

if not df.empty:
    models_dict, model_metrics, X_train, y_train = get_models(df)

# Sidebar Navigation
st.sidebar.markdown("<h2>🌍 AirSight AI</h2>", unsafe_allow_html=True)
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigation", 
    ["📊 Dashboard (Overview)", 
     "🔍 City Explorer", 
     "🌡️ Seasonal Analysis", 
     "🤖 AQI Predictor (AI Insights)", 
     "📈 Model Performance"])

# Helper function for KPI
def kpi_card(title, value):
    return f"""
    <div class="kpi-card">
        <div class="kpi-title">{title}</div>
        <div class="kpi-value">{value}</div>
    </div>
    """

# Page implementations
if df.empty:
    st.warning("No data available. Please check the data path.")
else:
    if page == "📊 Dashboard (Overview)":
        st.markdown('<div class="section-header">Dashboard Overview</div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.markdown(kpi_card("Total Records", f"{len(df):,}"), unsafe_allow_html=True)
        with col2:
            st.markdown(kpi_card("Cities Covered", f"{df['City'].nunique()}"), unsafe_allow_html=True)
        with col3:
            st.markdown(kpi_card("Avg AQI", f"{df['AQI'].mean():.1f}"), unsafe_allow_html=True)
        with col4:
            most_polluted = df.groupby('City')['AQI'].mean().idxmax()
            st.markdown(kpi_card("Most Polluted", most_polluted), unsafe_allow_html=True)
        with col5:
            cleanest = df.groupby('City')['AQI'].mean().idxmin()
            st.markdown(kpi_card("Cleanest City", cleanest), unsafe_allow_html=True)
            
        colA, colB = st.columns(2)
        
        with colA:
            # AQI trend line chart
            monthly_aqi = df.groupby(df['Date'].dt.to_period('M'))['AQI'].mean().reset_index()
            monthly_aqi['Date'] = monthly_aqi['Date'].dt.to_timestamp()
            fig1 = px.line(monthly_aqi, x='Date', y='AQI', title='Monthly Average AQI Trend (All Cities)')
            fig1.update_layout(plot_bgcolor='white')
            st.plotly_chart(fig1, use_container_width=True)
            
        with colB:
            # Top 10 most polluted cities
            top10 = df.groupby('City')['AQI'].mean().sort_values(ascending=False).head(10).reset_index()
            fig2 = px.bar(top10, x='AQI', y='City', orientation='h', title='Top 10 Most Polluted Cities (Avg AQI)', color='AQI', color_continuous_scale='Reds')
            fig2.update_layout(yaxis={'categoryorder':'total ascending'}, plot_bgcolor='white')
            st.plotly_chart(fig2, use_container_width=True)
            
        colC, colD = st.columns(2)
        
        with colC:
            # AQI category distribution
            cat_counts = df['AQI_Bucket'].value_counts().reset_index()
            cat_counts.columns = ['Category', 'Count']
            color_map = {'Good': '#00B050', 'Satisfactory': '#92D050', 'Moderate': '#FFFF00', 'Poor': '#FF9900', 'Very Poor': '#FF0000', 'Severe': '#C00000'}
            fig3 = px.pie(cat_counts, names='Category', values='Count', title='AQI Category Distribution', hole=0.4, color='Category', color_discrete_map=color_map)
            st.plotly_chart(fig3, use_container_width=True)
            
        with colD:
            # Year-over-year AQI comparison
            yearly_aqi = df.groupby('Year')['AQI'].mean().reset_index()
            fig4 = px.bar(yearly_aqi, x='Year', y='AQI', title='Year-over-Year Average AQI', text_auto='.1f', color='AQI', color_continuous_scale='Blues')
            fig4.update_layout(plot_bgcolor='white', xaxis_type='category')
            st.plotly_chart(fig4, use_container_width=True)

    elif page == "🔍 City Explorer":
        st.markdown('<div class="section-header">City Explorer</div>', unsafe_allow_html=True)
        
        st.sidebar.markdown("### Filters")
        selected_city = st.sidebar.selectbox("Select City", df['City'].unique())
        
        min_year = int(df['Year'].min())
        max_year = int(df['Year'].max())
        selected_years = st.sidebar.slider("Select Year Range", min_year, max_year, (min_year, max_year))
        
        all_categories = df['AQI_Bucket'].dropna().unique().tolist()
        selected_cats = st.sidebar.multiselect("AQI Categories", all_categories, default=all_categories)
        
        # Filter data
        city_data = df[(df['City'] == selected_city) & 
                       (df['Year'] >= selected_years[0]) & 
                       (df['Year'] <= selected_years[1]) & 
                       (df['AQI_Bucket'].isin(selected_cats))]
                       
        if city_data.empty:
            st.warning("No data found for the selected filters.")
        else:
            # Pollutant time-series
            pollutants = ['PM2.5', 'PM10', 'NO2', 'SO2', 'O3']
            fig_ts = px.line(city_data, x='Date', y=pollutants, title=f'Pollutant Time Series for {selected_city}')
            fig_ts.update_layout(plot_bgcolor='white', hovermode='x unified')
            st.plotly_chart(fig_ts, use_container_width=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Correlation heatmap
                corr_data = city_data[pollutants + ['AQI']].corr()
                fig_corr = px.imshow(corr_data, text_auto=".2f", aspect="auto", title=f"Pollutant Correlation in {selected_city}", color_continuous_scale='RdBu_r')
                st.plotly_chart(fig_corr, use_container_width=True)
                
            with col2:
                # Monthly AQI box plot
                fig_box = px.box(city_data, x='Month', y='AQI', title=f'Monthly AQI Distribution in {selected_city}', color='Month')
                fig_box.update_layout(plot_bgcolor='white', showlegend=False)
                st.plotly_chart(fig_box, use_container_width=True)
                
            # Data table
            st.subheader("Raw Data")
            st.dataframe(city_data[['Date', 'City', 'AQI', 'AQI_Bucket'] + pollutants].head(100))
            
            # Download button
            csv = city_data.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download data as CSV",
                data=csv,
                file_name=f'{selected_city}_air_quality.csv',
                mime='text/csv',
            )

    elif page == "🌡️ Seasonal Analysis":
        st.markdown('<div class="section-header">Seasonal Analysis</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Season-wise AQI
            season_aqi = df.groupby('Season')['AQI'].mean().reset_index()
            fig_season = px.bar(season_aqi, x='Season', y='AQI', title='Average AQI by Season', color='Season')
            fig_season.update_layout(plot_bgcolor='white')
            st.plotly_chart(fig_season, use_container_width=True)
            
        with col2:
            # Weekend vs Weekday
            wknd_aqi = df.groupby('IsWeekend')['AQI'].mean().reset_index()
            wknd_aqi['Day Type'] = wknd_aqi['IsWeekend'].map({0: 'Weekday', 1: 'Weekend'})
            fig_wknd = px.pie(wknd_aqi, names='Day Type', values='AQI', title='Average AQI: Weekend vs Weekday', hole=0.4)
            st.plotly_chart(fig_wknd, use_container_width=True)
            
        # Month-wise AQI heatmap (cities × months)
        city_month = df.pivot_table(index='City', columns='Month', values='AQI', aggfunc='mean')
        fig_heatmap = px.imshow(city_month, title='Average AQI Heatmap (Cities x Months)', 
                                labels=dict(x="Month", y="City", color="AQI"),
                                color_continuous_scale='Reds', aspect="auto")
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Festival period analysis (Oct-Nov vs others)
        df['Period'] = np.where(df['Month'].isin([10, 11]), 'Festival Season (Oct-Nov)', 'Other Months')
        festival_aqi = df.groupby(['City', 'Period'])['AQI'].mean().reset_index()
        fig_fest = px.bar(festival_aqi, x='City', y='AQI', color='Period', barmode='group', title='Festival Season vs Other Months AQI')
        fig_fest.update_layout(plot_bgcolor='white', xaxis={'categoryorder':'total descending'})
        st.plotly_chart(fig_fest, use_container_width=True)

    elif page == "🤖 AQI Predictor (AI Insights)":
        st.markdown('<div class="section-header">AQI Predictor</div>', unsafe_allow_html=True)
        
        st.write("Adjust the pollutant levels below to predict the Air Quality Index (AQI) using our trained XGBoost model.")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Input Parameters")
            pm25 = st.slider("PM2.5", min_value=0.0, max_value=500.0, value=50.0)
            pm10 = st.slider("PM10", min_value=0.0, max_value=800.0, value=100.0)
            no2 = st.slider("NO2", min_value=0.0, max_value=200.0, value=30.0)
            so2 = st.slider("SO2", min_value=0.0, max_value=100.0, value=15.0)
            o3 = st.slider("O3", min_value=0.0, max_value=200.0, value=40.0)
            co = st.slider("CO", min_value=0.0, max_value=20.0, value=1.0)
            
        with col2:
            st.subheader("Prediction")
            
            input_df = pd.DataFrame({
                'PM2.5': [pm25], 'PM10': [pm10], 'NO2': [no2],
                'SO2': [so2], 'O3': [o3], 'CO': [co]
            })
            
            xgb_model = models_dict.get('XGBoost')
            if xgb_model:
                pred_aqi = xgb_model.predict(input_df)[0]
                
                # Determine bucket
                if pred_aqi <= 50: bucket, color = 'Good', '#00B050'
                elif pred_aqi <= 100: bucket, color = 'Satisfactory', '#92D050'
                elif pred_aqi <= 200: bucket, color = 'Moderate', '#FFFF00'
                elif pred_aqi <= 300: bucket, color = 'Poor', '#FF9900'
                elif pred_aqi <= 400: bucket, color = 'Very Poor', '#FF0000'
                else: bucket, color = 'Severe', '#C00000'
                
                st.markdown(f"""
                <div style="background-color: {color}; padding: 30px; border-radius: 10px; text-align: center; color: {'black' if pred_aqi<=200 else 'white'};">
                    <h2 style="margin:0;">Predicted AQI: {pred_aqi:.1f}</h2>
                    <h3 style="margin:0;">Category: {bucket}</h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Feature importance
                st.subheader("Feature Importance")
                importance = pd.DataFrame({
                    'Feature': input_df.columns,
                    'Importance': xgb_model.feature_importances_
                }).sort_values('Importance', ascending=True)
                
                fig_imp = px.bar(importance, x='Importance', y='Feature', orientation='h', title='What drives the AQI prediction?')
                fig_imp.update_layout(plot_bgcolor='white')
                st.plotly_chart(fig_imp, use_container_width=True)
                
            else:
                st.error("Model not available.")

    elif page == "📈 Model Performance":
        st.markdown('<div class="section-header">Model Performance Metrics</div>', unsafe_allow_html=True)
        
        # Display metrics
        metrics_df = pd.DataFrame({
            model: {
                'R² Score': m['R2'],
                'RMSE': m['RMSE'],
                'MAE': m['MAE']
            } for model, m in model_metrics.items()
        }).T.reset_index().rename(columns={'index': 'Model'})
        
        st.dataframe(metrics_df.style.format({'R² Score': '{:.4f}', 'RMSE': '{:.2f}', 'MAE': '{:.2f}'}), use_container_width=True)
        
        # Side-by-side comparison
        fig_r2 = px.bar(metrics_df, x='Model', y='R² Score', title='R² Score Comparison', color='Model')
        fig_r2.update_layout(plot_bgcolor='white')
        st.plotly_chart(fig_r2, use_container_width=True)
        
        # Select model to visualize
        st.subheader("Detailed Analysis")
        sel_model = st.selectbox("Select Model", list(models_dict.keys()))
        
        col1, col2 = st.columns(2)
        
        y_test = model_metrics[sel_model]['y_test']
        y_pred = model_metrics[sel_model]['y_pred']
        
        with col1:
            # Actual vs Predicted
            fig_scatter = px.scatter(x=y_test, y=y_pred, labels={'x': 'Actual AQI', 'y': 'Predicted AQI'}, title=f'{sel_model}: Actual vs Predicted')
            fig_scatter.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()], mode='lines', name='Ideal'))
            fig_scatter.update_layout(plot_bgcolor='white')
            st.plotly_chart(fig_scatter, use_container_width=True)
            
        with col2:
            # Residual distribution
            residuals = y_test - y_pred
            fig_resid = px.histogram(residuals, title=f'{sel_model}: Residual Distribution', labels={'value': 'Residual'})
            fig_resid.update_layout(plot_bgcolor='white')
            st.plotly_chart(fig_resid, use_container_width=True)

# Footer
st.markdown('<div class="footer">Built with ❤️ | AirQuality Intelligence Project</div>', unsafe_allow_html=True)
