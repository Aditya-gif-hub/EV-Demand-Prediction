EV Vehicle Demand Prediction 🚗⚡
📊 Project Overview
Predicting Electric Vehicle (EV) demand across US counties using machine learning techniques. This internship project analyzes historical data (2019-2022) to forecast future EV adoption trends.
📅 Week 1: Data Exploration & Feature Engineering ✅
🎯 Objectives Completed

Comprehensive Exploratory Data Analysis (EDA)
Data cleaning and preprocessing
Feature engineering for temporal patterns
Created 8+ visualizations for insights

📈 Key Findings

Dataset: 20,819 records spanning Nov 2019 - Sept 2022
Coverage: 50 US states with county-level granularity
EV Growth: Identified strong upward trend in EV adoption
Top States: California, Texas, and Florida lead in EV adoption
Vehicle Types: Passenger vehicles dominate (85%+) over trucks

🛠️ Technical Implementation
Data Preprocessing

Handled comma-separated numeric values
Converted date strings to datetime objects
Created time-based features (Year, Month, Quarter, Season)
Managed missing values and outliers

Visualizations Created

Temporal Analysis: EV adoption trends over time
Geographic Distribution: State-wise EV concentration
Vehicle Type Analysis: Passenger vs Truck distribution
BEV vs PHEV Comparison: Battery vs Plug-in Hybrid trends
Correlation Matrix: Feature relationships
Seasonal Patterns: Quarterly adoption variations
Growth Rate Analysis: Month-over-month changes
Comprehensive Dashboard: Multi-metric overview

Feature Engineering

Extracted temporal features (Year, Month, Quarter, Season)
Created YearMonth periods for time-series analysis
Prepared foundation for lag features and rolling averages

📁 Deliverables

✅ 01_data_exploration.ipynb - Complete EDA notebook
✅ data_preprocessing.py - Reusable preprocessing functions
✅ 8+ visualizations in visualizations/ folder
✅ Processed dataset in data/processed/
✅ Summary statistics report

🔍 Key Insights for Modeling

Strong temporal patterns suggest time-series approaches
State-level variations indicate need for geographic features
BEV growing faster than PHEV - separate models may help
Seasonal patterns present - important for predictions

📊 Sample Visualizations

Monthly EV adoption showing consistent growth with seasonal variations
