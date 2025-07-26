# Week 2: Model Development - Visualization Generator
# ===================================================
# Run this after your Week 2 model development notebook to create professional visualizations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Set style for professional plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Create directories
import os
os.makedirs('visualizations/week2', exist_ok=True)
os.makedirs('results/week2', exist_ok=True)

print("🎨 Week 2: Creating Model Development Visualizations...")
print("=" * 60)

# ============================================================================
# 1. MODEL PERFORMANCE COMPARISON CHART
# ============================================================================

# Sample model results (replace with your actual results)
model_results = {
    'Model': ['Linear Regression', 'Ridge Regression', 'Lasso Regression', 
              'Random Forest', 'Gradient Boosting', 'XGBoost', 'LightGBM'],
    'R2_Score': [0.742, 0.756, 0.751, 0.823, 0.834, 0.867, 0.871],
    'RMSE': [2847.3, 2765.2, 2798.1, 2356.7, 2284.3, 2043.8, 2015.4],
    'MAE': [2156.8, 2089.3, 2123.7, 1789.2, 1732.1, 1534.2, 1498.7],
    'MAPE': [8.92, 8.64, 8.78, 7.41, 7.16, 6.35, 6.19],
    'Training_Time': [0.02, 0.03, 0.05, 1.23, 2.45, 3.67, 2.89]
}

df_results = pd.DataFrame(model_results)

# Create comprehensive model comparison visualization
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Week 2: Model Performance Comparison Dashboard', fontsize=20, fontweight='bold', y=0.98)

# R² Score comparison
colors = sns.color_palette("viridis", len(df_results))
bars1 = ax1.bar(df_results['Model'], df_results['R2_Score'], color=colors, alpha=0.8)
ax1.set_title('R² Score by Model', fontsize=14, fontweight='bold')
ax1.set_ylabel('R² Score')
ax1.tick_params(axis='x', rotation=45)
ax1.set_ylim(0.7, 0.9)

# Add value labels on bars
for bar, value in zip(bars1, df_results['R2_Score']):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
             f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

# RMSE comparison
bars2 = ax2.bar(df_results['Model'], df_results['RMSE'], color=colors, alpha=0.8)
ax2.set_title('Root Mean Square Error (RMSE)', fontsize=14, fontweight='bold')
ax2.set_ylabel('RMSE')
ax2.tick_params(axis='x', rotation=45)

# Add value labels
for bar, value in zip(bars2, df_results['RMSE']):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50, 
             f'{value:.0f}', ha='center', va='bottom', fontweight='bold')

# MAE comparison
bars3 = ax3.bar(df_results['Model'], df_results['MAE'], color=colors, alpha=0.8)
ax3.set_title('Mean Absolute Error (MAE)', fontsize=14, fontweight='bold')
ax3.set_ylabel('MAE')
ax3.tick_params(axis='x', rotation=45)

# MAPE comparison
bars4 = ax4.bar(df_results['Model'], df_results['MAPE'], color=colors, alpha=0.8)
ax4.set_title('Mean Absolute Percentage Error (MAPE)', fontsize=14, fontweight='bold')
ax4.set_ylabel('MAPE (%)')
ax4.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('visualizations/week2/model_performance_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ Model Performance Comparison Chart created!")

# ============================================================================
# 2. FEATURE IMPORTANCE ANALYSIS
# ============================================================================

# Sample feature importance data (replace with your actual results)
feature_importance = {
    'Feature': ['EV_Sales_Lag_1', 'EV_Sales_MA_6', 'Month', 'EV_Sales_Lag_3', 
                'Gas_Price', 'GDP_Growth', 'Charging_Stations', 'EV_Sales_MA_12',
                'Quarter', 'Year', 'EV_Sales_MA_3', 'Gas_Price_MA_3',
                'Population', 'Income_Per_Capita', 'EV_Price_Index'],
    'Importance': [0.284, 0.156, 0.089, 0.078, 0.067, 0.058, 0.052, 0.048,
                   0.041, 0.038, 0.034, 0.029, 0.026, 0.023, 0.019],
    'Category': ['Lag Features', 'Moving Average', 'Time', 'Lag Features',
                'Economic', 'Economic', 'Infrastructure', 'Moving Average',
                'Time', 'Time', 'Moving Average', 'Economic',
                'Demographic', 'Economic', 'Economic']
}

df_importance = pd.DataFrame(feature_importance)

# Create feature importance visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
fig.suptitle('Week 2: Feature Importance Analysis', fontsize=20, fontweight='bold')

# Top 15 features horizontal bar chart
top_features = df_importance.head(15)
category_colors = {'Lag Features': '#FF6B6B', 'Moving Average': '#4ECDC4', 
                   'Time': '#45B7D1', 'Economic': '#96CEB4', 
                   'Infrastructure': '#FFEAA7', 'Demographic': '#DDA0DD'}

colors_list = [category_colors[cat] for cat in top_features['Category']]

bars = ax1.barh(range(len(top_features)), top_features['Importance'], 
                color=colors_list, alpha=0.8)
ax1.set_yticks(range(len(top_features)))
ax1.set_yticklabels(top_features['Feature'])
ax1.set_xlabel('Feature Importance')
ax1.set_title('Top 15 Most Important Features', fontsize=14, fontweight='bold')
ax1.invert_yaxis()

# Add value labels
for i, (bar, value) in enumerate(zip(bars, top_features['Importance'])):
    ax1.text(value + 0.005, bar.get_y() + bar.get_height()/2, 
             f'{value:.3f}', va='center', fontweight='bold')

# Feature importance by category
category_importance = df_importance.groupby('Category')['Importance'].sum().sort_values(ascending=False)
colors_cat = [category_colors[cat] for cat in category_importance.index]

wedges, texts, autotexts = ax2.pie(category_importance.values, labels=category_importance.index, 
                                   autopct='%1.1f%%', colors=colors_cat, startangle=90)
ax2.set_title('Feature Importance by Category', fontsize=14, fontweight='bold')

# Enhance pie chart text
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')

plt.tight_layout()
plt.savefig('visualizations/week2/feature_importance_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ Feature Importance Analysis created!")

# ============================================================================
# 3. BEST MODEL PREDICTIONS VS ACTUAL
# ============================================================================

# Generate sample prediction data (replace with your actual predictions)
np.random.seed(42)
n_samples = 100
actual_values = np.random.normal(15000, 5000, n_samples)
actual_values = np.maximum(actual_values, 1000)  # Ensure positive values

# Best model predictions (LightGBM with some realistic noise)
predicted_values = actual_values * (1 + np.random.normal(0, 0.15, n_samples))
predicted_values = np.maximum(predicted_values, 1000)

# Calculate metrics
r2 = r2_score(actual_values, predicted_values)
mae = mean_absolute_error(actual_values, predicted_values)
rmse = np.sqrt(mean_squared_error(actual_values, predicted_values))

# Create prediction visualization
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle(f'Week 2: Best Model (LightGBM) - Prediction Analysis\nR² = {r2:.3f} | MAE = {mae:.0f} | RMSE = {rmse:.0f}', 
             fontsize=18, fontweight='bold')

# Actual vs Predicted scatter plot
ax1.scatter(actual_values, predicted_values, alpha=0.6, color='#3498db', s=50)
min_val = min(min(actual_values), min(predicted_values))
max_val = max(max(actual_values), max(predicted_values))
ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
ax1.set_xlabel('Actual Values')
ax1.set_ylabel('Predicted Values')
ax1.set_title('Actual vs Predicted Values', fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Residuals plot
residuals = predicted_values - actual_values
ax2.scatter(predicted_values, residuals, alpha=0.6, color='#e74c3c', s=50)
ax2.axhline(y=0, color='black', linestyle='--', alpha=0.8)
ax2.set_xlabel('Predicted Values')
ax2.set_ylabel('Residuals')
ax2.set_title('Residuals Plot', fontweight='bold')
ax2.grid(True, alpha=0.3)

# Prediction error distribution
ax3.hist(residuals, bins=30, alpha=0.7, color='#9b59b6', edgecolor='black')
ax3.set_xlabel('Prediction Error')
ax3.set_ylabel('Frequency')
ax3.set_title('Distribution of Prediction Errors', fontweight='bold')
ax3.axvline(x=0, color='red', linestyle='--', alpha=0.8)
ax3.grid(True, alpha=0.3)

# Time series prediction (sample)
time_index = pd.date_range('2017-01-01', periods=len(actual_values), freq='M')
ax4.plot(time_index, actual_values, label='Actual', color='#2ecc71', linewidth=2)
ax4.plot(time_index, predicted_values, label='Predicted', color='#f39c12', linewidth=2, alpha=0.8)
ax4.set_xlabel('Date')
ax4.set_ylabel('EV Sales')
ax4.set_title('Time Series: Actual vs Predicted', fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)
plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)

plt.tight_layout()
plt.savefig('visualizations/week2/best_model_predictions.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ Best Model Predictions Analysis created!")

# ============================================================================
# 4. CROSS-VALIDATION RESULTS
# ============================================================================

# Sample cross-validation results
cv_results = {
    'Model': ['Linear Regression', 'Ridge', 'Lasso', 'Random Forest', 'Gradient Boosting', 'XGBoost', 'LightGBM'],
    'CV_Mean': [0.728, 0.741, 0.736, 0.812, 0.821, 0.854, 0.858],
    'CV_Std': [0.089, 0.076, 0.082, 0.061, 0.058, 0.045, 0.042],
    'Fold_1': [0.651, 0.678, 0.663, 0.768, 0.779, 0.823, 0.829],
    'Fold_2': [0.745, 0.759, 0.751, 0.834, 0.841, 0.867, 0.871],
    'Fold_3': [0.789, 0.801, 0.794, 0.856, 0.863, 0.889, 0.893],
    'Fold_4': [0.712, 0.731, 0.723, 0.798, 0.806, 0.845, 0.851],
    'Fold_5': [0.743, 0.736, 0.749, 0.804, 0.816, 0.846, 0.848]
}

df_cv = pd.DataFrame(cv_results)

# Create cross-validation visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
fig.suptitle('Week 2: Cross-Validation Results Analysis', fontsize=18, fontweight='bold')

# Box plot of CV scores
cv_data = []
model_names = []
for _, row in df_cv.iterrows():
    scores = [row['Fold_1'], row['Fold_2'], row['Fold_3'], row['Fold_4'], row['Fold_5']]
    cv_data.append(scores)
    model_names.append(row['Model'])

bp = ax1.boxplot(cv_data, labels=model_names, patch_artist=True)
colors = sns.color_palette("Set3", len(model_names))
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax1.set_title('Cross-Validation Score Distribution', fontweight='bold')
ax1.set_ylabel('R² Score')
ax1.tick_params(axis='x', rotation=45)
ax1.grid(True, alpha=0.3)

# Mean CV scores with error bars
x_pos = range(len(df_cv))
bars = ax2.bar(x_pos, df_cv['CV_Mean'], yerr=df_cv['CV_Std'], 
               capsize=5, color=colors, alpha=0.7, edgecolor='black')
ax2.set_xlabel('Model')
ax2.set_ylabel('Mean R² Score')
ax2.set_title('Mean Cross-Validation Scores (±1 std)', fontweight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(df_cv['Model'], rotation=45)
ax2.grid(True, alpha=0.3)

# Add value labels
for bar, mean_val, std_val in zip(bars, df_cv['CV_Mean'], df_cv['CV_Std']):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std_val + 0.01, 
             f'{mean_val:.3f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('visualizations/week2/cross_validation_results.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ Cross-Validation Results created!")

# ============================================================================
# 5. LEARNING CURVES
# ============================================================================

# Generate sample learning curve data
train_sizes = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
n_samples_total = 1000

# Sample learning curves for different models
models_lc = {
    'Random Forest': {
        'train_scores': [0.95, 0.94, 0.93, 0.92, 0.91, 0.90, 0.89, 0.88, 0.87, 0.86],
        'val_scores': [0.70, 0.74, 0.77, 0.79, 0.80, 0.81, 0.82, 0.82, 0.82, 0.82]
    },
    'XGBoost': {
        'train_scores': [0.92, 0.91, 0.90, 0.89, 0.88, 0.87, 0.87, 0.86, 0.86, 0.85],
        'val_scores': [0.75, 0.79, 0.82, 0.84, 0.85, 0.86, 0.86, 0.87, 0.87, 0.87]
    },
    'LightGBM': {
        'train_scores': [0.91, 0.90, 0.89, 0.88, 0.87, 0.87, 0.86, 0.86, 0.85, 0.85],
        'val_scores': [0.76, 0.80, 0.83, 0.85, 0.86, 0.87, 0.87, 0.87, 0.87, 0.87]
    }
}

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Week 2: Learning Curves for Best Performing Models', fontsize=16, fontweight='bold')

for idx, (model_name, curves) in enumerate(models_lc.items()):
    ax = axes[idx]
    
    train_scores = np.array(curves['train_scores'])
    val_scores = np.array(curves['val_scores'])
    
    # Add some noise to make it realistic
    train_std = np.random.uniform(0.01, 0.03, len(train_scores))
    val_std = np.random.uniform(0.01, 0.04, len(val_scores))
    
    ax.plot(train_sizes * n_samples_total, train_scores, 'o-', color='#3498db', 
            label='Training Score', linewidth=2, markersize=6)
    ax.fill_between(train_sizes * n_samples_total, train_scores - train_std, 
                    train_scores + train_std, alpha=0.2, color='#3498db')
    
    ax.plot(train_sizes * n_samples_total, val_scores, 'o-', color='#e74c3c', 
            label='Validation Score', linewidth=2, markersize=6)
    ax.fill_between(train_sizes * n_samples_total, val_scores - val_std, 
                    val_scores + val_std, alpha=0.2, color='#e74c3c')
    
    ax.set_xlabel('Training Set Size')
    ax.set_ylabel('R² Score')
    ax.set_title(f'{model_name}', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations/week2/learning_curves.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ Learning Curves created!")

# ============================================================================
# 6. MODEL COMPLEXITY ANALYSIS
# ============================================================================

# Sample hyperparameter tuning results
hyperparams = {
    'XGBoost': {
        'n_estimators': [50, 100, 200, 300, 400, 500],
        'scores': [0.834, 0.856, 0.867, 0.869, 0.867, 0.865]
    },
    'Random Forest': {
        'n_estimators': [50, 100, 200, 300, 400, 500],
        'scores': [0.812, 0.823, 0.828, 0.829, 0.828, 0.827]
    }
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle('Week 2: Hyperparameter Tuning Results', fontsize=16, fontweight='bold')

# XGBoost tuning
ax1.plot(hyperparams['XGBoost']['n_estimators'], hyperparams['XGBoost']['scores'], 
         'o-', color='#2ecc71', linewidth=3, markersize=8, label='XGBoost')
ax1.set_xlabel('Number of Estimators')
ax1.set_ylabel('R² Score')
ax1.set_title('XGBoost: n_estimators Tuning', fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Find and mark best score
best_idx = np.argmax(hyperparams['XGBoost']['scores'])
ax1.annotate(f'Best: {hyperparams["XGBoost"]["scores"][best_idx]:.3f}', 
             xy=(hyperparams['XGBoost']['n_estimators'][best_idx], hyperparams['XGBoost']['scores'][best_idx]),
             xytext=(10, 10), textcoords='offset points', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

# Random Forest tuning
ax2.plot(hyperparams['Random Forest']['n_estimators'], hyperparams['Random Forest']['scores'], 
         'o-', color='#9b59b6', linewidth=3, markersize=8, label='Random Forest')
ax2.set_xlabel('Number of Estimators')
ax2.set_ylabel('R² Score')
ax2.set_title('Random Forest: n_estimators Tuning', fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend()

# Find and mark best score
best_idx_rf = np.argmax(hyperparams['Random Forest']['scores'])
ax2.annotate(f'Best: {hyperparams["Random Forest"]["scores"][best_idx_rf]:.3f}', 
             xy=(hyperparams['Random Forest']['n_estimators'][best_idx_rf], hyperparams['Random Forest']['scores'][best_idx_rf]),
             xytext=(10, 10), textcoords='offset points', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

plt.tight_layout()
plt.savefig('visualizations/week2/hyperparameter_tuning.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ Hyperparameter Tuning Analysis created!")

# ============================================================================
# 7. SAVE SUMMARY RESULTS
# ============================================================================

# Create Week 2 summary
week2_summary = {
    "week": 2,
    "focus": "Model Development & Evaluation",
    "models_trained": len(df_results),
    "best_model": {
        "name": df_results.loc[df_results['R2_Score'].idxmax(), 'Model'],
        "r2_score": float(df_results['R2_Score'].max()),
        "rmse": float(df_results.loc[df_results['R2_Score'].idxmax(), 'RMSE']),
        "mae": float(df_results.loc[df_results['R2_Score'].idxmax(), 'MAE']),
        "mape": float(df_results.loc[df_results['R2_Score'].idxmax(), 'MAPE'])
    },
    "top_features": feature_importance['Feature'][:5],
    "key_insights": [
        f"Best performing model: {df_results.loc[df_results['R2_Score'].idxmax(), 'Model']} with R² = {df_results['R2_Score'].max():.3f}",
        f"Tree-based models outperformed linear models significantly",
        f"Lag features and moving averages are most predictive",
        f"Cross-validation shows stable performance across folds",
        f"Model achieved {df_results['MAPE'].min():.1f}% MAPE on test set"
    ],
    "visualizations_created": [
        "model_performance_comparison.png",
        "feature_importance_analysis.png", 
        "best_model_predictions.png",
        "cross_validation_results.png",
        "learning_curves.png",
        "hyperparameter_tuning.png"
    ]
}

# Save results
df_results.to_csv('results/week2/model_comparison.csv', index=False)
df_importance.to_csv('results/week2/feature_importance.csv', index=False)
df_cv.to_csv('results/week2/cross_validation_results.csv', index=False)

# Save predictions data
predictions_df = pd.DataFrame({
    'Actual': actual_values,
    'Predicted': predicted_values,
    'Residuals': residuals,
    'Date': time_index
})
predictions_df.to_csv('results/week2/best_model_predictions.csv', index=False)

# Save summary as JSON
import json
with open('results/week2/week2_summary.json', 'w') as f:
    json.dump(week2_summary, f, indent=2)

print("\n" + "="*60)
print("🎉 WEEK 2 VISUALIZATION SUITE COMPLETE!")
print("="*60)
print("\n📊 Created Visualizations:")
for viz in week2_summary["visualizations_created"]:
    print(f"   ✅ {viz}")

print(f"\n📈 Best Model: {week2_summary['best_model']['name']}")
print(f"   • R² Score: {week2_summary['best_model']['r2_score']:.3f}")
print(f"   • RMSE: {week2_summary['best_model']['rmse']:.0f}")
print(f"   • MAE: {week2_summary['best_model']['mae']:.0f}")
print(f"   • MAPE: {week2_summary['best_model']['mape']:.1f}%")

print(f"\n🔑 Top 5 Features:")
for i, feature in enumerate(week2_summary['top_features'][:5], 1):
    print(f"   {i}. {feature}")

print(f"\n📁 Files Saved:")
print(f"   • results/week2/model_comparison.csv")
print(f"   • results/week2/feature_importance.csv") 
print(f"   • results/week2/cross_validation_results.csv")
print(f"   • results/week2/best_model_predictions.csv")
print(f"   • results/week2/week2_summary.json")

print(f"\n🚀 Ready for Week 3: Advanced Optimization & Deployment!")
print("="*60)