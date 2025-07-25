# 🚗 Week 2: Model Development and Evaluation – EV Demand Prediction 🔋⚡

This notebook focuses on building predictive models to forecast Electric Vehicle (EV) demand using historical data and engineered features from Week 1. Several machine learning models are trained, tuned, and evaluated to identify the most effective approach.

---

## 📌 Objectives

- Train multiple regression models to predict EV demand
- Fine-tune model parameters for better performance
- Evaluate and compare models using key metrics
- Visualize model results and residuals

---

## 📂 Notebook Structure

### `02_model_development.ipynb`
Key tasks covered:

1. **Importing Cleaned Dataset**
   - Data from Week 1 (after EDA and feature engineering) is used
2. **Train-Test Split**
   - Time-series aware splitting for better generalization
3. **Model Building**
   - Linear Regression
   - Random Forest Regressor
   - Gradient Boosting Regressor
   - XGBoost (if available)
4. **Hyperparameter Tuning**
   - GridSearchCV and RandomizedSearchCV (for applicable models)
5. **Evaluation Metrics**
   - Mean Absolute Error (MAE)
   - Root Mean Squared Error (RMSE)
   - R² Score
6. **Visualization**
   - Actual vs. Predicted
   - Residual plots
   - Feature importance charts

---




---

## 🛠 Dependencies

- Python 3.8+
- pandas, numpy, matplotlib, seaborn
- scikit-learn
- xgboost (optional)

```bash
pip install -r requirements.txt
