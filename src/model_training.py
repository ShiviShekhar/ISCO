import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import os

# --- Main Execution ---
def train_model():
    """Load features and historical data, train a model, and save it."""
    print("Starting model training...")

    # Load data
    try:
        features_df = pd.read_csv("data/feature_dataset.csv")
        deals_df = pd.read_csv("data/historical_deals.csv")
    except FileNotFoundError as e:
        print(f"Error: {e}. Make sure feature_dataset.csv and historical_deals.csv exist.")
        return

    # 1. Create the training dataset by merging features with historical outcomes
    training_data = pd.merge(features_df, deals_df, on=['seller_id', 'account_id'])

    # Filter out deals with no value, as they don't represent successful sales
    training_data = training_data[training_data['booking_value'] > 0]

    if training_data.empty:
        print("No historical deal data to train on. Please check the data simulation and feature engineering steps.")
        return

    print(f"Created a training dataset with {len(training_data)} historical deals.")

    # 2. Define features (X) and target (y)
    features = ['technology_match_score', 'geo_match', 'potential_revenue']
    target = 'booking_value'

    X = training_data[features]
    y = training_data[target]

    # 3. Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Split data into {len(X_train)} training samples and {len(X_test)} testing samples.")

    # 4. Initialize and train the XGBoost model
    print("Training the XGBoost Regressor model...")
    xgb_regressor = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    xgb_regressor.fit(X_train, y_train)

    # 5. Evaluate the model
    print("Evaluating the model...")
    y_pred = xgb_regressor.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print(f"Model evaluation complete. Root Mean Squared Error (RMSE): ${rmse:,.2f}")

    # 6. Save the trained model
    if not os.path.exists("models"):
        os.makedirs("models")
        
    model_path = "models/success_score_model.joblib"
    joblib.dump(xgb_regressor, model_path)
    print(f"\nModel training complete!")
    print(f"Trained model saved to {model_path}")

if __name__ == "__main__":
    train_model()
