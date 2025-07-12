import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Define constants
DATA_PATH = "C:/btc-data/output"
MODEL_SAVE_PATH = "model/btc_model.pkl"

# Features used for training (should match what your streaming script computes)
FEATURE_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
TARGET_COLUMN = 'close'  # or 'price_change' depending on your goal

# Load CSV files safely
def load_data(data_path):
    all_files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith(".csv")]
    df_list = []

    for file in all_files:
        try:
            df = pd.read_csv(file, header=None)
            if not df.empty and df.shape[1] >= 7:
                df.columns = ['symbol', 'timestamp', 'open', 'close', 'high', 'low', 'volume']
                df_list.append(df)
            else:
                print(f"[SKIP] Empty or invalid file: {file}")
        except pd.errors.EmptyDataError:
            print(f"[SKIP] Corrupt or unreadable file: {file}")
            continue

    if df_list:
        return pd.concat(df_list, ignore_index=True)
    else:
        return pd.DataFrame()

# Train model
def train_model(data):
    # Add derived features
    data['price_change'] = data['close'] - data['open']
    data['volatility'] = data['high'] - data['low']
    data['close/open'] = data['close'] / data['open']

    features = data[['open', 'high', 'low', 'close', 'volume', 'price_change', 'volatility', 'close/open']]
    target = data['close']

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    #rmse = mean_squared_error(y_test, predictions, squared=False)
    r2 = r2_score(y_test, predictions)

    print(f"[✅] Model trained successfully!")
   # print(f"[📊] RMSE: {rmse:.4f}")
    print(f"[📊] R² Score: {r2:.4f}")

    return model

# Main function
def main():
    print("[INFO] Loading data from CSV...")
    data = load_data(DATA_PATH)

    if data.empty:
        print("[⚠️] No valid data found in CSVs. Skipping training.")
        return

    print(f"[INFO] Loaded {len(data)} rows.")

    print("[INFO] Training model...")
    model = train_model(data)

    print(f"[INFO] Saving model to {MODEL_SAVE_PATH}")
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    joblib.dump(model, MODEL_SAVE_PATH)

if __name__ == "__main__":
    main()
