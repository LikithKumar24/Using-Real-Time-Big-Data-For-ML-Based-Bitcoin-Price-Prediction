import os
import pandas as pd
import joblib

MODEL_PATH = "model/btc_model.pkl"
CSV_DIR = "C:/btc-data/output"

def get_latest_csv(csv_dir):
    csv_files = [os.path.join(csv_dir, f) for f in os.listdir(csv_dir) if f.endswith('.csv')]
    if not csv_files:
        raise FileNotFoundError("No CSV files found.")
    return max(csv_files, key=os.path.getctime)

def main():
    print("[INFO] Loading trained model...")
    model = joblib.load(MODEL_PATH)

    latest_csv = get_latest_csv(CSV_DIR)
    print(f"[INFO] Using latest file: {latest_csv}")
    df = pd.read_csv(latest_csv, header=None)
    df.columns = ['symbol', 'timestamp', 'open', 'close', 'high', 'low', 'volume']

    df['price_change'] = df['close'] - df['open']
    df['volatility'] = df['high'] - df['low']
    df['close/open'] = df['close'] / df['open']

    features = df[['open', 'high', 'low', 'close', 'volume', 'price_change', 'volatility', 'close/open']]

    predictions = model.predict(features)
    print("[📈] Predictions:")
    print(predictions)

if __name__ == "__main__":
    main()
