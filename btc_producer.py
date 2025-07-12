import json
import time
from kafka import KafkaProducer
import requests
from datetime import datetime

# Kafka Configuration
KAFKA_BROKER = 'localhost:9092'
TOPIC_NAME = 'btc_prices'

# Initialize Kafka Producer
producer = KafkaProducer(
    bootstrap_servers=KAFKA_BROKER,
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# Binance API Endpoint
BINANCE_URL = "https://api.binance.com/api/v3/klines"

def fetch_latest_btc_data(interval='1m'):
    params = {
        "symbol": "BTCUSDT",
        "interval": interval,
        "limit": 1  # Only latest candle
    }
    response = requests.get(BINANCE_URL, params=params)
    if response.status_code == 200:
        kline = response.json()[0]
        return {
            "symbol": "BTC",
            "timestamp": datetime.fromtimestamp(kline[0] / 1000).isoformat(),
            "open": float(kline[1]),
            "close": float(kline[4]),
            "high": float(kline[2]),
            "low": float(kline[3]),
            "volume": float(kline[5])
        }
    else:
        print(f"Error fetching data: {response.status_code}")
        return None

def stream_data_to_kafka(interval='1m', sleep_duration=60):
    print(f"[INFO] Starting Kafka producer with interval {interval}")
    while True:
        data = fetch_latest_btc_data(interval)
        if data:
            print(f"[Kafka ➜] {data}")
            producer.send(TOPIC_NAME, value=data)
        time.sleep(sleep_duration)  # Wait for next interval

if __name__ == "__main__":
    # You can change interval to '5m' and sleep_duration to 300 for 5-minute updates
    stream_data_to_kafka(interval='1m', sleep_duration=60)
