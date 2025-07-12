# 🚀 Using Real-Time Big Data for ML-Based Bitcoin Price Prediction

A real-time data pipeline and machine learning system that ingests live Bitcoin prices, processes them with Apache Spark, and predicts future price trends using trained ML models.

---

## 📌 Table of Contents
- [💡 Overview](#-overview)
- [⚙️ Tech Stack](#-tech-stack)
- [📊 Architecture](#-architecture)
- [🔧 Installation](#-installation)
- [🚀 Running the Project](#-running-the-project)
- [📈 Model & Results](#-model--results)
- [📁 Project Structure](#-project-structure)
- [🛡️ License](#-license)

---

## 💡 Overview

This project demonstrates how to leverage **Apache Kafka**, **Spark Structured Streaming**, and **Machine Learning** to build a **real-time Bitcoin price monitoring and prediction system**.

📉 Real-time data is streamed via **Kafka**  
🔥 Processed using **Spark**  
🤖 Predicted using **ML models** trained on historical data

> ⚡ Built for scalable, low-latency financial analytics.

---

## ⚙️ Tech Stack

| Component        | Technology                          |
|------------------|--------------------------------------|
| Programming      | Python 🐍                            |
| Data Streaming   | Apache Kafka (via Redpanda) 🔴        |
| Stream Processing| Apache Spark Structured Streaming ⚡ |
| Storage          | HDFS 🗄️                              |
| Containerization | Docker 🐳                             |
| ML Model         | scikit-learn, pandas, numpy          |
| Visualization    | matplotlib, seaborn                  |

---

## 📊 Architecture

```mermaid
graph TD;
    A[Bitcoin Price Source] --> B(Kafka Producer)
    B --> C(Spark Structured Streaming)
    C --> D(ML Model)
    D --> E[Predicted Price Output]
    C --> F[HDFS Storage]
🔧 Installation
✅ Prerequisites: Python 3.8+, Docker, Java 8+, Spark, Kafka/Redpanda

bash
Copy
Edit
# Clone the repository
git clone https://github.com/LikithKumar24/Using-Real-Time-Big-Data-For-ML-Based-Bitcoin-Price-Prediction.git
cd Using-Real-Time-Big-Data-For-ML-Based-Bitcoin-Price-Prediction

# (Optional) Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # For Linux/macOS
# .venv\Scripts\activate    # For Windows

# Install required packages
pip install -r requirements.txt
🚀 Running the Project
1. 🧱 Start Kafka Broker (via Redpanda or Docker)
bash
Copy
Edit
docker-compose up -d
2. 📡 Start Kafka Producer to Stream Live BTC Price
bash
Copy
Edit
python kafka_producer.py
3. ⚡ Run Spark Structured Streaming
bash
Copy
Edit
spark-submit spark_streaming_job.py
4. 📈 Train and Run ML
