# 💳 Credit Card Fraud Detection & Curation Dashboard

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-EE4C2C)
![Dash](https://img.shields.io/badge/Dash-Web_App-008DE4)
![Scikit-Learn](https://img.shields.io/badge/Scikit_Learn-Data_Curation-F7931E)

## 📌 Project Overview
An end-to-end Data Curation and Visualization (DCV) pipeline and interactive dashboard engineered to analyze and detect anomalies in highly skewed financial transaction datasets. 

This project bridges the gap between exploratory data analysis and production-level AI deployment. It features a robust preprocessing engine to handle extreme class imbalances, an interactive web interface for visual analytics, and a backend inference engine powered by a custom PyTorch neural network for real-time anomaly detection.

🌐 **Live Demo:** [Insert your Render link here, e.g., https://credit-fraud-dash.onrender.com]

## 🚀 Key Features
* **Interactive Visual Analytics:** Dynamic Dash-based frontend utilizing Plotly for high-density visualizations, including comparative violin plots, box plots, and multivariate scatter projections.
* **Advanced Data Curation:** Intelligent pipeline utilizing Scikit-Learn's `RobustScaler` to neutralize extreme statistical outliers in financial features (e.g., Transaction Amount, Time) prior to modeling.
* **Real-Time PyTorch Inference:** A globally initialized `torch.nn.Module` architecture integrated directly into the web application callbacks to evaluate transaction risk profiles instantly without blocking the main server thread.
* **Custom Dataset Pipeline:** Built-in `torch.utils.data.Dataset` mapping to feed scaled pandas DataFrames directly into gradient-based training loops.

## 🛠️ Tech Stack
* **Frontend:** Dash, Plotly Express, HTML/CSS
* **Data Engineering:** Pandas, Scikit-Learn, NumPy
* **Deep Learning:** PyTorch
* **Deployment:** Gunicorn (WSGI HTTP Server)

## 📂 Project Structure
```text
credit-card-fraud-curation/
├── .gitignore             # Git exclusions (ignores raw dataset)
├── app.py                 # Main Dash application and PyTorch callbacks
├── fraud_weights.pth      # Saved state_dict for the PyTorch inference model
├── requirements.txt       # Production dependencies
├── train.py               # Model training script
└── README.md              # Project documentation
