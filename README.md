# Computer-Science-311-Assignment-1

# 🪙 Gold Price Prediction Application

A machine learning application built with Python to predict the future price of Gold (GLD) using historical market data and macroeconomic indicators.



## 📌 Project Overview
This project was developed as part of the Applied AI course. It aims to predict commodity prices by analyzing the relationship between Gold and other market factors such as the S&P 500, Crude Oil prices (USO), and US Dollar exchange rates.

The application uses a **Random Forest Regressor** to handle the non-linear relationships often found in financial time-series data.

## 📊 Dataset Information
The model is trained on the [Gold Price Prediction Dataset](https://www.kaggle.com/datasets/sid321axn/gold-price-prediction-dataset) from Kaggle.
* **Features:** 80 columns including S&P 500 (`SP_`), Dow Jones (`DJ_`), Silver (`SLV`), and US Dollar Index.
* **Target Variable:** `Adj Close` (Adjusted Closing Price of Gold).
* **Size:** ~1,718 records covering over a decade of daily market data.

## 🛠️ Technical Stack
* **Language:** Python 3.x
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** Scikit-learn (Random Forest)
* **Visualization:** Matplotlib, Seaborn

## 🚀 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/gold-price-prediction.git](https://github.com/yourusername/gold-price-prediction.git)