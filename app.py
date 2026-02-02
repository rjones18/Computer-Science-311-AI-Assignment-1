import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# Ingesting and Cleaning Data
def ingesting_data(filepath):
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

# Begin analyzing dataset
def plotting(df):
    cols = ['Adj Close', 'SP_Ajclose', 'DJ_Ajclose', 'USO_Adj Close', 'EU_Price']
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[cols].corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation between Gold and Macroeconomics')
    plt.show()

# Start to train model
def train_model(df):
    X = df.drop(['Date', 'Adj Close'], axis=1)
    y = df['Adj Close']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(f"Error: {mean_absolute_error(y_test, predictions):.2f}")
    print(f"R-squared Score: {r2_score(y_test, predictions):.2f}")
    return y_test, predictions

# Main Function
if __name__ == "__main__":
    data_path = 'FINAL_USO.csv'
    gold_df = ingesting_data(data_path)
    
    plotting(gold_df)
    y_test, preds = train_model(gold_df)

    # Visualize Results
    plt.figure(figsize=(10, 5))
    plt.plot(y_test.values, label='Actual Price', color='gold')
    plt.plot(preds, label='Predicted Price', color='black', linestyle='--')
    plt.legend()
    plt.title('Gold Price Prediction: Actual vs Predicted')
    plt.show()