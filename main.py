import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings

# Ignore warnings for clean output
warnings.filterwarnings('ignore')

# Load Dataset from local file
dataset_filename = 'GlobalWeatherRepository.csv'
data = pd.read_csv(dataset_filename)

# Display column names for reference
print("Available columns in the dataset:")
print(data.columns)

# User input for country or city name
location = input("Enter the name of the country or city: ")

# Check for column names that might represent location
def find_location_column(columns):
    possible_names = ['location', 'city', 'country', 'region']
    for name in possible_names:
        if name in columns:
            return name
    return None

location_column = find_location_column(data.columns)

if location_column is None:
    print("The dataset does not contain a column for location (e.g., 'location', 'city', 'country'). Please verify the dataset.")
    exit()

# Filter data by country or city
filtered_data = data[data[location_column].str.contains(location, case=False, na=False)]

if filtered_data.empty:
    print(f"No data found for the specified location: {location}")
    exit()

# Data Cleaning & Preprocessing
# Handling Missing Values
filtered_data.fillna(filtered_data.select_dtypes(include=[np.number]).mean(numeric_only=True), inplace=True)

# Handling Outliers using IQR Method
def cap_outliers(df, column_name):
    Q1 = df[column_name].quantile(0.25)
    Q3 = df[column_name].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[column_name] = np.where(df[column_name] < lower_bound, lower_bound, df[column_name])
    df[column_name] = np.where(df[column_name] > upper_bound, upper_bound, df[column_name])
    return df

for column in ['temperature_celsius', 'precip_mm']:
    if column in filtered_data.columns:
        cap_outliers(filtered_data, column)

# Data Normalization
scaler = MinMaxScaler()
if 'temperature_celsius' in filtered_data.columns and 'wind_kph' in filtered_data.columns:
    filtered_data[['temperature_celsius', 'wind_kph']] = scaler.fit_transform(filtered_data[['temperature_celsius', 'wind_kph']])

# Exploratory Data Analysis (EDA)
# Temperature and Precipitation Trends
if 'date' in filtered_data.columns:
    plt.figure(figsize=(10, 6))
    plt.plot(filtered_data['date'], filtered_data['temperature_celsius'], label='Temperature (Celsius)')
    plt.plot(filtered_data['date'], filtered_data['precip_mm'], label='Precipitation (mm)')
    plt.xlabel('Date')
    plt.ylabel('Values')
    plt.title(f'Temperature and Precipitation Trends for {location}')
    plt.legend()
    plt.show()

# Correlation Analysis
if {'temperature_celsius', 'precip_mm', 'humidity', 'wind_kph'}.issubset(filtered_data.columns):
    plt.figure(figsize=(8, 6))
    sns.heatmap(filtered_data[['temperature_celsius', 'precip_mm', 'humidity', 'wind_kph']].corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.show()

# Model Building: Time Series Analysis
# Converting 'date' column to datetime format
if 'date' in filtered_data.columns:
    filtered_data['date'] = pd.to_datetime(filtered_data['date'])
    filtered_data.set_index('date', inplace=True)

    # Building ARIMA Model
    model = ARIMA(filtered_data['temperature_celsius'], order=(5, 1, 0))
    model_fit = model.fit()

    # Forecasting
    forecast = model_fit.forecast(steps=30)

    # Evaluation
    mae = mean_absolute_error(filtered_data['temperature_celsius'][-30:], forecast)
    rmse = np.sqrt(mean_squared_error(filtered_data['temperature_celsius'][-30:], forecast))

    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")

# Advanced EDA: Anomaly Detection
from scipy.stats import zscore

if 'temperature_celsius' in filtered_data.columns:
    filtered_data['zscore_temp'] = zscore(filtered_data['temperature_celsius'])
    threshold = 3
    anomalies = filtered_data[np.abs(filtered_data['zscore_temp']) > threshold]

    # Plotting anomalies
    plt.figure(figsize=(10, 6))
    plt.plot(filtered_data.index, filtered_data['temperature_celsius'], label='Temperature (Celsius)')
    plt.scatter(anomalies.index, anomalies['temperature_celsius'], color='red', label='Anomalies')
    plt.xlabel('Date')
    plt.ylabel('Temperature')
    plt.title(f'Temperature Anomalies for {location}')
    plt.legend()
    plt.show()

# Advanced Model Building: LSTM
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# Prepare data for LSTM
if 'temperature_celsius' in filtered_data.columns:
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(filtered_data['temperature_celsius'].values.reshape(-1, 1))
    train_size = int(len(data_scaled) * 0.8)
    train, test = data_scaled[0:train_size], data_scaled[train_size:]

    # Create sequences
    def create_sequences(dataset, look_back=1):
        X, y = [], []
        for i in range(len(dataset) - look_back - 1):
            a = dataset[i:(i + look_back), 0]
            X.append(a)
            y.append(dataset[i + look_back, 0])
        return np.array(X), np.array(y)

    look_back = 3
    X_train, y_train = create_sequences(train, look_back)
    X_test, y_test = create_sequences(test, look_back)

    # Reshape for LSTM input
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # Build LSTM Model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(look_back, 1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train LSTM Model
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), verbose=2)

    # Make Predictions
    lstm_predictions = model.predict(X_test)

    # Inverse transform to get actual values
    lstm_predictions = scaler.inverse_transform(lstm_predictions)
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Evaluation
    lstm_rmse = np.sqrt(mean_squared_error(y_test_actual, lstm_predictions))
    print(f"LSTM Model RMSE: {lstm_rmse}")
