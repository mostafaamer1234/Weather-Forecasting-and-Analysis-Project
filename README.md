# Weather Forecasting and Analysis Project

## Project Overview
This project aims to analyze and forecast weather data using ARIMA and LSTM models. It also includes anomaly detection for temperature data, providing insights that can support decision-making.

## Dataset
The dataset used is `GlobalWeatherRepository.csv`, which contains weather metrics like temperature, wind speed, and humidity for various countries and cities.

## Methodology
1. **Data Cleaning**: Handled missing values and outliers.
2. **Exploratory Data Analysis**: Conducted time-series analysis and correlation analysis.
3. **Forecasting Models**: Built ARIMA and LSTM models for temperature prediction.
4. **Anomaly Detection**: Detected anomalies using z-score analysis.

## Results
- ARIMA Model MAE: `x`
- ARIMA Model RMSE: `y`
- LSTM Model RMSE: `z`

## Insights
- Seasonal Patterns: The time-series analysis revealed clear seasonal patterns in temperature and precipitation, with higher temperatures observed during summer months.

Correlation Insights: A strong correlation was observed between humidity and temperature, suggesting that higher temperatures tend to coincide with higher humidity levels.

Anomalies: The anomaly detection analysis identified several instances of unusually high temperatures, which could be linked to specific climatic events or data recording errors.

## Getting Started
1. Clone the repository.
2. Install dependencies listed in `requirements.txt`.
3. Run the script using `python main.py`.

## Dependencies
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- tensorflow
- statsmodels

## Contact
For any questions, please reach out at amermostafa.official477@gmail.com.
