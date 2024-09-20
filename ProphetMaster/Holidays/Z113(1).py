import pandas as pd
import matplotlib.pyplot as plt
import logging
from prophet import Prophet

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

#Z-016
# Load data from Z113.csv
file_path = 'Z113.csv'
df = pd.read_csv(file_path)

# Convert the 'ДАТА' column to datetime format
df['ДАТА'] = pd.to_datetime(df['ДАТА'], format='%d.%m.%Y')

# Adjust the 'HOUR' column where HOUR is 24
df.loc[df['HOUR'] == 24, 'HOUR'] = 0
df.loc[df['HOUR'] == 0, 'ДАТА'] = df['ДАТА'] + pd.Timedelta(days=1)

# Filter data for specific product and tank
filtered_df = df[(df['PRODNAME'] == 3300000002) & (df['TANKNUM'] == 4)].copy()

# Check for missing values in 'КОЛИЧЕСТВО' and fill them with zeros
if filtered_df['КОЛИЧЕСТВО'].isnull().any():
    logging.warning("Есть пустые значения в столбце 'КОЛИЧЕСТВО'. Заполняем нулями.")
    filtered_df['КОЛИЧЕСТВО'] = filtered_df['КОЛИЧЕСТВО'].fillna(0)

# Create a datetime column combining date and hour
filtered_df['ds'] = pd.to_datetime(filtered_df['ДАТА'].astype(str) + ' ' + filtered_df['HOUR'].astype(str) + ':00:00')

# Generate a complete range of dates and hours
start_date = filtered_df['ds'].min().normalize()
end_date = filtered_df['ds'].max().normalize() + pd.Timedelta(days=1)  # Ensure the last day is included
complete_index = pd.date_range(start=start_date, end=end_date, freq='h')

# Remove or aggregate duplicate 'ds' values before reindexing
filtered_df = filtered_df.groupby('ds', as_index=False).agg({'КОЛИЧЕСТВО': 'mean'})

# Now reindex after handling duplicates
filtered_df = filtered_df.set_index('ds').reindex(complete_index).fillna({'КОЛИЧЕСТВО': 0}).reset_index()
filtered_df.rename(columns={'index': 'ds'}, inplace=True)

# Add weekday column for finding similar days
filtered_df['weekday'] = filtered_df['ds'].dt.weekday

# Aggregate data by taking the average of duplicates for each day and hour
filtered_df = filtered_df.groupby(['ds', 'weekday']).agg({'КОЛИЧЕСТВО': 'mean'}).reset_index()

# Prepare historical data up to a specific date
historical_df = filtered_df[filtered_df['ds'] < '2024-08-31']

# Prepare actual data from a specific date
actual_df = filtered_df[(filtered_df['ds'] >= '2024-08-31') & (filtered_df['ds'] <= '2024-09-01')]

# Function to find similar days and calculate rolling mean for a given weekday and hour
def find_similar_days(target_weekday, target_hour, data, window_size=3):
    # Increased window_size to 10 for better smoothing
    similar_days = data[
        (data['weekday'] == target_weekday) & 
        (data['ds'].dt.hour == target_hour)
    ]
    if not similar_days.empty:
        rolling_mean = similar_days['КОЛИЧЕСТВО'].rolling(window=window_size, min_periods=1).mean().iloc[-1]
    else:
        rolling_mean = 0
    return rolling_mean

# Preparing data for Prophet model with tuned parameters
prophet_df = historical_df[['ds', 'КОЛИЧЕСТВО']].rename(columns={'КОЛИЧЕСТВО': 'y'})

# Fine-tuning the Prophet model
model = Prophet(
    seasonality_mode='multiplicative',  # Changed to multiplicative for better handling of variations
    yearly_seasonality=True,  # Enable yearly seasonality
    weekly_seasonality=True,  # Enable weekly seasonality
    daily_seasonality=True   # Disable daily seasonality
)

# Add holidays or special events to improve the model (if applicable)
# Example: Add known holidays or events to the model
holidays = pd.DataFrame({
    'holiday': 'special_event',
   'ds': pd.to_datetime(['2024-08-30', '2024-08-31']),
   'lower_window': 0,
     'upper_window': 1,
 })
model = Prophet(holidays=holidays)

model.fit(prophet_df)

# Forecasting with Prophet
future = model.make_future_dataframe(periods=48, freq='h')
forecast = model.predict(future)
forecast = forecast[['ds', 'yhat']]

# Forecast for the period from 30 August to 1 September 2024 using hybrid method
forecast_start_date = '2024-08-31 17:00:00'
forecast_end_date = '2024-09-01 23:59:59'
forecast_dates = pd.date_range(start=forecast_start_date, end=forecast_end_date, freq='h')
forecast_values = []

for date in forecast_dates:
    target_weekday = date.weekday()
    target_hour = date.hour
    rolling_mean_value = find_similar_days(target_weekday, target_hour, historical_df)
    prophet_pred = forecast.loc[forecast['ds'] == date, 'yhat'].values[0] if not forecast.loc[forecast['ds'] == date].empty else rolling_mean_value
    
    # Weighted combination of rolling mean and Prophet predictions
    combined_forecast = (0.3 * rolling_mean_value) + (0.6 * prophet_pred)  # Weighted to favor Prophet's predictions more
    forecast_values.append(combined_forecast)

# Create DataFrame for hybrid forecast
forecast_df = pd.DataFrame({'ds': forecast_dates, 'yhat': forecast_values})

# Initial level meter
initial_volume = 20883  
forecast_df['DATE'] = forecast_df['ds'].dt.date

# Forecast level meter and check for dead stock level
forecast_df['уровнемер'] = initial_volume - forecast_df['yhat'].cumsum()
dead_stock = 2973   
max_iterations = 1000
iteration_count = 0

while forecast_df['уровнемер'].iloc[-1] > dead_stock:
    if iteration_count >= max_iterations:
        logging.warning("Reached the maximum number of iterations. Exiting loop to prevent infinite run.")
        break

    next_date = forecast_df['ds'].iloc[-1] + pd.Timedelta(hours=1)
    next_volume = forecast_df['уровнемер'].iloc[-1] - find_similar_days(next_date.weekday(), next_date.hour, historical_df)
    
    next_forecast = pd.DataFrame({
        'ds': [next_date],
        'yhat': [find_similar_days(next_date.weekday(), next_date.hour, historical_df)],
        'уровнемер': [next_volume],
        'DATE': [next_date.date()]
    })
    
    forecast_df = pd.concat([forecast_df, next_forecast], ignore_index=True)
    iteration_count += 1

# Check for dead stock level
below_dead_stock = forecast_df[forecast_df['уровнемер'] <= dead_stock]

if not below_dead_stock.empty:
    first_date_reach_dead_stock = below_dead_stock.iloc[0]
    print(f"Мертвый остаток будет достигнут к: {first_date_reach_dead_stock['ds'].date()} в {first_date_reach_dead_stock['ds'].time()}")
else:
    print("Мертвый остаток не будет достигнут в пределах 3-4 сентября 2024 года.")

# Set index to 'ds' for merging
forecast_df = forecast_df.set_index('ds')
actual_df = actual_df.set_index('ds')

# Merge actual data with forecast for accuracy calculation
merged_df = forecast_df.join(actual_df[['КОЛИЧЕСТВО']], how='left', rsuffix='_actual')

# Function to calculate accuracy
def calculate_simple_accuracy(actual, forecast):
    return (min(actual, forecast) / max(actual, forecast)) * 100 if pd.notnull(actual) and pd.notnull(forecast) else None

# Calculate accuracy for all data
merged_df['accuracy'] = merged_df.apply(lambda row: calculate_simple_accuracy(row['КОЛИЧЕСТВО'], row['yhat']), axis=1)

# Filter data for accuracy calculation for 30-31 August 2024
forecast_days_df = merged_df[(merged_df.index >= '2024-08-31') & (merged_df.index < '2024-09-01')]

# Calculate daily accuracy for 30-31 August
forecast_days_df['daily_accuracy'] = forecast_days_df.apply(
    lambda row: calculate_simple_accuracy(row['КОЛИЧЕСТВО'], row['yhat']), axis=1
)

# Output the average accuracy for 30-31 August
daily_accuracy = forecast_days_df['daily_accuracy'].mean()
print(f"Средняя точность прогноза на 8-10 сентября 2024 года: {daily_accuracy:.2f}%")

# Save forecast and actual data to CSV for comparison
comparison_df = merged_df[['yhat', 'КОЛИЧЕСТВО', 'accuracy']]
comparison_df.to_csv('forecast_dates.csv')

print("CSV файл с прогнозом, фактическими данными и точностью создан: forecast_dates.csv")

# Visualization
plt.figure(figsize=(16, 8))

# Historical data up to 20 August (blue line)
plt.plot(historical_df['ds'], historical_df['КОЛИЧЕСТВО'], label='Фактический объем до 3 сентября', color='blue')

# Hybrid forecast for the period 30-31 August
plt.plot(forecast_df.index, forecast_df['yhat'], label='Гибридный прогноз (скользящее среднее + Prophet)', color='orange')

# Forecast accuracy line
plt.plot(merged_df.index, merged_df['accuracy'], label='Точность прогноза', color='black', linestyle='--')

# Level meter line
plt.plot(forecast_df.index, forecast_df['уровнемер'], label='Уровнемер', color='purple', linestyle='--')

# Dead stock line
plt.axhline(y=dead_stock, color='green', linestyle='--', label='Мертвый остаток (2929 литров)')

# Actual data for 30-31 August
plt.plot(actual_df.index, actual_df['КОЛИЧЕСТВО'], label='Фактический объем за 3-4 сентября', color='green')

# Добавление легенды и заголовка
plt.legend()
plt.title('Прогноз на 3-4 сентября 2024 года с фактическими данными и гибридным методом (скользящее среднее + Prophet)')
plt.xlabel('Дата и время')
plt.ylabel('Объем реализации (литры)')

# Показать график
plt.show()

