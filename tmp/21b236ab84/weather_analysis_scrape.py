import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

# Load the dataset
df = pd.read_csv("/home/archer/projects/grasper/tmp/21b236ab84/sample-weather.csv")

# Convert 'Date' column to datetime objects
df['Date'] = pd.to_datetime(df['Date'])

# Calculate average temperature in Celsius
average_temp_c = df['Temp_C'].mean()

# Find the date with the highest precipitation
max_precip_date = df.loc[df['Precip_mm'].idxmax(), 'Date'].strftime('%Y-%m-%d')

# Find the minimum temperature recorded
min_temp_c = df['Temp_C'].min()

# Calculate the correlation between temperature and precipitation
temp_precip_correlation = df['Temp_C'].corr(df['Precip_mm'])

# Calculate the average precipitation in millimeters
average_precip_mm = df['Precip_mm'].mean()

# Plot temperature over time as a line chart
plt.figure(figsize=(10, 6))
sns.lineplot(x='Date', y='Temp_C', data=df, color='red')
plt.title('Temperature Over Time')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.tight_layout()

# Save temperature line chart to a base64 string
buf_temp_line = io.BytesIO()
plt.savefig(buf_temp_line, format='png', bbox_inches='tight')
plt.close()
temp_line_chart = base64.b64encode(buf_temp_line.getvalue()).decode('utf-8')

# Plot precipitation as a histogram
plt.figure(figsize=(10, 6))
sns.histplot(df['Precip_mm'], bins=20, color='orange', kde=False)
plt.title('Precipitation Distribution')
plt.xlabel('Precipitation (mm)')
plt.ylabel('Frequency')
plt.tight_layout()

# Save precipitation histogram to a base64 string
buf_precip_hist = io.BytesIO()
plt.savefig(buf_precip_hist, format='png', bbox_inches='tight')
plt.close()
precip_histogram = base64.b64encode(buf_precip_hist.getvalue()).decode('utf-8')

# Construct the JSON object
result = {
    "average_temp_c": average_temp_c,
    "max_precip_date": max_precip_date,
    "min_temp_c": min_temp_c,
    "temp_precip_correlation": temp_precip_correlation,
    "average_precip_mm": average_precip_mm,
    "temp_line_chart": temp_line_chart,
    "precip_histogram": precip_histogram
}

print(result)