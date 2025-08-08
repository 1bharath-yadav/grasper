
import pandas as pd
import matplotlib.pyplot as plt
import base64
from io import BytesIO

# Load the data
df = pd.read_csv("/home/archer/projects/grasper/tmp/21b236ab84/sample-weather.csv")

# Convert date to datetime objects
df["date"] = pd.to_datetime(df["date"])

# 1. Average temperature in Celsius
average_temp_c = df["temperature_c"].mean()

# 2. Date with highest precipitation
max_precip_date = df.loc[df["precip_mm"].idxmax(), "date"].strftime("%Y-%m-%d")

# 3. Minimum temperature recorded
min_temp_c = df["temperature_c"].min()

# 4. Correlation between temperature and precipitation
temp_precip_correlation = df["temperature_c"].corr(df["precip_mm"])

# 5. Average precipitation in millimeters
average_precip_mm = df["precip_mm"].mean()

# 6. Plot temperature over time as a line chart (red line), base64 PNG
plt.figure(figsize=(10, 6))
plt.plot(df["date"], df["temperature_c"], color="red")
plt.title("Temperature Over Time")
plt.xlabel("Date")
plt.ylabel("Temperature (°C)")
plt.grid(True)
buf = BytesIO()
plt.savefig(buf, format="png", bbox_inches="tight", dpi=150)
buf.seek(0)
temp_line_chart = base64.b64encode(buf.read()).decode("utf-8")
plt.close()

# 7. Plot precipitation as a histogram (orange bars), base64 PNG
plt.figure(figsize=(10, 6))
plt.hist(df["precip_mm"], bins=5, color="orange", edgecolor="black")
plt.title("Precipitation Histogram")
plt.xlabel("Precipitation (mm)")
plt.ylabel("Frequency")
plt.grid(axis="y", alpha=0.75)
buf = BytesIO()
plt.savefig(buf, format="png", bbox_inches="tight", dpi=150)
buf.seek(0)
precip_histogram = base64.b64encode(buf.read()).decode("utf-8")
plt.close()

# Print the results in the specified JSON format
print({
    "average_temp_c": average_temp_c,
    "max_precip_date": max_precip_date,
    "min_temp_c": min_temp_c,
    "temp_precip_correlation": temp_precip_correlation,
    "average_precip_mm": average_precip_mm,
    "temp_line_chart": temp_line_chart,
    "precip_histogram": precip_histogram
})
