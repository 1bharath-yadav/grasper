import pandas as pd
import matplotlib.pyplot as plt
import base64
from io import BytesIO

# Load the data
df = pd.read_csv('/home/archer/projects/grasper/tmp/215b70bdbc/sample-sales.csv')

# 1. Total sales across all regions
total_sales = df['sales'].sum()

# 2. Region with the highest total sales
top_region = df.groupby('region')['sales'].sum().idxmax()

# 3. Correlation between day of month and sales
df['date'] = pd.to_datetime(df['date'])
df['day_of_month'] = df['date'].dt.day
day_sales_correlation = df['day_of_month'].corr(df['sales'])

# 4. Plot total sales by region as a bar chart (base64 PNG)
sales_by_region = df.groupby('region')['sales'].sum()
plt.figure(figsize=(8, 6))
plt.bar(sales_by_region.index, sales_by_region.values, color='blue')
plt.xlabel('Region')
plt.ylabel('Total Sales')
plt.title('Total Sales by Region')
plt.grid(axis='y', linestyle='--')
buf_bar = BytesIO()
plt.savefig(buf_bar, format='png', bbox_inches='tight', dpi=150)
buf_bar.seek(0)
bar_chart_base64 = base64.b64encode(buf_bar.read()).decode('utf-8')
plt.close()

# 5. Median sales amount across all orders
median_sales = df['sales'].median()

# 6. Total sales tax if the tax rate is 10%
total_sales_tax = total_sales * 0.10

# 7. Plot cumulative sales over time as a line chart (base64 PNG)
df_sorted = df.sort_values(by='date')
df_sorted['cumulative_sales'] = df_sorted['sales'].cumsum()

plt.figure(figsize=(10, 6))
plt.plot(df_sorted['date'], df_sorted['cumulative_sales'], color='red')
plt.xlabel('Date')
plt.ylabel('Cumulative Sales')
plt.title('Cumulative Sales Over Time')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
buf_line = BytesIO()
plt.savefig(buf_line, format='png', bbox_inches='tight', dpi=150)
buf_line.seek(0)
cumulative_sales_chart_base64 = base64.b64encode(buf_line.read()).decode('utf-8')
plt.close()

# Convert numpy types to native Python types
result = {
    'total_sales': float(total_sales),
    'top_region': str(top_region),
    'day_sales_correlation': float(day_sales_correlation) if pd.notna(day_sales_correlation) else None,
    'bar_chart': bar_chart_base64,
    'median_sales': float(median_sales),
    'total_sales_tax': float(total_sales_tax),
    'cumulative_sales_chart': cumulative_sales_chart_base64
}

print(result)