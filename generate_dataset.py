"""
Generate a synthetic retail sales dataset with realistic patterns,
including trends, distributions, correlations, and outliers.
"""
import numpy as np
import pandas as pd

np.random.seed(42)
N = 1000

regions = ["North", "South", "East", "West"]
categories = ["Electronics", "Clothing", "Groceries", "Furniture", "Sports"]
channels = ["Online", "In-Store", "Wholesale"]

region_arr = np.random.choice(regions, N)
category_arr = np.random.choice(categories, N, p=[0.25, 0.20, 0.30, 0.10, 0.15])
channel_arr = np.random.choice(channels, N, p=[0.45, 0.40, 0.15])

# Unit price varies by category
price_map = {
    "Electronics": (200, 80),
    "Clothing":    (45,  20),
    "Groceries":   (15,   8),
    "Furniture":   (350, 120),
    "Sports":      (60,  25),
}
unit_price = np.array([
    abs(np.random.normal(price_map[c][0], price_map[c][1]))
    for c in category_arr
])

# Quantity: mostly 1-5, right-skewed
quantity = np.random.gamma(shape=1.5, scale=2, size=N).astype(int) + 1

# Discount: 0-30%, higher for Wholesale
discount_base = np.random.beta(2, 8, N) * 0.30
discount = np.where(channel_arr == "Wholesale", discount_base * 1.5, discount_base)
discount = np.clip(discount, 0, 0.50)

# Revenue
revenue = unit_price * quantity * (1 - discount)

# Inject ~2% outliers (bulk orders / data errors)
outlier_idx = np.random.choice(N, size=20, replace=False)
revenue[outlier_idx] *= np.random.uniform(5, 15, size=20)

# Customer satisfaction score 1-10, correlated with discount
satisfaction = np.clip(
    6 + discount * 10 + np.random.normal(0, 1.2, N), 1, 10
).round(1)

# Days to deliver: Online slower than In-Store pickup
delivery_days = np.where(
    channel_arr == "In-Store", 0,
    np.where(channel_arr == "Online",
             np.random.poisson(4, N),
             np.random.poisson(7, N))
)

# Date: 2023 full year
dates = pd.to_datetime(
    np.random.choice(pd.date_range("2023-01-01", "2023-12-31"), N)
)

df = pd.DataFrame({
    "date":           dates,
    "region":         region_arr,
    "category":       category_arr,
    "channel":        channel_arr,
    "unit_price":     unit_price.round(2),
    "quantity":       quantity,
    "discount_pct":   (discount * 100).round(1),
    "revenue":        revenue.round(2),
    "satisfaction":   satisfaction,
    "delivery_days":  delivery_days,
})

df.to_csv("retail_sales.csv", index=False)
print(f"Dataset saved: {df.shape[0]} rows x {df.shape[1]} columns")
print(df.dtypes)
