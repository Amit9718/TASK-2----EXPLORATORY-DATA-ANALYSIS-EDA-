# Retail Sales - Exploratory Data Analysis

A Python EDA project that generates a synthetic retail sales dataset and analyzes it using descriptive statistics, outlier detection, trend analysis, and visualizations.

## Project Structure

```
.
├── generate_dataset.py   # Generates retail_sales.csv
├── eda_analysis.py       # Full EDA: stats, charts, report
├── requirements.txt      # Python dependencies
├── retail_sales.csv      # Generated dataset (1,000 rows)
├── eda_charts.png        # 8-panel visualization output
└── eda_report.txt        # Written summary report
```

## Dataset

1,000 synthetic retail transactions across 2023 with 10 features:

| Column | Description |
|---|---|
| `date` | Transaction date (2023) |
| `region` | North, South, East, West |
| `category` | Electronics, Clothing, Groceries, Furniture, Sports |
| `channel` | Online, In-Store, Wholesale |
| `unit_price` | Price per unit ($) |
| `quantity` | Units purchased |
| `discount_pct` | Discount applied (%) |
| `revenue` | Final revenue after discount |
| `satisfaction` | Customer satisfaction score (1-10) |
| `delivery_days` | Days to deliver (0 for In-Store) |

## Setup

```bash
pip install -r requirements.txt
```

## Usage

Run in order:

```bash
# 1. Generate the dataset
python generate_dataset.py

# 2. Run the full EDA
python eda_analysis.py
```

Outputs: `retail_sales.csv`, `eda_charts.png`, `eda_report.txt`

## Analysis Covers

- **Descriptive statistics** — mean, median, std, skewness, kurtosis for all numeric features
- **Distributions** — revenue histogram, category box plots
- **Trends** — monthly revenue bar chart, channel revenue share
- **Correlations** — heatmap across all numeric variables
- **Outlier detection** — IQR method and Z-score (>3) for revenue, price, and quantity
- **Business insights** — top category, best channel, regional performance, discount impact

## Key Findings

- Electronics drives the most total revenue ($238K); Furniture has the highest avg. order ($1,408)
- Groceries has the most transactions (303) but lowest avg. value ($60) — upsell opportunity
- Online channel leads avg. order value ($538 vs $423 In-Store)
- Discount positively correlates with satisfaction (r ~ 0.30)
- 107 revenue outliers detected — likely bulk orders or data entry errors
- Wholesale delivery averages ~7 days vs ~4 days for Online
