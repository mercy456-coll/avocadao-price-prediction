# Avocado Price Prediction & Type Classification

## Project Overview

This is a group machine learning project that explores the Avocado dataset to answer two key questions:
1. Can we **predict avocado prices** using regression and time series models?
2. Can we **classify avocado type** (Organic vs Conventional) using logistic regression and PCA?

The project covers the full machine learning workflow — from data cleaning and feature engineering to model building, evaluation, and interpretation.

---

## Group Members

- Esther Akinboye
- Mercy Inameti
- Victoria Eghonghon Eronlan
- Khadijat Oyedeji
- Nkiru Johanna Ezedinma
- Sarah David
- Adaeze Chime

---

## Dataset

The dataset contains historical avocado sales data across different US regions (2015–2018), including:

| Feature | Description |
|--------|-------------|
| `Date` | Date of observation |
| `AveragePrice` | Average price of avocados |
| `Total Volume` | Total number of avocados sold |
| `PLU Codes` | Different avocado size categories (4046, 4225, 4770) |
| `Total Bags` | Total number of bags sold |
| `Type` | Organic or Conventional |
| `Region` | US sales region |
| `Year` | Year of observation |

---

## Project Structure

```
avocado-price-prediction/
│
├── README.md
├── Group_D_Mini_Project_Final.ipynb   # Full notebook with code, visualizations and analysis
└── avocado.csv                        # Dataset
```

---

## Part 1: Price Prediction

### Approach
- Reshaped the dataset from wide to long format to expose hidden relationships
- Engineered new features: `month`, `day`, and `season`
- Built **4 Linear Regression models** progressively, adding more features each time
- Built **Time Series (AR) models** — compared AR(1) vs AR(2)

### Key Finding
The **AR(1) time series model** outperformed all linear regression models, achieving a lower RMSE (Root Mean Squared Error). Linear regression was limited by the weak and scattered relationship between volume and price.

---

## Part 2: Type Classification (Organic vs Conventional)

### Approach
- **Option A:** Logistic Regression with manual feature selection (Region, Volume, Season)
- **Option B:** PCA (dimensionality reduction) + Logistic Regression

### Key Finding
**Logistic Regression with manual feature selection** outperformed the PCA model, achieving **92% accuracy** on both training and test data — confirming the model generalises well with no overfitting.

> Interesting insight: 1,648 conventional avocados were misclassified as organic. This reflects real-world market behaviour — in high-demand regions or peak seasons, conventional avocados can reach prices similar to organic ones, making them statistically hard to distinguish.

---

## Key Drivers of Avocado Prices

- **Type** — Organic avocados consistently command higher prices
- **Supply (Total Volume)** — Higher supply generally lowers prices
- **Seasonality** — Fall has the highest average prices; Winter the lowest
- **Region** — Location has a strong influence on price, often overriding type

---

## Tools & Libraries

- Python 3
- `pandas`, `numpy` — data manipulation
- `plotnine`, `matplotlib`, `seaborn` — visualizations
- `scikit-learn` — machine learning models
- `statsmodels` — time series analysis
- `janitor` — data cleaning

---

## How to Run

1. Clone this repository or download the files
2. Install the required libraries:
   ```bash
   pip install pandas numpy matplotlib seaborn plotnine scikit-learn statsmodels pyjanitor
   ```
3. Open the notebook:
   ```bash
   jupyter notebook Group_D_Mini_Project_Final.ipynb
   ```
4. Run all cells from top to bottom

> **Note:** Make sure `avocado.csv` is in the **same folder** as the notebook before running.
