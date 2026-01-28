# House Price Prediction using Linear Regression

This project builds a **complete end-to-end regression pipeline** to predict house prices using **Linear Regression**. It includes data preprocessing, exploratory data analysis (EDA), model training, evaluation, and error analysis.

---

## 📌 Project Objective

To predict **house prices** based on property features such as area, location, bedrooms, and amenities, and to evaluate how well a linear regression model explains price variation.

This project is designed as a **strong Data Science portfolio project**.

---

## 🛠️ Tools & Libraries

* **Python**
* **Pandas** – data loading and preprocessing
* **Matplotlib** – visualization
* **Seaborn** – statistical plots
* **Scikit-learn** – machine learning model and evaluation

---

## 📂 Dataset

**Input File:** `house_price_prediction_.csv`

### Key Columns

* `area_sqft`
* `bedrooms`
* `bathrooms`
* `floors`
* `year_built`
* `location` (categorical)
* `has_garage` (categorical)
* `price` (target variable)

---

## 🔄 Project Workflow

### 1. Data Cleaning

* Removed unnecessary `id` column
* Handled categorical variables using **Label Encoding**

---

### 2. Exploratory Data Analysis (EDA)

* **Correlation Heatmap** to understand feature relationships
* **Scatter plots** for numerical features vs price
* **Box plots** to analyze price distribution by:

  * Location
  * Garage availability

---

### 3. Feature & Target Selection

* Features (`X`): all columns except `price`
* Target (`y`): house price

---

### 4. Model Training

* Algorithm: **Linear Regression**
* Train-test split: 80% training, 20% testing
* Random state fixed for reproducibility

---

### 5. Model Evaluation

* Metric used: **R² Score**
* Performance displayed as percentage

Example:

```
Linear Regression R² Score: XX.XX%
```

---

### 6. Model Diagnostics

* **Actual vs Predicted Price** scatter plot
* **Residual distribution** to check error behavior

---

## 📈 Visual Outputs

* Correlation heatmap
* Feature vs price scatter plots
* Price distribution box plots
* Actual vs predicted price comparison
* Residual error distribution

---

## 🚀 How to Run

1. Clone the repository
2. Install dependencies:

   ```bash
   pip install pandas matplotlib seaborn scikit-learn
   ```
3. Place `house_price_prediction_.csv` in the project directory
4. Run the Python script

---

## 📌 Use Cases

* Real estate price estimation
* Feature impact analysis on housing prices
* Baseline regression model for advanced ML
* Academic and portfolio demonstration

---

## 👤 Author

**Khubaib**
Aspiring AI Engineer | Machine Learning & Predictive Modeling

---

## ⭐ Notes

* Linear Regression assumes linear relationships
* Label Encoding is used for simplicity; One-Hot Encoding may improve performance
* Model can be extended using Regularization or Tree-based algorithms

---

If this project helped you, feel free to ⭐ the repository and build upon it for more advanced models.
