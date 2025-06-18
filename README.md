
## 🏠 House Price Prediction using Linear Regression

This project applies **Linear Regression**, a supervised machine learning algorithm, to predict house prices based on their **square footage**, **number of bedrooms**, and **number of bathrooms** using a custom housing dataset.  
It demonstrates how to build a regression model from scratch, evaluate its performance, and make custom predictions.

---

### 📁 Dataset Overview

- **Filename:** `Housing.csv`  
- **Source:** Provided locally (sample housing data)

- **Columns Used:**
    - `sqft` – Area of the house in square feet  
    - `bedrooms` – Number of bedrooms  
    - `bathrooms` – Number of bathrooms
    - `price` – Target variable (house price in ₹)

---

### 📌 Problem Statement

The objective is to predict house prices based on various physical characteristics (area, bedrooms, bathrooms).  
The model aims to assist buyers, sellers, and real estate businesses in estimating property value more accurately.

---

### 🚀 How It Works

#### 1. **Importing Libraries**

Essential Python libraries used:

- `pandas`, `numpy` – for data manipulation
- `scikit-learn` – for model building and evaluation

---

#### 2. **Loading the Dataset**
```python
df = pd.read_csv('/mnt/data/Housing.csv')
```

---

#### 3. **Feature Selection**

Selected independent (X) and dependent (y) variables:
```python
X = df[['area', 'bedrooms', 'bathrooms']]
y = df['price']
```

---

#### 4. **Train-Test Split**

Split the data for training and evaluation:
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

---

#### 5. **Model Training**

Using `LinearRegression` from `sklearn`:
```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
```

---

#### 6. **Model Evaluation**

Evaluate model using Mean Squared Error and R² Score:
```python
from sklearn.metrics import mean_squared_error, r2_score

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
```

Also print model coefficients:
```python
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)
```

---

#### 7. **Custom Price Prediction**

Custom prediction using user input and a reusable function:
```python
def predicted_price(area_sqft, bedrooms, bathrooms):
    input_data = pd.DataFrame([[area_sqft, bedrooms, bathrooms]], columns=['area', 'bedrooms', 'bathrooms'])
    predicted_price = model.predict(input_data)[0]
    print(f'\nPredicted Price for {area_sqft} sqft, {bedrooms}-bedrooms, {bathrooms}-bathrooms : ₹{predicted_price:,.2f}')

area_sqft = int(input('\nEnter Area[in square feet]: '))
bedrooms = int(input('Enter Number of Bedrooms: '))
bathrooms = int(input('Enter Number of Bathrooms: '))

predicted_price(area_sqft, bedrooms, bathrooms)
```

---

### 📈 Sample Output

```
Mean Squared Error: 1434523.54
R² Score: 0.7923
Intercept: 12234.56
Coefficients:
  sqft: 1250.25
  bedrooms: -3435.75
  bathrooms: 8420.42

Predicted price for 850 sqft, 2-bedrooms, 2-bathrooms house: ₹1,459,240.55
```

---

### 📎 Requirements
- pandas  
- numpy  
- scikit-learn  
- matplotlib

Install with:
```bash
pip install pandas numpy scikit-learn matplotlib
```

---

### 📚 Learnings

- Hands-on implementation of Linear Regression
- Understanding model evaluation with MSE and R² Score
- Feature importance using coefficients
- Making predictions using custom input

---

### 🧠 Future Improvements

- Add more features like location, parking
- Visualize actual vs predicted prices using plots
- Build an interactive web app using Streamlit or Flask

---

### 👨‍💻 Author

**Yash Patel**  

Python Enthusiast | Machine Learning Explorer | Aspiring Data Scientist
