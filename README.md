🏠 **House Price Prediction using Linear Regression**
This project builds a linear regression model to predict house prices using features like square footage and number of bathrooms, based on a dataset with prices in Indian currency (₹) from a CSV file.

📌 **Project Overview**
🎯 **Goal:** Predict housing prices from simple numerical features
📊 **Model Used:** LinearRegression from scikit-learn

🧹 **Data Cleaning:**

- Converted strings like **"1.2 Cr"**, **"50 Lac"** into numeric values
- Extracted square footage from Carpet Area
- Cleaned Bathroom column and dropped NaNs

🔍 **Steps Performed**
1. Preprocessed the dataset for numerical analysis
2. Split data into training and testing sets (80/20)
3. Trained a linear regression model
4. Evaluated model using:
    - Mean Squared Error (MSE)
    - R² Score
5. Predicted the price of a sample house with given sqft and bathroom count
