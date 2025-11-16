import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Step 1: Load Dataset
data = pd.read_csv("house_data.csv")

# Step 2: Select Features & Target
X = data[['area', 'bedrooms', 'bathrooms', 'age']]
y = data['price']

# Step 3: Split Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Create Model
model = LinearRegression()

# Step 5: Train Model
model.fit(X_train, y_train)

# Step 6: Test Model
predictions = model.predict(X_test)
error = mean_squared_error(y_test, predictions)

print("Model Trained Successfully!")
print("Mean Squared Error:", error)

# ----------- Prediction Section -----------
print("\n--- Predict House Price ---")

area = float(input("Enter area (sq ft): "))
bedrooms = int(input("Enter number of bedrooms: "))
bathrooms = int(input("Enter number of bathrooms: "))
age = int(input("Enter age of the house: "))

# FIX: Create DataFrame with correct column names
new_data = pd.DataFrame([[area, bedrooms, bathrooms, age]],
                        columns=['area', 'bedrooms', 'bathrooms', 'age'])

predicted_price = model.predict(new_data)[0]

print(f"\nEstimated House Price: ₹{predicted_price:.2f}")
