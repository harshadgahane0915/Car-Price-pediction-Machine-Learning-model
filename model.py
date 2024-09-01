# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib 
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Loading the dataset
data = pd.read_csv('dataset.csv', na_values='null ')

# Selecting relevant features for prediction
features = ['Kilometers_Driven_Scaled', 'mileage', 'Age', 'engine_size', 'horsepower', 'Seats', 'Fuel_Type', 'Transmission', 'Owner_Type']

# Dropping rows containing null values
data.dropna(inplace=True)

# Assuming 'current_year' is defined
current_year = 2023

# Create 'Age' feature
data['Age'] = current_year - data['Year']
data.drop(['Year'], axis=1, inplace=True)

# Scale 'Kilometers_Driven'
scaler = StandardScaler()
data['Kilometers_Driven_Scaled'] = scaler.fit_transform(data[['Kilometers_Driven']])
joblib.dump(scaler, 'fitted_scaler.pkl')

# Label encode 'Fuel_Type'
fuel_type_mapping = {'CNG': 1, 'Diesel': 2, 'Petrol': 3}
data['Fuel_Type'] = data['Fuel_Type'].map(fuel_type_mapping)

# Convert 'Transmission' to binary (1 for Manual, 0 for Automatic)
data['Transmission'] = (data['Transmission'] == 'Manual').astype(int)

# Map 'Owner_Type' to numerical values
owner_type_mapping = {'First': 1, 'Second': 2, 'Third': 3, 'Fourth & Above': 4}
data['Owner_Type'] = data['Owner_Type'].map(owner_type_mapping)


# Extract features and target variable
X = data[features]
y = data['Price']

imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Regressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Save the model to a file
joblib.dump(model, 'car_price_model.pkl')

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
