from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
scaler = joblib.load('fitted_scaler.pkl')

# Loading our machine learning model
model = joblib.load('car_price_model.pkl')

@app.route('/')
def home():
    return render_template('/web.html')

@app.route('/predict', methods=['POST'])
def predict_car_price():
    try:
        # Reading input values from the JSON data
        data = request.get_json()

        # Checking if required fields are present
        required_fields = ['kilometers', 'fuelType', 'transmission', 'ownerType', 'mileage', 'age', 'engineSize', 'horsepower', 'seats']
        for field in required_fields:
            if field not in data:
                raise ValueError(f'Missing required field: {field}')

        # Converting data types
        mileage = float(data['mileage'])
        age = int(data['age'])
        engine_size = float(data['engineSize'])
        horsepower = float(data['horsepower'])
        
        # Handling categorical variables
        fuel_type_mapping = {'CNG': 1, 'Diesel': 2, 'Petrol': 3}
        Fuel_Type = fuel_type_mapping.get(data.get('fuelType', 'UnknownFuelType'), 0)
        
        Transmission = data.get('transmission', 'UnknownTransmission') == 'Manual'

        
        Kilometers_Driven_Scaled = scaler.transform([[float(data['kilometers'])]])[0, 0]
        
        owner_type_mapping = {'First': 1, 'Second': 2, 'Third': 3, 'Fourth & Above': 4}
        Owner_Type = owner_type_mapping.get(data.get('ownerType', 'UnknownOwnerType'), 0)
        
        Seats = int(data.get('seats', 0))

        # Print statements for debugging
        print("Debugging info:")
        print(f"mileage: {mileage}")
        print(f"age: {age}")
        print(f"engine_size: {engine_size}")
        print(f"horsepower: {horsepower}")
        print(f"Fuel_Type: {Fuel_Type}")
        print(f"Transmission: {Transmission}")
        print(f"Kilometers_Driven_Scaled: {Kilometers_Driven_Scaled}")
        print(f"Owner_Type: {Owner_Type}")
        print(f"Seats: {Seats}")

        # Preparing input data for prediction
        input_data = np.array([[Kilometers_Driven_Scaled, mileage, age, engine_size, horsepower, Seats, Fuel_Type, Transmission, Owner_Type]])

        # Making prediction
        predicted_price = model.predict(input_data)

        # Returning the predicted price as a JSON response
        return jsonify({'predicted_price': float(predicted_price[0])})

    except Exception as e:
        print(f"Exception details: {e}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
