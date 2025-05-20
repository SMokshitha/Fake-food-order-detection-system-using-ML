BACKEND 
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
# Assuming you have a pandas DataFrame df loaded from your dataset
# Example:
df = pd.read_csv('customer.csv')
# Preprocessing
df['payment_mode'] = df['payment_mode'].astype(str)  # Ensuring it's a string for LabelEncoder

# Use LabelEncoder to encode 'payment_mode'
label_encoder = LabelEncoder()
df['payment_mode'] = label_encoder.fit_transform(df['payment_mode'])
# Define features (X) and target (y)
X = df[['quantity', 'payment_mode', 'Rating']]
y = df['order_status']  # Assuming order_status is the target, 1 for fraudulent, 0 for not

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Train a RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
# Save the model and label encoder
with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)


with open('label_encoder.pkl', 'wb') as le_file:
    pickle.dump(label_encoder, le_file)
print("Model and label encoder saved!")

FLASK
from flask import Flask, render_template, request
import pickle
import numpy as np
import traceback  # For better error logging
app = Flask(_name_)
# Load the model and label encoder
try:
    with open('model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('label_encoder.pkl', 'rb') as le_file:
        label_encoder = pickle.load(le_file)
except Exception as e:
    print(f"Error loading model or label encoder: {e}")
    raise
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract input features from the form
        quantity = int(request.form.get('quantity'))
        payment_mode = request.form.get('payment_mode')
        rating = float(request.form.get('Rating'))  # Using 'Rating' with uppercase R
      

 # Debugging: Print the inputs for checking
        print(f"Received inputs - Quantity: {quantity}, Payment Mode: {payment_mode}, Rating: {Rating}")
  # Encode the payment_mode using the LabelEncoder
        try:
            payment_mode_encoded = label_encoder.transform([payment_mode])[0]
            print(f"Encoded Payment Mode: {payment_mode_encoded}")  # Debugging encoded value
        except Exception as e:
            print(f"Error encoding payment mode: {e}")
            raise ValueError(f"Invalid payment mode: {payment_mode}")
         # Prepare the input for the model (reshape for single sample)
        # Ensure the order of features matches your training data
        final_input = np.array([quantity, payment_mode_encoded, rating]).reshape(1, -1)
        # Debugging: Print the input array for prediction
        print(f"Final input for model prediction: {final_input}")

        # Predict the fraud probability
        prediction = model.predict_proba(final_input)
        fraud_probability = prediction[0][1]
        # Format the result for display
        output = '{0:.{1}f}'.format(fraud_probability, 2)
        # Generate result message based on probability
        if fraud_probability > 0.5:
            result_message = f"Warning: High likelihood of fraud! Fraud probability is {output}."
            advice = "Consider verifying this order."
        else:
            result_message = f"Low likelihood of fraud. Fraud probability is {output}."
            advice = "The order appears to be legitimate."


        # Render the result template
        return render_template('index.html', pred=result_message, advice=advice)
    except Exception as e:
        # Log the exception with traceback for debugging
        print(f"Error occurred: {e}")
        print("Full error traceback:", traceback.format_exc())  # Display the full error traceback
        return render_template('index.html', pred="An error occurred while processing your input.", advice="Please check your inputs and try again.")

if _name_ == "_main_":
    app.run(debug=True)
