# Import necessary libraries
import pandas as pd  # For data manipulation and analysis
import numpy as np  # For numerical computations
from sklearn.model_selection import StratifiedKFold  # For k-fold cross-validation
from sklearn.preprocessing import StandardScaler, LabelEncoder  # For feature scaling and label encoding
from tensorflow.keras.models import Sequential, load_model  # For building and loading neural network models
from tensorflow.keras.layers import Dense, Dropout  # For defining dense and dropout layers
from tensorflow.keras.optimizers import Adam  # For using the Adam optimizer
from sklearn.metrics import classification_report, roc_auc_score  # For model evaluation
import matplotlib.pyplot as plt  # For plotting training history
from flask import Flask, request, jsonify, render_template_string  # For creating the web app
import tensorflow as tf  # For TensorFlow functionalities










# Define column names 
column_names = [
    'GEO Accession #', 'Patient Group', 'Gender', 'IRF5 (ILMN_2312606)', 'SSB (ILMN_2219628)', 
    'TRIM21 (ILMN_1678054)', 'MX1 (ILMN_1662358)', 'IFI44 (ILMN_1760062)', 'IFIT1 (ILMN_1707695)', 
    'CD19 (ILMN_1782704)', 'TNFSF13B (ILMN_2066858)', 'IL6 (ILMN_1699651)', 'CXCL13 (ILMN_1718552)', 
    'FAS (ILMN_2319077)', 'STAT1 (ILMN_1690105)', 'AQP5 (ILMN_1807036)'
]

# Define selected features
selected_features = [
    'Gender', 'IRF5 (ILMN_2312606)', 'SSB (ILMN_2219628)', 'TRIM21 (ILMN_1678054)', 
    'MX1 (ILMN_1662358)', 'IFI44 (ILMN_1760062)'
]

# function for loading and preprocessing the data
def load_and_preprocess_data(file_path):
    """
    Load the dataset, preprocess it, and return features, labels, scaler, and label encoder.
    - Handles missing values by filling them with 0.
    - Encodes categorical variables (e.g., Gender).
    - Normalizes features using StandardScaler.
    """
    # loads dataset
    data = pd.read_csv(file_path)
    
    # assigns column names from the dataset
    data.columns = column_names
    
    # encodes gender, males as 0 and females as 1
    le = LabelEncoder()
    data['Gender'] = le.fit_transform(data['Gender'].astype(str))  # Ensure encoding works correctly
    
    # extract selected features and convert them to numeric values, filling missing values with 0
    features = data[selected_features].apply(pd.to_numeric, errors='coerce').fillna(0)
    
    # extract labels, encode sjogren's syndrome as 1 and 0 for others
    labels = data['Patient Group'].apply(lambda x: 1 if 'Sjogren' in x else 0)
    
    # normalizes features by using keras standardscaler, helps with calculations
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    return features_scaled, labels.values, scaler, le

# Step 2: Build the neural network model
def create_model(input_dim, learning_rate=0.001):
    """
    Neural network model that can be chnaged to speicfications
    L2 regularization to stop overfitting
    gives an output between 0 and 1 for probability of having sjogren's syndrome
    """
    model = Sequential([
        # hidden layer 1, has relu activation function, l2 regularizer with 64 nodes/neurons
        Dense(64, activation='relu', input_dim=input_dim, kernel_regularizer='l2'),
        # droupout layer randomly deactivates 30% of nodes/neurons
        Dropout(0.3),
        # 32 neurons in second layer, relu activation, l2 regularizer
        Dense(32, activation='relu', kernel_regularizer='l2'),
        # output layer with only 1 neuron, between 0 and 1 sigmoid function
        Dense(1, activation='sigmoid')
    ])
    
    # Compile the model using Adam optimizer and binary cross-entropy loss, twype of loss function used for this project
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

# Step 3: Train the model with cross-validation
def train_with_cross_validation(X, y, model_builder, epochs=100, batch_size=32, k=5):
    """
    k-fold validation is a way to train a neural network, like in this case
    Uses the area under the ROC curve (AUC) to track the best model made
    """
    # Starts utilizing the k-fold validation
    kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    scores = []  # stores AUC values for each k-fold
    best_model = None  # gets the best model by using AUC of ROC curve
    best_auc = 0  # tracks the best AUC curve for reference/comparision
    
    # iterates over each k-fold
    for train_index, val_index in kf.split(X, y):
        # to perform k-fold analysis, splits into training and evaluation
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        
        # builds a model for the current fold
        model = model_builder(X_train.shape[1])
        
        # trains the model based on the data
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,  
            validation_data=(X_val, y_val)
        )
        
        # predicts on the validation set, which will be used to evaluate the model
        val_preds = model.predict(X_val)
        
        # calculates the AUC score
        if len(set(y_val)) > 1:
            auc = roc_auc_score(y_val, val_preds)
            scores.append(auc)
            
            # Higher auc is better, so if the auc is higher, then it sets the best model to the new model
            if auc > best_auc:
                best_auc = auc
                best_model = model
    
    # Return the average AUC score for reference and the best model
    return np.mean(scores) if scores else 0, best_model if best_model else model_builder(X.shape[1])

# Flask web app
app = Flask(__name__)  # Initialize the Flask app

@app.route('/')
def home():
    """
    makes a form to track user input, styles the website
    """
    return render_template_string('''
    <html><head><title>Sjogren's Syndrome Risk Calculator</title>
    <style>
        body { font-family: 'Arial', sans-serif; text-align: center; background: #eef2f3; padding: 20px; }
        .container { max-width: 500px; margin: auto; background: white; padding: 20px; border-radius: 10px; box-shadow: 0px 0px 15px rgba(0, 0, 0, 0.2); }
        input, select { margin: 5px; padding: 10px; width: 100%; border-radius: 5px; border: 1px solid #ccc; }
        button { background: #3498db; color: white; padding: 10px; border: none; border-radius: 5px; cursor: pointer; width: 100%; }
        button:hover { background: #2980b9; }
    </style></head><body>
    <div class="container">
    <h1>Sjogren's Syndrome Risk Calculator</h1>
    <form action="/predict" method="post">
    <label>Gender:
        <select name="Gender">
            <option value="0">Male</option>
            <option value="1">Female</option>
        </select>
    </label><br>
    <label>IRF5 (ILMN_2312606): <input type="text" name="IRF5 (ILMN_2312606)"></label><br>
    <label>SSB (ILMN_2219628): <input type="text" name="SSB (ILMN_2219628)"></label><br>
    <label>TRIM21 (ILMN_1678054): <input type="text" name="TRIM21 (ILMN_1678054)"></label><br>
    <label>MX1 (ILMN_1662358): <input type="text" name="MX1 (ILMN_1662358)"></label><br>
    <label>IFI44 (ILMN_1760062): <input type="text" name="IFI44 (ILMN_1760062)"></label><br>
    <button type="submit">Calculate Risk</button>
    </form>
    </div>
    </body></html>
    ''')

@app.route('/predict', methods=['POST'])
def predict():
    """
    handles form submission, preprocess input, and makes predictions
    """
    try:
        # Get user input from the form
        input_data = [
            int(request.form['Gender']),
            float(request.form['IRF5 (ILMN_2312606)']),
            float(request.form['SSB (ILMN_2219628)']),
            float(request.form['TRIM21 (ILMN_1678054)']),
            float(request.form['MX1 (ILMN_1662358)']),
            float(request.form['IFI44 (ILMN_1760062)'])
        ]
        
        # Scale the input data using the pre-trained scaler
        input_data_scaled = scaler.transform([input_data])
        
        # Make a prediction using the pre-trained model
        prediction = model.predict(input_data_scaled)[0][0] * 100  # Convert to percentage
        
        # shows the results, styled well
        return render_template_string('''
        <html><head><title>Prediction Result</title>
        <style>
            body { font-family: 'Arial', sans-serif; text-align: center; background: #eef2f3; padding: 20px; }
            .container { max-width: 500px; margin: auto; background: white; padding: 20px; border-radius: 10px; box-shadow: 0px 0px 15px rgba(0, 0, 0, 0.2); }
            a { color: #3498db; text-decoration: none; }
            a:hover { text-decoration: underline; }
        </style></head><body>
        <div class="container">
        <h1>Prediction Result</h1>
        <p><strong>Risk Percentage:</strong> {{ prediction }}%</p>
        <a href="/">Back to Home</a>
        </div>
        </body></html>
        ''', prediction=f"{prediction:.2f}")
    
    except Exception as e:
        # error handling, if the input is not correct format
        return render_template_string('''
        <html><head><title>Error</title>
        <style>
            body { font-family: 'Arial', sans-serif; text-align: center; background: #eef2f3; padding: 20px; }
            .container { max-width: 500px; margin: auto; background: white; padding: 20px; border-radius: 10px; box-shadow: 0px 0px 15px rgba(0, 0, 0, 0.2); }
            a { color: #3498db; text-decoration: none; }
            a:hover { text-decoration: underline; }
        </style></head><body>
        <div class="container">
        <h1>Error</h1>
        <p>{{ error }}</p>
        <a href="/">Back to Home</a>
        </div>
        </body></html>
        ''', error=str(e))

# main execution
if __name__ == '__main__':
    # loads and preprocesses the data using function above
    file_path = '/Users/jaybhatia/Documents/SS Project/SSDataset2.csv'
    features_scaled, labels, scaler, le = load_and_preprocess_data(file_path)
    
    #  trains the model with the function abvoe
    avg_auc, model = train_with_cross_validation(features_scaled, labels, create_model, epochs=100, batch_size=32, k=5)
    print(f"Cross-validated AUC: {avg_auc:.2f}")
    
    # saves the model, required step to ensure that it will keep running
    if model:
        model.save('sjogrens_risk_model.h5')
    
    # runs the flask app, standard
    app.run(debug=True)