# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, render_template_string

# Define column names globally
column_names = [
    'GEO Accession #', 'Patient Group', 'Gender', 'ILMN_2312606', 'ILMN_2219628', 
    'ILMN_1678054', 'ILMN_1662358', 'ILMN_1760062', 'ILMN_1707695', 'ILMN_1782704', 
    'ILMN_2066858', 'ILMN_1699651', 'ILMN_1718552', 'ILMN_2319077', 'ILMN_1690105', 
    'ILMN_1807036'
]

# Define selected features for the model
selected_features = [
    'Gender', 'ILMN_2312606', 'ILMN_2219628', 'ILMN_1678054', 
    'ILMN_1662358', 'ILMN_1760062'
]

# Step 1: Load and preprocess the data
def load_and_preprocess_data(file_path):
    """
    Load the dataset, preprocess it, and return features, labels, scaler, and label encoder.
    """
    # Load the dataset
    try:
        data = pd.read_csv(file_path, sep='\t')  # Use tab separator
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None, None, None, None
    
    # Check if the dataset has the expected number of columns
    if data.shape[1] != len(column_names):
        print(f"Error: Dataset has {data.shape[1]} columns, but {len(column_names)} column names were provided.")
        return None, None, None, None
    
    # Assign column names to the dataset
    data.columns = column_names
    
    # Encode gender (Male: 0, Female: 1)
    le = LabelEncoder()
    data['Gender'] = le.fit_transform(data['Gender'].str.replace('gender: ', ''))
    
    # Extract selected features and labels
    features = data[selected_features].apply(pd.to_numeric, errors='coerce').fillna(0)
    labels = data['Patient Group'].apply(lambda x: 1 if 'Control' in x else 0)  # 1 for Control, 0 for Sjogren's
    
    # Normalize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    return features_scaled, labels.values, scaler, le

# Step 2: Build the neural network model
def create_model(input_dim, learning_rate=0.001):
    """
    Create a neural network model with configurable architecture.
    """
    model = Sequential([
        Dense(64, activation='relu', input_dim=input_dim, kernel_regularizer='l2'),
        Dropout(0.3),
        Dense(32, activation='relu', kernel_regularizer='l2'),
        Dense(1, activation='sigmoid')
    ])
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Step 3: Train the model using the entire dataset
def train_model(X, y):
    """
    Train the model on the entire dataset.
    Uses SMOTE to handle class imbalance.
    """
    # Apply SMOTE to the entire dataset to handle class imbalance
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    
    # Build the model
    model = create_model(X_res.shape[1])
    
    # Train the model
    history = model.fit(
        X_res, y_res,
        epochs=100,
        batch_size=32,
        verbose=1
    )
    
    # Plot training history
    plot_history(history)
    
    return model

# Step 4: Plot training history
def plot_history(history):
    """
    Plot training loss and accuracy over epochs.
    """
    plt.figure(figsize=(12, 6))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.legend()
    plt.title('Loss')
    
    # Plot accuracy
    if 'accuracy' in history.history:
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.legend()
        plt.title('Accuracy')
    
    plt.tight_layout()
    plt.show()

# Step 5: Flask web app
app = Flask(__name__)

@app.route('/')
def home():
    """
    Render the home page with a form for user input.
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
    <label>ILMN_2312606: <input type="text" name="ILMN_2312606"></label><br>
    <label>ILMN_2219628: <input type="text" name="ILMN_2219628"></label><br>
    <label>ILMN_1678054: <input type="text" name="ILMN_1678054"></label><br>
    <label>ILMN_1662358: <input type="text" name="ILMN_1662358"></label><br>
    <label>ILMN_1760062: <input type="text" name="ILMN_1760062"></label><br>
    <button type="submit">Calculate Risk</button>
    </form>
    </div>
    </body></html>
    ''')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle form submission, preprocess input, and make predictions.
    """
    try:
        # Get user input from the form
        input_data = [
            int(request.form['Gender']),
            float(request.form['ILMN_2312606']),
            float(request.form['ILMN_2219628']),
            float(request.form['ILMN_1678054']),
            float(request.form['ILMN_1662358']),
            float(request.form['ILMN_1760062'])
        ]
        
        # Scale the input data using the pre-trained scaler
        input_data_scaled = scaler.transform([input_data])
        
        # Make a prediction using the pre-trained model
        prediction = model.predict(input_data_scaled)[0][0] * 100  # Convert to percentage
        
        # Render the result page
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
        # Handle errors gracefully
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

# Main execution
if __name__ == '__main__':
    # Load and preprocess data
    file_path = '/Users/jaybhatia/Documents/SS Project/SSDataset2.csv'
    features_scaled, labels, scaler, le = load_and_preprocess_data(file_path)
    
    if features_scaled is not None:
        # Print class distribution
        print("Class Distribution:")
        print(np.unique(labels, return_counts=True))  # Should print (array([0, 1]), array([131, 29]))
        
        # Train the model using the entire dataset
        model = train_model(features_scaled, labels)
        
        # Save the model
        if model:
            model.save('sjogrens_risk_model.h5')
            print("Model saved successfully.")
        
        # Run the Flask app
        app.run(debug=True)
    else:
        print("Data loading failed. Please check the dataset and file path.")