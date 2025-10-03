import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify

# Step 1: Load and preprocess the data
# def load_and_preprocess_data(file_path):
#     # Load dataset
#     data = pd.read_csv(file_path, sep='\t', header=None)

#     # Ensure correct column parsing
#     column_count = data.shape[1]
#     data.columns = ['GEO Accession #', 'Patient Group', 'Gender'] + [f'ILMN_{i}' for i in range(column_count - 3)]

#     # Extract and encode gender
#     data['Gender'] = data['Gender'].str.split(': ').str[1]
#     le = LabelEncoder()
#     data['gender_encoded'] = le.fit_transform(data['Gender'])  # Encode Female=1, Male=0

#     # Extract features and labels
#     features = data.iloc[:, 3:].apply(pd.to_numeric, errors='coerce').fillna(0)
#     labels = data['Patient Group'].str.split(': ').str[1].apply(lambda x: 1 if 'Sjogren' in x else 0)  # Binary classification

#     # Add gender as a feature
#     features['gender'] = data['gender_encoded']

#     # Normalize features
#     scaler = StandardScaler()
#     features_scaled = scaler.fit_transform(features)

#     return features_scaled, labels.values, scaler

# Step 1: Load and preprocess the data
def load_and_preprocess_data(file_path):
    # Load dataset
    data = pd.read_csv(file_path)

    # Validate expected columns
    expected_columns = [
        'GEO Accession #', 'Patient Group', 'Gender',
        'ILMN_2312606', 'ILMN_2219628', 'ILMN_1678054', 'ILMN_1662358', 
        'ILMN_1760062', 'ILMN_1707695', 'ILMN_1782704', 'ILMN_2066858',
        'ILMN_1699651', 'ILMN_1718552', 'ILMN_2319077', 'ILMN_1690105', 'ILMN_1807036'
    ]
    if list(data.columns) != expected_columns:
        raise ValueError(f"Dataset columns do not match the expected format: {expected_columns}")

    # Remove 'GEO Accession #' column
    data = data.drop(columns=['GEO Accession #'])

    # Encode Gender (Male = 0, Female = 1)
    data['Gender'] = data['Gender'].str.split(': ').str[1]
    data['gender_encoded'] = data['Gender'].map({'Male': 0, 'Female': 1})
    if data['gender_encoded'].isnull().any():
        raise ValueError("Gender column contains unexpected values. Expected 'Male' or 'Female'.")

    # Encode Patient Group (Control = 0, Patient = 1)
    data['Patient Group'] = data['Patient Group'].str.split(': ').str[1]
    labels = data['Patient Group'].map({'Control': 0, 'Patient': 1}).values
    if pd.isnull(labels).any():
        raise ValueError("Patient Group column contains unexpected values. Expected 'Control' or 'Patient'.")

    # Extract features (ILMN columns + gender)
    features = data.iloc[:, 2:-1].apply(pd.to_numeric, errors='coerce').fillna(0)
    features['Gender'] = data['gender_encoded']

    # Normalize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    return features_scaled, labels, scaler


# Step 2: Build the neural network model
def create_model(input_dim, learning_rate=0.001):
    model = Sequential([
        Dense(128, activation='relu', input_dim=input_dim, kernel_regularizer='l2'),
        Dropout(0.5),
        Dense(64, activation='relu', kernel_regularizer='l2'),
        Dropout(0.3),
        Dense(32, activation='relu', kernel_regularizer='l2'),
        Dense(1, activation='sigmoid')  # Output layer for binary classification
    ])

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Step 3: Train the model with cross-validation
def train_with_cross_validation(X, y, model_builder, epochs=100, batch_size=32, k=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    scores = []
    best_model = None
    best_auc = 0

    for train_index, val_index in kf.split(X):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # Create a new model for each fold
        model = model_builder(X_train.shape[1])
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0, validation_data=(X_val, y_val))

        # Evaluate on the validation set
        val_preds = model.predict(X_val)
        auc = roc_auc_score(y_val, val_preds)
        scores.append(auc)

        if auc > best_auc:
            best_auc = auc
            best_model = model

        # Plot training history
        plot_history(history)

    return np.mean(scores), best_model

# Step 4: Plot training history
def plot_history(history):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title('Loss')

    if 'accuracy' in history.history and 'val_accuracy' in history.history:
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.legend()
        plt.title('Accuracy')

    plt.tight_layout()
    plt.show()

# Step 5: Flask web app
app = Flask(__name__)

@app.route('/')
def home():
    return '''<html><body>
    <h1>Sjogren's Syndrome Risk Calculator</h1>
    <form action="/predict" method="post">
    <label>Enter Gene Expression Values (comma-separated):</label><br>
    <textarea name="input_data" rows="5" cols="60"></textarea><br><br>
    <label>Select Gender:</label><br>ç
    <select name="gender">
        <option value="0">Male</option>
        <option value="1">Female</option>
    </select><br><br>
    <label><strong>Input Details:</strong></label><br>
    <p>Please provide values for the following genes:</p>
    <ul>
        <li>ILMN_2312606</li>
        <li>ILMN_2219628</li>
        <li>ILMN_1678054</li>
        <li>ILMN_1662358</li>
        <li>ILMN_1760062</li>
        <li>ILMN_1707695</li>
        <li>ILMN_1782704</li>
    </ul>
    <input type="submit" value="Calculate Risk">
    </form>
    </body></html>'''

@app.route('/predict', methods=['POST'])
def predict():
    # Parse input data
    input_data = request.form['input_data']
    gender = int(request.form['gender'])  # Gender as binary (Male=0, Female=1)

    try:
        input_data = np.array([float(val) for val in input_data.split(',')])
    except ValueError:
        return '''<html><body>
        <h1>Error</h1>
        <p>Please enter valid numeric values separated by commas.</p>
        <a href="/">Back to Home</a>
        </body></html>'''

    # Add gender to the input data
    input_data = np.append(input_data, gender)

    # Preprocess input data
    input_data_scaled = scaler.transform([input_data])

    # Predict risk percentage
    prediction = model.predict(input_data_scaled)[0][0] * 100  # Convert to percentage

    return f'''<html><body>
    <h1>Prediction Result</h1>
    <p><strong>Risk Percentage:</strong> {prediction:.2f}%</p>
    <a href="/">Back to Home</a>
    </body></html>'''

if __name__ == '__main__':
    # Load and preprocess data
    file_path = '/Users/jaybhatia/Documents/SS Project/SSDataset2.csv'
    features_scaled, labels, scaler = load_and_preprocess_data(file_path)

    # Train the model using k-fold cross-validation
    avg_auc, model = train_with_cross_validation(features_scaled, labels, create_model, epochs=100, batch_size=32, k=5)
    print(f"Cross-validated AUC: {avg_auc:.2f}")

    # Save the model
    model.save('sjogrens_risk_model.h5')

    # Run the Flask app
    app.run(debug=True)
