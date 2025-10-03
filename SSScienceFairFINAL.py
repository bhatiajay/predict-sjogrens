import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify
import tensorflow as tf

# Define column names globally
column_names = ['GEO Accession #', 'Patient Group', 'Gender', 'ILMN_2312606', 'ILMN_2219628', 'ILMN_1678054',
                'ILMN_1662358', 'ILMN_1760062', 'ILMN_1707695', 'ILMN_1782704', 'ILMN_2066858', 'ILMN_1699651',
                'ILMN_1718552', 'ILMN_2319077', 'ILMN_1690105', 'ILMN_1807036']

# Step 1: Load and preprocess the data
def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    data.columns = column_names
    
    # Encode gender
    le = LabelEncoder()
    data['Gender'] = le.fit_transform(data['Gender'].astype(str))  # Ensure encoding works correctly

    # Extract features and labels
    features = data.drop(columns=['GEO Accession #', 'Patient Group']).apply(pd.to_numeric, errors='coerce').fillna(0)
    labels = data['Patient Group'].apply(lambda x: 1 if 'Sjogren' in x else 0)  # Binary classification

    # Normalize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    return features_scaled, labels.values, scaler, le

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
    kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    scores = []
    best_model = None
    best_auc = 0
    
    initial_model = model_builder(X.shape[1])

    for train_index, val_index in kf.split(X, y):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        model = model_builder(X_train.shape[1])
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0, validation_data=(X_val, y_val))

        val_preds = model.predict(X_val)
        if len(set(y_val)) > 1:  # Ensure both classes are present
            auc = roc_auc_score(y_val, val_preds)
            scores.append(auc)

            if auc > best_auc:
                best_auc = auc
                best_model = model
        
        plot_history(history)
    
    return np.mean(scores) if scores else 0, best_model if best_model else initial_model

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
    return '''<html><head><title>Sjogren's Syndrome Risk Calculator</title>
    <style>
        body { font-family: 'Arial', sans-serif; text-align: center; background: #f4f4f4; padding: 20px; }
        form { display: inline-block; text-align: left; padding: 20px; background: white; box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1); border-radius: 10px; }
        input, select { margin: 5px; padding: 8px; width: 100%; border-radius: 5px; border: 1px solid #ddd; }
        button { background-color: #4CAF50; color: white; padding: 10px; border: none; border-radius: 5px; cursor: pointer; }
        button:hover { background-color: #45a049; }
    </style></head><body>
    <h1>Sjogren's Syndrome Risk Calculator</h1>
    <form action="/predict" method="post">
    <label>Gender:
        <select name="Gender">
            <option value="0">Male</option>
            <option value="1">Female</option>
        </select>
    </label><br>
    ''' + ''.join([f'<label>{col}: <input type="text" name="{col}" placeholder="Enter {col}"></label><br>' for col in column_names[3:]]) + '''
    <button type="submit">Calculate Risk</button>
    </form>
    </body></html>'''

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = [int(request.form['Gender'])] + [float(request.form[col]) for col in column_names[3:]]
        input_data_scaled = scaler.transform([input_data])
        prediction = model.predict(input_data_scaled)[0][0] * 100

        return f'''<html><body>
        <h1>Prediction Result</h1>
        <p><strong>Risk Percentage:</strong> {prediction:.2f}%</p>
        <a href="/">Back to Home</a>
        </body></html>'''
    except Exception as e:
        return f'''<html><body>
        <h1>Error</h1>
        <p>{str(e)}</p>
        <a href="/">Back to Home</a>
        </body></html>'''

if __name__ == '__main__':
    file_path = '/Users/jaybhatia/Documents/SS Project/SSDataset2.csv'
    features_scaled, labels, scaler, le = load_and_preprocess_data(file_path)
    avg_auc, model = train_with_cross_validation(features_scaled, labels, create_model, epochs=100, batch_size=32, k=5)
    print(f"Cross-validated AUC: {avg_auc:.2f}")
    if model:
        model.save('sjogrens_risk_model.h5')
    app.run(debug=True)