import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score  # Import accuracy_score
from pytorch_tabnet.tab_model import TabNetClassifier
import joblib
# Load data from JSONL
df = pd.read_json('../dataset/unprocessed-dataset.jsonl', lines=True)

# Identify target column
target_col = df.columns[-1]
X = df.drop(columns=[target_col])
y = df[target_col]

# Identify column types
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Define preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ]
)

# Fit and transform
X_processed = preprocessor.fit_transform(X)

# Convert to DataFrame with column names
feature_names = (
    numeric_cols +
    preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols).tolist()
)
X_processed_df = pd.DataFrame(X_processed, columns=feature_names)

# Add target column back
X_processed_df[target_col] = y.values

# Export to JSON Lines format
X_processed_df.to_json("processed-data.jsonl", orient="records", lines=True)

print("✅ Preprocessing complete and saved to processed-data.jsonl")

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_processed_df.drop(columns=[target_col]), y, test_size=0.2, random_state=42)

# Convert data to NumPy arrays (not torch tensors yet)
X_train_np = X_train.values.astype(np.float32)
X_test_np = X_test.values.astype(np.float32)
y_train_np = y_train.values.astype(np.int64)
y_test_np = y_test.values.astype(np.int64)

# Initialize TabNet model
tabnet_model = TabNetClassifier()

# Train the model using NumPy arrays
tabnet_model.fit(X_train_np, y_train_np)

# Make predictions on the test set
y_pred = tabnet_model.predict(X_test_np)

# Calculate accuracy
accuracy = accuracy_score(y_test_np, y_pred)
print(f"Model accuracy: {accuracy}")

# Function to preprocess new input data and predict
def predict_new_data(new_data):
    # Preprocess the new data using the same preprocessor
    new_data_processed = preprocessor.transform(new_data)

    # Convert to NumPy array
    new_data_processed_np = new_data_processed.astype(np.float32)

    # Make prediction
    prediction = tabnet_model.predict(new_data_processed_np)
    return prediction

# Example usage: Take user input and predict
def get_user_input_and_predict():
    # Collecting input from the user (make sure this matches your column names and order)
    user_input = {}
    for col in numeric_cols:
        user_input[col] = float(input(f"Enter value for {col}: "))
    
    for col in categorical_cols:
        user_input[col] = input(f"Enter value for {col}: ")

    # Convert user input to DataFrame (same format as the model was trained on)
    new_data = pd.DataFrame(user_input, index=[0])

    # Predict using the pre-trained model
    prediction = predict_new_data(new_data)
    print(f"Predicted Output: {prediction[0]}")

# Uncomment the line below to run the prediction after taking user input
get_user_input_and_predict()

# Save the model
tabnet_model.save_model("tabnet_model")

joblib.dump(preprocessor, "preprocessor.pkl")
