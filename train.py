import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 

def run():
    # Load dataset
    df = pd.read_csv("data/Email_spam_numeric.csv")

    # Drop NaN values in target column
    df = df.dropna(subset=["spam"])

    # Separate features (X) and target variable (Y)
    X = df.drop(columns=["spam"])  
    Y = df["spam"]

    # Split data into training, validation, and test sets
    X_val_train, X_test, Y_val_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    X_train, X_val, Y_train, Y_val = train_test_split(X_val_train, Y_val_train, test_size=0.2, random_state=42)

    # Train Logistic Regression Model
    model = LogisticRegression(max_iter=1000)  # Increase iterations for better convergence
    model.fit(X_train, Y_train)

    # Make predictions on test set
    y_pred = model.predict(X_test)

    # Evaluate model performance
    accuracy = accuracy_score(Y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2%}")

    # Ensure directories exist before saving
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    # Save model and validation data
    joblib.dump(model, "models/model.pkl")
    joblib.dump(X_val, "data/X_val.pkl")
    joblib.dump(Y_val, "data/Y_val.pkl")

    print("TRAINING completed ...")

# Run the function
if __name__ == "__main__":
    run()
