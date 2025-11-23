"""
Real Estate Price Prediction - Main Training Pipeline
Authors: Colin Pietri & Cristian Larrain
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import lightgbm as lgb
import joblib
import warnings
warnings.filterwarnings('ignore')


def load_data(filepath='clean_dataset'):
    """Load the cleaned dataset"""
    print("Loading dataset...")
    df = pd.read_parquet(filepath)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def preprocess_data(df):
    """Preprocess the data for training"""
    print("Preprocessing data...")
    
    # Select features
    feature_columns = ['surface_m2', 'nombre_pieces_principales', 
                      'code_postal', 'annee', 'mois']
    target_column = 'valeur_fonciere'
    
    # Filter data
    X = df[feature_columns].copy()
    y = df[target_column].copy()
    
    # One-hot encode postal code
    X = pd.get_dummies(X, columns=['code_postal'], prefix='postal')
    
    print(f"Features shape: {X.shape}")
    return X, y


def train_model(X_train, y_train):
    """Train LightGBM model"""
    print("Training LightGBM model...")
    
    model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=10,
        random_state=42,
        verbose=-1
    )
    
    model.fit(X_train, y_train)
    print("Training complete!")
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    print("\nEvaluating model...")
    
    y_pred = model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    print("\n" + "="*50)
    print("MODEL PERFORMANCE METRICS")
    print("="*50)
    print(f"R² Score:  {r2:.3f}")
    print(f"RMSE:      {rmse:,.0f}€")
    print(f"MAE:       {mae:,.0f}€")
    print(f"MAPE:      {mape:.1f}%")
    print("="*50 + "\n")
    
    return {'r2': r2, 'rmse': rmse, 'mae': mae, 'mape': mape}


def main():
    parser = argparse.ArgumentParser(description='Real Estate Price Prediction')
    parser.add_argument('--train', action='store_true', 
                       help='Train a new model')
    parser.add_argument('--evaluate', action='store_true', 
                       help='Evaluate existing model')
    parser.add_argument('--data', type=str, default='clean_dataset',
                       help='Path to dataset file')
    
    args = parser.parse_args()
    
    if args.train:
        # Load and preprocess data
        df = load_data(args.data)
        X, y = preprocess_data(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        model = train_model(X_train, y_train)
        
        # Evaluate
        metrics = evaluate_model(model, X_test, y_test)
        
        # Save model
        joblib.dump(model, 'champion_model_paris.joblib')
        print("Model saved as 'champion_model_paris.joblib'")
        
    elif args.evaluate:
        # Load existing model
        print("Loading existing model...")
        model = joblib.load('champion_model_paris.joblib')
        
        # Load and preprocess data
        df = load_data(args.data)
        X, y = preprocess_data(df)
        
        # Split data (same split as training)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Evaluate
        metrics = evaluate_model(model, X_test, y_test)
        
    else:
        print("Please specify --train or --evaluate")
        print("Example: python main.py --train")


if __name__ == "__main__":
    main()
