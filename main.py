import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import yaml

from src.model import RidgeRegression
from src.data_preparation import generate_sample_data, prepare_data

def load_config(config_path):
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def main():
    # Load configuration
    config = load_config('configs/model_config.yaml')
    
    # Generate sample data
    X, y = generate_sample_data(
        n_samples=config['data']['n_samples'],
        n_features=config['data']['n_features'],
        noise=config['data']['noise'],
        random_state=config['random_state']
    )
    
    # Prepare data
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = prepare_data(
        X, y,
        test_size=config['data']['test_size'],
        random_state=config['random_state']
    )
    
    # Initialize and train the model
    ridge = RidgeRegression(alpha=config['model']['alpha'])
    ridge.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = ridge.predict(X_test_scaled)
    
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Print results
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R-squared Score: {r2:.4f}")
    print("Coefficients:", ridge.coefficients)
    print("Intercept:", ridge.intercept)
    
    # Optional: Save the model
    # You could implement a save_model function in your model.py file
    # ridge.save_model('models/ridge_model.pkl')

if __name__ == "__main__":
    main()