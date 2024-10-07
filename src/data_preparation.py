import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def generate_sample_data(n_samples=100, n_features=5, noise=0.1, random_state=42):
    """
    Generate sample data for ridge regression.
    
    Parameters:
    - n_samples: Number of samples to generate
    - n_features: Number of features to generate
    - noise: Standard deviation of Gaussian noise
    - random_state: Random state for reproducibility
    
    Returns:
    - X: Feature matrix
    - y: Target vector
    """
    np.random.seed(random_state)
    
    X = np.random.randn(n_samples, n_features)
    
    # True coefficients
    true_coefficients = np.array([3, 2, 1.5, -1, 0.5])
    
    # Generate target values with some noise
    y = np.dot(X[:, :len(true_coefficients)], true_coefficients) + np.random.randn(n_samples) * noise
    
    return X, y

def prepare_data(X, y, test_size=0.2, random_state=42):
    """
    Prepare data for ridge regression by splitting into train and test sets,
    and scaling features.
    
    Parameters:
    - X: Feature matrix
    - y: Target vector
    - test_size: Proportion of the dataset to include in the test split
    - random_state: Random state for reproducibility
    
    Returns:
    - X_train_scaled: Scaled training features
    - X_test_scaled: Scaled test features
    - y_train: Training target values
    - y_test: Test target values
    - scaler: Fitted StandardScaler object
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def load_data(filepath):
    """
    Load data from a file.
    
    Parameters:
    - filepath: Path to the data file
    
    Returns:
    - X: Feature matrix
    - y: Target vector
    """
    # This is a placeholder function. In a real scenario, you would implement
    # the actual data loading logic here, which depends on your data format.
    # For now, we'll just raise a NotImplementedError.
    raise NotImplementedError("Data loading from file not implemented yet.")

# Example usage
if __name__ == "__main__":
    # Generate sample data
    X, y = generate_sample_data()
    
    # Prepare data
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = prepare_data(X, y)
    
    print("Sample data generated and prepared.")
    print(f"Training set shape: {X_train_scaled.shape}")
    print(f"Test set shape: {X_test_scaled.shape}")