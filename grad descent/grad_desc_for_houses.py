import numpy as np
import pandas as pd
import copy

# Preprocess the data
def process_data(dataset):
    # Make a copy to avoid modifying original data
    data = copy.deepcopy(dataset)

    # Separate features and target
    if 'SalePrice' in data.columns:
        X = data.drop("SalePrice", axis=1)
        Y = data["SalePrice"]
        has_target = True
    else:
        X = data
        Y = None        
        has_target = False
    
    # Keep only numeric columns
    X = X.select_dtypes(include=[np.number])

    # Handle missing values
    X = X.fillna(X.mean())

    # Scale the features to prevent overflow
    X_std = X.std()
    X_std = X_std.replace(0, 1e-8)  # Prevent division by zero
    X = (X - X.mean()) / X_std

    # Store scaling parameters for target variable
    y_mean = None
    y_std = None
    if Y is not None:
        y_mean = Y.mean()
        y_std = Y.std()
        Y = (Y - y_mean) / y_std
    
    # Convert to numpy arrays
    X = X.values
    if has_target:
        Y = Y.values
        
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    # Return scaling parameters as well
    return X, Y, y_mean, y_std


def compute_cost_reg (X, y, w, b, lambda_):
    m = X.shape[0]
    predictions = X.dot(w) + b
    errors = predictions - y

    cost = (1/(2*m))*(np.sum(errors**2))
    reg_cost = (lambda_/(2*m)) * (np.sum(w**2))
    return reg_cost + cost

def compute_grad_reg (X, y, w, b, lambda_):
    m, n = X.shape
    dj_dw = np.zeros(n)
    dj_db = 0

    predictions = X.dot(w) + b
    errors = predictions - y

    dj_dw = (1/m) * (X.T.dot(errors)) + (lambda_/m) * w
    dj_db = (1/m) * np.sum(errors)

    return dj_dw, dj_db

def gradient_desc (w_in, b_in, X, y, alpha, num_iterations, lambda_):
    
    w = copy.deepcopy(w_in)
    b = copy.deepcopy(b_in)
    for i in range(num_iterations):
        dj_dw, dj_db = compute_grad_reg(X,y,w,b,lambda_)
        w -= alpha*dj_dw
        b -= alpha*dj_db

    cost_at_end = compute_cost_reg(X, y, w, b, lambda_)
    return w, b, cost_at_end

def make_predictions(X, w, b):
    return X.dot(w) + b

def denormalize_predictions(y_pred_normalized, y_mean, y_std):
    return y_pred_normalized * y_std + y_mean

# Load the datasets
training_dataset = pd.read_csv("train.csv")
testing_dataset = pd.read_csv("test.csv")

# Preprocess the data
X_train, y_train, y_mean_train, y_std_train = process_data(training_dataset)
X_test, y_test, y_mean_test, y_std_test = process_data(testing_dataset)
# Initialize parameters
n_features = X_train.shape[1]
w = np.random.randn(n_features) * 0.01
b = 0.0
alpha = 0.005
lambda_ = 0.1
num_iters = 10000
cost = 0.0

# Perform gradient descent
w, b, cost = gradient_desc(w, b, X_train, y_train, alpha, num_iters, lambda_)

# Output the training results
print(" TRAINING RESULTS ")
print("Trained weights shape:", w.shape)
print("Trained bias:", b)
print("Cost at the end:", cost)

# Make predictions on test set (normalized)
y_pred_normalized = make_predictions(X_test, w, b)

# Denormalize predictions to original scale
y_pred_original = denormalize_predictions(y_pred_normalized, y_mean_train, y_std_train)

print("\n TEST SET PREDICTIONS ")
print("First 10 normalized predictions:", y_pred_normalized[:10])
print("First 10 denormalized predictions:", y_pred_original[:10])

# test 
# test 2