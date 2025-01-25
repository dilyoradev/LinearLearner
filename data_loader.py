import torch
from sklearn.model_selection import train_test_split

def generate_data():
    # Generate synthetic data
    x = torch.linspace(0, 10, 100).unsqueeze(1)
    true_slope = 2
    true_intercept = 3
    noise = torch.randn_like(x) * 0.5
    y = true_slope * x + true_intercept + noise
    return x, y

def split_data(x, y):
    # Split the data into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    return x_train, x_test, y_train, y_test
