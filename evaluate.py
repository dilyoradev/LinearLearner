import torch
import matplotlib.pyplot as plt
from model import RegressionModel
from data_loader import generate_data

# Load model
model = RegressionModel()
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Generate data
x, y = generate_data()

# Make predictions
with torch.no_grad():
    y_pred = model(x)

# Visualize the results
plt.scatter(x.numpy(), y.numpy(), label="True Data")
plt.scatter(x.numpy(), y_pred.numpy(), label="Predicted Data", color="red")
plt.legend()
plt.show()
