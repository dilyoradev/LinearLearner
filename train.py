import torch
from model import RegressionModel
from data_loader import generate_data, split_data
from torch import nn

# Initialize data
x, y = generate_data()
x_train, x_test, y_train, y_test = split_data(x, y)

# Initialize model, loss, and optimizer
model = RegressionModel()
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01)

# Training loop
epochs = 1000
for epoch in range(epochs):
    model.train()  # Set the model to training mode
    y_preds = model(x_train)
    loss = loss_fn(y_preds, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Test the model every 100 epochs
    if epoch % 100 == 0:
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            test_preds = model(x_test)
            test_loss = loss_fn(test_preds, y_test)
        print(f"Epoch: {epoch} | Loss: {loss:.5f} | Test loss: {test_loss:.5f}")

# Save model
torch.save(model.state_dict(), "model.pth")
