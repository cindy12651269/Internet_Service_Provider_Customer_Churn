# -*- coding: utf-8 -*-
"""Pytorch.py

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1mO4sqHGKGKgvSaA9T81A-EZUFIrGa7th
"""

import torch

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model with Batch Normalization
class Model(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.hidden1 = torch.nn.Linear(input_size, 64)
        self.bn1 = torch.nn.BatchNorm1d(64)
        self.hidden2 = torch.nn.Linear(64, 32)
        self.bn2 = torch.nn.BatchNorm1d(32)
        self.predict = torch.nn.Linear(32, output_size)

    def forward(self, x):
        output1 = torch.relu(self.bn1(self.hidden1(x)))
        output2 = torch.relu(self.bn2(self.hidden2(output1)))
        output = torch.sigmoid(self.predict(output2))
        return output

# Initialize model and optimizer
model = Model(X_test.shape[1], 1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_func = torch.nn.BCELoss()

# Learning rate scheduler (reduces LR when validation loss plateaus)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

# Early stopping parameters
best_val_loss = float('inf')
patience = 5
patience_counter = 0

# Convert data and move to GPU if available
x_data = torch.tensor(X_train.values, dtype=torch.float32).to(device)
y_data = torch.tensor(np.expand_dims(y_train, axis=1), dtype=torch.float32).to(device)

x_val_data = torch.tensor(X_val.values, dtype=torch.float32).to(device)
y_val_data = torch.tensor(np.expand_dims(y_val, axis=1), dtype=torch.float32).to(device)

# Hyperparameters
batch_size = 64  # Increased batch size
num_epochs = 200

training_losses = []
val_losses = []

# Training loop
for epoch in range(num_epochs):
    model.train()
    for i in range(0, len(x_data), batch_size):
        prediction = model(x_data[i:i+batch_size])
        loss = loss_func(prediction, y_data[i:i+batch_size])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Calculate training and validation loss
    model.eval()
    with torch.no_grad():
        loss = loss_func(model(x_data), y_data)
        training_losses.append(float(loss))

        y_pred = model(x_val_data)
        val_loss = loss_func(y_pred, y_val_data)
        val_losses.append(float(val_loss))

        print("Epoch {}: training loss: {}, val loss: {}, val acc: {}".format(
            epoch+1, float(loss), float(val_loss), accuracy_score(y_val_data.cpu(), np.where(y_pred.cpu() >= 0.5, 1, 0))))

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        # Reduce learning rate if validation loss plateaus
        scheduler.step(val_loss)

        # Trigger early stopping if no improvement in patience epochs
        if patience_counter >= patience:
            print("Early stopping triggered at epoch", epoch + 1)
            break

# Convert test data and move to GPU if available
x_test_data = torch.tensor(X_test.values, dtype=torch.float32).to(device)

# Make predictions on the test set
y_pred = model(x_test_data)

# Convert the predicted values to binary labels (0 or 1), detach from the graph before converting to numpy
y_pred = np.where(y_pred.detach().cpu().numpy() >= 0.5, 1, 0)

print(classification_report(y_test, y_pred))

plt.plot(training_losses)
plt.plot(val_losses)
plt.legend(("Training loss", "val loss"))
plt.xlabel("epoch")
plt.ylabel("loss")