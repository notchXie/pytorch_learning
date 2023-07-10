import torch

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = torch.tensor([1.0])
w.requires_grad = True

def forward(x):
    return torch.tensor([x]) * w

def loss(x, y):
    y_pred = forward(x)
    return pow(y-y_pred, 2)

print("Predict (before training)", 4, forward(4).item())

for epoch in range(200):
    for x, y in zip(x_data, y_data):
        l = loss(x, y)
        l.backward()
        print("\tgrad: ", x, y, w.grad.item())
        w.data = w.data - 0.01 * w.grad.data

        w.grad.data.zero_()

    print(f"Epoch: {epoch} | Loss: {l.item()}")

print("Predict (after training)", 4, forward(4).item())