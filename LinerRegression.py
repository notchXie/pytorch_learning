import torch
import matplotlib.pyplot as plt

x_data = torch.tensor([[1.0], [2.0], [3.0]])
y_data = torch.tensor([[2.0], [4.0], [6.0]])

loss_list = []
epoch_list = []


class LinerModel(torch.nn.Module):
    def __init__(self):
        super(LinerModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)  # 构造对象，contains two parameters: weight and bias

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred


model = LinerModel()

criterion = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.ASGD(model.parameters(), lr=0.01)

for epoch in range(200):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 保存loss和epoch
    loss_list.append(loss.item())
    epoch_list.append(epoch)

plt.plot(epoch_list, loss_list)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()

print("w = ", model.linear.weight.item())
print("b = ", model.linear.bias.item())

x_test = torch.tensor([[4.0]])
y_test = model(x_test)
print("y_pred = ", y_test.data)
