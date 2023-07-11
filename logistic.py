import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt

x_data = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0]])
y_data = torch.tensor([[0.0], [0.0], [1.0], [0.0], [1.0], [1.0], [1.0]])

loss_list = []
epoch_list = []


class LogisticModel(torch.nn.Module):
    def __init__(self):
        super(LogisticModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        y_pred = F.sigmoid(self.linear(x))
        return y_pred

model = LogisticModel()

criterion = torch.nn.BCELoss(size_average=False)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(1000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 保存loss和epoch
    loss_list.append(loss.item())
    epoch_list.append(epoch)

print("w = ", model.linear.weight.item())
print("b = ", model.linear.bias.item())

x_test = torch.tensor([[1.0]])
y_test = model(x_test)
print("y_pred = ", y_test.data)

plt.plot(epoch_list, loss_list)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()

function_pred_x = list(range(0, 10))
function_pred_y = []
for i in function_pred_x:
    x_test = torch.tensor([[i]])
    y_test = model(x_test)
    function_pred_y.append(y_test.data)

plt.plot(function_pred_x, function_pred_y)
plt.xlabel("x")
plt.ylabel("y")
plt.show()