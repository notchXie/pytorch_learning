import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


xy = np.loadtxt('diabetes.csv.gz', delimiter=',', dtype=np.float32)
x_data = torch.from_numpy(xy[:, 0:-1])
y_data = torch.from_numpy(xy[:, [-1]])


# 划分训练集和测试集
train_x = x_data[:600]
train_y = y_data[:600]
test_x = x_data[600:]
test_y = y_data[600:]


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x


model = Model()


criterion = torch.nn.BCELoss(size_average=False)
optimizer = torch.optim.ASGD(model.parameters(), lr=0.01)

loss_list = []
epoch_list = []

for epoch in range(1000):
    y_pred = model(train_x)
    loss = criterion(y_pred, train_y)
    print("Now epoch is: ", epoch, "loss is: ", loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_list.append(loss.item())
    epoch_list.append(epoch)

y_pred = model(test_x)
y_pred = (y_pred > 0.5).float()


# 计算iou
TP = ((y_pred == 1) & (test_y == 1)).sum().item()
FP = ((y_pred == 1) & (test_y == 0)).sum().item()
FN = ((y_pred == 0) & (test_y == 1)).sum().item()
TN = ((y_pred == 0) & (test_y == 0)).sum().item()
print("TP: ", TP)
print("FP: ", FP)
print("FN: ", FN)
print("TN: ", TN)
iou = TP / (TP + FP + FN)
print("iou: ", iou)


plt.plot(epoch_list, loss_list)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()