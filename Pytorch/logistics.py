import torch

#Prepare dataset
x_data = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y_data = torch.tensor([[0.0], [0.0], [1.0], [1.0]])

#Design model using class, inherit from torch.nn.Module
class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred
    
model = LogisticRegression(input_dim=1, output_dim=1)

#Construct loss and optimizer, using PyTorch API
criterion = torch.nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

#Training cycle
for epoch in range(100):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    if (epoch + 1) % 10 == 0:
        print('epoch {}, loss {}'.format(epoch + 1, loss.item()))

    optimizer.zero_grad()    #梯度清零，清零的是上一轮的梯度值，避免累加
    loss.backward()     #反向传播，计算新梯度，写入 .grad
    optimizer.step()    #用 .grad 更新参数