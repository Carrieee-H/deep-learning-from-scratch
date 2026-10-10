import torch
import numpy as np

# Prepare dataset
x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168], 
                    [9.779], [6.182], [7.59], [2.167], [7.042], 
                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)

y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573], 
                    [3.366], [2.596], [2.53], [1.221], [2.827], 
                    [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)

#Design model using class, inherit from torch.nn.Module
class LinearModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

input_dim = 1   #每个样本有几个输入特征
output_dim = 1   #每个样本要预测几个输出值
model = LinearModel(input_dim, output_dim)

#Construct loss and optimizer, using PyTorch API
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

#Training cycle
for epoch in range(100):
    #Convert numpy array to torch Variable
    inputs = torch.from_numpy(x_train)
    labels = torch.from_numpy(y_train)

    outputs = model(inputs)
    loss = criterion(outputs, labels)

    optimizer.zero_grad()    #梯度清零，清零的是上一轮的梯度值，避免累加
    loss.backward()     #反向传播，计算新梯度，写入 .grad
    optimizer.step()    #用 .grad 更新参数

    if (epoch + 1) % 10 == 0:
        print('epoch {}, loss {}'.format(epoch + 1, loss.item()))