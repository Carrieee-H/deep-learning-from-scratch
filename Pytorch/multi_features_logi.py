import numpy as np
import torch

# Prepare dataset
x_data = np.loadtxt('diabetes_data_raw.csv.gz', dtype=np.float32)   #文件是空格分隔，不指定delimiter即可
y_data = np.loadtxt('diabetes_target.csv.gz', dtype=np.float32)
#把连续的病情指标转成二分类标签：大于中位数为1，否则为0
y_data = (y_data > np.median(y_data)).astype(np.float32).reshape(-1, 1)   #reshape成(442, 1)，与模型输出形状一致
#convert numpy array to torch Variable
x_data = torch.from_numpy(x_data)
y_data = torch.from_numpy(y_data)

#Design model using class, inherit from torch.nn.Module
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        #多层神经网络
        self.linear1 = torch.nn.Linear(10, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        #正常应该写o1=self.sigmoid(self.linear1(x))，o2=self.sigmoid(self.linear2(o1))
        #但是在实际工程中对于这种纯序列的神经网络，通常会直接写成上面这种形式，省略中间变量防止出错
        return x

model = Model()

#Construct loss and optimizer, using PyTorch API
criterion = torch.nn.BCELoss()  
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

#Training cycle
for epoch in range(100):
    #forward
    y_pred = model(x_data)  #这里并没有用mini-batch
    loss = criterion(y_pred, y_data)
    if (epoch + 1) % 10 == 0:
        print('epoch {}, loss {}'.format(epoch + 1, loss.item()))

    #backward
    optimizer.zero_grad()    #梯度清零，清零的是上一轮的梯度值，避免累加
    loss.backward()     #反向传播，计算新梯度，写入 .grad

    #update
    optimizer.step()    #用 .grad 更新参数

    