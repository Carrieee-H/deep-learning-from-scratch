#多分类问题
import torch
from torchvision import transforms  #针对图像数据的预处理
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim     #优化器

#Prepare dataset
batch_size = 64
transform = transforms.Compose([
    transforms.ToTensor(),    #将PIL图像数据转为tensor，并归一化到[0,1]
    transforms.Normalize((0.1307,), (0.3081,))    #标准化，均值0.1307和标准差0.3081是通过整体MNIST训练集计算出来的
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

#Design model using Class
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = torch.nn.Linear(28*28, 512)
        self.l2 = torch.nn.Linear(512, 256)
        self.l3 = torch.nn.Linear(256, 10)  #这里取10是因为MNIST数据集有10个类别，分别是0-9的数字

    def forward(self, x):
        x = x.view(-1, 28*28)    #将输入的图像数据展平为一维向量，-1表示自适应求数，
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.l3(x)    #最后一层不使用激活函数，因为后续会使用softmax计算概率分布

model = Net()

#Construct loss and optimizer
criterion = torch.nn.CrossEntropyLoss()    #交叉熵损失函数，适用于多分类问题
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)    #随机梯度下降优化器，学习率为0.01 

#Training cycle
def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()    #梯度清零，清零的是上一轮的梯度值，避免累加

        #forward + backward + update
        outputs = model(inputs)
        loss = criterion(outputs, labels)    
        loss.backward()    
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 100 == 99:
            print(f'[{epoch + 1}, {batch_idx + 1}] loss: {running_loss / 100:.3f}')
            running_loss = 0.0

def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)    #返回每行最大值的索引，即预测的类别
            total += labels.size(0)
            correct += (predicted == labels).sum().item()    #.item() 把只含一个元素的 tensor 转成 Python 的 int 或 float
        print(f'Accuracy on the test set: {100 * correct / total:.2f} %')

if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        test()