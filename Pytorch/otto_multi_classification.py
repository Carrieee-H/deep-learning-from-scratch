#多分类问题：kaggle-Otto Group Product Classification Challenge
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch.optim as optim     #优化器
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

#Preprocessing dataset
df = pd.read_csv('otto-group-product-classification-challenge/train.csv')
X = df.drop(['id', 'target'], axis=1).values.astype(np.float32)
y = df['target'].str.split('_').str[1].astype(int).values - 1   #将类别标签Class_1-Class_9转为0-8的数字

X = np.log1p(X)    #对特征数据进行log1p变换，log1p(x) = log(1+x)，可以减小数据的偏态分布
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
mean,std = X_train.mean(axis=0), X_train.std(axis=0)
X_train = (X_train - mean) / std    #对训练数据进行标准化处理
X_test = (X_test - mean) / std      #对测试数据进行标准化处理

#Prepare dataset
class OttoDataset(Dataset):
    def __init__(self, X, y):
        self.len = X.shape[0]
        self.x_data = torch.from_numpy(X)
        self.y_data = torch.from_numpy(y)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

train_dataset = OttoDataset(X_train, y_train)
test_dataset = OttoDataset(X_test, y_test)
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=256, shuffle=False)


#Design model using Class
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(93, 64)
        self.linear2 = torch.nn.Linear(64, 32)
        self.linear3 = torch.nn.Linear(32, 9)    #这里取9是因为Otto数据集有9个类别，分别是Class_1-Class_9

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return self.linear3(x)    #最后一层不使用激活函数，因为后续会使用softmax计算概率分布

model = Model()

#Construct loss and optimizer
criterion = torch.nn.CrossEntropyLoss()    #交叉熵损失函数，适用于多分类问题
optimizer = optim.Adam(model.parameters(), lr=0.001)    #Adam优化器

#Training cycle
for epoch in range(10):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader):
        inputs, labels = data
        optimizer.zero_grad()    #梯度清零，清零的是上一轮的梯度值，避免累加

        #forward + backward + update
        outputs = model(inputs)
        loss = criterion(outputs, labels)    
        loss.backward()    
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    print(f'Epoch [{epoch+1}/10], Loss: {running_loss / len(train_dataset):.4f}')

def evaluate(loader):
    model.eval()
    total_loss,correct = 0.0,0
    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs)
            total_loss += criterion(outputs, labels).item() * inputs.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
        n = len(loader.dataset)
        return total_loss/n, correct/n

test_loss, test_acc = evaluate(test_loader)
print(f'Epoch [{epoch+1}/10], Train Loss:{running_loss/len(train_dataset):.4f},'
      f'Test Loss: {test_loss:.4f}, Val Acc: {100 * test_acc:.2f}%')


#Predict test.csv and write submission
test_df = pd.read_csv('otto-group-product-classification-challenge/test.csv')
X_sub = np.log1p(test_df.drop(['id'], axis=1).values.astype(np.float32))
X_sub = (X_sub - mean)/std

model.eval()
with torch.no_grad():
    probs = F.softmax(model(torch.from_numpy(X_sub)), dim=1).numpy()

submission = pd.DataFrame(probs, columns=[f'Class_{i}' for i in range(1,10)])
submission.insert(0,'id',test_df['id'])
submission.to_csv('submission.csv', index=False)