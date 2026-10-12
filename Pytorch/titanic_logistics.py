import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import pandas as pd

#Prepare dataset
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

def preprocess(df, mean=None, std=None):
    df = df[features].copy()
    #神经网络只能处理数字数据，对于简单的分类可以用标签编码，但对于自由文本需要做embedding处理
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Embarked'] = df['Embarked'].fillna('S').map({'C': 0, 'Q': 1, 'S': 2})
    #缺失值如果不处理，之后的反向传播会出现nan，模型直接崩溃
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    x = df.values.astype(np.float32)
    #标准化：mean和std只从训练集计算，测试集传入训练集的值，保证同一个数在两份数据里含义一致
    if mean is None:    #训练集计算mean和std
        mean = x.mean(axis=0)
        std = x.std(axis=0)
    x = (x - mean) / std    #训练集用传进来的测试集的参数
    return x, mean, std

class TitanicDataset(Dataset):
    def __init__(self, file_path):
        df = pd.read_csv(file_path)
        x, self.mean, self.std = preprocess(df)    #保存训练集的mean和std，供测试集使用
        self.len = x.shape[0]
        self.x_data = torch.from_numpy(x)
        self.y_data = torch.from_numpy(df[['Survived']].values.astype(np.float32))

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

train_dataset = TitanicDataset('titanic_train.csv')
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)

#Design model using Class
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(7, 5)
        self.linear2 = torch.nn.Linear(5, 3)
        self.linear3 = torch.nn.Linear(3, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x

model = Model()

#Define loss and optimizer
criterion = torch.nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

#Training cycle
for epoch in range(100):
    for i, data in enumerate(train_loader):
        inputs, labels = data
        y_pred = model(inputs)
        loss = criterion(y_pred, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

#Predict on test set
#测试集没有Survived列，不能用TitanicDataset，直接用preprocess处理特征
test_df = pd.read_csv('titanic_test.csv')
x_test, _, _ = preprocess(test_df, train_dataset.mean, train_dataset.std)    #用训练集的mean和std标准化
x_test = torch.from_numpy(x_test)

model.eval()    #切换到推理模式
with torch.no_grad():    #预测时不需要计算梯度
    y_prob = model(x_test)    #模型最后一层已经是sigmoid，输出就是生还概率
    y_test_pred = (y_prob > 0.5).int()    #概率大于0.5判为生还

submission = pd.DataFrame({
    'PassengerId': test_df['PassengerId'],
    'Survived': y_test_pred.view(-1).numpy()
})
submission.to_csv('titanic_submission.csv', index=False)
print(submission.head())
print('预测生还人数：', submission['Survived'].sum(), '/', len(submission))
#提交分数为0.77511