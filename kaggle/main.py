# 导入必要的库
import numpy as np
import pandas as pd
import os# 用于遍历文件夹
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import torch# 用于深度学习框架
import torch.nn as nn# 神经网络库
import torch.nn.functional as F# 神经网络函数库
import torch.optim as optim # 优化器库
# 读取训练数据集
train_df = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
train_df
# 读取测试数据集
test_df = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
test_df
# 从训练数据集中分离特征和标签
X_train = train_df.drop('label', axis = 1)# 特征，排除标签列
Y_train = train_df.label.to_numpy()# 标签，转换为numpy数组
X_train /= 255# 将像素值归一化到0-1之间
X_train = X_train.to_numpy() # 转换为numpy数组

x_train = []# 初始化存储处理后图像的列表
x_test = []# 初始化存储处理后图像的列表

X_test = test_df
X_test /= 255# 将像素值归一化到0-1之间
X_test = X_test.to_numpy()# 标签，转换为numpy数组

for i in range(len(X_train)):# 遍历每个测试样本
    x_train.append(X_train[i].reshape((28,28)).astype(np.float32))# 将图像重塑为28x28，并转换为float32类型，然后添加到列表中

for i in range(len(X_test)):# 遍历每个测试样本
    x_test.append(X_test[i].reshape((28,28)).astype(np.float32))# 将图像重塑为28x28，并转换为float32类型，然后添加到列表中

x_train = np.array(x_train) # 将列表转换为numpy数组，以便于后续操作
x_test = np.array(x_test) # 将列表转换为numpy数组，以便于后续操作
# 使用matplotlib库显示一个处理后的训练样本图像
import matplotlib.pyplot as plt# 导入matplotlib库
plt.imshow(x_train[10])# 选择第11个训练样本图像（索引从0开始）进行显示
plt.show()# 显示图像
# 定义CNN类，继承自nn.Module
class CNN(nn.Module):
    def __init__(self, in_channels, num_classes):# 初始化函数，接收输入通道数和类别数作为参数
        super(CNN, self).__init__()# 调用父类nn.Module的初始化函数
        # 定义第一个卷积层，输入通道数为in_channels，输出通道数为32，卷积核大小为5x5
        self.conv1 = nn.Conv2d(in_channels, out_channels=32, kernel_size=5)
        # 定义最大池化层，池化核大小为2x2
        self.pool = nn.MaxPool2d(kernel_size=2)
        # 定义第二个卷积层，输入通道数为32（与第一个卷积层的输出通道数一致），输出通道数为64，卷积核大小为3x3
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        # 定义丢弃层，丢弃率为0.25
        self.dropout = nn.Dropout(0.25)
        # 定义全连接层1，输入特征数为1600（根据前一层输出大小计算得出），输出特征数为16
        self.fc1 = nn.Linear(in_features=1600, out_features=16)
        # 定义全连接层2，输入特征数为16，输出特征数为num_classes
        self.fc2 = nn.Linear(in_features=16, out_features=num_classes)

    def forward(self, x):# 前向传播函数
        # 通过第一个卷积层，激活函数为ReLU
        x = F.relu(self.conv1(x))
        # 通过最大池化层
        x = self.pool(x)
        # 通过第二个卷积层，激活函数为ReLU
        x = F.relu(self.conv2(x))
        # 通过最大池化层（与上一步的池化层输出大小一致）
        x = self.pool(x)
        # 通过丢弃层，随机丢弃部分神经元，防止过拟合
        x = self.dropout(x)
        # 将特征图展平为一维向量，方便全连接层处理
        x = x.reshape(x.shape[0], -1)
        # 通过全连接层1，激活函数为ReLU
        x = F.relu(self.fc1(x))
        # 通过全连接层2，输出最终的分类结果
        x = self.fc2(x)

        return x# 返回分类结果
# 定义一个函数，用于创建数据批量
def create_batch(x_data, y_data, batch_size=32):
    # 初始化索引为0
    i = 0
    # 初始化一个空列表，用于存储批量数据
    batch = []
    # 当索引加上批量大小小于数据总长度时，继续循环
    while (i + batch_size < len(x_data)):
        # 从x_data中提取一个批量数据，并转换为PyTorch张量
        x = torch.from_numpy(x_data[i: i + batch_size])
        # 从y_data中提取一个批量数据，并转换为PyTorch张量
        y = torch.from_numpy(y_data[i: i + batch_size])
        # 索引增加批量大小
        i += batch_size
        # 将提取的批量数据添加到batch列表中
        batch.append([x, y])
    # 如果x_data的长度不能被batch_size整除，则从x_data中提取剩余的数据，并转换为PyTorch张量
    x = torch.from_numpy(x_data[i:])
    # 如果y_data的长度不能被batch_size整除，则从y_data中提取剩余的数据，并转换为PyTorch张量
    y = torch.from_numpy(y_data[i:])
    # 将剩余的批量数据添加到batch列表中
    batch.append([x, y])

    return batch# 返回包含所有批量数据的列表
# 检查是否有可用的GPU，如果有则使用GPU，否则使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# 设置训练周期数为25
epochs = 25
# 设置批量大小为32
batch_size = 32
# 设置学习率为0.01
lr = 0.01
# 使用create_batch函数创建训练批量数据
train_batch = create_batch(x_train, Y_train, batch_size)
# 定义一个函数，用于创建测试数据批量
def create_test_batch(x_data, batch_size):
    i = 0# 初始化索引为0
    batch = []# 初始化一个空列表，用于存储批量数据
    while (i + batch_size < len(x_data)):# 当索引加上批量大小小于数据总长度时，继续循环
        x = torch.from_numpy(x_data[i: i + batch_size])# 从x_data中提取一个批量数据，并转换为PyTorch张量
        i += batch_size# 索引增加批量大小
        batch.append(x)# 将提取的批量数据添加到batch列表中
    x = torch.from_numpy(x_data[i:])# 处理剩余的数据（如果存在）
    batch.append(x)# 将剩余的批量数据添加到batch列表中

    return batch # 返回包含所有批量数据的列表
# 使用定义的函数创建测试批量数据
test_batch = create_test_batch(x_test, batch_size)
# 加载已训练好的CNN模型
model = CNN(1, 10).to(device)
# 定义交叉熵损失函数
criterion = nn.CrossEntropyLoss()
# 定义Adam优化器
optimizer = optim.Adam(model.parameters(), lr)
# 初始化一个空列表，用于存储模型预测的结果
results = []
# 定义一个函数，用于计算模型的准确率
def get_accuracy(model, data):
        model.eval()# 设置模型为评估模式，关闭dropout和batch normalization层的学习状态

        with torch.no_grad():# 不计算梯度，减少内存使用
            for x in data:# 遍历数据集中的每一个样本
                x = x.to(device)# 将数据转移到指定的设备（如GPU）上

                x = x.reshape(32, 1, 28, 28)# 将数据重新塑形为CNN所需的输入形状

                scores = model(x)# 将样本输入到模型中得到得分或输出
                _, predictions = scores.max(1)# 通过最大值操作获取预测类别（假设有10个类别）
                results.append(predictions)# 将预测结果添加到results列表中
# 遍历所有的训练周期
for epoch in range(epochs):
    # 初始化一个累加器来计算损失
    running_loss = 0.0
    # 遍历每一个训练批量
    for x, y in train_batch:
        # 将数据移动到指定的设备（如GPU）上
        x = x.to(device)
        y = y.to(device)
        # 重新塑形输入数据以匹配模型的期望输入形状
        x = x.reshape(x.shape[0], 1, 28, 28)
        # 使用模型进行预测
        scores = model(x)
        # 计算损失
        loss = criterion(scores, y)
        # 累加损失
        running_loss += loss
        # 清零梯度缓存，为反向传播做准备
        optimizer.zero_grad()
        # 进行反向传播，计算梯度
        loss.backward()
        # 使用优化器更新权重
        optimizer.step()
# 打印当前周期的训练损失
print(f"Epoch-{epoch + 1} | Training Loss {running_loss / len(train_batch)}")
# 使用模型进行测试并计算准确率
get_accuracy(model, test_batch)
# 收集模型的预测结果（类别标签）
y_preds = []
for i in results:
    for j in i:
        y_preds.append(j.item())
# 创建一个索引列表，与预测的标签相对应（假设有28000个样本）
index = []
for i in range(1, 28001):
    index.append(i)
# 读取样本提交文件，以便将预测结果写入到该文件中
df = pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')
df# 显示数据框内容，这里df包含了提交文件的格式信息，但不包含真实标签。
# 将模型的预测结果赋值给数据框的'Label'列
df['Label'] = y_preds
df.to_csv('submission.csv', index=False)# 将预测结果写入到提交文件中，不包括索引列。
df# 再次显示数据框内容，现在它包含了预测的标签。
