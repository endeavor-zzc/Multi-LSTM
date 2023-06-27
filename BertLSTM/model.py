"""
@Project : Multi-LSTM (1)
@File    : model.py
@Author  : endeavor
@Brief   : This is a file for building models that includes three classes: LSTM, CNN, and LSTM_CNN
"""

import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        # print(x.shape)
        # 前向传播 LSTM
        out, _ = self.lstm(x, (h0, c0))
        # out = out.reshape(out.size(0), -1)

        # 取最后一个时间步的输出
        # out = self.fc(out[:, -1, :])
        # out, (h0, c0) = self.lstm(x)
        # print(out.shape)
        return out


# 定义CNN模型
class CNNModel(nn.Module):
    def __init__(self, input_channels, output_size):
        super(CNNModel, self).__init__()

        # 定义卷积层和池化层
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=4, padding=1)  # 输出通道，卷积核3，
        self.pool1 = nn.AvgPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=4, padding=1)
        self.pool2 = nn.AvgPool1d(kernel_size=2)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=4, padding=1)
        self.pool3 = nn.AvgPool1d(kernel_size=2)


        # 定义全连接层
        self.fc1 = nn.Linear(896, 512)  # !!! before: self.fc1 = nn.Linear(128*6*6, 512)
        self.fc2 = nn.Linear(512, output_size)
        self.dropout = nn.Dropout(p=0.5)
        # 定义激活函数
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 卷积层和池化层
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.relu(self.conv3(x))
        x = self.pool3(x)


        # 将张量展平为向量
        x = x.view(x.size(0), -1)

        # 全连接层和激活函数
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x


class LSTM_CNN_Model(nn.Module):
    def __init__(self, lstm_input_size, lstm_hidden_size, lstm_num_layers, lstm_num_classes,
                 cnn_input_channels, cnn_output_size):
        super(LSTM_CNN_Model, self).__init__()

        # 定义LSTM模型
        self.lstm = LSTMModel(lstm_input_size, lstm_hidden_size, lstm_num_layers, lstm_num_classes)

        # 定义CNN模型
        self.cnn = CNNModel(cnn_input_channels, cnn_output_size)

    def forward(self, x1, x2, x3):
        # 输入三批数据到LSTM模型提取特征
        lstm_out1 = self.lstm(x1)
        lstm_out2 = self.lstm(x2)
        lstm_out3 = self.lstm(x3)

        # 将LSTM模型的输出拼接在一起
        lstm_out = torch.cat((lstm_out1, lstm_out2, lstm_out3), dim=1)

        # 输入LSTM模型的输出到CNN模型训练
        cnn_out = self.cnn(lstm_out)

        return cnn_out
