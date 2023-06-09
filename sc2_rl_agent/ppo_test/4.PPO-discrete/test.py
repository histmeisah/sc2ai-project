import torch
import torch.nn as nn


class AlphaStarLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(AlphaStarLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden=None):
        if hidden is None:
            batch_size = x.size(1)
            hidden = (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                      torch.zeros(self.num_layers, batch_size, self.hidden_size))
        lstm_out, hidden = self.lstm(x, hidden)
        output = self.fc(lstm_out[-1][-1].unsqueeze(0))
        return output, hidden


x = torch.randn((100, 128))
print(x.shape)
input_dim = 128
hidden_dim = 10
num_layers = 5
output_size = 16
seq_len, input_size = x.shape
print(f'seq_len:{seq_len},input_size={input_size}')

# 将数据切分为n个样本
samples = []
for i in range(seq_len):
    sample = x[i].unsqueeze(0)  # 将第i个时间步的输入转换为1x128的张量
    samples.append(sample)

# 将样本拼接为一个张量，形状为(n, 1, 128)
x = torch.cat(samples, dim=0).unsqueeze(1)
print(type(x))
lstm = AlphaStarLSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, output_size=output_size)
y = lstm(x)

y, _ = y
print(y.shape)
