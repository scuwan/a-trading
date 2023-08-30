# https://medium.com/@prajjwalchauhan94017/stock-prediction-and-forecasting-using-lstm-long-short-term-memory-9ff56625de73
import pandas as pd
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
from datetime import datetime
import fire

class TradingDataset(Dataset):
    def __init__(self, data, time_step = 1):
          self.data_x = []
          self.data_y = []
          for i in range(len(data) - time_step - 1):
            self.data_x.append(data[i:(i+time_step)])
            self.data_y.append(data[i+1: i + time_step + 1])
    def __len__(self):
        return len(self.data_x)
    def __getitem__(self, index):
        feature = self.data_x[index]
        label = self.data_y[index]
        return torch.unsqueeze(feature, 1), torch.unsqueeze(label, 1)

class TradingModel(nn.Module):
    def __init__(self, time_step):
        super(TradingModel, self).__init__()
        self.lstm = nn.LSTM(1, 50, 3, batch_first=True).double()
        self.linear = nn.Linear(50, 1).double()
    def forward(self, x):
        output,(hn,cn) = self.lstm(x)
        logits = self.linear(output)
        return logits

def process_data(path):
    data = pd.read_csv(path)
    data = data.reset_index()['收盘']
    data = torch.tensor(data.values)
    min_val, max_val = data.aminmax()
    data = (data - min_val) / (max_val - min_val)
    return data, min_val, max_val

def prepare_dataset(data, time_step=1):
    train_size = int(len(data) * 0.65)
    train_data = data[0:train_size]
    test_data = data[train_size:]
    train_dataset = TradingDataset(train_data, time_step=time_step)
    test_dataset = TradingDataset(test_data, time_step=time_step)
    whole_dataset = TradingDataset(data, time_step)
    return train_dataset, test_dataset, whole_dataset


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (features, labels) in enumerate(dataloader):
        # compute prediction and loss
        pred = model(features)
        loss = loss_fn(pred, labels)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            loss, current = loss.item(), batch*len(features) 
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def eval_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    test_loss = 0
    with torch.no_grad():
        for features, labels in dataloader:
            pred = model(features)
            test_loss += loss_fn(pred, labels).item()
    test_loss /= size
    print(f"Avg loss: {test_loss:>8f}\n")

def train(path="./a_2023_08_23/601398_hfq.csv", batch_size=64, learning_rate=1e-3, epochs=100, time_step=100):
    data, _, _ = process_data(path)
    train_dataset, test_dataset, _ = prepare_dataset(data, time_step=time_step)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle = True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    model = TradingModel(time_step=time_step)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        print(f'epoch {epoch}>>>')
        train_loop(train_dataloader, model, loss_fn, optimizer)
        eval_loop(test_dataloader, model, loss_fn)
    if not os.path.exists("model"):
        os.mkdir("model")
    time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    torch.save(model.state_dict(), f'model/601398_lstm_{time}.pth')
    print(f'model_pth: model/601398_lstm_{time}.pth')

def predict(model_pth='model/601398_lstm_2023-08-29_11-51-03.pth', data_path = "./a_2023_08_23/601398_hfq.csv", time_step=100):
    model = TradingModel(time_step=time_step)
    model.load_state_dict(torch.load(model_pth))
    model.eval()

    data, min_val, max_val = process_data(data_path)
    start_index = int(len(data) * 0.4)
    eval_data = data[start_index:]
    eval_data = torch.unsqueeze(eval_data, 1)

    predicts = model(eval_data)
    predicts = predicts * (max_val - min_val) + min_val
    print(predicts.shape)

    data = data * (max_val - min_val) + min_val
    print(data.shape)

    predicts_plot = np.random.rand(len(data) + 1)
    predicts_plot[:] = np.nan
    print(len(predicts))
    predicts_plot[start_index + 1 : start_index + 1 + len(predicts)] = predicts.flatten().detach().numpy()

    plt.plot(data)
    plt.plot(predicts_plot)
    plt.show()

if __name__ == "__main__":
    fire.Fire({
        'train': train,
        'predict': predict
    })