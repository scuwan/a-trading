import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
#from tqdm.auto import tqdm
from datetime import datetime

import stock_analysis as sa


class TradingDataset(Dataset):
    def __init__(self, features, labels, transform=None, target_transform=None):
        self.features = features
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, index):
        feature = self.features[index]
        label = self.labels[index]
        if self.transform:
            feature = self.transform(feature)
        if self.target_transform:
            label = self.target_transform(label)
        return feature, label


class TradingModel(nn.Module):
    def __init__(self):
        super(TradingModel, self).__init__()
        self.linear_relu_stack = nn.Sequential( 
            nn.Linear(27, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 5),
            nn.ReLU()
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (features, labels) in enumerate(dataloader):
        # compute prediction and loss
        pred = model(features)
        loss = loss_fn(pred, labels)
        print(pred)
        print(labels)
        # print(loss.item())

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch*len(features) 
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    # print_weights(model)

def eval_loop(dataloader, model, loss_fn):
    # print_weights(model)
    size = len(dataloader.dataset)
    test_loss = 0
    with torch.no_grad():
        for features, labels in dataloader:
            print(features)
            pred = model(features)
            print(pred)
            print(labels)
            test_loss += loss_fn(pred, labels).item()
    test_loss /= size
    print(f"Avg loss: {test_loss:>8f}\n")

# print weights
def print_weights(model):
    for param in model.parameters():
        print(param.data)
        break

def train():
    hist = sa.fetch_hist('601398')
    features, labels = sa.aggregate_stock_hist_sliding_window(hist, ndays= 5)
    data_length = len(features)
    train_data_length = int(data_length * 0.9)
    train_features = features[:train_data_length]
    train_labels = labels[:train_data_length]
    eval_features = features[train_data_length:]
    eval_labels = labels[train_data_length:]
    train_dataset = TradingDataset(train_features, train_labels, torch.FloatTensor, torch.FloatTensor)
    eval_dataset = TradingDataset(eval_features, eval_labels, torch.FloatTensor, torch.FloatTensor)
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle = True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=16)

    # setting hyperparameters
    learning_rate = 1e-4
    epochs = 20

    # 创建模型
    model = TradingModel()
    # print_weights(model)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        train_loop(train_dataloader, model, loss_fn, optimizer)
        eval_loop(eval_dataloader, model, loss_fn)
    
    # print_weights(model)
    time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    torch.save(model.state_dict(), f'model/601398_{time}.pth')

train()