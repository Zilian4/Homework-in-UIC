import numpy as np
import torch.nn as nn
import torch
from torch.utils.data.dataset import Dataset as Dataset
import torch.nn.functional as F
from sklearn.preprocessing import OneHotEncoder as OneHotEncoder
from torch.optim.lr_scheduler import StepLR as StepLR
import matplotlib.pyplot as plt


class NameDataset(Dataset):
    def __init__(self, path):
        super(NameDataset, self).__init__()
        names = []
        data = load_data(path)
        for i in range(len(data[1])):
            x = data[0][i]
            y = data[1][i]
            names.append((x, y))
        self.names = names

    def __getitem__(self, index):
        x, y = self.names[index]
        x = torch.tensor(x, dtype=torch.float).resize(11, 27)
        y = torch.tensor(y, dtype=torch.float).resize(11, 27)
        return x, y

    def __len__(self):
        return len(self.names)


class NameGenerator(nn.Module):
    def __init__(self):
        super(NameGenerator, self).__init__()
        self.hidden_layer_size = 10
        self.lstm = nn.LSTM(input_size=27, hidden_size=self.hidden_layer_size)
        self.fc1 = nn.Linear(self.hidden_layer_size, 300)
        self.fc2 = nn.Linear(300, 27)

        self.hidden = torch.zeros(1, 11, 10)
        self.cell = torch.zeros(1, 11, 10)

    def forward(self, x):
        out, (self.hidden, self.cell) = self.lstm(x, (self.hidden, self.cell))
        self.hidden = self.hidden.detach()
        self.cell = self.cell.detach()
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        return out


# the longest name has 11 strs
def load_data(file_path):
    # transfer to onehot

    onehot_map = [['@'], ['a'], ['b'], ['c'], ['d'], ['e'], ['f'], ['g'], ['h'], ['i'], ['j'], ['k'], ['l'], ['m'],
                  ['n'], ['o'], ['p'], ['q'], ['r'], ['s'], ['t'], ['u'], ['v'], ['w'], ['x'], ['y'], ['z']]
    enc = OneHotEncoder()
    enc.fit(onehot_map)
    # read file
    with open(file_path) as file:
        data_x = file.read()
        data_y = []
    data_x = data_x.split('\n')
    for i in range(len(data_x)):
        name_x = data_x[i].lower()
        name_y = name_x[1:]
        while len(name_x) < 11:
            name_x = name_x + '@' * (11 - len(name_x))
        while len(name_y) < 11:
            name_y = name_y + '@'
        list_x = []
        list_y = []
        for j in range(len(name_x)):
            x_str = []
            y_str = []
            x_str.append(name_x[j])
            y_str.append(name_y[j])
            list_x.append(x_str)
            list_y.append(y_str)
        list_x = enc.transform(list_x).toarray()
        list_y = enc.transform(list_y).toarray()
        data_x[i] = list_x
        data_y.append(list_y)
    data_x = np.array(data_x)
    data_y = np.array(data_y)
    return (data_x, data_y)


def train(batch_size, log_interval, model, device, train_loader, optimizer, epoch, loss_list):
    print("Training Start")
    model.train()
    tot_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        output = model(data)
        loss = torch.nn.CrossEntropyLoss()(output, target)
        torch.autograd.set_detect_anomaly(True)
        loss.backward(retain_graph=True)
        optimizer.step()
        tot_loss = tot_loss + loss.item()

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]'.format(epoch, batch_idx * len(data), len(train_loader.dataset),
                                                             100. * batch_idx / len(train_loader)))
    loss_list.append(tot_loss / (len(train_loader)))

    print('End of Epoch: {}'.format(epoch))
    print('Training Loss: {:.6f}, '.format(tot_loss / (len(train_loader))))


def plot_loss_acc_img(epochs, line1):
    plt.rc('axes', facecolor='white')
    plt.rc('figure', figsize=(6, 4))
    plt.rc('axes', grid=False)
    plt.plot(range(1, epochs + 1), line1, '.:r')
    plt.legend(['Train', "Test"], loc='upper right')

    plt.title('Loss-Epoch')
    plt.ylabel('Loss')

    plt.xlabel('Epoch')
    plt.show()


# setting hyper-parameters
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
path = './names.txt'
batch_size = 64
epochs = 30
lr = 0.01
gamma = 0.8
seed = 1
log_interval = 10
save_model = True
torch.manual_seed(seed)

train_data = NameDataset(path)
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

model = NameGenerator().to(device)
model.cell, model.hidden = model.cell.to(device), model.hidden.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
# optimizer = torch.optim.RAdam(model.parameters(), lr=lr)
scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

loss_list = []
for epoch in range(1, epochs + 1):
    train(batch_size, log_interval, model, device, train_loader, optimizer, epoch, loss_list)
    scheduler.step()
torch.save(model.state_dict(), "./0702-667647397-Hong.pt")
