import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import cv2 as cv
import os


class This_is_a_Net(nn.Module):
    def __init__(self):
        super(This_is_a_Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.conv2 = nn.Conv2d(32, 128, 5)
        self.conv3 = nn.Conv2d(128, 512, 5)
        self.conv4 = nn.Conv2d(512, 1024, 3)
        self.fc1 = nn.Linear(9216, 120)
        self.fc2 = nn.Linear(120, 36)
        self.fc3 = nn.Linear(36, 9)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 4)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 3)
        out = F.relu(self.conv3(out))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv4(out))
        print(out.shape)
        out = out.view(out.size(0), -1)
        print(out.shape)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


def train(batch_size, log_interval, model, device, train_loader, optimizer, epoch, loss_list, acc_list):
    model.train()
    tot_loss = 0
    correct = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        output = model(data)
        loss = torch.nn.CrossEntropyLoss()(output, target)
        loss.backward()
        optimizer.step()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

        tot_loss = tot_loss + loss.item()
        acc = correct / ((batch_idx + 1) * batch_size)
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accuracy: {:.2f}%'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), tot_loss / (batch_idx + 1), 100.0 * acc))
    loss_list.append(tot_loss / (len(train_loader)))
    acc_list.append(correct / (len(train_loader) * batch_size))
    print('End of Epoch: {}'.format(epoch))
    print('Training Loss: {:.6f}, Training Accuracy: {:.2f}%'.format(
        tot_loss / (len(train_loader)), 100.0 * correct / (len(train_loader) * batch_size)))


def test(test_batch_size, model, device, test_loader, loss_list, acc_list):
    model.eval()
    tot_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            tot_loss += torch.nn.CrossEntropyLoss()(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss = tot_loss / (len(test_loader))
    test_acc = correct / (len(test_loader) * test_batch_size)
    loss_list.append(test_loss)
    acc_list.append(test_acc)
    print('Test Loss: {:.6f}, Test Accuracy: {:.2f}%'.format(
        test_loss, 100.0 * test_acc))


def read_directory(path):
    class_count = 1
    if not os.path.exists('./train_data'):
        os.mkdir('./train_data')
    if not os.path.exists('./test_data'):
        os.mkdir('./test_data')
    split = 1

    for img_name in os.listdir(path):
        label = img_name.split('_')[0]
        if not os.path.exists('./train_data/' + label):
            os.mkdir('./train_data/' + label)
        if not os.path.exists('./test_data/' + label):
            os.mkdir('./test_data/' + label)
        img = cv.imread(path + '/' + img_name)

        if split <= 8000:
            cv.imwrite('./train_data/' + label + '/' + img_name, img)
        else:
            cv.imwrite('./test_data/' + label + '/' + img_name, img)

        if split >= 10000:
            split = 1
            print("Progress:", class_count, "/9")
            class_count += 1
        split += 1


img_path = './geometry_dataset/output'
read_directory(img_path)

batch_size = 32
test_batch_size = 1000
epochs = 3
lr = 4e-4
gamma = 0.7
seed = 1
log_interval = 100
save_model = True
torch.manual_seed(seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_data = torchvision.datasets.ImageFolder('./datasets/train', transform=transforms.ToTensor())
test_data = torchvision.datasets.ImageFolder('./datasets/test', transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=test_batch_size, shuffle=True)
model = This_is_a_Net().to(device)

optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
train_loss_list = []
train_acc_list = []
test_loss_list = []
test_acc_list = []
for epoch in range(1, epochs + 1):
    train(batch_size, log_interval, model, device, train_loader, optimizer, epoch, train_loss_list, train_acc_list)
    test(test_batch_size, model, device, test_loader, test_loss_list, test_acc_list)
    scheduler.step()

torch.save(model.state_dict(), "0602-667647397-Hong.pt")

def plot_loss_acc_img(epochs, line1, line2):
    plt.rc('axes', facecolor='white')
    plt.rc('figure', figsize=(6, 4))
    plt.rc('axes', grid=False)
    plt.plot(range(1, epochs + 1), line1, '.:r', range(1, epochs + 1), line2, ':b')
    plt.legend(['Train', "Test"], loc='upper right')

    #     plt.title('Loss-Epoch')
    #     plt.ylabel('Loss')
    plt.title('Accuracy-Epoch')
    plt.ylabel('Accuracy')

    plt.xlabel('Epoch')
    plt.show()
