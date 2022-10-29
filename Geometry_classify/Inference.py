import cv2 as cv
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


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
        self.flatten = nn.Flatten()

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 4)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 3)
        out = F.relu(self.conv3(out))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv4(out))
        out = out.flatten()
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


model = This_is_a_Net()
model.load_state_dict(torch.load('0602-667647397-Hong.pt'))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
transform = transforms.ToTensor()

model.eval()
geo_class = {0: 'Circle', 1: 'Heptagon', 2: 'Hexagon', 3: 'Nonagon', 4: 'Octagon', 5: 'Pentagon', 6: 'Square',
             7: 'Star', 8: 'Triangle'}
path = "./datasets/validation"
for img_name in os.listdir(path):
    img_path = path + '/' + img_name
    img = cv.imread(img_path)
    img_in = transform(img)
    prediction = int(model(img_in).argmax(0))
    print('Image:{},  Class:{}'.format(img_name, geo_class[prediction]))
