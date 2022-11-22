import random
import torch.nn as nn
import torch
import torch.nn.functional as F
from sklearn.preprocessing import OneHotEncoder as OneHotEncoder
import numpy as np
import warnings

warnings.filterwarnings("ignore")


class NameGenerator(nn.Module):
    def __init__(self):
        super(NameGenerator, self).__init__()
        self.hidden_layer_size = 10
        self.lstm = nn.LSTM(input_size=27, hidden_size=self.hidden_layer_size)
        self.fc1 = nn.Linear(self.hidden_layer_size, 300)
        self.fc2 = nn.Linear(300, 27)


    def forward(self, x):
        out, (self.hidden, self.cell) = self.lstm(x)
        # self.hidden = self.hidden.detach()
        # self.cell = self.cell.detach()
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.sigmoid(out)
        return out


model = NameGenerator()
model.load_state_dict(torch.load('0702-667647397-Hong.pt'))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# a

onehot_map = [['@'], ['a'], ['b'], ['c'], ['d'], ['e'], ['f'], ['g'], ['h'], ['i'], ['j'], ['k'], ['l'], ['m'], ['n'],
              ['o'], ['p'], ['q'], ['r'], ['s'], ['t'], ['u'], ['v'], ['w'], ['x'], ['y'], ['z']]
enc = OneHotEncoder()
enc.fit(onehot_map)


def generate_letter(letter, model):
    while len(letter) < 11:
        letter += '@'
    list_letter = list(letter)
    for i in range(len(list_letter)):
        list_letter[i] = [list_letter[i]]
    letter_array = enc.transform(list_letter).toarray()
    letter_array = torch.tensor(letter_array, dtype=torch.float).resize(11, 27)
    model.eval()
    predict = model(letter_array)
    predict = predict.detach().numpy()
    set_rand = np.random.normal(0, 0.5, (11, 27)).astype(np.float32)+1
    predict = np.multiply(predict, set_rand)
    name_list = []
    for item in predict:
        name_list.append(onehot_map[int(item.argmax())][0])
    name_str = "".join(name_list)
    return name_str


def generate_name(letter, model):
    name_list = []
    for j in range(20):
        name = letter
        length_of_name = random.randint(2, 10)
        for i in range(length_of_name):
            next_letter = generate_letter(name, model)
            if next_letter[i] == '@':
                continue
            name = name + next_letter[i]
        name_list.append(name)
    for item in name_list:
        print(item)

