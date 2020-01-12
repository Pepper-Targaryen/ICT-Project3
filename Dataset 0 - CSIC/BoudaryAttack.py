import torch
import numpy as np
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim
from tqdm import *
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset, TensorDataset
# from utils import *
from foolbox.models import PyTorchModel
from foolbox.attacks import BoundaryAttack


class Args:
    def __init__(self):
        self.embed_num = 128
        self.embed_dim = 5
        self.class_num = 2
        self.kernel_num = 100
        self.kernel_sizes = [3, 5, 7, 9]
        #self.dropout = 0.3
        self.static = True


class CNN_Text_dropout(nn.Module):

    def __init__(self, args):
        super(CNN_Text_dropout, self).__init__()
        self.args = args

        V = args.embed_num
        D = args.embed_dim
        C = args.class_num
        Ci = 1
        Co = args.kernel_num
        Ks = args.kernel_sizes

        #self.embed = nn.Embedding(V, D)
        # self.convs1 = [nn.Conv2d(Ci, Co, (K, D)) for K in Ks]
        self.convs1 = nn.ModuleList([nn.Conv2d(
            in_channels=Ci, out_channels=Co, kernel_size=(K, D), stride=1) for K in Ks])

        #self.dropout = nn.Dropout(args.dropout)
        self.dropout1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(len(Ks)*Co, 256)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 64)
        self.dropout3 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(64, C)
        self.relu = nn.ReLU()

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def weight_init(self):
        for module in self.convs1:
            torch.nn.init.xavier_uniform_(module.weight)
            torch.nn.init.constant_(module.bias, 0.1)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.constant_(self.fc1.bias, 0.1)

    def forward(self, x):

        x = x.unsqueeze(1)  # (N, Ci, W, D)

        x = [F.relu(conv(x)).squeeze(3)
             for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)

        x = [F.max_pool1d(i, i.size(2)).squeeze(2)
             for i in x]  # [(N, Co), ...]*len(Ks)

        x = torch.cat(x, 1)

        self.features = x
        x = self.dropout1(x)
        x = self.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.relu(self.fc2(x))
        x = self.dropout3(x)
        logit = self.fc3(x)
        #print('in the FP')
        # print(x.shape)
        #logit = self.fc3(self.relu(self.fc2()))
        return logit


def Tensor_to_Log(symbol_dict, input_tensor, find_nearest=True, end_filter=True):
    count = 0
    string = ''
    input_tensor = input_tensor.squeeze()
    for i in range(input_tensor.shape[0]):
        if not find_nearest:
            for ele in symbol_dict:
                # precision problem

                if torch.sum(abs(symbol_dict[ele] - input_tensor[i].cpu()) > 1e-5).item() == 0:
                    # if ele != ' ':
                    string += ele
                    count += 1
                    # print(ele)
                    break
        else:
            # find the nearest vector
            error = 1  # should decrease the error
            good_symbol = ''
            for symbol in symbol_dict:
                temp = torch.sum(
                    (symbol_dict[symbol] - input_tensor[i].cpu()) ** 2)
                if error > temp:
                    error = temp
                    good_symbol = symbol
                    # print(error)
            count += 1
            string += good_symbol

            # filter other characters at the end
            if end_filter and good_symbol == '\x00':
                for _ in range(400 - count):
                    string += '\x00'
                break
    return string


def Log_to_Tensor(symbol_dict, log):
    Tensor = torch.zeros([400, 5])
    for i in range(400):
        Tensor[i] = symbol_dict['\x00']
    for i, e in enumerate(log):
        for ele in symbol_dict:
            if e == ele:
                Tensor[i] = symbol_dict[ele]
    return Tensor


def find_string_from_tensor(X):
    result = []
    for x in X:
        string = ""
        for v in x:
            string += chr(int(v * 128))
        result.append(string)
    return result


def find_nearest_adversial(origins, candidate, distance_metric):
    dic = {}
    assert (len(candidate) > 0), "There should be at least one candidate"
    for i in tqdm.tqdm_notebook(origins):
        min_norm = None
        min_candidate = None
        for j in candidate:
            if min_norm is None:
                min_norm = distance_metric(i, j)
                min_candidate = j
            else:
                new_distance = distance_metric(i, j)
                min_norm, min_candidate = (
                    min_norm, min_candidate) if new_distance < min_norm else (new_distance, j)
        dic[i] = min_candidate
    return dic


def str_similarity(str_a, str_b):
    #str_a = Tensor_to_Log(symbol_dict, tensor_a)
    #str_b = Tensor_to_Log(symbol_dict, tensor_b)
    return difflib.SequenceMatcher(a=str_a, b=str_b).ratio()


def main():
    X_train = np.load("./Data/sGrid/X_train.npy")
    X_test = np.load("./Data/sGrid/X_test.npy")
    X_vaild = np.load("./Data/sGrid/X_vaild.npy")
    Y_train = np.load("./Data/sGrid/Y_train.npy")
    Y_test = np.load("./Data/sGrid/Y_test.npy")
    Y_vaild = np.load("./Data/sGrid/Y_vaild.npy")

    torch.manual_seed(1)
    embedding = nn.Embedding(128, 5, max_norm=1)

    Y_train = torch.from_numpy(Y_train)
    Y_test = torch.from_numpy(Y_test)
    Y_vaild = torch.from_numpy(Y_vaild)

    input = Variable(torch.from_numpy(X_train*128).long())
    X_train_embed = embedding(input)
    X_train_embed = X_train_embed.detach()

    input = Variable(torch.from_numpy(X_test*128).long())
    X_test_embed = embedding(input)
    X_test_embed = X_test_embed.detach()

    input = Variable(torch.from_numpy(X_vaild*128).long())
    X_vaild_embed = embedding(input)
    X_vaild_embed = X_vaild_embed.detach()

    dic = {}
    count = 0
    for i in range(X_train.shape[0]):
        for j in range(400):
            if chr(int(X_train[i, j]*128)) not in dic.keys():
                dic[chr(int(X_train[i, j]*128))] = X_train_embed[i, j]

    symbol_dict = dic

    args = Args()

    net = CNN_Text_dropout(args).cuda()
    print(net)

    pretrained_dict = torch.load(
        'Parameters/cnn_text_kernel3.5.7.9_128_embed_dropout.pkl').state_dict()
    model_dict = net.state_dict()
    pretrained_dict = {k: v for k,
                       v in pretrained_dict.items() if k in model_dict}
    # 更新现有的model_dict
    model_dict.update(pretrained_dict)
    # 加载我们真正需要的state_dict
    net.load_state_dict(model_dict)

    batch_size = 500
    Train_data = Data.TensorDataset(X_train_embed, Y_train)
    Test_data = Data.TensorDataset(X_test_embed, Y_test)
    train_data = Data.DataLoader(
        dataset=Train_data, batch_size=batch_size, shuffle=False)
    test_data = Data.DataLoader(dataset=Test_data, batch_size=1, shuffle=False)
    optimizer = optim.Adam(net.parameters(), lr=0.0001, weight_decay=1e-9)
    loss_function = nn.CrossEntropyLoss()

    attack_log_list = None
    attack_log_string_list = []

    net.eval()

    # This is the begin of the attack
    # model and boudary attack
    model = PyTorchModel(net, (-1, 1), 2, )
    attack = BoundaryAttack(model)

    # find the nearest attack sample as the starting point
    X_test_string = find_string_from_tensor(X_test)
    dict_attack_string_tensor = {}
    for i in range(len(X_train)):
        x, label = X_train[i], int(Y_train[i].numpy()[0])
        # the prediction of an attack sample should be an attack
        if label == 1 and np.argmax(model.predictions(X_train_embed[i].numpy())) == 1:
            string = ""
            for v in x:
                string += chr(int(v * 128))
            ''' duplication of attack
            if string in dict_attack_string_tensor:
                print(string)
            '''
            dict_attack_string_tensor[string] = X_train_embed[i]

    n_test = 100
    dict_nearest_str = find_nearest_adversial(X_test_string[:n_test], list(
        dict_attack_string_tensor.keys()), str_similarity)
    list_X_test_nearest_tensor = []
    for log in X_test_string[:n_test]:
        list_X_test_nearest_tensor.append(
            dict_attack_string_tensor[dict_nearest_str[log]])

    # begin the attack
    try_time = 1
    max_iteration = 50
    n_success = 0
    n_total = 0
    iterations = []

    file = open(
        f'./Data/boundary_attack_unfixed_iteration_nearest_starting_max_{max_iteration}_test_{n_test}.txt', "w")
    for i in tqdm.tqdm_notebook(range(n_test)):
        url, label = X_test_embed[i].numpy(), int(Y_test[i].numpy()[0])
        prediction = np.argmax(model.predictions(url))

        if label == 0 and prediction == 0:
            n_total += 1
            good_adversarial = None
            good_iteration = 0
            for iteration in range(max_iteration+1):
                adversarial = attack(url, label, starting_point=list_X_test_nearest_tensor[i].numpy(
                ), log_every_n_steps=20, iterations=iteration)

                # adversarial log
                str_adversarial = Tensor_to_Log(
                    symbol_dict, torch.from_numpy(adversarial))
                # need to change the adversarial string back to the tensor
                prediction = np.argmax(model.predictions(
                    Log_to_Tensor(symbol_dict, str_adversarial).numpy()))
                if prediction == 1:
                    good_iteration = iteration
                    good_adversarial = adversarial

            if not good_adversarial is None:
                n_success += 1
                iterations.append(good_iteration)
                # original log
                file.write(X_test_string[i])
                file.write("\n")

                # adversarial log
                file.write(Tensor_to_Log(
                    symbol_dict, torch.from_numpy(good_adversarial)))
                file.write("\n\n")

    file.close()


main()
