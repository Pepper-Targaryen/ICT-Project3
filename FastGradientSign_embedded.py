import torch
import numpy as np
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim
from tqdm import *
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
from torch.utils.data import DataLoader,Dataset,TensorDataset
from utils import *

class Args:
    def __init__(self):
        self.embed_num = 128
        self.embed_dim = 5
        self.class_num = 2
        self.kernel_num = 100
        self.kernel_sizes = [3,5,7,9]
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
        self.convs1 = nn.ModuleList([nn.Conv2d(in_channels=Ci, out_channels=Co, kernel_size=(K, D),stride=1) for K in Ks])
        
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

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)

        x = torch.cat(x, 1)

        self.features = x
        x = self.dropout1(x)
        x = self.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.relu(self.fc2(x))
        x = self.dropout3(x)
        logit = self.fc3(x)
        #print('in the FP')
        #print(x.shape)
        #logit = self.fc3(self.relu(self.fc2()))
        return logit
    

def main():
    X_train=np.load("./Data/sGrid/X_train.npy")
    X_test=np.load("./Data/sGrid/X_test.npy")
    X_vaild=np.load("./Data/sGrid/X_vaild.npy")
    Y_train=np.load("./Data/sGrid/Y_train.npy")
    Y_test=np.load("./Data/sGrid/Y_test.npy")
    Y_vaild=np.load("./Data/sGrid/Y_vaild.npy")
    
    torch.manual_seed(1)
    embedding = nn.Embedding(128, 5, max_norm = 1)

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
            if chr(int(X_train[i,j]*128)) not in dic.keys():
                dic[chr(int(X_train[i,j]*128))] = X_train_embed[i,j]
                
    symbol_dict = dic
    
    args = Args()
    
    net = CNN_Text_dropout(args).cuda()
    print(net)

    pretrained_dict = torch.load('Parameters/cnn_text_kernel3.5.7.9_128_embed_dropout.pkl').state_dict()
    model_dict = net.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 更新现有的model_dict
    model_dict.update(pretrained_dict)
    # 加载我们真正需要的state_dict
    net.load_state_dict(model_dict)

    
    batch_size = 500
    Train_data = Data.TensorDataset(X_train_embed, Y_train)
    Test_data = Data.TensorDataset(X_test_embed, Y_test)
    train_data = Data.DataLoader(dataset=Train_data, batch_size=batch_size, shuffle=False)
    test_data = Data.DataLoader(dataset=Test_data, batch_size=1, shuffle=False)
    optimizer = optim.Adam(net.parameters(), lr=0.0001, weight_decay=1e-9)
    loss_function = nn.CrossEntropyLoss()

    attack_log_list = None
    attack_log_string_list = []
    
    net.eval()

    file = open('Data/attack_log_list_embed_dropout_FGSM_3.txt',"w") 
    for data in tqdm(test_data):
        inputs, labels = data
        labels = Variable(labels.cuda())
        inputs=Variable(inputs.cuda(), requires_grad =True)
        optimizer.zero_grad()
        outputs = net(inputs.float())
        #print(outputs.shape)
        #print(labels.long().squeeze(1).shape)
        loss = loss_function(outputs, labels.long().squeeze(1))
        loss.backward()
        log_grad = torch.sign(inputs.grad.data).cuda()
        _, predicted = torch.max(outputs.data, 1)
        #print(predicted.data[0])
        #print(labels.data[0].item())
        #print(log_grad[0][1][1])
        #print(inputs.data.shape)

        if predicted.data[0] == labels.data[0].item() and labels.data[0] == 0:
            new_grad, slash_index, end_index = mask_perturbation(inputs, log_grad)

            if end_index - slash_index > 10:
                size = 10
            elif end_index - slash_index < 10 and end_index - slash_index > 0:
                size = end_index - slash_index
            else:
                size = 0

            new_grad=torch.zeros(1,400,5).cuda()

            for j in range(1):
                table = np.random.randint(slash_index, end_index, size=(1,size))
                for e in table:
                    new_grad[j][e]=log_grad[j][e]

            for norm in range(1, 10):
                epsilon = norm / 10 
                log_adversarial = inputs.data + epsilon * new_grad
                log_adversarial = Change_Symbol(symbol_dict, log_adversarial)

                conf = F.softmax(net(log_adversarial).data, dim=1)
                probability, label_pred_adversarial = torch.max(conf, 1)
                if label_pred_adversarial.item() == 1:
                    log_string = Tensor_to_Log(symbol_dict, inputs)
                    attack_log_string = Tensor_to_Log(symbol_dict, log_adversarial)

                    if attack_log_list is None:
                        print('original log:', log_string)
                        print('adversarial log:', attack_log_string)
                        attack_log_list = log_adversarial
                        file.writelines(attack_log_string + '\n')
                        attack_log_string_list.append(attack_log_string)
                        np.save('./Data/attack_log_embed_dropout_FGSM_3.npy',attack_log_list.data.cpu().numpy())
                    elif attack_log_string not in attack_log_string_list:
                        print('original log:', log_string)
                        print('adversarial log:', attack_log_string)
                        attack_log_list = torch.cat([attack_log_list, log_adversarial], dim = 0)
                        file.writelines(attack_log_string + '\n')
                        attack_log_string_list.append(attack_log_string)
                        np.save('./Data/attack_log_embed_dropout_FGSM_3.npy',attack_log_list.data.cpu().numpy())
                    break
                    #print(log_adversarial.shape)
                    #print('original log:', log_string)
                    #print('adversarial log:', attack_log_string)

    file.close() 

main()