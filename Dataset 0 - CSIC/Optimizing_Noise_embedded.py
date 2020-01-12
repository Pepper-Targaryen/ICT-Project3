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
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class Args:
    def __init__(self):
        self.embed_num = 128
        self.embed_dim = 5
        self.class_num = 2
        self.kernel_num = 100
        self.kernel_sizes = [3,5,7,9]
        #self.dropout = 0.3
        self.static = True

def main():
    # load data
    X_train=np.load("./Data/sGrid/X_train.npy")
    X_test=np.load("./Data/sGrid/X_test.npy")
    X_vaild=np.load("./Data/sGrid/X_vaild.npy")
    Y_train=np.load("./Data/sGrid/Y_train.npy")
    Y_test=np.load("./Data/sGrid/Y_test.npy")
    Y_vaild=np.load("./Data/sGrid/Y_vaild.npy")
    
    # character embedding
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
    
    symbol_dict = {}
    count = 0
    for i in range(X_train.shape[0]):
        for j in range(X_train.shape[1]):
            if chr(int(X_train[i,j]*128)) not in symbol_dict.keys():
                symbol_dict[chr(int(X_train[i,j]*128))] = X_train_embed[i,j]
        
    args = Args()
    #net = CNN_Text_dropout(args).cuda()
    
    attack_net = CNN_Text_attack_dropout(args).cuda()
    model_dict = attack_net.state_dict()
    print(attack_net)
    pretrained_dict = torch.load('Parameters/cnn_text_kernel3.5.7.9_128_embed_dropout.pkl').state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 更新现有的model_dict
    model_dict.update(pretrained_dict)
    # 加载我们真正需要的state_dict
    attack_net.load_state_dict(model_dict)
    
    batch_size = 1
    Test_data = Data.TensorDataset(X_test_embed, Y_test)
    test_data = Data.DataLoader(dataset=Test_data, batch_size=1, shuffle=False)
    optimizer = optim.Adam(params=[attack_net.r], lr=0.05)
    loss_function = nn.CrossEntropyLoss()
    
    regularization = None
    attack_log_list = None
    attack_log_string_list = []
    i = 0
    num = 0
    attack_net.eval()
    with open('Data/attack_log_list_embed_dropout_2.txt',"w") as f:
        for data in tqdm(test_data):
            attack_net.r.data = torch.zeros(1,400,5).cuda()
            #print(attack_net.r)
            log, labels = data
            log = log.float().cuda()
            labels = labels.long().cuda()
            attack_net.r.data = torch.zeros(1,400,5).cuda()
            outputs = attack_net(log)
            _, predicted = torch.max(outputs.data, 1)

            i += 1
            if labels.data.item() == 0 and predicted.data.item() ==0:
                print('log No.', i)
                attack_net.r.data = torch.zeros(1,400,5).cuda()
                _y_target = torch.LongTensor([1]).cuda()
                flag = True
                for iteration in range(100):
                    #print(i)
                    #i += 1
                    optimizer.zero_grad() 
                    outputs = attack_net(log)
                    xent_loss = loss_function(outputs, _y_target) 
                    if regularization == "l1":
                        adv_loss = xent_loss + 10 * torch.mean(torch.abs(attack_net.r))
                    elif regularization == "l2":
                        adv_loss  = xent_loss + torch.mean(torch.pow(attack_net.r,2))
                    elif regularization == None:
                        adv_loss = xent_loss
                    else:
                        raise Exception("regularization method {} is not implemented, please choose one of l1, l2 or None".format(regularization))

                    adv_loss.backward() 
                    optimizer.step() 
                    _, predicted = torch.max(attack_net(log).data, 1)
                    if predicted == _y_target:
                        #print(Tensor_to_Log(symbol_dict, log))
                        perturbation, slash_index, end_index = mask_perturbation(log, attack_net.r)
                        attack_log = Change_Symbol(symbol_dict, log + perturbation)
                        #print(Tensor_to_Log(symbol_dict, log))
                        #print(Tensor_to_Log(symbol_dict, attack_log))
                        #print(attack_log.shape)
                        #print(MSE(log + attack_net.r,log))
                        #print(log)
                        #print(attack_net.r)
                        if Tensor_to_Log(symbol_dict, attack_log) != Tensor_to_Log(symbol_dict, log):
                            #attack_log = torch.clamp(torch.round((log + attack_net.r) * 128) / 128,0,1)

                            outputs = attack_net(attack_log)
                            _, predicted = torch.max(outputs.data, 1)
                            if predicted == _y_target:
                                #print('find 1')
                                #print('predicted', predicted)
                                #print('target', _y_target)
                                log_string = Tensor_to_Log(symbol_dict, log)
                                attack_log_string = Tensor_to_Log(symbol_dict, attack_log)
                                #print('original log:', log_string)
                                #print('adversarial log:', attack_log_string)
                                count = 0
                                for k in range(400):
                                    if log_string[k] != attack_log_string[k]:
                                        count += 1

                                if attack_log_list is None and count >= 5:
                                    print('original log:', log_string)
                                    print('adversarial log:', attack_log_string)
                                    print('changed symbols: ', count)
                                    attack_log_list = attack_log
                                    f.writelines(attack_log_string + '\n')
                                    attack_log_string_list.append(attack_log_string)
                                    break
                                    np.save('./Data/attack_log_embed_dropout_2.npy',attack_log_list.data.cpu().numpy())
                                elif attack_log_string not in attack_log_string_list and count >= 5:
                                    print('original log:', log_string)
                                    print('adversarial log:', attack_log_string)
                                    print('changed symbols: ', count)
                                    attack_log_list = torch.cat([attack_log_list, attack_log], dim = 0)
                                    f.writelines(attack_log_string + '\n')
                                    attack_log_string_list.append(attack_log_string)
                                    np.save('./Data/attack_log_embed_dropout_2.npy',attack_log_list.data.cpu().numpy())
                                    break
                                if count >= 10:
                                    break

main()