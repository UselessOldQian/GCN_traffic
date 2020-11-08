# -*- coding: utf-8 -*-
"""
@Time   : 2020/6/4

@Author : Shen Fang
"""
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from GAT import GAT_model
from traffic_dataset import LoadData
from chebnet import ChebNet
import numpy as np
import math
import random
import matplotlib.pyplot as plt
seed = 1001
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

class GCN(nn.Module):
    def __init__(self, in_c, hid_c, out_c):
        super(GCN, self).__init__()
        self.linear_1 = nn.Linear(in_c, hid_c)
        self.linear_2 = nn.Linear(hid_c, out_c)
        self.act = nn.ReLU()

    def forward(self, data, device):
        graph_data = data["graph"].to(device)[0]  # [N, N]
        graph_data = GCN.process_graph(graph_data)

        flow_x = data["flow_x"].to(device)  # [B, N, H, D]

        B, N = flow_x.size(0), flow_x.size(1)

        flow_x = flow_x.view(B, N, -1)  # [B, N, H*D]  H = 6, D = 1

        output_1 = self.linear_1(flow_x)  # [B, N, hid_C]
        output_1 = self.act(torch.matmul(graph_data, output_1))  # [N, N], [B, N, Hid_C]

        output_2 = self.linear_2(output_1)
        output_2 = self.act(torch.matmul(graph_data, output_2))  # [B, N, 1, Out_C]

        return output_2.unsqueeze(2)

    @staticmethod
    def process_graph(graph_data):
        N = graph_data.size(0)
        matrix_i = torch.eye(N, dtype=graph_data.dtype, device=graph_data.device)
        graph_data += matrix_i  # A~ [N, N]

        degree_matrix = torch.sum(graph_data, dim=-1, keepdim=False)  # [N]
        degree_matrix = degree_matrix.pow(-1)
        degree_matrix[degree_matrix == float("inf")] = 0.  # [N]

        degree_matrix = torch.diag(degree_matrix)  # [N, N]

        return torch.mm(degree_matrix, graph_data)  # D^(-1) * A = \hat(A)


def MAE(y_true,y_pre):
    y_true=(y_true).detach().numpy().copy().reshape((-1,1))
    y_pre=(y_pre).detach().numpy().copy().reshape((-1,1))
    re = np.abs(y_true-y_pre).mean()
    return re

def RMSE(y_true,y_pre):
    y_true=(y_true).detach().numpy().copy().reshape((-1,1))
    y_pre=(y_pre).detach().numpy().copy().reshape((-1,1))
    re = math.sqrt(((y_true-y_pre)**2).mean())
    return re

def MAPE(y_true,y_pre):
    y_true=(y_true).detach().numpy().copy().reshape((-1,1))
    y_pre=(y_pre).detach().numpy().copy().reshape((-1,1))
    e = (y_true+y_pre)/2+1e-2
    re = (np.abs(y_true-y_pre)/(np.abs(y_true)+e)).mean()
    return re


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Loading Dataset
    train_data = LoadData(data_path=["PeMS_04/PeMS04.csv", "PeMS_04/PeMS04.npz"], num_nodes=307, divide_days=[45, 14],
                          time_interval=5, history_length=6,
                          train_mode="train")

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

    test_data = LoadData(data_path=["PeMS_04/PeMS04.csv", "PeMS_04/PeMS04.npz"], num_nodes=307, divide_days=[45, 14],
                         time_interval=5, history_length=6,
                         train_mode="test")

    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    # Loading Model
    # TODO:  Construct the GAT (must) and DCRNN (optional) Model

    #my_net = None
    my_net = GAT_model(6,6,1)
    #my_net = GCN(6,6,1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    my_net = my_net.to(device)

    criterion = nn.MSELoss()

    optimizer = optim.Adam(params=my_net.parameters())

    # Train model
    Adam_Epoch = 5
    my_net.train()
    for epoch in range(Adam_Epoch):
        epoch_loss = 0.0
        epoch_mae = 0.0
        epoch_rmse = 0.0
        epoch_mape = 0.0
        num = 0
        start_time = time.time()
        for data in train_loader:  # ["graph": [B, N, N] , "flow_x": [B, N, H, D], "flow_y": [B, N, 1, D]]
            my_net.zero_grad()
            predict_value = my_net(data, device).to(torch.device("cpu"))  # [0, 1] -> recover

            loss = criterion(predict_value, data["flow_y"])      
            epoch_mae += MAE( data["flow_y"],predict_value)
            epoch_rmse += RMSE( data["flow_y"],predict_value)
            epoch_mape += MAPE( data["flow_y"],predict_value)
            
            epoch_loss += loss.item()
            num += 1
            loss.backward()
            optimizer.step()
        end_time = time.time()
        epoch_mae = epoch_mae/num
        epoch_rmse = epoch_rmse/num
        epoch_mape = epoch_mape/num
        print("Epoch: {:04d}, Loss: {:02.4f}, mae: {:02.4f}, rmse: {:02.4f}, mape: {:02.4f}, Time: {:02.2f} mins".format(epoch+1,  10*epoch_loss / (len(train_data)/64),
                                                                                                              epoch_mae,epoch_rmse,epoch_mape,(end_time-start_time)/60))
    SGD_epoch = 15
    optimizer = torch.optim.SGD(my_net.parameters(), lr=0.2,weight_decay=0.005)
    for epoch in range(SGD_epoch):
        if (epoch % 3 == 0):
            for p in optimizer.param_groups:
                p['lr'] *= 0.5
        epoch_loss = 0.0
        epoch_mae = 0.0
        epoch_rmse = 0.0
        epoch_mape = 0.0
        num = 0
        start_time = time.time()
        for data in train_loader:  # ["graph": [B, N, N] , "flow_x": [B, N, H, D], "flow_y": [B, N, 1, D]]
            my_net.zero_grad()
            predict_value = my_net(data, device).to(torch.device("cpu"))  # [0, 1] -> recover

            loss = criterion(predict_value, data["flow_y"])      
            epoch_mae += MAE( data["flow_y"],predict_value)
            epoch_rmse += RMSE( data["flow_y"],predict_value)
            epoch_mape += MAPE( data["flow_y"],predict_value)
            
            epoch_loss += loss.item()
            num += 1
            loss.backward()
            optimizer.step()
        end_time = time.time()
        epoch_mae = epoch_mae/num
        epoch_rmse = epoch_rmse/num
        epoch_mape = epoch_mape/num
        print("Epoch: {:04d}, Loss: {:02.4f}, mae: {:02.4f}, rmse: {:02.4f}, mape: {:02.4f}, Time: {:02.2f} mins".format(epoch+Adam_Epoch+1, 10*epoch_loss / (len(train_data)/64),
                                                                                                              epoch_mae,epoch_rmse,epoch_mape,(end_time-start_time)/60))


    my_net.eval()
    with torch.no_grad():
        epoch_mae = 0.0
        epoch_rmse = 0.0
        epoch_mape = 0.0
        total_loss = 0.0
        num = 0
        all_predict_value = 0
        all_y_true = 0
        for data in test_loader:
            predict_value = my_net(data, device).to(torch.device("cpu"))
            if num == 0:
                all_predict_value = predict_value
                all_y_true = data["flow_y"]
            else:
                all_predict_value = torch.cat([all_predict_value, predict_value], dim=0)
                all_y_true = torch.cat([all_y_true, data["flow_y"]], dim=0)
            loss = criterion(predict_value, data["flow_y"])
            total_loss += loss.item()
            num += 1
        epoch_mae = MAE( all_y_true,all_predict_value)
        epoch_rmse = RMSE( all_y_true,all_predict_value)
        epoch_mape = MAPE( all_y_true,all_predict_value)
        print("Test Loss: {:02.4f}, mae: {:02.4f}, rmse: {:02.4f}, mape: {:02.4f}".format( 10*total_loss / (len(test_data)/64) ,epoch_mae,epoch_rmse,epoch_mape))

    #保存模型
    torch.save(my_net,'model_GAT_6.pth')
    
    
    ####选择节点进行流量可视化
    node_id = 120

    plt.title(str(node_id)+" 号节点交通流量可视化(第一天)")
    plt.xlabel("时刻/5min")
    plt.ylabel("交通流量")
    plt.plot(test_data.recover_data(test_data.flow_norm[0],test_data.flow_norm[1],all_y_true)[:24*12,node_id,0,0],label='真实值')
    plt.plot(test_data.recover_data(test_data.flow_norm[0],test_data.flow_norm[1],all_predict_value)[:24*12,node_id,0,0],label = '预测值')
    plt.legend()
    plt.savefig(str(node_id)+" 号节点交通流量可视化(第一天).png", dpi=400)
    plt.show()
    
    plt.title(str(node_id)+" 号节点交通流量可视化(两周)")
    plt.xlabel("时刻/5min")
    plt.ylabel("交通流量")
    plt.plot(test_data.recover_data(test_data.flow_norm[0],test_data.flow_norm[1],all_y_true)[:,node_id,0,0],label = '真实值')
    plt.plot(test_data.recover_data(test_data.flow_norm[0],test_data.flow_norm[1],all_predict_value)[:,node_id,0,0],label = '预测值')
    plt.legend()
    plt.savefig(str(node_id)+" 号节点交通流量可视化(两周).png", dpi=400)
    plt.show()

    mae = MAE(test_data.recover_data(test_data.flow_norm[0],test_data.flow_norm[1],all_y_true),test_data.recover_data(test_data.flow_norm[0],test_data.flow_norm[1],all_predict_value))
    rmse = RMSE(test_data.recover_data(test_data.flow_norm[0],test_data.flow_norm[1],all_y_true),test_data.recover_data(test_data.flow_norm[0],test_data.flow_norm[1],all_predict_value))
    mape = MAPE(test_data.recover_data(test_data.flow_norm[0],test_data.flow_norm[1],all_y_true),test_data.recover_data(test_data.flow_norm[0],test_data.flow_norm[1],all_predict_value))
    print("基于原始值的精度指标  mae: {:02.4f}, rmse: {:02.4f}, mape: {:02.4f}".format(mae,rmse,mape))
                           
if __name__ == '__main__':
    main()
