import copy
from collections import OrderedDict
import torch


def FedAvg(ws):
    w_avg = copy.deepcopy(ws[0])
    for k in w_avg.keys():
        for i in range(1, len(ws)):
            w_avg[k] += ws[i][k]
        w_avg[k] = torch.div(w_avg[k], len(ws))
    return w_avg
#实现经典的联邦平均算法（Federated Averaging, FedAvg）

def LGFedAvg(args, wg, ws, wc):
    '''
    - 所有客户端的通用特征提取器的权重进行平均
    - 异质性头部特征提取器分别取平均值
    
    wg: 全局权重 (ws_glob)
    ws: 本地权重 (ws_local)
    wc: 公共权重 (w_comm)
    '''
    # 初始化
    w_com = OrderedDict()
    for k in wc.keys():
        w_com[k] = 0*copy.deepcopy(wc[k])

    # 本次循环轮次的总客户数量。用于计算平均值
    num = 0
    for j in range(args.num_models):
        num += len(ws[j])
    
    # 平均公共特征提取器
    for k in wc.keys():
        for j in range(args.num_models):
            if ws[j]:
                for i in range(len(ws[j])):
                    w_com[k] += ws[j][i][k]
        w_com[k] = torch.div(w_com[k], num)

    # 对异质头部进行平均，并嵌入通用特征提取器。
    w_avg = [None for _ in range(args.num_models)]
    for j in range(args.num_models):
        if ws[j]:
            w_avg[j] = copy.deepcopy(ws[j][0])
            for k in w_avg[j].keys():
                if k not in wc.keys():
                    for i in range(1, len(ws[j])):
                        w_avg[j][k] += ws[j][i][k]
                    w_avg[j][k] = torch.div(w_avg[j][k], len(ws[j]))
                else:
                    w_avg[j][k] = w_com[k]
        else:
            w_avg[j] = copy.deepcopy(wg[j]) # 如果某个模型类型没有客户端参与训练，则使用上一轮的全局模型参数。
            for k in wc.keys():
                w_avg[j][k] = w_com[k]
     
    return w_avg, w_com # 返回平均后的聚合模型权重和平均后的公共特征提取器的权重
#实现局部全局联邦平均算法（Local-Global Federated Averaging, LG-FedAvg），用于处理异构模型的情况。

def LGFedAvg_frozen_FE(args, wg, ws, wc):
    '''
    wg: 全局权重(ws_glob)
    ws: 本地权重 (ws_local)
    wc: 公共权重 (w_comm)
    '''
    w_com = wc
    
    '''
    平均非公共部分的权重
    '''
    w_avg = [None for _ in range(args.num_models)]
    for j in range(args.num_models):
        if ws[j]:
            w_avg[j] = copy.deepcopy(ws[j][0])
            for k in w_avg[j].keys():
                if k not in wc.keys():
                    for i in range(1, len(ws[j])):
                        w_avg[j][k] += ws[j][i][k]
                    w_avg[j][k] = torch.div(w_avg[j][k], len(ws[j]))
                else:
                    w_avg[j][k] = w_com[k]
        else:
            w_avg[j] = copy.deepcopy(wg[j]) # 如果本轮没有如果某个模型类型没有客户端参与训练，则使用上一轮的全局模型参数。
            for k in wc.keys():
                w_avg[j][k] = w_com[k]
     
    return w_avg, w_com
#与LGFedAvg类似，但冻结公共特征提取器（Frozen Feature Extractor），不更新其参数。

def model_wise_FedAvg(args, wg, ws):
    '''
    Average model-by-model (equivalent local models)
    
    wg: 全局权重(ws_glob)
    ws: 本地权重 (ws_local)
    '''

    w_avg = [None for _ in range(args.num_models)]
    for j in range(args.num_models):
        if ws[j]:
            w_avg[j] = copy.deepcopy(ws[j][0])
            for k in w_avg[j].keys():
                for i in range(1, len(ws[j])):
                    w_avg[j][k] += ws[j][i][k]
                w_avg[j][k] = torch.div(w_avg[j][k], len(ws[j]))
        else:
            w_avg[j] = copy.deepcopy(wg[j]) # get weights from previous
     
    return w_avg
#实现按模型类型联邦平均（Model-wise Federated Averaging），用于处理多个模型类型的情况。