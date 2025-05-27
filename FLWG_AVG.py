'''
Conditional Convolutional VAE
nn.embedding for one_hot (label)
'''
import random
import copy
import numpy as np
import torch
import wandb
from torchsummary import summary  
from args import parse_args
import random
import copy
import time
import numpy as np
import torch
import wandb

from utils.user_sampling import user_select
from utils.setup import setup_experiment
from utils.localUpdateTarget import LocalUpdate, LocalUpdate_onlyGen, LocalUpdate_avgKD, LocalUpdate_fedprox
from utils.localUpdateGen import LocalUpdate_CVAE
from utils.avg import LGFedAvg, model_wise_FedAvg, FedAvg
from utils.getGenTrainData import generator_traindata
from utils.util import save_generated_images, evaluate_models
from generators32.CCVAE import *

def count_parameters(model):
    """统计模型的总参数量（可训练+不可训练）"""
    return sum(p.numel() for p in model.parameters())

def count_trainable_parameters(model):
    """统计模型的可训练参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    total_start = time.time()
    init_start = time.time()
    init_time = time.time() - init_start
    dataset_train, dataset_test, dict_users, local_models, common_net, w_comm, ws_glob, run = setup_experiment(args)
    print(args)

    print("\n===== 参数量统计 =====")
    for i, model in enumerate(local_models):
        total_params = count_parameters(model)
        trainable_params = count_trainable_parameters(model)
        print(f"模型 {i+1} - 总参数量: {total_params:,} | 可训练参数量: {trainable_params:,}")
    
    loss_train = []
    gen_glob = CCVAE(args).to(args.device)
    gen_total_params = count_parameters(gen_glob)
    gen_trainable_params = count_trainable_parameters(gen_glob)
    print(f"生成模型 (CCVAE) - 总参数量: {gen_total_params:,} | 可训练参数量: {gen_trainable_params:,}\n")
    opt = torch.optim.Adam(gen_glob.parameters(), lr=1e-3, weight_decay=0.001).state_dict()
    opts = [copy.deepcopy(opt) for _ in range(args.num_users)]

    ''' ---------------------------
    Federated Training generative model
    --------------------------- '''
    gen_train_start = time.time()
    for iter in range(1, args.gen_wu_epochs+1):
        gen_w_local, loss_locals = [], []
        
        idxs_users = user_select(args)
        for idx in idxs_users:
            local = LocalUpdate_CVAE(args, dataset=train_data, idxs=dict_users[idx])
            gen_weight, loss, opts[idx] = local.train(net=copy.deepcopy(gen_glob), opt=opts[idx])
            gen_w_local.append(copy.deepcopy(gen_weight))
            loss_locals.append(loss)
        
        gen_w_glob = FedAvg(gen_w_local)
        gen_glob.load_state_dict(gen_w_glob)
        loss_avg = sum(loss_locals) / len(loss_locals)
        print("\n训练后参数量统计（生成模型）:")
        print(f"生成模型 (CCVAE) - 总参数量: {count_parameters(gen_glob):,}")

        if args.save_imgs and (iter % args.sample_test == 0 or iter == args.gen_wu_epochs):
            save_generated_images(args.save_dir, gen_glob, args, iter)
        print('Warm-up Gen Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))

    best_perf = [0 for _ in range(args.num_models)]
    gen_train_time = time.time() - gen_train_start
    ''' ----------------------------------------
    Train main networks by local sample and generated samples
    ---------------------------------------- '''
    target_train_start = time.time()
    for iter in range(1,args.epochs+1):
        ws_local = [[] for _ in range(args.num_models)]
        gen_w_local, loss_locals, gen_loss_locals  = [], [], []

        idxs_users = user_select(args)
        for idx in idxs_users:
            dev_spec_idx = min(idx//(args.num_users//args.num_models), args.num_models-1)
            model = local_models[dev_spec_idx]
            model.load_state_dict(ws_glob[dev_spec_idx])

            if args.only_gen:
                local = LocalUpdate_onlyGen(args, dataset=dataset_train, idxs=dict_users[idx])
                weight, loss, gen_loss = local.train(net=copy.deepcopy(model).to(args.device), gennet=copy.deepcopy(gen_glob), learning_rate=args.lr)
            else:
                if args.avgKD == True and iter > 1:
                    local = LocalUpdate_avgKD(args, dataset=dataset_train, idxs=dict_users[idx])
                    weight, loss, gen_loss = local.train(local_models, ws_glob, dev_spec_idx, learning_rate=args.lr)
                elif args.fedprox:
                    local = LocalUpdate_fedprox(args, dataset=dataset_train, idxs=dict_users[idx])
                    weight, loss, gen_loss = local.train(net=copy.deepcopy(model).to(args.device), learning_rate=args.lr)                        
                else:
                    local = LocalUpdate(args, dataset=dataset_train, idxs=dict_users[idx])                    
                    if args.aid_by_gen: # synthetic data updates header & real data updates whole target network
                        weight, loss, gen_loss = local.train(net=copy.deepcopy(model).to(args.device), gennet=copy.deepcopy(gen_glob), learning_rate=args.lr)
                    else:
                        weight, loss, gen_loss = local.train(net=copy.deepcopy(model).to(args.device), learning_rate=args.lr)

            ws_local[dev_spec_idx].append(weight)
            loss_locals.append(loss)
            gen_loss_locals.append(gen_loss)

        if args.avg_FE:
            ws_glob, w_comm = LGFedAvg(args, ws_glob, ws_local, w_comm) # main net, feature extractor weight update
        else:
            ws_glob = model_wise_FedAvg(args, ws_glob, ws_local)
        loss_avg = sum(loss_locals) / len(loss_locals)
        try:
            gen_loss_avg = sum(gen_loss_locals) / len(gen_loss_locals)
        except:
            gen_loss_avg = -1
        print('Round {:3d}, Average loss {:.3f}, Average loss by Gen {:.3f}'.format(iter, loss_avg, gen_loss_avg))

        loss_train.append(loss_avg)
        if iter == 1 or iter % args.sample_test == 0 or iter == args.epochs:
            best_perf = evaluate_models(local_models, ws_glob, dataset_test, args, iter, best_perf)

    print(best_perf, 'AVG'+str(args.rs), sum(best_perf)/len(best_perf))
    target_train_time = time.time() - target_train_start
    total_time = time.time() - total_start 
    print(f"""
    ======= Time Report =======
    Total Time: {total_time:.2f}s
    - Initialization: {init_time:.2f}s
    - Generative Training: {gen_train_time:.2f}s
    - Target Training: {target_train_time:.2f}s
    ===========================
    """)      
    if args.wandb:
        run.finish()

    return sum(best_perf)/len(best_perf)


if __name__ == "__main__":
    args = parse_args()
    args.gen_model = 'vae'
    train_data = generator_traindata(args)    

    results = []
    for i in range(args.num_experiment):
        torch.manual_seed(args.rs)
        torch.cuda.manual_seed(args.rs)
        torch.cuda.manual_seed_all(args.rs)
        np.random.seed(args.rs)
        random.seed(args.rs)
        results.append(main())
        args.rs = args.rs+1
        print(results)
        from args import parse_args

