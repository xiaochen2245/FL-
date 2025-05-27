
# from math import ceil as up
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
import logging
import copy
from torchvision.utils import save_image
import wandb
from sklearn.metrics import f1_score, recall_score

def one_hot(labels, class_size):
    targets = torch.zeros(labels.size(0), class_size)
    for i, label in enumerate(labels):
        targets[i, label] = 1
    return targets


def test_img(net_g, datatest, args):
    net_g.eval()
    net_g.to(args.device)
    # testing
    test_loss = 0
    correct = 0

    data_loader = DataLoader(datatest, batch_size=args.bs)
    with torch.no_grad():
      for idx, (data, target) in enumerate(data_loader):
          if 'cuda' in args.device:
              data, target = data.to(args.device), target.to(args.device)
          logits, log_probs = net_g(data)
          test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
          y_pred = log_probs.data.max(1, keepdim=True)[1]

          correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    net_g.to('cpu')
    return accuracy, test_loss


def get_logger(logpath, filepath, package_files=[], displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if len(logger.handlers)>0:
        logger.handlers.clear()
        
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        # info_file_handler.terminator = ""
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        # console_handler.terminator = ""
        logger.addHandler(console_handler)
    logger.info(filepath)
    
    # with open(filepath, "r") as f:
        # logger.info(f.read())

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())

    return logger


def save_generated_images(dir, gen_model, args, iter):
    gen_model.eval()
    sample_num = 40
    samples = gen_model.sample_image_4visualization(sample_num)
    save_image(samples.view(sample_num, args.output_channel, args.img_size, args.img_size),
                dir + str(args.name)+ str(args.rs) +'_' + str(iter) + '.png', nrow=10)  # normalize=True
    gen_model.train()


def evaluate_models(local_models, ws_glob, dataset_test, args, iter, best_perf):
    acc_test_tot = []
    f1_scores = []  
    recall_scores = []  

    for i in range(args.num_models):
        model_e = local_models[i]
        model_e.load_state_dict(ws_glob[i])
        model_e.to(args.device)  # Move model to the same device as data
        model_e.eval()
        
        y_true = []
        y_pred = []
        test_loss = 0
        correct = 0
        
        data_loader = DataLoader(dataset_test, batch_size=args.bs)
        with torch.no_grad():
            for data, target in data_loader:
                # Move data to the same device as model
                data, target = data.to(args.device), target.to(args.device)
                
                logits, log_probs = model_e(data)
                test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
                pred = log_probs.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
                
                # Store for metrics calculation
                y_true.extend(target.cpu().numpy())
                y_pred.extend(pred.cpu().numpy())

        # Calculate metrics
        test_loss /= len(data_loader.dataset)
        acc_test = 100. * correct / len(data_loader.dataset)
        
        from sklearn.metrics import f1_score, recall_score
        f1 = f1_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        
        f1_scores.append(f1)
        recall_scores.append(recall)
        acc_test_tot.append(acc_test)

        if acc_test > best_perf[i]:
            best_perf[i] = float(acc_test)

        print(f"Model {i} - Accuracy: {acc_test:.2f}%, F1: {f1:.4f}, Recall: {recall:.4f}")

        if args.wandb:
            wandb.log({
                "Communication round": iter,
                f"Local model {i} test accuracy": acc_test,
                f"Local model {i} F1 score": f1,
                f"Local model {i} recall": recall
            })
    
    # Calculate and log mean metrics
    mean_acc = sum(acc_test_tot) / len(acc_test_tot)
    mean_f1 = sum(f1_scores) / len(f1_scores)
    mean_recall = sum(recall_scores) / len(recall_scores)

    if args.wandb:
        wandb.log({
            "Communication round": iter,
            "Mean test accuracy": mean_acc,
            "Mean F1 score": mean_f1,
            "Mean recall": mean_recall
        })

    print(f"\nMean metrics - Accuracy: {mean_acc:.2f}%, F1: {mean_f1:.4f}, Recall: {mean_recall:.4f}")
    
    model_e.to('cpu')  # Move model back to CPU to free GPU memory
    return best_perf
