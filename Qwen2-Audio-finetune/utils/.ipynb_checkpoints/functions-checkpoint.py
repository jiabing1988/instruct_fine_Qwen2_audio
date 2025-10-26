import torch
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
def compute_acc(logits,labels):
    _,labels_len = labels.shape
    preds = torch.argmax(logits,dim=-1)
    labels_indices = labels != -100 
    acc = torch.sum(preds[:,-labels_len-1:-1][labels_indices] == labels[labels_indices]).float() /torch.sum(labels_indices).float()
    return acc

def compute_metrics(logits, labels):
    preds = torch.argmax(logits, dim=-1)
    labels_indices = labels != -100
    
    seq_len = min(preds.size(1), labels.size(1))
    preds_aligned = preds[:, -seq_len:]
    labels_aligned = labels[:, -seq_len:]
    labels_indices_aligned = labels_indices[:, -seq_len:]
    
    valid_preds = preds_aligned[labels_indices_aligned]
    valid_labels = labels_aligned[labels_indices_aligned]
    
    device = labels.device
    
    if len(valid_labels) == 0:
        return (torch.tensor(0.0, device=device), 
                torch.tensor(0.0, device=device), 
                torch.tensor(0.0, device=device), 
                torch.tensor(0.0, device=device))
    
    # 计算准确率
    accuracy = torch.sum(valid_preds == valid_labels).float() / len(valid_labels)
    
    # 获取所有类别
    all_classes = torch.unique(torch.cat([valid_preds, valid_labels]))
    
    # 计算每个类别的指标
    precision_list = []
    recall_list = []
    
    for class_id in all_classes:
        tp = torch.sum((valid_preds == class_id) & (valid_labels == class_id)).float()
        fp = torch.sum((valid_preds == class_id) & (valid_labels != class_id)).float()
        fn = torch.sum((valid_preds != class_id) & (valid_labels == class_id)).float()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else torch.tensor(0.0, device=device)
        recall = tp / (tp + fn) if (tp + fn) > 0 else torch.tensor(0.0, device=device)
        
        precision_list.append(precision)
        recall_list.append(recall)
    
    # 宏平均 - 确保所有张量在同一设备上
    if precision_list:
        precision_macro = torch.mean(torch.stack(precision_list))
        recall_macro = torch.mean(torch.stack(recall_list))
    else:
        precision_macro = torch.tensor(0.0, device=device)
        recall_macro = torch.tensor(0.0, device=device)
    
    # 计算F1
    if precision_macro + recall_macro > 0:
        f1_macro = 2 * precision_macro * recall_macro / (precision_macro + recall_macro)
    else:
        f1_macro = torch.tensor(0.0, device=device)
    
    return accuracy, precision_macro, recall_macro, f1_macro