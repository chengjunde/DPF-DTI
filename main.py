import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, precision_recall_curve, auc, accuracy_score
from models import *
from dataloader import DTIDataset
from torch.utils.data import DataLoader
from utils import set_seed, graph_collate_func
from trainer import Trainer
import os
import dgl

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

seed = 1 # [1, 2, 123, 2025, 5000]
set_seed(seed)

def read_sequences(file_path):
    """
    从文件中读取一行 id 一行 sequence 的数据，只保存序列到列表中
    :param file_path: 文件路径
    :return: 包含 sequence 的列表
    """
    sequences = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for i in range(1, len(lines), 2):  # 从第二行开始，步长为 2
            sequence = lines[i].strip()
            sequences.append(sequence)
    return sequences

def split_interaction_file(ratio=1, seed=seed):
    """
    划分训练集、验证集、测试集（比例：7:1:2）
    """
    M = np.loadtxt("data/mat_drug_protein.txt")
    pos_idx = np.array(np.where(M == 1)).T
    neg_idx = np.array(np.where(M == 0)).T
    _rng = np.random.RandomState(seed)
    neg_sampled_idx = _rng.choice(neg_idx.shape[0], int(pos_idx.shape[0] * ratio), replace=False)
    neg_idx = neg_idx[neg_sampled_idx]

    # 第一步：划分训练集（70%）和临时集（验证+测试=30%）
    train_pos_idx, temp_pos_idx = train_test_split(pos_idx, test_size=0.3, random_state=seed)
    train_neg_idx, temp_neg_idx = train_test_split(neg_idx, test_size=0.3, random_state=seed)
    
    # 第二步：从临时集中划分验证集（10%）和测试集（20%）
    # 临时集占30%，验证集取1/3（总数据10%），测试集取2/3（总数据20%）
    val_pos_idx, test_pos_idx = train_test_split(temp_pos_idx, test_size=2/3, random_state=seed)
    val_neg_idx, test_neg_idx = train_test_split(temp_neg_idx, test_size=2/3, random_state=seed)

    # 组装训练集、验证集、测试集索引
    train_idx = np.concatenate((train_pos_idx, train_neg_idx), axis=0)
    val_idx = np.concatenate((val_pos_idx, val_neg_idx), axis=0)
    test_idx = np.concatenate((test_pos_idx, test_neg_idx), axis=0)

    # 训练集交互矩阵
    train_P = np.zeros(M.shape)
    train_P[train_idx[:, 0], train_idx[:, 1]] = 1
    
    # 验证集和测试集的真实标签
    val_y = np.concatenate((np.ones(val_pos_idx.shape[0]), np.zeros(val_neg_idx.shape[0])))
    test_y = np.concatenate((np.ones(test_pos_idx.shape[0]), np.zeros(test_neg_idx.shape[0])))

    return train_P, train_idx, val_idx, test_idx, val_y, test_y

def evaluate(y_true, y_pred):
    """
    扩展评估函数，计算 AUROC、AUPRC、Accuracy、Sensitivity、Specificity
    :param y_true: 真实标签 (np.array)
    :param y_pred: 预测概率 (np.array)
    :return: 各类指标字典
    """
    # 1. 计算 AUROC（ROC曲线下面积）
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    
    # 2. 计算 AUPRC（精确率-召回率曲线下面积）
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    pr_auc = auc(recall, precision)
    
    # 3. 计算 Accuracy（准确率）：需要将概率转为0/1标签（阈值0.5）
    y_pred_label = (y_pred >= 0.5).astype(int)
    accuracy = accuracy_score(y_true, y_pred_label)
    
    # 4. 计算 Sensitivity（灵敏度/召回率/TPR）和 Specificity（特异度/TNR）
    # 混淆矩阵核心元素
    TP = np.sum((y_true == 1) & (y_pred_label == 1))  # 真阳性
    TN = np.sum((y_true == 0) & (y_pred_label == 0))  # 真阴性
    FP = np.sum((y_true == 0) & (y_pred_label == 1))  # 假阳性
    FN = np.sum((y_true == 1) & (y_pred_label == 0))  # 假阴性
    
    # 处理分母为0的边界情况（避免除以0错误）
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0  # 灵敏度 = TP/(TP+FN)
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0  # 特异度 = TN/(TN+FP)
    
    return {
        'AUROC': round(roc_auc, 4),
        'AUPRC': round(pr_auc, 4),
        'Accuracy': round(accuracy, 4),
        'Sensitivity': round(sensitivity, 4),
        'Specificity': round(specificity, 4)
    }

# ========== 修复后的预测函数 ==========
def predict(model, data_loader, device):
    """
    适配DTIDataset的返回格式：(v_d, v_p, drug_idx, prot_idx, labels)
    """
    model.eval()
    y_pred = []
    y_true = []  # 可选：同时收集真实标签
    with torch.no_grad():
        for batch in data_loader:
            # 正确解包batch（5个元素）
            v_d, v_p, drug_idx, prot_idx, labels = batch
            
            # 迁移到设备（适配DGLGraph/Tensor）
            if isinstance(v_d, dgl.DGLGraph):
                v_d = v_d.to(device)
            elif isinstance(v_d, torch.Tensor):
                v_d = v_d.to(device)
            elif isinstance(v_d, dict):
                v_d = {k: v.to(device) for k, v in v_d.items()}
            
            if isinstance(v_p, dgl.DGLGraph):
                v_p = v_p.to(device)
            elif isinstance(v_p, torch.Tensor):
                v_p = v_p.to(device)
            elif isinstance(v_p, dict):
                v_p = {k: v.to(device) for k, v in v_p.items()}
            
            drug_idx = drug_idx.to(device)
            prot_idx = prot_idx.to(device)
            
            # 前向传播（注意model的输入参数和返回值）
            # 参考Trainer的eval_epoch，model返回：v_d_out, v_p_out, score, att
            _, _, score, _ = model(v_d, v_p, drug_idx, prot_idx, mode='eval')
            
            # 将score转为概率（如果是logits则用sigmoid）
            # 参考Trainer中的处理：binary_cross_entropy返回的n是概率值
            # 如果score是logits：
            pred_probs = torch.sigmoid(score).cpu().numpy()
            # 如果score已经是概率（如Trainer中n），则直接取：
            # pred_probs = score.cpu().numpy()
            
            y_pred.extend(pred_probs.flatten())
            y_true.extend(labels.cpu().numpy().flatten())
    
    return np.array(y_pred), np.array(y_true)

if __name__ == '__main__':
    set_seed(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    
    print("Preparing interaction data (train/val/test split 7:1:2)...")
    # 划分数据集（7:1:2）
    train_P, train_idx, val_idx, test_idx, y_val_true, y_test_true = split_interaction_file()
    
    print("Preparing drug/protein data...")
    # 读取药物SMILES序列
    drug_file = os.path.join(script_dir, "data", "drug_smiles.txt")
    drug_smiles = read_sequences(drug_file)
    # 读取蛋白质序列
    protein_file = os.path.join(script_dir, "data", "protein_seq.txt")
    protein_seqs = read_sequences(protein_file)
    # 读取DTI交互矩阵
    interaction_matrix = np.loadtxt(os.path.join(script_dir, "data", "mat_drug_protein.txt"))

    # 数据加载器参数
    batch_size = 64
    # 创建训练/验证/测试数据集和加载器
    train_dataset = DTIDataset(train_idx, drug_smiles, protein_seqs, interaction_matrix)
    val_dataset = DTIDataset(val_idx, drug_smiles, protein_seqs, interaction_matrix)
    test_dataset = DTIDataset(test_idx, drug_smiles, protein_seqs, interaction_matrix)
    
    train_generator = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=graph_collate_func)
    val_generator = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=graph_collate_func)
    test_generator = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=graph_collate_func)

    # 模型参数配置
    # 药物相关参数
    drug_in_feats = 75
    drug_embedding = 128
    drug_hidden_feats = [128,128,128]
    drug_padding = True
    # 蛋白质相关参数
    protein_emb_dim = 128
    # MLP解码器参数
    mlp_in_dim = 384
    mlp_hidden_dim = 128
    mlp_out_dim = 64

    # 训练参数
    learning_rate = 1e-4
    epochs = 100
    n_class = 1
    output_dir = "result"
    
    # 加载预训练嵌入
    drug_rw_emb = torch.load(os.path.join(script_dir, 'data', 'drug_rw_emb.pt'))
    prot_rw_emb = torch.load(os.path.join(script_dir, 'data', 'prot_rw_emb.pt'))

    # 初始化模型
    model = DPFDTI(
        drug_in_feats, drug_embedding, drug_hidden_feats, protein_emb_dim,
        mlp_in_dim, mlp_hidden_dim, mlp_out_dim, drug_padding,
        drug_rw_emb, prot_rw_emb
    ).to(device)
    
    # 优化器（损失函数在Trainer中已定义，无需重复定义）
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # 初始化训练器（传入验证集加载器）
    trainer = Trainer(
        model, opt, device, train_generator, val_generator, test_generator,
        epochs, n_class, batch_size, output_dir, save_test_att_maps=True
    )
    
    # 开始训练（Trainer会自动完成训练、验证、测试、早停）
    print("Starting training...")
    result = trainer.train()

    # 注：Trainer已经返回了验证集最佳指标和测试集最终指标，可直接使用
    print("\n===== Final Results from Trainer =====")
    print("Best Validation Metrics:")
    for k, v in result["val_best_metrics"].items():
        print(f"{k}: {v:.4f}")
    
    print("\nFinal Test Metrics:")
    for k, v in result["test_final_metrics"].items():
        print(f"{k}: {v:.4f}")
    