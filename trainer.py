import torch
import torch.nn as nn
import copy
import os
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, confusion_matrix, precision_recall_curve, precision_score
from models import binary_cross_entropy
from prettytable import PrettyTable
from tqdm import tqdm
import torch.nn.functional as F
import dgl
import pickle

class Trainer(object):
    def __init__(self, model, optim, device, train_dataloader, val_dataloader, test_dataloader, 
                 epochs, n_class, batch_size, output_dir, save_test_att_maps=False):
        """
        :param save_test_att_maps: bool, 控制是否在训练结束后保存测试集的注意力图
        """
        self.model = model
        self.optim = optim
        self.device = device
        self.epochs = epochs
        self.current_epoch = 0
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.n_class = n_class
        self.batch_size = batch_size
        self.nb_training = len(self.train_dataloader)
        self.step = 0
        self.output_dir = output_dir
        
        # <--- 新增控制参数
        self.save_test_att_maps = save_test_att_maps
        self.test_att_maps = None  # 用于存储测试集注意力图
        
        # 早停相关参数
        self.early_stop_patience = 15
        self.early_stop_counter = 0
        self.best_val_auroc = 0
        self.best_model_weights = None
        self.best_epoch = 0
        
        # 指标记录
        self.train_loss_epoch = []
        self.val_metrics = {}    
        self.test_metrics = {}   
        
        # 表格表头
        test_metric_header = ["# Best Epoch", "AUROC", "AUPRC", "F1", "Sensitivity", "Specificity", "Accuracy",
                              "Threshold", "Test_loss"]
        val_metric_header = ["# Epoch", "Val_AUROC", "Val_AUPRC", "Val_F1", "Val_Sensitivity", "Val_Specificity", "Val_Accuracy", "Val_Loss"]
        train_metric_header = ["# Epoch", "Train_loss", "Train_Accuracy"]
        self.test_table = PrettyTable(test_metric_header)
        self.val_table = PrettyTable(val_metric_header)
        self.train_table = PrettyTable(train_metric_header)

    def train(self):
        float2str = lambda x: '%0.4f' % x
        for i in range(self.epochs):
            self.current_epoch += 1
            
            # 1. 训练
            train_loss, train_accuracy = self.train_epoch()
            train_lst = [f"epoch {self.current_epoch}"] + list(map(float2str, [train_loss, train_accuracy]))
            self.train_table.add_row(train_lst)
            self.train_loss_epoch.append(train_loss)
            print(f'\n===== Epoch {self.current_epoch} Training =====')
            print(f'Training loss: {train_loss:.4f} | Training accuracy: {train_accuracy*100:.2f}%')
            
            # 2. 验证 (默认不返回 att maps 以节省内存)
            val_results = self.eval_epoch(dataloader="val", return_att_maps=False)
            val_auroc, val_auprc, val_f1, val_sensitivity, val_specificity, val_accuracy, val_loss, val_thred, val_precision = val_results
            
            print(f'===== Epoch {self.current_epoch} Validation =====')
            print(f'Val Loss={val_loss:.4f} | Val AUROC={val_auroc:.4f} | Val Acc={val_accuracy:.2%}')
            
            val_lst = [f"epoch {self.current_epoch}"] + list(map(float2str, [val_auroc, val_auprc, val_f1, val_sensitivity, val_specificity, val_accuracy, val_loss]))
            self.val_table.add_row(val_lst)
            
            # 3. 早停逻辑
            if val_auroc > self.best_val_auroc:
                self.best_val_auroc = val_auroc
                self.best_epoch = self.current_epoch
                self.best_model_weights = copy.deepcopy(self.model.state_dict())
                self.val_metrics.update({
                    "auroc": val_auroc, "auprc": val_auprc, "F1": val_f1,
                    "sensitivity": val_sensitivity, "specificity": val_specificity,
                    "accuracy": val_accuracy, "loss": val_loss, "threshold": val_thred,
                    "precision": val_precision, "best_epoch": self.best_epoch
                })
                self.early_stop_counter = 0
            else:
                self.early_stop_counter += 1
                if self.early_stop_counter >= self.early_stop_patience:
                    print(f"\nEarly stopping triggered at epoch {self.current_epoch}! Best epoch: {self.best_epoch}")
                    break

        # 4. 加载最佳模型
        print(f'\n===== Final Evaluation (Best Epoch {self.best_epoch}) =====')
        if self.best_model_weights is not None:
            self.model.load_state_dict(self.best_model_weights)
            print("Loaded best model weights.")
        
        # 5. 测试集评估
        # <--- 关键：仅当 save_test_att_maps=True 时，才请求返回注意力图
        return_atts = self.save_test_att_maps
        test_results = self.eval_epoch(dataloader="test", return_att_maps=return_atts)
        
        if return_atts:
            test_auroc, test_auprc, test_f1, test_sensitivity, test_specificity, test_accuracy, test_loss, test_thred, test_precision, test_atts = test_results
            self.test_att_maps = test_atts  # 保存注意力图
        else:
            test_auroc, test_auprc, test_f1, test_sensitivity, test_specificity, test_accuracy, test_loss, test_thred, test_precision = test_results

        self.test_metrics.update({
            "auroc": test_auroc, "auprc": test_auprc, "F1": test_f1,
            "sensitivity": test_sensitivity, "specificity": test_specificity,
            "accuracy": test_accuracy, "loss": test_loss, "threshold": test_thred,
            "precision": test_precision, "best_epoch": self.best_epoch
        })
        
        test_lst = [f"epoch {self.best_epoch}"] + list(map(float2str, [test_auroc, test_auprc, test_f1, test_sensitivity, test_specificity, test_accuracy, test_thred, test_loss]))
        self.test_table.add_row(test_lst)
        
        print(f'===== Final Test Metrics =====')
        print(f'Test AUROC={test_auroc:.4f} | Test AUPRC={test_auprc:.4f} | Test F1={test_f1:.4f} | Test Acc={test_accuracy:.2%}')
        
        # 6. 保存结果
        self.save_result()
        
        return {
            "val_best_metrics": self.val_metrics,
            "test_final_metrics": self.test_metrics
        }

    def save_result(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # 保存基础指标
        state = {
            "train_epoch_loss": self.train_loss_epoch,
            "val_best_metrics": self.val_metrics,
            "test_final_metrics": self.test_metrics,
        }
        torch.save(state, os.path.join(self.output_dir, "all_metrics.pt"))
        
        # <--- 关键：如果开启了保存且有数据，单独保存注意力图
        if self.save_test_att_maps and self.test_att_maps is not None:
            att_save_path = os.path.join(self.output_dir, "test_attention_maps.pt")
            try:
                torch.save({
                    "best_epoch": self.best_epoch,
                    "test_att_maps": self.test_att_maps
                }, att_save_path)
                print(f"Test attention maps saved to {att_save_path}")
            except Exception as e:
                # 如果 torch.save 失败（例如包含非 tensor 对象），回退到 pickle
                pkl_save_path = os.path.join(self.output_dir, "test_attention_maps.pkl")
                print(f"Warning: torch.save failed ({e}), saving as pickle to {pkl_save_path}")
                with open(pkl_save_path, "wb") as f:
                    pickle.dump({
                        "best_epoch": self.best_epoch,
                        "test_att_maps": self.test_att_maps
                    }, f)

        # 保存表格
        with open(os.path.join(self.output_dir, "train_table.txt"), "w") as fp:
            fp.write(self.train_table.get_string())
        with open(os.path.join(self.output_dir, "val_table.txt"), 'w') as fp:
            fp.write(self.val_table.get_string())
        with open(os.path.join(self.output_dir, "test_table.txt"), 'w') as fp:
            fp.write(self.test_table.get_string())
        
        torch.save(self.best_model_weights, os.path.join(self.output_dir, "best_model.pth"))
        print(f"\nAll results saved to {self.output_dir}")

    def train_epoch(self):
        self.model.train()
        loss_epoch = 0
        correct = 0    
        total = 0       
        num_batches = len(self.train_dataloader)
        
        for i, (v_d, v_p, drug_idx, prot_idx, labels) in enumerate(tqdm(self.train_dataloader, desc="Training")):
            self.step += 1
            # 设备迁移
            if isinstance(v_d, dgl.DGLGraph): v_d = v_d.to(self.device)
            elif isinstance(v_d, torch.Tensor): v_d = v_d.to(self.device)
            elif isinstance(v_d, dict): v_d = {k: v.to(self.device) for k, v in v_d.items()}
            if isinstance(v_p, dgl.DGLGraph): v_p = v_p.to(self.device)
            elif isinstance(v_p, torch.Tensor): v_p = v_p.to(self.device)
            elif isinstance(v_p, dict): v_p = {k: v.to(self.device) for k, v in v_p.items()}
            
            drug_idx = drug_idx.to(self.device)
            prot_idx = prot_idx.to(self.device)
            labels = labels.float().to(self.device)
            
            self.optim.zero_grad()
            v_d_out, v_p_out, f, score = self.model(v_d, v_p, drug_idx, prot_idx, mode='train')
            n, loss = binary_cross_entropy(score, labels)
            loss.backward()
            self.optim.step()
            
            loss_epoch += loss.item()
            with torch.no_grad():
                preds = (n >= 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)
            
        
        return loss_epoch / num_batches, (correct / total if total != 0 else 0.0)

    def eval_epoch(self, dataloader="val", return_att_maps=False):
        """
        :param return_att_maps: 是否收集并返回注意力图
        """
        loss_epoch = 0
        y_label, y_pred = [], []
        all_att_maps = [] if return_att_maps else None
        
        data_loader = self.val_dataloader if dataloader == "val" else self.test_dataloader
        num_batches = len(data_loader)
        
        with torch.no_grad():
            self.model.eval()
            for i, (v_d, v_p, drug_idx, prot_idx, labels) in enumerate(tqdm(data_loader, desc=f"Evaluating {dataloader}")):
                # 设备迁移
                if isinstance(v_d, dgl.DGLGraph): v_d = v_d.to(self.device)
                elif isinstance(v_d, torch.Tensor): v_d = v_d.to(self.device)
                elif isinstance(v_d, dict): v_d = {k: v.to(self.device) for k, v in v_d.items()}
                if isinstance(v_p, dgl.DGLGraph): v_p = v_p.to(self.device)
                elif isinstance(v_p, torch.Tensor): v_p = v_p.to(self.device)
                elif isinstance(v_p, dict): v_p = {k: v.to(self.device) for k, v in v_p.items()}
                
                drug_idx = drug_idx.to(self.device)
                prot_idx = prot_idx.to(self.device)
                labels = labels.float().to(self.device)
                
                # 前向传播
                v_d_out, v_p_out, score, att = self.model(v_d, v_p, drug_idx, prot_idx, mode='eval')
                
                n, loss = binary_cross_entropy(score, labels)
                
                loss_epoch += loss.item()
                y_label.extend(labels.cpu().tolist())
                y_pred.extend(n.cpu().tolist())
                
                # <--- 收集注意力图 (仅当需要时)
                if return_att_maps:
                    # 移至 CPU 以释放显存
                    if isinstance(att, torch.Tensor):
                        att_cpu = att.cpu()
                    elif isinstance(att, list):
                        att_cpu = [x.cpu() if isinstance(x, torch.Tensor) else x for x in att]
                    elif isinstance(att, dict):
                        att_cpu = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in att.items()}
                    else:
                        att_cpu = att
                    
                    all_att_maps.append({
                        "batch_idx": i,
                        "labels": labels.cpu(),
                        "drug_idx": drug_idx.cpu(),
                        "prot_idx": prot_idx.cpu(),
                        "att_map": att_cpu
                    })

        loss_epoch = loss_epoch / num_batches
        
        # 计算指标
        auroc = roc_auc_score(y_label, y_pred)
        auprc = average_precision_score(y_label, y_pred)
        
        fpr, tpr, thresholds = roc_curve(y_label, y_pred)
        prec, recall, _ = precision_recall_curve(y_label, y_pred)
        precision_arr = tpr / (tpr + fpr + 1e-8)
        f1_arr = 2 * precision_arr * tpr / (tpr + precision_arr + 1e-8)
        
        thred_optim = thresholds[5:][np.argmax(f1_arr[5:])] if len(thresholds) > 5 else 0.5
        y_pred_s = [1 if p >= thred_optim else 0 for p in y_pred]
        
        cm = confusion_matrix(y_label, y_pred_s)
        if cm.shape == (1, 1):
            cm = np.pad(cm, ((0,1),(0,1)), mode='constant')
        
        accuracy = (cm[0, 0] + cm[1, 1]) / np.sum(cm) if np.sum(cm) > 0 else 0.0
        specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0.0
        sensitivity = cm[1, 1] / (cm[1, 0] + cm[1, 1]) if (cm[1, 0] + cm[1, 1]) > 0 else 0.0
        precision1 = precision_score(y_label, y_pred_s, zero_division=0)
        max_f1 = np.max(f1_arr[5:]) if len(f1_arr) > 5 else 0.0
        
        if return_att_maps:
            return auroc, auprc, max_f1, sensitivity, specificity, accuracy, loss_epoch, thred_optim, precision1, all_att_maps
        else:
            return auroc, auprc, max_f1, sensitivity, specificity, accuracy, loss_epoch, thred_optim, precision1