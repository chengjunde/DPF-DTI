import torch.nn as nn
import torch.nn.functional as F
import torch
import math

from dgl.nn.pytorch import GINConv
from Intention import BiIntention

def binary_cross_entropy(pred_output, labels, pos_weight=1):
    """
    带正负样本加权的二元交叉熵损失计算
    
    Args:
        pred_output: 模型输出 (batch_size, 1) 或 (batch_size,)
        labels: 真实标签 (batch_size,)，值为0或1
        pos_weight: 正样本权重，若为None则自动计算当前批次的正负样本比
    
    Returns:
        sigmoid_output: 经过sigmoid后的预测概率 (batch_size,)
        loss: 加权后的交叉熵损失
    """
    if not isinstance(pos_weight, torch.Tensor):
        pos_weight = torch.tensor(pos_weight, device=labels.device)
    
    loss_fct = nn.BCEWithLogitsLoss(pos_weight=pos_weight)  
    pred_output_squeezed = pred_output.squeeze(1)
    loss = loss_fct(pred_output_squeezed, labels)
    sigmoid_output = torch.sigmoid(pred_output_squeezed)
    return sigmoid_output, loss

class DPFDTI(nn.Module):
    def __init__(self, drug_in_feats, drug_embedding, drug_hidden_feats, protein_emb_dim, 
                     mlp_in_dim, mlp_hidden_dim, mlp_out_dim, drug_padding, drug_rw_emb, prot_rw_emb):
        super(DPFDTI, self).__init__()

        self.drug_extractor = MolecularGIN(in_feats=drug_in_feats, dim_embedding=drug_embedding,
                                           padding=drug_padding,
                                           hidden_feats=drug_hidden_feats)
        

        self.protein_extractor = Protein_Encoder(dim_embedding=protein_emb_dim)

        self.cross_intention = BiIntention(embed_dim=128, num_head=8, layer=1)

        self.mlp_classifier = MLPDecoder(mlp_in_dim, mlp_hidden_dim, mlp_out_dim)
        
        self.register_buffer('drug_rw_emb', drug_rw_emb)
        self.register_buffer('prot_rw_emb', prot_rw_emb)

        # 替换随机游走层为结构感知交互模块
        self.struct_interaction = StructureAwareInteraction(
            drug_dim=384,
            protein_dim=384,
            hidden_dim=64,
            heads=2
        )


    def forward(self, bg_d, v_p, drug_idx, prot_idx, mode="train"):
        v_d = self.drug_extractor(bg_d) # [8, 290, 128]
        v_p = self.protein_extractor(v_p) # [8, 1000, 128]

        drug_feat = self.drug_rw_emb[drug_idx]
        prot_feat = self.prot_rw_emb[prot_idx]
        # 使用结构感知交互替代随机游走
        struct_d, struct_p = self.struct_interaction(drug_feat, prot_feat)

        f, v_d, v_p, att = self.cross_intention(drug=v_d, protein=v_p)
        f = torch.cat([v_d, v_p, struct_d, struct_p], dim=1)

        score = self.mlp_classifier(f) # [8, 1]
        if mode == "train":
            return v_d, v_p, f, score
        elif mode == "eval":
            return v_d, v_p, score, att

# drug first embedding

class MolecularGIN(nn.Module):
    def __init__(self, in_feats, dim_embedding=128, padding=True, hidden_feats=None):
        super().__init__()
        self.init_transform = nn.Linear(in_feats, dim_embedding, bias=False)
        if padding:
            with torch.no_grad():
                self.init_transform.weight[-1].fill_(0)
        
        # 使用DGL原生GINConv
        self.gin_layers = nn.ModuleList()
        in_dim = dim_embedding
        for out_dim in hidden_feats:
            self.gin_layers.append(
                GINConv(
                    apply_func=nn.Sequential(
                        nn.Linear(in_dim, out_dim),
                        nn.BatchNorm1d(out_dim),
                        nn.ReLU(),
                        nn.Linear(out_dim, out_dim)
                    ),
                    aggregator_type='sum',
                    learn_eps=True
                )
            )
            in_dim = out_dim
        
        self.output_feats = hidden_feats[-1]

    def forward(self, batch_graph):
        # 正确使用DGL图对象
        node_feats = batch_graph.ndata['h']
        node_feats = self.init_transform(node_feats)
        
        # 直接在图上进行消息传递
        for gin_layer in self.gin_layers:
            node_feats = gin_layer(batch_graph, node_feats)  # 传入DGLGraph对象
            node_feats = F.relu(node_feats)
        
        batch_size = batch_graph.batch_size
        return node_feats.view(batch_size, -1, self.output_feats)
    
# protein first embedding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) 
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class Protein_Encoder(nn.Module):
    def __init__(self, n_word=26, dim_embedding=128, d_model=128, nhead=8, 
                 dim_feedforward=1024, dropout=0.1, num_encoder_layers=3):
        super(Protein_Encoder, self).__init__()
        self.n_word = n_word
        self.dim = dim_embedding
        self.d_model = d_model
        self.embed_word = nn.Embedding(n_word, dim_embedding, padding_idx=0)  # 添加padding_idx
        self.pos_encoder = PositionalEncoding(d_model)
        prot_encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True
        )
        encoder_norm = nn.LayerNorm(d_model)
        self.prot_encoder = nn.TransformerEncoder(
            prot_encoder_layer, num_encoder_layers, encoder_norm
        )
    
    def forward(self, proteins):
        proteins = proteins.long()
        # 创建padding mask（假设padding值为0）
        # padding_mask = (proteins == 0)  # [batch_size, seq_len]
        
        protein_vector = self.embed_word(proteins)
        protein_vector = self.pos_encoder(protein_vector)
        protein_vector = F.layer_norm(protein_vector, [self.d_model]) 
        # 调整维度适配Transformer
        protein_vector = protein_vector.permute(1, 0, 2)  # [seq_len, batch_size, d_model]
        
        # 转换mask为Transformer需要的格式
        protein_vector = self.prot_encoder(protein_vector)
        
        # 恢复原始维度
        protein_vector = protein_vector.permute(1, 0, 2)
        return protein_vector

class MLPDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, binary=1):
        super(MLPDecoder, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.bn3 = nn.BatchNorm1d(out_dim)
        self.fc4 = nn.Linear(out_dim, binary)

    def forward(self, x):#x.shpae[64, 256]
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.bn2(F.relu(self.fc2(x)))
        x = self.bn3(F.relu(self.fc3(x)))
        x = self.fc4(x)
        return x

class StructureAwareInteraction(nn.Module):
    def __init__(self, drug_dim, protein_dim, hidden_dim=64, heads=4):
        super().__init__()
        # 图结构感知的注意力机制
        self.graph_aware_att = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=heads, batch_first=True
        )
    
        self.seq_aware_att = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=heads, batch_first=True
        )
        
        # 结构信息投影层
        self.drug_struct_proj = nn.Sequential(
            nn.Linear(drug_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim))
        
        self.prot_struct_proj = nn.Sequential(
            nn.Linear(protein_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim))
        
        # 交互融合门控
        self.fusion_gate = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.Sigmoid()
        )

    def forward(self, drug_feats, prot_feats):
        # 投影结构信息
        drug_struct = self.drug_struct_proj(drug_feats)
        prot_struct = self.prot_struct_proj(prot_feats)
        
        # 图结构感知的交叉注意力
        drug_att, _ = self.graph_aware_att(
            query=drug_struct,
            key=prot_struct,
            value=prot_struct
        )
        
        prot_att, _ = self.seq_aware_att(
            query=prot_struct,
            key=drug_struct,
            value=drug_struct
        )

        combined = torch.cat([drug_att, prot_att], dim=1)  
        
        # 门控融合机制
        fusion_coeff = self.fusion_gate(combined)
        fused_drug = fusion_coeff * drug_att
        fused_prot = (1 - fusion_coeff) * prot_att
        
        return fused_drug, fused_prot
    