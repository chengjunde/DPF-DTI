import torch
import torch.nn as nn
from einops import reduce

class Intention(nn.Module):
    def __init__(self, dim, num_heads, kqv_bias=False, device='cuda'):
        super(Intention, self).__init__()
        self.dim = dim
        self.head = num_heads
        self.head_dim = dim // num_heads
        self.device = device
        self.alpha = nn.Parameter(torch.rand(1))
        assert dim % num_heads == 0, 'dim must be divisible by num_heads!'

        self.wq = nn.Linear(dim, dim, bias=kqv_bias)
        self.wk = nn.Linear(dim, dim, bias=kqv_bias)
        self.wv = nn.Linear(dim, dim, bias=kqv_bias)

        self.softmax = nn.Softmax(dim=-2)
        self.out = nn.Linear(dim, dim)

    def forward(self, x, query=None):
        if query is None:
            query = x

        # 原始的qkv投影
        query = self.wq(query)
        key = self.wk(x)
        value = self.wv(x)

        b, n, c = x.shape
        key = key.reshape(b, n, self.head, self.head_dim).permute(0, 2, 1, 3)
        key_t = key.clone().permute(0, 1, 3, 2)
        value = value.reshape(b, n, self.head, self.head_dim).permute(0, 2, 1, 3)

        b, n, c = query.shape
        query = query.reshape(b, n, self.head, self.head_dim).permute(0, 2, 1, 3)

        # ========== 传统注意力机制计算 (不参与反向传播) ==========
        # 计算缩放的点积注意力 (Scaled Dot-Product Attention)
        with torch.no_grad():  # 禁用梯度计算
            # 传统注意力分数计算
            attn_scores = (query @ key_t) / torch.sqrt(torch.tensor(self.head_dim, device=self.device))
            # 计算传统注意力图
            traditional_attn_map = self.softmax(attn_scores)
            # 确保完全脱离计算图
            traditional_attn_map = traditional_attn_map.detach()
        
        # ========== 原有逻辑保持不变 ==========
        kk = key_t @ key
        kk = self.alpha * torch.eye(kk.shape[-1], device=self.device) + kk
        kk_inv = torch.inverse(kk)
        attn_map = (kk_inv @ key_t) @ value

        attn_map = self.softmax(attn_map)

        out = (query @ attn_map)
        out = out.permute(0, 2, 1, 3).reshape(b, n, c)
        out = self.out(out)

        # 返回原始输出 + 传统注意力图
        return out, traditional_attn_map

class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.):
        super(SelfAttention, self).__init__()
        self.wq = nn.Sequential(
             nn.Linear(dim, dim),
             nn.Dropout(p=dropout)
            )
        self.wk = nn.Sequential(
             nn.Linear(dim, dim),
             nn.Dropout(p=dropout)
            )
        self.wv = nn.Sequential(
             nn.Linear(dim, dim),
             nn.Dropout(p=dropout)
            )
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads)

    def forward(self, x):
        query = self.wq(x)
        key = self.wk(x)
        value = self.wv(x)
        att, _ = self.attn(query, key, value)
        out = att + x
        return out

class IntentionBlock(nn.Module):
    def __init__(self, dim, num_heads=8, kqv_bias=True, device='cuda'):
        super(IntentionBlock, self).__init__()
        self.norm_layer = nn.LayerNorm(dim)
        self.attn = Intention(dim=dim, num_heads=num_heads, kqv_bias=kqv_bias, device=device)
        self.softmax = nn.Softmax(dim=-2)
        self.beta = nn.Parameter(torch.rand(1))

    def forward(self, x, q):
        x = self.norm_layer(x)
        q_t = q.permute(0, 2, 1)
        # 接收传统注意力图，但不使用它进行计算
        att, traditional_attn_map = self.attn(x, q)
        att_map = self.softmax(att)
        out = self.beta * q_t @ att_map
        # 返回原始输出 + 传统注意力图
        return out, traditional_attn_map

class BiIntention(nn.Module):
    def __init__(self, embed_dim, layer=1, num_head=8, device='cuda'):
        super(BiIntention, self).__init__()

        self.layer = layer
        self.drug_intention = nn.ModuleList([
            IntentionBlock(dim=embed_dim, device=device, num_heads=num_head) for _ in range(layer)])
        self.protein_intention = nn.ModuleList([
            IntentionBlock(dim=embed_dim, device=device, num_heads=num_head) for _ in range(layer)])
        # self attention
        self.attn_drug = SelfAttention(dim=embed_dim, num_heads=num_head)
        self.attn_protein = SelfAttention(dim=embed_dim, num_heads=num_head)

    def forward(self, drug, protein):
        drug = self.attn_drug(drug)
        protein = self.attn_protein(protein)
        
        # 存储每层的传统注意力图
        drug_protein_att_maps = []
        protein_drug_att_maps = []

        for i in range(self.layer):
            # 获取传统注意力图但不参与计算
            v_p, drug_protein_att = self.drug_intention[i](drug, protein)
            v_d, protein_drug_att = self.protein_intention[i](protein, drug)
            
            # 保存注意力图
            drug_protein_att_maps.append(drug_protein_att)
            protein_drug_att_maps.append(protein_drug_att)
            
            drug, protein = v_d, v_p

        v_d = reduce(drug, 'B H W -> B H', 'max')
        v_p = reduce(protein, 'B H W -> B H', 'max')

        f = torch.cat((v_d, v_p), dim=1)

        # 返回原始输出 + 注意力图（可以按需使用）
        return f, v_d, v_p, protein_drug_att_maps
        

# 测试代码
if __name__ == "__main__":
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 创建模型实例
    model = BiIntention(embed_dim=128, layer=2, num_head=8, device=device).to(device)
    
    # 生成测试数据 (batch_size, seq_len, dim)
    batch_size = 4
    drug_seq_len = 20
    protein_seq_len = 30
    embed_dim = 128
    
    drug = torch.randn(batch_size, drug_seq_len, embed_dim).to(device)
    protein = torch.randn(batch_size, protein_seq_len, embed_dim).to(device)
    
    # 前向传播
    f, v_d, v_p, att_maps = model(drug, protein)
    
    # 打印输出形状
    print("融合特征形状:", f.shape)
    print("药物特征形状:", v_d.shape)
    print("蛋白质特征形状:", v_p.shape)
    print("药物->蛋白质注意力图层数:", len(att_maps['drug_protein_att_maps']))
    print("注意力图形状:", att_maps['drug_protein_att_maps'][0].shape)
    
    # 验证梯度是否隔离
    # 检查注意力图是否有梯度
    print("\n注意力图是否有梯度:", att_maps['drug_protein_att_maps'][0].requires_grad)
    # 检查模型输出是否有梯度（应该有）
    print("模型输出是否有梯度:", f.requires_grad)