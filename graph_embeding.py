import torch
import numpy as np
import networkx as nx
from node2vec import Node2Vec

def prepare_similarity_networks(device):
    drug_association_files = ["mat_drug_drug.txt", "mat_drug_disease_proc.txt", "mat_drug_similarity_proc.txt"]
    prot_association_files = ["mat_protein_protein.txt", "mat_protein_disease_proc.txt", "mat_protein_similarity_proc.txt"]

    drug_sim_nets = []
    for f in drug_association_files:
        mat = np.loadtxt(f"data/graphs/{f}")
        mat_tensor = torch.from_numpy(mat).float().to(device)
        drug_sim_nets.append(mat_tensor)
        # 验证索引一致性
        if len(drug_sim_nets) > 1:
            assert drug_sim_nets[0].shape == drug_sim_nets[-1].shape, "药物矩阵维度不一致"

    prot_sim_nets = []
    for f in prot_association_files:
        mat = np.loadtxt(f"data/graphs/{f}")
        mat_tensor = torch.from_numpy(mat).float().to(device)
        prot_sim_nets.append(mat_tensor)
        if len(prot_sim_nets) > 1:
            assert prot_sim_nets[0].shape == prot_sim_nets[-1].shape, "蛋白矩阵维度不一致"

    return drug_sim_nets, prot_sim_nets

def generate_rw_embeddings(adj_list, output_path, dimensions=128):
    all_embeddings = []
    for adj in adj_list:
        # 移动到 CPU 并二值化
        adj = adj.cpu().numpy()
        G = nx.from_numpy_array(adj)
        # 生成 Node2Vec 嵌入
        node2vec = Node2Vec(G, dimensions=dimensions, walk_length=30, num_walks=200, workers=4)
        model = node2vec.fit(window=10, min_count=1)
        embeddings = torch.tensor(model.wv.vectors, dtype=torch.float32)
        all_embeddings.append(embeddings)
    # 拼接嵌入并保存
    combined_emb = torch.cat(all_embeddings, dim=1)
    torch.save(combined_emb, output_path)
    return combined_emb

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        drug_sim_nets, prot_sim_nets = prepare_similarity_networks(device)
        drug_rw_emb = generate_rw_embeddings(drug_sim_nets, "drug_rw_emb.pt")
        prot_rw_emb = generate_rw_embeddings(prot_sim_nets, "prot_rw_emb.pt")
        print("嵌入生成成功！")
    except Exception as e:
        print(f"错误: {str(e)}")