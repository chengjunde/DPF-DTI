import pandas as pd
import torch.utils.data as data
import torch
import numpy as np
from functools import partial
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer
from utils import integer_label_protein


class DTIDataset(data.Dataset):
    def __init__(self, idx_list, drug_smiles_list, protein_seq_list, interaction_matrix, max_drug_nodes=290):
        self.idx_list = idx_list
        self.drug_smiles_list = drug_smiles_list
        self.protein_seq_list = protein_seq_list
        self.interaction_matrix = interaction_matrix
        self.max_drug_nodes = max_drug_nodes

        self.atom_featurizer = CanonicalAtomFeaturizer()
        self.bond_featurizer = CanonicalBondFeaturizer(self_loop=True)
        self.fc = partial(smiles_to_bigraph, add_self_loop=True)

    def __len__(self):
        return len(self.idx_list)

    def __getitem__(self, index):
        real_index = self.idx_list[index]
        drug_index, protein_index = real_index

        v_d = self.drug_smiles_list[drug_index]
        v_d = self.fc(smiles=v_d, node_featurizer=self.atom_featurizer, edge_featurizer=self.bond_featurizer)
        actual_node_feats = v_d.ndata.pop('h')
        num_actual_nodes = actual_node_feats.shape[0]
        num_virtual_nodes = self.max_drug_nodes - num_actual_nodes
        virtual_node_bit = torch.zeros([num_actual_nodes, 1])
        actual_node_feats = torch.cat((actual_node_feats, virtual_node_bit), 1)
        v_d.ndata['h'] = actual_node_feats
        virtual_node_feat = torch.cat((torch.zeros(num_virtual_nodes, 74), torch.ones(num_virtual_nodes, 1)), 1)
        v_d.add_nodes(num_virtual_nodes, {"h": virtual_node_feat})
        v_d = v_d.add_self_loop()

        v_p = self.protein_seq_list[protein_index]
        v_p = integer_label_protein(v_p)
       
        y = self.interaction_matrix[drug_index, protein_index]
        return  v_d, v_p, drug_index, protein_index, y