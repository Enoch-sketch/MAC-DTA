"""
Dataset classes for PMHGT-DTA-CLIFF
"""
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data as DATA
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import os
from torch_geometric.data import DataLoader

import re
from smiles2graph import smile2graph4drugood

AA_TO_IDX = {aa: i for i, aa in enumerate('ACDEFGHIKLMNPQRSTVWYX')}


def load_mutation_info(csv_path='./mutation_info.csv'):
    """加载突变信息表，返回 target_key -> info 的字典"""
    if not os.path.exists(csv_path):
        print(f"WARNING: {csv_path} not found, mutation mask disabled")
        return {}
    mi = pd.read_csv(csv_path)
    info = {}
    for _, row in mi.iterrows():
        info[row['target_key']] = {
            'is_mutant': bool(row['is_mutant']),
            'mutated_idx': int(row['mutated_idx']),
            'in_window': bool(row['in_window']),
        }
    return info


def build_seq_mutation_flag(target_key, seq_len, mutation_info):
    """为序列路径构建突变标记张量 [seq_len], 突变位点=1"""
    flag = torch.zeros(seq_len, dtype=torch.long)
    info = mutation_info.get(target_key)
    if info and info['is_mutant'] and info['in_window']:
        idx = info['mutated_idx']
        if 0 <= idx < seq_len:
            flag[idx] = 1
    return flag


def build_mutation_flag(target_key, num_nodes, mutation_info):
    """为蛋白质图节点构建突变标记张量 (0=normal, 1=mutated)"""
    flag = torch.zeros(num_nodes, dtype=torch.long)
    info = mutation_info.get(target_key)
    if info and info['is_mutant'] and info['in_window']:
        idx = info['mutated_idx']
        if 0 <= idx < num_nodes:
            flag[idx] = 1
    return flag


def build_native_x_from_sequence(sequence, num_nodes):
    """
    根据当前样本实际使用的蛋白序列，重建图节点的残基身份索引。

    这里不修改 edge_index / contact map，只保证节点语义（第 i 个节点代表什么残基）
    与当前 `fixed_target_sequence` / `target_sequence` 一致。

    当序列长度与图节点数不一致时，采取安全截断/补 X 的策略。
    """
    aa_idx = [AA_TO_IDX.get(aa, 20) for aa in sequence]
    if len(aa_idx) >= num_nodes:
        aa_idx = aa_idx[:num_nodes]
    else:
        aa_idx = aa_idx + [20] * (num_nodes - len(aa_idx))
    return torch.tensor(aa_idx, dtype=torch.long)


def create_fold_setting_cold(df, fold_seed, frac, entities):
    """create cold-split where given one or multiple columns, it first splits based on
    entities in the columns and then maps all associated data points to the partition

    Args:
            df (pd.DataFrame): dataset dataframe
            fold_seed (int): the random seed
            frac (list): a list of train/valid/test fractions
            entities (Union[str, List[str]]): either a single "cold" entity or a list of
                    "cold" entities on which the split is done

    Returns:
            dict: a dictionary of splitted dataframes, where keys are train/valid/test and values correspond to each dataframe
    """
    if isinstance(entities, str):
        entities = [entities]

    train_frac, val_frac, test_frac = frac

    # For each entity, sample the instances belonging to the test datasets
    test_entity_instances = [
        df[e]
        .drop_duplicates()
        .sample(frac=test_frac, replace=False, random_state=fold_seed)
        .values
        for e in entities
    ]

    # Select samples where all entities are in the test set
    test = df.copy()
    for entity, instances in zip(entities, test_entity_instances):
        test = test[test[entity].isin(instances)]

    if len(test) == 0:
        raise ValueError(
            "No test samples found. Try another seed, increasing the test frac or a "
            "less stringent splitting strategy."
        )

    # Proceed with validation data
    train_val = df.copy()
    for i, e in enumerate(entities):
        train_val = train_val[~train_val[e].isin(test_entity_instances[i])]

    val_entity_instances = [
        train_val[e]
        .drop_duplicates()
        .sample(frac=val_frac / (1 - test_frac), replace=False, random_state=fold_seed)
        .values
        for e in entities
    ]
    val = train_val.copy()
    for entity, instances in zip(entities, val_entity_instances):
        val = val[val[entity].isin(instances)]

    if len(val) == 0:
        raise ValueError(
            "No validation samples found. Try another seed, increasing the test frac "
            "or a less stringent splitting strategy."
        )

    train = train_val.copy()
    for i, e in enumerate(entities):
        train = train[~train[e].isin(val_entity_instances[i])]

    return {
        "train": train.reset_index(drop=True),
        "valid": val.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }


class DTADataset(InMemoryDataset):
    """
    Drug-Target Affinity Dataset
    
    Args:
        root: Root directory
        path: Path to CSV data file
        smiles_emb: SMILES embedding dictionary
        target_emb: Target embedding dictionary
        smiles_idx: SMILES index dictionary
        smiles_graph: SMILES graph dictionary
        target_graph: Target graph dictionary (not used, loaded from files)
        smiles_len: SMILES length dictionary
        target_len: Target length dictionary
        protein_graph_dir: Directory containing protein graph .pt files
        drug_graph_file: Path to drug graph .pt file
        dataset_name: Name of the dataset (davis/kiba)
    """
    def __init__(self, root, path, smiles_emb, target_emb, smiles_idx, smiles_graph, 
                 target_graph, smiles_len, target_len, 
                 protein_graph_dir=None, drug_graph_file=None, dataset_name='davis',
                 mutation_info_path='./mutation_info.csv'):

        super(DTADataset, self).__init__(root)
        self.path = path
        df = pd.read_csv(path)
        self.data = df
        self.dataset_name = dataset_name
        
        if protein_graph_dir is None:
            protein_graph_dir = f'./graphs_{dataset_name}'
        if drug_graph_file is None:
            drug_graph_file = f'./unimol_compounds_{dataset_name}.pt'
        
        self.target_graph_dict = {}
        uniprot_list = df['uniprot'].unique()
        print(f"加载蛋白质图数据从: {protein_graph_dir}")
        
        loaded_count = 0
        for uniprot in tqdm(uniprot_list, desc="加载蛋白质图"):
            graph_path = os.path.join(protein_graph_dir, f'{uniprot}.pt')
            if os.path.exists(graph_path):
                try:
                    pro_graph = torch.load(graph_path, map_location='cpu', weights_only=False)
                    self.target_graph_dict[uniprot] = pro_graph
                    loaded_count += 1
                except Exception as e:
                    print(f"加载蛋白质图失败 {uniprot}: {e}")
        
        print(f"成功加载 {loaded_count}/{len(uniprot_list)} 个蛋白质图")
        
        print(f"加载药物图数据从: {drug_graph_file}")
        if os.path.exists(drug_graph_file):
            self.drug_graph_dict = torch.load(drug_graph_file, map_location='cpu', weights_only=False)
            print(f"加载了 {len(self.drug_graph_dict)} 个药物图")
        else:
            print(f"警告: 药物图文件不存在 {drug_graph_file}")
            self.drug_graph_dict = {}

        self.mutation_info = load_mutation_info(mutation_info_path)

        self.smiles_emb = smiles_emb
        self.target_emb = target_emb
        self.smiles_len = smiles_len
        self.target_len = target_len

        self.process(df, smiles_emb, target_emb, smiles_idx, smiles_graph, target_graph, smiles_len, target_len)
        
        self._filter_valid_samples()

    def _filter_valid_samples(self):
        """过滤掉没有对应图数据的样本，并将 DataFrame 转为 list-of-dicts 加速 __getitem__"""
        valid_indices = []
        for idx in range(len(self.data)):
            da = self.data.iloc[idx, :]
            uniprot = da['uniprot']
            smiles = da['compound_iso_smiles']
            
            if uniprot in self.target_graph_dict and smiles in self.drug_graph_dict:
                valid_indices.append(idx)
        
        if len(valid_indices) < len(self.data):
            print(f"过滤后样本数: {len(valid_indices)}/{len(self.data)}")
            self.data = self.data.iloc[valid_indices].reset_index(drop=True)
        
        # 转为 list-of-dicts，__getitem__ 中用 O(1) 访问代替慢速 pandas iloc
        self._data_records = self.data.to_dict('records')

    @property
    def raw_file_names(self):
        pass

    @property
    def processed_file_names(self):
        return ['process.pt']

    def download(self):
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def process(self, df, smiles_emb, target_emb, smiles_idx, smiles_graph, target_graph, smiles_len, target_len):
        if self.pre_filter is not None:
            self.data = [data for data in self.data if self.pre_filter(self.data)]

        if self.pre_transform is not None:
            self.data = [self.pre_transform(data) for data in self.data]
        pass

    def off_adj(self, adj, size):
        adj1 = adj.copy()
        for i in range(adj1.shape[0]):
            adj1[i][0] += size
            adj1[i][1] += size
        return adj1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        da = self._data_records[idx]
        uniprot = da['uniprot']
        sm = da['compound_iso_smiles']
        
        if uniprot in self.target_graph_dict:
            target_graph = self.target_graph_dict[uniprot].clone()
        else:
            target_graph = DATA(
                x=torch.zeros(1, 1280),
                edge_index=torch.zeros(2, 0, dtype=torch.long),
                native_x=torch.zeros(1, dtype=torch.long)
            )
        
        if sm in self.drug_graph_dict:
            drug_graph = self.drug_graph_dict[sm]
        else:
            drug_graph = DATA(
                x=torch.zeros(1, 75),
                edge_index=torch.zeros(2, 0, dtype=torch.long),
                edge_attr=torch.zeros(0, 6)
            )

        target = da['target_key']
        seq = da.get('fixed_target_sequence', da['target_sequence'])
        if pd.isna(seq):
            seq = da['target_sequence']
        label = da['affinity']

        num_nodes = target_graph.x.size(0)
        # 用当前样本真正使用的序列对齐图节点残基身份。
        # 这一步不重建图拓扑，只修复 node identity / native_x 语义。
        target_graph.native_x = build_native_x_from_sequence(seq, num_nodes)
        mut_flag = build_mutation_flag(target, num_nodes, self.mutation_info)
        target_graph.mut_flag = mut_flag
        seq_fixed = bool(da.get('seq_fixed', False))
        target_graph.seq_fixed_graph = torch.tensor([1 if seq_fixed else 0], dtype=torch.float32)
        # 旧图上的 ESM 节点特征很可能仍来自旧序列；对修复过序列的样本打标，
        # 供模型侧进行保守降权，而不是直接删除。
        target_graph.esm_stale_graph = torch.tensor([1 if seq_fixed else 0], dtype=torch.float32)

        smiles = self.smiles_emb.get(sm, torch.zeros(100, dtype=torch.long))
        protein = self.target_emb.get(seq, torch.zeros(1000, dtype=torch.long))
        smiles_lengths = self.smiles_len.get(sm, 1)
        protein_lengths = self.target_len.get(seq, 1)

        # mut_flag_seq 必须 pad 到与 protein token 序列相同的固定长度 tar_len
        # 否则 PyG Batch 拼接后无法 view(batchsize, -1)
        tar_len = len(protein)  # protein 已经是 tar_len 长度的 tensor
        raw_mut_flag = build_seq_mutation_flag(target, len(seq), self.mutation_info)
        if len(raw_mut_flag) < tar_len:
            pad = torch.zeros(tar_len - len(raw_mut_flag), dtype=torch.long)
            mut_flag_seq = torch.cat([raw_mut_flag, pad])
        else:
            mut_flag_seq = raw_mut_flag[:tar_len]

        Data = DATA(y=torch.FloatTensor([label]),
                    sm=sm,
                    target=target,
                    uniprot=uniprot,
                    smiles=smiles,
                    protein=protein,
                    smiles_lengths=smiles_lengths,
                    protein_lengths=protein_lengths,
                    seq=seq,
                    mut_flag_seq=mut_flag_seq,
                    # Global dataset index: used to look up pre-computed teacher
                    # predictions (KD) and cliff-sample masks without ambiguity.
                    # Stored here in the base class so it is available regardless
                    # of whether DTADatasetLocal or DTADataset is used (i.e. with
                    # or without --no_local_branch).
                    sample_idx=torch.tensor([idx], dtype=torch.long),
                    )
        return Data, drug_graph, target_graph


class SimpleDTADataset(torch.utils.data.Dataset):
    """
    简化版的DTA数据集，不依赖InMemoryDataset
    """
    def __init__(self, csv_path, smiles_emb, target_emb, smiles_len, target_len,
                 protein_graph_dir, drug_graph_file, dataset_name='davis'):
        
        self.data = pd.read_csv(csv_path)
        self.smiles_emb = smiles_emb
        self.target_emb = target_emb
        self.smiles_len = smiles_len
        self.target_len = target_len
        
        # 加载蛋白质图
        print(f"加载蛋白质图从: {protein_graph_dir}")
        self.target_graph_dict = {}
        for uniprot in tqdm(self.data['uniprot'].unique(), desc="加载蛋白质图"):
            graph_path = os.path.join(protein_graph_dir, f'{uniprot}.pt')
            if os.path.exists(graph_path):
                try:
                    self.target_graph_dict[uniprot] = torch.load(graph_path, map_location='cpu', weights_only=False)
                except:
                    pass
        
        # 加载药物图
        print(f"加载药物图从: {drug_graph_file}")
        if os.path.exists(drug_graph_file):
            self.drug_graph_dict = torch.load(drug_graph_file, map_location='cpu', weights_only=False)
        else:
            self.drug_graph_dict = {}
        
        # 过滤有效样本
        self._filter_valid_samples()
        print(f"有效样本数: {len(self.data)}")
    
    def _filter_valid_samples(self):
        valid_mask = []
        for idx in range(len(self.data)):
            row = self.data.iloc[idx]
            valid = (row['uniprot'] in self.target_graph_dict and 
                    row['compound_iso_smiles'] in self.drug_graph_dict)
            valid_mask.append(valid)
        self.data = self.data[valid_mask].reset_index(drop=True)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        uniprot = row['uniprot']
        sm = row['compound_iso_smiles']
        seq = row['target_sequence']
        label = row['affinity']
        
        target_graph = self.target_graph_dict[uniprot]
        drug_graph = self.drug_graph_dict[sm]
        
        smiles = self.smiles_emb.get(sm, torch.zeros(100, dtype=torch.long))
        protein = self.target_emb.get(seq, torch.zeros(1000, dtype=torch.long))
        smiles_lengths = self.smiles_len.get(sm, 1)
        protein_lengths = self.target_len.get(seq, 1)
        
        data = DATA(
            y=torch.FloatTensor([label]),
            sm=sm,
            target=row['target_key'],
            smiles=smiles,
            protein=protein,
            smiles_lengths=smiles_lengths,
            protein_lengths=protein_lengths,
            seq=seq
        )
        
        return data, drug_graph, target_graph
