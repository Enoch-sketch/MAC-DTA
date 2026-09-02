"""
Cliff-aware ranking loss components:
- CliffPairDataset: 加载预构建的悬崖对
- CliffPairIterator: 无限循环迭代器，每次返回一组悬崖对
- cliff_ranking_loss: 计算 MarginRankingLoss
"""
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch_geometric.data import Data as DATA


class CliffPairDataset:
    """从 cliff_pairs.csv 和主数据集中构建悬崖对"""

    def __init__(self, cliff_csv_path, main_dataset, split='train'):
        cliff_df = pd.read_csv(cliff_csv_path)
        self.pairs = cliff_df[cliff_df['pair_split'] == split].reset_index(drop=True)
        self.main_dataset = main_dataset
        print(f"CliffPairDataset ({split}): {len(self.pairs)} pairs loaded")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        row = self.pairs.iloc[idx]
        wt_idx = int(row['wt_idx'])
        mut_idx = int(row['mut_idx'])

        wt_data, wt_drug, wt_prot = self.main_dataset[wt_idx]
        mut_data, mut_drug, mut_prot = self.main_dataset[mut_idx]

        sign = 1.0 if row['wt_affinity'] > row['mut_affinity'] else -1.0
        target_sign = torch.FloatTensor([sign])

        return (wt_data, wt_drug, wt_prot, mut_data, mut_drug, mut_prot, target_sign)


class CliffPairIterator:
    """无限循环的悬崖对迭代器，每次返回 batch_size 个对"""

    def __init__(self, cliff_dataset, batch_size=16):
        self.dataset = cliff_dataset
        self.batch_size = batch_size
        self.indices = np.arange(len(self.dataset))
        np.random.shuffle(self.indices)
        self.pos = 0

    def get_batch(self):
        """返回一组悬崖对的数据"""
        if self.pos + self.batch_size > len(self.indices):
            np.random.shuffle(self.indices)
            self.pos = 0

        batch_indices = self.indices[self.pos:self.pos + self.batch_size]
        self.pos += self.batch_size

        items = [self.dataset[i] for i in batch_indices]

        from torch_geometric.data import Batch
        wt_data_list = [item[0] for item in items]
        wt_drug_list = [item[1] for item in items]
        wt_prot_list = [item[2] for item in items]
        mut_data_list = [item[3] for item in items]
        mut_drug_list = [item[4] for item in items]
        mut_prot_list = [item[5] for item in items]
        signs = torch.cat([item[6] for item in items])

        wt_data = Batch.from_data_list(wt_data_list)
        wt_drug = Batch.from_data_list(wt_drug_list)
        wt_prot = Batch.from_data_list(wt_prot_list)
        mut_data = Batch.from_data_list(mut_data_list)
        mut_drug = Batch.from_data_list(mut_drug_list)
        mut_prot = Batch.from_data_list(mut_prot_list)

        return wt_data, wt_drug, wt_prot, mut_data, mut_drug, mut_prot, signs


def cliff_ranking_loss(model, cliff_batch, device, margin=0.5):
    """计算悬崖对的 MarginRankingLoss"""
    wt_data, wt_drug, wt_prot, mut_data, mut_drug, mut_prot, signs = cliff_batch

    wt_data = wt_data.to(device)
    wt_drug = wt_drug.to(device)
    wt_prot = wt_prot.to(device)
    mut_data = mut_data.to(device)
    mut_drug = mut_drug.to(device)
    mut_prot = mut_prot.to(device)
    signs = signs.to(device)

    pred_wt, _ = model(wt_data, wt_drug, wt_prot)
    pred_mut, _ = model(mut_data, mut_drug, mut_prot)

    loss_fn = nn.MarginRankingLoss(margin=margin)
    rank_loss = loss_fn(pred_wt.view(-1), pred_mut.view(-1), signs)

    return rank_loss
