"""
Cliff-specific evaluation metrics:
- cliff_rmse: 悬崖样本上的 RMSE
- cliff_hit_rate: 模型预测排序方向是否正确的比例
- cliff_delta_correlation: corr(pred_wt - pred_mut, label_wt - label_mut)
"""
import numpy as np
import pandas as pd
from scipy import stats
from math import sqrt


def evaluate_cliff(all_labels, all_preds, dataset_df, cliff_csv_path, split='test'):
    """
    计算 cliff 专项指标
    Args:
        all_labels: 全量标签 (numpy array, 按 dataset 顺序)
        all_preds: 全量预测 (numpy array, 按 dataset 顺序)
        dataset_df: 数据集 DataFrame (含 split 列)
        cliff_csv_path: cliff_pairs.csv 路径
        split: 评估哪个 split 的悬崖对
    """
    cliff_df = pd.read_csv(cliff_csv_path)
    pairs = cliff_df[cliff_df['pair_split'] == split]

    if len(pairs) == 0:
        print(f"No cliff pairs in {split} split")
        return {}

    split_indices = dataset_df[dataset_df['split'] == split].index.tolist()
    idx_to_pos = {idx: pos for pos, idx in enumerate(split_indices)}

    pred_wt_list, pred_mut_list = [], []
    label_wt_list, label_mut_list = [], []
    valid_pairs = 0

    for _, row in pairs.iterrows():
        wt_idx = int(row['wt_idx'])
        mut_idx = int(row['mut_idx'])

        if wt_idx not in idx_to_pos or mut_idx not in idx_to_pos:
            continue

        wt_pos = idx_to_pos[wt_idx]
        mut_pos = idx_to_pos[mut_idx]

        pred_wt_list.append(all_preds[wt_pos])
        pred_mut_list.append(all_preds[mut_pos])
        label_wt_list.append(all_labels[wt_pos])
        label_mut_list.append(all_labels[mut_pos])
        valid_pairs += 1

    if valid_pairs == 0:
        print(f"No valid cliff pairs found in {split}")
        return {}

    pred_wt = np.array(pred_wt_list)
    pred_mut = np.array(pred_mut_list)
    label_wt = np.array(label_wt_list)
    label_mut = np.array(label_mut_list)

    cliff_preds = np.concatenate([pred_wt, pred_mut])
    cliff_labels = np.concatenate([label_wt, label_mut])
    cliff_rmse = sqrt(((cliff_labels - cliff_preds) ** 2).mean())

    all_rmse = sqrt(((all_labels - all_preds) ** 2).mean())

    label_sign = np.sign(label_wt - label_mut)
    pred_sign = np.sign(pred_wt - pred_mut)
    hit_rate = (label_sign == pred_sign).mean()

    delta_label = label_wt - label_mut
    delta_pred = pred_wt - pred_mut
    if np.std(delta_pred) > 1e-8:
        delta_corr = np.corrcoef(delta_label, delta_pred)[0, 1]
        delta_spearman = stats.spearmanr(delta_label, delta_pred)[0]
    else:
        delta_corr = 0.0
        delta_spearman = 0.0

    results = {
        'split': split,
        'n_cliff_pairs': valid_pairs,
        'general_rmse': all_rmse,
        'cliff_rmse': cliff_rmse,
        'rmse_gap': cliff_rmse - all_rmse,
        'cliff_hit_rate': hit_rate,
        'cliff_delta_pearson': delta_corr,
        'cliff_delta_spearman': delta_spearman,
    }

    print(f"\n{'='*50}")
    print(f"Cliff Evaluation ({split}, {valid_pairs} pairs)")
    print(f"{'='*50}")
    print(f"  General RMSE:        {all_rmse:.4f}")
    print(f"  Cliff RMSE:          {cliff_rmse:.4f}  (gap: +{cliff_rmse - all_rmse:.4f})")
    print(f"  Cliff Hit Rate:      {hit_rate:.4f}  ({int(hit_rate*valid_pairs)}/{valid_pairs})")
    print(f"  Cliff ΔPred Pearson: {delta_corr:.4f}")
    print(f"  Cliff ΔPred Spearman:{delta_spearman:.4f}")
    print(f"{'='*50}\n")

    return results
