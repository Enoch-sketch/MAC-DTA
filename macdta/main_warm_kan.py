"""MAC-DTA training, validation, and test entry point.

The model combines the PMHGT multimodal backbone with mutation indicators,
a mutation-centred local graph branch, pairwise ranking supervision, KD Guard,
and a KAN prediction head. Data paths are resolved from the repository root so
the public package does not depend on machine-specific absolute paths.
"""
import sys, os, math

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
DATA_ROOT = os.path.abspath(
    os.environ.get('MAC_DTA_DATA_ROOT', os.path.join(REPO_ROOT, 'data', 'davis'))
)
sys.path.insert(0, DATA_ROOT)

import numpy as np
import pandas as pd
import rdkit
import rdkit.Chem as Chem
import networkx as nx
import torch
torch.cuda.empty_cache()

from tqdm import tqdm
from torch_geometric.loader import DataLoader
from build_vocab import WordVocab
import utils as base_utils
from utils import *

from dataset import DTADataset
from model import *
from torch import nn as nn
import torch.nn.functional as F
from utils import seed_torch
from torch_geometric.utils import to_dense_batch
from argparse import ArgumentParser
from cliff_loss import CliffPairDataset, CliffPairIterator, cliff_ranking_loss
from cliff_eval import evaluate_cliff

# Local imports (from this directory)
sys.path.insert(0, SCRIPT_DIR)
from mut_local import MutLocalBranch, MutCondGating, build_local_mask, DeltaHead, MutLocalDeltaBranch

#############################################################################

parser = ArgumentParser(description="MAC-DTA training and evaluation")
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--epochs', type=int, default=180)
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--cuda_device', type=int, default=0,
                    help='Torch CUDA device index for this process, e.g. 0 when '
                         'CUDA_VISIBLE_DEVICES is set to a single physical GPU, '
                         'or the physical GPU index when CUDA_VISIBLE_DEVICES is unset.')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--num_workers', type=int, default=4,
                    help='Training DataLoader workers. Set to 0 in restricted '
                         'environments that do not permit multiprocessing IPC. '
                         'Validation/test use at most half this value.')
parser.add_argument('--dataset', type=str, default='davis', choices=['davis', 'kiba'])
parser.add_argument('--exp_name', type=str, default='mutlocal_v1')
parser.add_argument('--no_cliff_loss', action='store_true', default=False)
parser.add_argument('--cliff_lambda', type=float, default=0.1)
parser.add_argument('--cliff_margin', type=float, default=0.84)
parser.add_argument('--cliff_batch_size', type=int, default=16)
parser.add_argument('--cliff_warmup_epochs', type=int, default=3)
parser.add_argument('--cliff_rampup_epochs', type=int, default=20,
                    help='Epochs to linearly ramp up cliff_lambda after warmup (0=instant full lambda)')
parser.add_argument('--cliff_freq', type=int, default=10)
parser.add_argument('--patience', type=int, default=30)
parser.add_argument('--resume', type=str, default=None)
parser.add_argument('--eval_only', action='store_true', default=False)
parser.add_argument('--save_test_predictions', action='store_true', default=False,
                    help='Save per-sample test predictions to CSV for downstream analysis')
# mutation local branch
parser.add_argument('--no_local_branch', action='store_true', default=False,
                    help='Disable mutation-center local branch AND conditioned gating (ablation)')
parser.add_argument('--local_k_hops', type=int, default=2,
                    help='Neighborhood radius for local branch BFS (default 2)')
parser.add_argument('--no_seq_mut_inject', action='store_true', default=False,
                    help='Disable mutation flag injection into sequence branch (ablation: '
                         'keep only graph-branch mutation signal, reduces dual-path disruption)')
parser.add_argument('--no_graph_mut_inject', action='store_true', default=False,
                    help='Disable direct mutation flag injection into graph node features '
                         '(x_mut = 0). Combined with --no_local_branch and '
                         '--no_seq_mut_inject this gives the cleanest PMHGT-equivalent '
                         'baseline for ablation (C-group experiments).')
parser.add_argument('--use_delta_head', action='store_true', default=False,
                         help='Enable DeltaHead: predicts Δaffinity on cliff pairs; '
                              'adds delta regression loss on top of main MSE.')
parser.add_argument('--delta_lambda', type=float, default=0.5,
                         help='Weight for the delta-head loss term.')
parser.add_argument('--residual_gate_scale', type=float, default=0.1,
                    help='Scale factor for residual mutation corrections (default 0.1, '
                         'set higher to allow larger mutation-induced adjustments)')
# ── teacher distillation ──────────────────────────────────────────────────────
parser.add_argument('--teacher_pred_file', type=str, default=None,
                    help='Path to .npy file of pre-computed frozen-teacher predictions '
                         'for ALL dataset samples, indexed by global dataset index. '
                         'Generate with: python gen_teacher_preds.py --split all '
                         '--output teacher_preds_all.npy. NaN entries are automatically '
                         'skipped in the KD loss. Used for knowledge distillation.')
parser.add_argument('--kd_lambda', type=float, default=0.3,
                    help='Weight for teacher-distillation loss term (default 0.3). '
                         'Only active when --teacher_pred_file is set.')
parser.add_argument('--kd_noncliff_only', action='store_true', default=False,
                    help='Apply KD loss only on non-cliff samples (those not appearing in '
                         'cliff_pairs.csv). Cliff samples are allowed to deviate from teacher.')
parser.add_argument('--train_pair_csv', type=str, default=None,
                    help='Optional pair-supervision CSV for training. '
                         'Defaults to DATA_ROOT/cliff_pairs.csv. Can point to an augmented '
                         'pair file containing near-cliff or mut-mut relations.')
parser.add_argument('--eval_pair_csv', type=str, default=None,
                    help='Optional pair CSV used only for test-time cliff evaluation. '
                         'Defaults to DATA_ROOT/cliff_pairs.csv so evaluation protocol stays fixed.')
parser.add_argument('--pair_sampling', type=str, default='uniform',
                    choices=['uniform', 'gene_balanced', 'hybrid'],
                    help='Sampling strategy for cliff pair batches. '
                         '"gene_balanced" samples genes uniformly before drawing pairs; '
                         '"hybrid" mixes uniform and gene-balanced sampling.')
parser.add_argument('--gene_balance_ratio', type=float, default=0.5,
                    help='Only used when pair_sampling=hybrid. Probability of drawing '
                         'a pair from gene-balanced sampling instead of uniform sampling.')
parser.add_argument('--pair_reweight_alpha', type=float, default=0.0,
                    help='Inverse-frequency reweighting strength for cliff/delta pair losses. '
                         '0=uniform weights, 0.5=inverse-sqrt frequency, 1.0=inverse frequency.')
parser.add_argument('--pair_weight_cap', type=float, default=2.0,
                    help='Maximum per-pair weight after normalization. '
                         'Used to avoid over-amplifying ultra-rare genes.')
parser.add_argument('--pair_density_weighting', type=str, default='none',
                    choices=['none', 'abs_delta', 'signed_delta'],
                    help='Density-aware weighting for continuous cliff magnitude. '
                         '"abs_delta" reweights rare |delta_affinity| ranges; '
                         '"signed_delta" reweights rare signed delta ranges.')
parser.add_argument('--pair_density_alpha', type=float, default=0.0,
                    help='Strength of density-aware weighting on pair loss. '
                         '0=disabled; larger values emphasize rarer delta ranges.')
parser.add_argument('--pair_density_bins', type=int, default=6,
                    help='Number of histogram bins used for density-aware pair weighting.')
parser.add_argument('--output_dir', type=str, default='./KAN_TRY',
                    help='Directory for checkpoints, logs, predictions, and results.')
parser.add_argument('--ckpt_base_name', type=str, default=None,
                    help='Optional checkpoint base filename. If omitted, uses '
                         'checkpoint_<dataset>_seed<seed>_<exp_name>.')
parser.add_argument('--force_raw', action='store_true', default=False,
                    help='Force use of raw (unfixed) CSV even when fixed CSV exists.')
parser.add_argument('--csv_path', type=str, default=None,
                    help='Optional explicit CSV path with split column.')
parser.add_argument('--trainable_modules', type=str, default='all',
                    help='Comma-separated module prefixes to train, e.g. '
                         '"mut_local,mut_gating,fusion_graph_seq,kan_head". '
                         'Use "all" to train full model.')
parser.add_argument('--max_train_batches', type=int, default=-1,
                    help='If >0, limit number of train mini-batches per epoch '
                         '(useful for support-set adaptation runs).')

args = parser.parse_args()
args.use_cliff_loss = not args.no_cliff_loss
args.use_local_branch = not args.no_local_branch
args.seq_mut_inject = not args.no_seq_mut_inject
args.graph_mut_inject = not args.no_graph_mut_inject

if args.cuda_device < 0:
    device = torch.device('cpu')
elif torch.cuda.is_available():
    device = torch.device(f'cuda:{args.cuda_device}')
else:
    raise RuntimeError('CUDA is not available but --cuda_device >= 0 was given.')

# The legacy helper module used a hard-coded cuda:0 device in `predicting`.
# Keep its global device synchronized with the CLI-selected training device so
# CPU runs and nonzero CUDA devices evaluate on the same device as training.
base_utils.device = device

LR = args.lr
NUM_EPOCHS = args.epochs
seed = args.seed
batch_size = args.batch_size
dataset_name = args.dataset

seed_torch(seed)
#############################################################################
_vis = os.environ.get('CUDA_VISIBLE_DEVICES', '(unset)')
print(f"Seed: {seed}, Exp: {args.exp_name}, device={device}, "
      f"cuda_device_arg={args.cuda_device}, CUDA_VISIBLE_DEVICES={_vis}, "
      f"Cliff: {args.use_cliff_loss}, Lambda: {args.cliff_lambda}, Margin: {args.cliff_margin}, "
      f"LocalBranch+Gating: {args.use_local_branch}, K-hops: {args.local_k_hops}, "
      f"Patience: {args.patience}, Resume: {args.resume}")


# ── DTADatasetLocal: wraps DTADataset, adds local_mask ────────
class DTADatasetLocal(DTADataset):
    """Extends DTADataset by computing a k-hop local_mask per protein graph."""

    def __init__(self, *args, k_hops=2, **kwargs):
        self._k_hops = k_hops
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx):
        data, drug_graph, target_graph = super().__getitem__(idx)

        num_nodes = target_graph.x.size(0)
        info = self.mutation_info.get(data.target)
        if info and info['is_mutant'] and info['in_window']:
            mut_idx = info['mutated_idx']
            local_mask = build_local_mask(
                target_graph.edge_index, mut_idx, num_nodes, self._k_hops
            )
        else:
            local_mask = torch.zeros(num_nodes, dtype=torch.bool)
        target_graph.local_mask = local_mask

        # sample_idx is already set in DTADataset.__getitem__ (base class),
        # so it is available regardless of whether DTADatasetLocal or DTADataset
        # is in use.  No override needed here.

        return data, drug_graph, target_graph


class PairBatchIterator:
    """Iterator for cliff pairs with optional gene-balanced sampling and reweighting."""

    def __init__(self, cliff_dataset, batch_size=16, sampling='uniform',
                 reweight_alpha=0.0, seed=123, gene_balance_ratio=0.5,
                 weight_cap=2.0, density_weighting='none',
                 density_alpha=0.0, density_bins=6):
        self.dataset = cliff_dataset
        self.batch_size = batch_size
        self.sampling = sampling
        self.reweight_alpha = reweight_alpha
        self.gene_balance_ratio = gene_balance_ratio
        self.weight_cap = weight_cap
        self.density_weighting = density_weighting
        self.density_alpha = density_alpha
        self.density_bins = density_bins
        self.rng = np.random.default_rng(seed)
        self.indices = np.arange(len(self.dataset))
        self.pos = 0

        pairs_df = self.dataset.pairs.copy()
        if 'base_gene' in pairs_df.columns:
            pair_genes = pairs_df['base_gene'].fillna('UNK').astype(str)
        else:
            pair_genes = pd.Series(['ALL'] * len(pairs_df))
        self.pair_genes = pair_genes.to_numpy()
        self.genes = sorted(pd.unique(self.pair_genes).tolist())
        self.gene_to_indices = {
            gene: np.flatnonzero(self.pair_genes == gene)
            for gene in self.genes
        }
        gene_counts = pd.Series(self.pair_genes).value_counts().to_dict()
        if reweight_alpha > 0:
            gene_weights = np.array(
                [(1.0 / (gene_counts[g] ** reweight_alpha)) for g in self.pair_genes],
                dtype=np.float32
            )
            gene_weights /= gene_weights.mean()
        else:
            gene_weights = np.ones(len(self.pair_genes), dtype=np.float32)

        if density_alpha > 0 and density_weighting != 'none' and 'delta_affinity' in pairs_df.columns:
            if density_weighting == 'abs_delta':
                density_values = pairs_df['delta_affinity'].abs().astype(float).to_numpy()
            else:
                density_values = pairs_df['delta_affinity'].astype(float).to_numpy()
            hist_counts, bin_edges = np.histogram(density_values, bins=max(2, density_bins))
            hist_counts = np.maximum(hist_counts, 1)
            bin_ids = np.clip(np.digitize(density_values, bin_edges[1:-1], right=False), 0, len(hist_counts) - 1)
            density_weights = np.array(
                [(1.0 / (hist_counts[b] ** density_alpha)) for b in bin_ids],
                dtype=np.float32
            )
            density_weights /= density_weights.mean()
        else:
            density_weights = np.ones(len(self.pair_genes), dtype=np.float32)

        raw_weights = gene_weights * density_weights
        if 'pair_base_weight' in pairs_df.columns:
            base_weights = pairs_df['pair_base_weight'].astype(float).to_numpy(dtype=np.float32)
            if base_weights.mean() > 0:
                base_weights = base_weights / base_weights.mean()
                raw_weights = raw_weights * base_weights
        raw_weights /= raw_weights.mean()
        if weight_cap > 0:
            raw_weights = np.clip(raw_weights, a_min=None, a_max=weight_cap)
            raw_weights /= raw_weights.mean()
        self.pair_weights = raw_weights

        self.rng.shuffle(self.indices)

    def _sample_gene_balanced(self):
        if len(self.genes) <= 1:
            return None
        chosen_genes = self.rng.choice(self.genes, size=self.batch_size, replace=True)
        return np.array(
            [self.rng.choice(self.gene_to_indices[gene]) for gene in chosen_genes],
            dtype=np.int64
        )

    def _sample_uniform(self):
        if self.pos + self.batch_size > len(self.indices):
            self.rng.shuffle(self.indices)
            self.pos = 0
        batch_indices = self.indices[self.pos:self.pos + self.batch_size]
        self.pos += self.batch_size
        return batch_indices.astype(np.int64)

    def _sample_indices(self):
        if self.sampling == 'gene_balanced':
            batch_indices = self._sample_gene_balanced()
            if batch_indices is not None:
                return batch_indices

        if self.sampling == 'hybrid':
            if self.rng.random() < self.gene_balance_ratio:
                batch_indices = self._sample_gene_balanced()
                if batch_indices is not None:
                    return batch_indices

        return self._sample_uniform()

    def get_batch(self):
        batch_indices = self._sample_indices()
        items = [self.dataset[int(i)] for i in batch_indices]

        from torch_geometric.data import Batch
        wt_data_list = [item[0] for item in items]
        wt_drug_list = [item[1] for item in items]
        wt_prot_list = [item[2] for item in items]
        mut_data_list = [item[3] for item in items]
        mut_drug_list = [item[4] for item in items]
        mut_prot_list = [item[5] for item in items]
        signs = torch.cat([item[6] for item in items])
        pair_weights = torch.tensor(self.pair_weights[batch_indices], dtype=torch.float32)

        wt_data = Batch.from_data_list(wt_data_list)
        wt_drug = Batch.from_data_list(wt_drug_list)
        wt_prot = Batch.from_data_list(wt_prot_list)
        mut_data = Batch.from_data_list(mut_data_list)
        mut_drug = Batch.from_data_list(mut_drug_list)
        mut_prot = Batch.from_data_list(mut_prot_list)
        return wt_data, wt_drug, wt_prot, mut_data, mut_drug, mut_prot, signs, pair_weights


def weighted_cliff_ranking_loss(model, cliff_batch, device, margin=0.5):
    """MarginRankingLoss with optional inverse-frequency pair weights."""
    wt_data, wt_drug, wt_prot, mut_data, mut_drug, mut_prot, signs, pair_weights = cliff_batch

    wt_data = wt_data.to(device)
    wt_drug = wt_drug.to(device)
    wt_prot = wt_prot.to(device)
    mut_data = mut_data.to(device)
    mut_drug = mut_drug.to(device)
    mut_prot = mut_prot.to(device)
    signs = signs.to(device)
    pair_weights = pair_weights.to(device)

    try:
        pred_wt, _ = model(wt_data, wt_drug, wt_prot)
        pred_mut, _ = model(mut_data, mut_drug, mut_prot)
    except Exception as e:
        print(
            "ERROR[cliff_forward]: "
            f"{type(e).__name__}: {e}; "
            f"wt_data_batch={getattr(wt_data, 'num_graphs', 'NA')}, "
            f"mut_data_batch={getattr(mut_data, 'num_graphs', 'NA')}, "
            f"wt_drug_x_shape={tuple(wt_drug.x.shape) if hasattr(wt_drug, 'x') else 'NA'}, "
            f"wt_prot_x_shape={tuple(wt_prot.x.shape) if hasattr(wt_prot, 'x') else 'NA'}, "
            f"mut_drug_x_shape={tuple(mut_drug.x.shape) if hasattr(mut_drug, 'x') else 'NA'}, "
            f"mut_prot_x_shape={tuple(mut_prot.x.shape) if hasattr(mut_prot, 'x') else 'NA'}"
        )
        raise

    if pred_wt.ndim == 1:
        print(f"WARN[cliff_pred_shape]: pred_wt ndim=1, shape={tuple(pred_wt.shape)}; unsqueeze to [B,1].")
        pred_wt = pred_wt.unsqueeze(-1)
    if pred_mut.ndim == 1:
        print(f"WARN[cliff_pred_shape]: pred_mut ndim=1, shape={tuple(pred_mut.shape)}; unsqueeze to [B,1].")
        pred_mut = pred_mut.unsqueeze(-1)

    pair_losses = F.margin_ranking_loss(
        pred_wt.view(-1), pred_mut.view(-1), signs,
        margin=margin, reduction='none'
    )
    return (pair_losses * pair_weights).mean()


# ── KAN prediction head (replaces final nn.Linear) ───────────


class KANLinear(nn.Module):
    """
    Single KAN layer with B-spline activations on each edge.
    Based on: KAN: Kolmogorov-Arnold Networks (Liu et al., ICLR 2025 Oral).
    Implemented from scratch; no external KAN library required.

    Architecture:
        output = scale_base * SiLU(x) @ base_weight.T
               + scale_spline * einsum('bin,oin->bo', B_splines(x), spline_weight)

    The base_activation residual (SiLU branch) provides a stable gradient path
    from the start of training, while the spline branch learns fine-grained
    nonlinear corrections as training progresses.
    """

    def __init__(self, in_features: int, out_features: int,
                 grid_size: int = 5, spline_order: int = 3,
                 scale_noise: float = 0.1, scale_base: float = 1.0,
                 scale_spline: float = 1.0,
                 grid_range: tuple = (-1.0, 1.0)):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.base_activation = nn.SiLU()

        # Extended grid: grid_size + 2*spline_order + 1 knots
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            torch.arange(-spline_order, grid_size + spline_order + 1).float() * h
            + grid_range[0]
        )
        # grid: [in_features, grid_size + 2*spline_order + 1]
        self.register_buffer('grid',
                             grid.unsqueeze(0).expand(in_features, -1).contiguous())
        self._grid_min = float(grid_range[0])
        self._grid_max = float(grid_range[1])

        n_basis = grid_size + spline_order  # number of basis functions per edge
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.empty(out_features, in_features, n_basis))

        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        # Small noise init for spline weights: near-zero at start so the
        # linear (SiLU) branch dominates early training, exactly like a regular
        # Linear layer, and spline corrections grow only as needed.
        nn.init.normal_(self.spline_weight, 0.0,
                        scale_noise / math.sqrt((grid_size + 1) * in_features))

    def _b_splines(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute B-spline basis values via de Boor recurrence.
        x:       [B, in_features]  — already clamped to grid range
        returns: [B, in_features, grid_size + spline_order]
        """
        x = x.unsqueeze(-1)           # [B, in, 1]
        grid = self.grid.unsqueeze(0)  # [1, in, num_knots]

        # Order-0: piecewise constant indicator functions
        bases = ((x >= grid[:, :, :-1]) & (x < grid[:, :, 1:])).float()

        # de Boor recurrence for orders 1 … spline_order
        for k in range(1, self.spline_order + 1):
            denom_l = grid[:, :, k:-1] - grid[:, :, :-(k + 1)]
            denom_r = grid[:, :, k + 1:] - grid[:, :, 1:-k]
            left  = (x - grid[:, :, :-(k + 1)]) / (denom_l + 1e-8) * bases[:, :, :-1]
            right = (grid[:, :, k + 1:] - x)    / (denom_r + 1e-8) * bases[:, :, 1:]
            bases = left + right

        return bases.contiguous()  # [B, in, n_basis]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, in_features]  →  [B, out_features]"""
        # Clamp to grid range to prevent extrapolation blow-up
        x = x.clamp(self._grid_min, self._grid_max - 1e-5)

        # SiLU residual branch (always-active linear-like path)
        base_out = F.linear(self.base_activation(x),
                            self.scale_base * self.base_weight)

        # Spline branch: learnable nonlinear activations on each edge
        splines = self._b_splines(x)            # [B, in, n_basis]
        spline_out = self.scale_spline * torch.einsum(
            'bin,oin->bo', splines, self.spline_weight)  # [B, out]

        return base_out + spline_out


class KANHead(nn.Module):
    """
    KAN prediction head replacing the final nn.Linear(hidden_dim*2, 1).

    Input:  fused representation [B, in_features] — values in [0, ∞) after ReLU.
    Output: scalar affinity prediction [B].

    Pipeline:
        fused → LayerNorm → clamp(−3, 3) → KANLinear(in, 1) → squeeze

    LayerNorm centers and scales the post-ReLU representation so that the
    B-spline grid (range [−3, 3]) is well-utilized.  The clamp matches the
    grid range and prevents order-0 basis functions from returning all-zeros.
    Small-noise init of spline weights ensures the head starts as a near-linear
    predictor (equivalent to the replaced nn.Linear) and adapts gradually.
    """

    def __init__(self, in_features: int,
                 grid_size: int = 5, spline_order: int = 3):
        super().__init__()
        self.input_norm = nn.LayerNorm(in_features)
        self.kan = KANLinear(in_features, 1,
                             grid_size=grid_size,
                             spline_order=spline_order,
                             scale_noise=0.1,
                             scale_base=1.0,
                             scale_spline=1.0,
                             grid_range=(-3.0, 3.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, in_features]  →  [B]"""
        x = self.input_norm(x)
        x = x.clamp(-3.0, 3.0 - 1e-5)
        return self.kan(x).squeeze(-1)


# ── PMHGT with local branch ──────────────────────────────────
class PMHGT(nn.Module):
    def __init__(self, embedding_dim, lstm_dim, hidden_dim, dropout_rate,
                 alpha, n_heads, bilstm_layers=2, protein_vocab=26,
                 smile_vocab=45, theta=0.5, use_local_branch=True,
                 seq_mut_inject=True, graph_mut_inject=True,
                 residual_gate_scale=0.1, use_delta_head=False):
        super(PMHGT, self).__init__()
        self.is_bidirectional = True
        self.theta = theta
        self.dropout = nn.Dropout(dropout_rate)
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        self.bilstm_layers = bilstm_layers
        self.n_heads = n_heads
        self.MGNN = GINConvNet(num_features_xd=512, n_output=hidden_dim)
        self.seq_mut_inject = seq_mut_inject
        self.graph_mut_inject = graph_mut_inject
        self.residual_gate_scale = residual_gate_scale

        # SMILES
        self.smiles_vocab = smile_vocab
        self.smiles_embed = nn.Embedding(smile_vocab + 1, 256, padding_idx=0)
        self.is_bidirectional = True
        self.smiles_input_fc = nn.Linear(256, lstm_dim)
        self.smiles_lstm = nn.LSTM(lstm_dim, lstm_dim, self.bilstm_layers, batch_first=True,
                                   bidirectional=self.is_bidirectional, dropout=dropout_rate)
        self.ln1 = torch.nn.LayerNorm(lstm_dim * 2)
        self.enhance1 = SpatialGroupEnhance_for_1D(groups=20)
        self.out_attentions3 = LinkAttention(hidden_dim, n_heads)

        # protein
        self.protein_vocab = protein_vocab
        self.protein_embed = nn.Embedding(protein_vocab + 1, embedding_dim, padding_idx=0)
        self.mut_seq_embed = nn.Embedding(2, embedding_dim, padding_idx=0)
        nn.init.zeros_(self.mut_seq_embed.weight[0])
        nn.init.normal_(self.mut_seq_embed.weight[1], std=0.02)
        self.is_bidirectional = True
        self.protein_input_fc = nn.Linear(embedding_dim, lstm_dim)
        self.protein_lstm = nn.LSTM(lstm_dim, lstm_dim, self.bilstm_layers, batch_first=True,
                                    bidirectional=self.is_bidirectional, dropout=dropout_rate)
        self.ln2 = torch.nn.LayerNorm(lstm_dim * 2)
        self.enhance2 = SpatialGroupEnhance_for_1D(groups=200)
        self.protein_head_fc = nn.Linear(lstm_dim * n_heads, lstm_dim)
        self.protein_out_fc = nn.Linear(2 * lstm_dim, hidden_dim)
        self.out_attentions2 = LinkAttention(hidden_dim, n_heads)

        # link
        self.out_attentions = LinkAttention(hidden_dim, n_heads)
        self.out_fc1 = nn.Linear(hidden_dim * 3, 256 * 8)
        self.out_fc2 = nn.Linear(256 * 8, hidden_dim * 2)

        self.kan_head = KANHead(hidden_dim * 2, grid_size=5, spline_order=3)
        self.layer_norm = nn.LayerNorm(lstm_dim * 2)

        # cross attention
        self.d_gca = GuidedCrossAttention(embed_dim=2 * lstm_dim, num_heads=1)
        self.p_gca = GuidedCrossAttention(embed_dim=2 * lstm_dim, num_heads=1)

        self.one_hot_embed = nn.Embedding(21, 96)
        self.proj_aa = nn.Linear(96, 512)
        self.proj_esm = nn.Linear(1280, 512)

        self.mut_embed = nn.Embedding(2, 512)
        nn.init.zeros_(self.mut_embed.weight[0])
        nn.init.normal_(self.mut_embed.weight[1], std=0.03)

        self.proj_uni = nn.Linear(512, 512)
        self.gcn_d = GraphCNN_D(pooling='MTP')
        self.gcn_p = GraphCNN_P(pooling='MTP')

        self.d_stru_gca = GuidedCrossAttention(embed_dim=2 * lstm_dim, num_heads=1)
        self.p_stru_gca = GuidedCrossAttention(embed_dim=2 * lstm_dim, num_heads=1)

        self.can_layer_emb = CAN_Layer(hidden_dim=256, num_heads_d=4, num_heads_p=4,
                                       group_size_d=1, group_size_p=1, agg_mode='mean_all_tok')

        self.lin_d1 = nn.Linear(512, 256)
        self.act_d = nn.GELU()
        self.d_norm = nn.LayerNorm(256)
        self.lin_d2 = nn.Linear(256, 256)

        self.p_adaptor_wo_skip_connect = FeedForwardLayer(1280, 512)
        self.lin_p1 = nn.Linear(1280, 512)
        self.act_p = nn.GELU()
        self.p_norm = nn.LayerNorm(512)
        self.lin_p2 = nn.Linear(512, 256)

        # ── mutation-center local branch + residual correction ────────────────
        self.use_local_branch = use_local_branch
        self.use_delta_head = use_delta_head
        # fusion_dim stays at hidden_dim * 4 = 1024 regardless of whether the
        # local branch is active.
        fusion_dim = hidden_dim * 4
        if use_local_branch:
            self.mut_local = MutLocalBranch(
                in_dim=512, hidden_dim=hidden_dim, out_dim=hidden_dim
            )
            # MutCondGating now produces residual deltas (not multiplicative
            # gates). Zero-init ensures the module starts as identity.
            self.mut_gating = MutCondGating(
                local_dim=hidden_dim, hidden_dim=hidden_dim,
                scale=residual_gate_scale
            )

        # ── DeltaHead: predicts Δaffinity from fused representation diff ─────
        # Uses MutLocalDeltaBranch when local_branch is on, else falls back to
        # fused representation arithmetic (diff + hadamard only).
        if use_delta_head:
            self.mut_local_delta = MutLocalDeltaBranch(
                in_dim=512, hidden_dim=hidden_dim, out_dim=hidden_dim
            )
            self.delta_head = DeltaHead(
                delta_dim=hidden_dim,   # local_delta_emb dim
                hidden_dim=hidden_dim * 2,  # fused repr dim (out_fc1 output)
                dropout=0.2
            )

        self.fusion_graph_seq = nn.Linear(fusion_dim, hidden_dim * 2)
        self.pwff_1 = nn.Linear(fusion_dim, fusion_dim)
        self.pwff_2 = nn.Linear(fusion_dim, fusion_dim)

    def forward(self, data, stu_d, stu_p, reset=False):
        batchsize = len(data.sm)
        smiles = data.smiles.to(device).view(batchsize, -1)
        protein = data.protein.to(device).view(batchsize, -1)
        smiles_lengths = data.smiles_lengths
        protein_lengths = data.protein_lengths

        # ── sequence branches ──
        smiles_emb = self.smiles_embed(smiles)
        smiles_emb = self.smiles_input_fc(smiles_emb)
        smiles_emb = self.enhance1(smiles_emb)

        protein_emb = self.protein_embed(protein)
        # Sequence-side mutation injection (controlled by --no_seq_mut_inject).
        # When disabled: mutation signal enters ONLY through the graph branch
        # (x_mut node embedding + MutLocalBranch residual correction).
        # This avoids the LSTM temporal spreading of mutation flags into every
        # position of protein_out, which can overwhelm the backbone's learned
        # sequence representation on non-mutant samples.
        if self.seq_mut_inject and hasattr(data, 'mut_flag_seq'):
            mut_flag_seq = data.mut_flag_seq.to(device).view(batchsize, -1)
            protein_emb = protein_emb + self.mut_seq_embed(mut_flag_seq)
        protein_emb = self.protein_input_fc(protein_emb)
        protein_emb = self.enhance2(protein_emb)

        smiles_emb, _ = self.smiles_lstm(smiles_emb)
        smiles_emb = self.ln1(smiles_emb)
        protein_emb, _ = self.protein_lstm(protein_emb)
        protein_emb = self.ln2(protein_emb)

        smiles_mask = self.generate_masks(smiles_emb, smiles_lengths, self.n_heads)
        protein_mask = self.generate_masks(protein_emb, protein_lengths, self.n_heads)

        prot, drug, mask_prot_grouped, mask_drug_grouped, _, _ = \
            self.can_layer_emb(protein_emb, smiles_emb, protein_mask, smiles_mask)
        mask_prot_grouped = mask_prot_grouped.float().unsqueeze(1).expand(-1, 8, -1)
        mask_drug_grouped = mask_drug_grouped.float().unsqueeze(1).expand(-1, 8, -1)

        smiles_out, _ = self.out_attentions3(drug, mask_drug_grouped)
        protein_out, _ = self.out_attentions2(prot, mask_prot_grouped)

        # ── graph branches ──
        x_aa = self.proj_aa(self.one_hot_embed(stu_p.native_x.long()))
        stu_p.x = stu_p.x.to(torch.float32)
        x_esm = self.proj_esm(stu_p.x)
        # --no_graph_mut_inject: set x_mut to 0 so graph nodes receive no direct
        # mutation signal.  Combined with --no_local_branch and --no_seq_mut_inject
        # this gives the cleanest "PMHGT backbone" ablation baseline (C-group).
        if self.graph_mut_inject:
            x_mut = self.mut_embed(stu_p.mut_flag.long())
        else:
            x_mut = torch.zeros(x_aa.size(0), x_aa.size(-1), device=x_aa.device)
        x_prot = F.relu(x_aa + x_esm + x_mut)

        gcn_n_featp, gcn_g_featp = self.gcn_p(x_prot, stu_p)

        # ── mutation-conditioned residual correction ──────────────────────────
        # Design principles:
        #
        # 1. Identity-preserving multiplicative gate:
        #    protein_out  = protein_out  * (1 + delta_seq)
        #    gcn_g_featp  = gcn_g_featp  * (1 + delta_graph)
        #    When delta=0 (zero-init of fc2), output == input strictly.
        #    Using (1+delta) rather than pure additive (x+delta) preserves the
        #    "per-dimension scale modulation" semantic and is robust to the
        #    magnitude of protein_out.
        #
        # 2. WT masking — the most critical fix:
        #    For WT samples local_mask is all-False; MutLocalBranch returns the
        #    learnable no_mut_token.  Under the old design, *every* WT sample
        #    was gated by this random token, corrupting backbone representations
        #    for the vast majority of the training data.
        #    Fix: compute graph_has_local (True only for samples that have ≥1
        #    node in their k-hop mutation neighbourhood), and apply delta only
        #    to those samples.  WT samples get strict identity.
        #
        # 3. fc2 zero-init (MutCondGating): already applied in mut_local.py.
        if self.use_local_branch:
            if hasattr(stu_p, 'local_mask'):
                local_emb = self.mut_local(
                    x_prot, stu_p.edge_index, stu_p.local_mask,
                    stu_p.batch, batchsize
                )
                # Determine which graphs in this batch actually have a mutant
                # k-hop neighbourhood (local_mask has at least one True node).
                graph_has_local = torch.zeros(batchsize, dtype=torch.bool,
                                              device=x_prot.device)
                if stu_p.local_mask.any():
                    mutant_graph_ids = stu_p.batch[stu_p.local_mask].unique()
                    graph_has_local[mutant_graph_ids] = True
            else:
                local_emb = self.mut_local.no_mut_token.unsqueeze(0).expand(batchsize, -1)
                graph_has_local = torch.zeros(batchsize, dtype=torch.bool,
                                              device=x_prot.device)

            delta_seq, delta_graph = self.mut_gating(local_emb)

            # Apply delta only to mutant samples; WT walks strict identity.
            # mask: [B, 1] float, 1.0 for mutant, 0.0 for WT.
            mut_mask = graph_has_local.float().unsqueeze(-1)  # [B, 1]
            masked_delta_seq   = delta_seq   * mut_mask
            masked_delta_graph = delta_graph * mut_mask

            protein_out = protein_out * (1.0 + masked_delta_seq)
            gcn_g_featp = gcn_g_featp * (1.0 + masked_delta_graph)

        xd = self.proj_uni(stu_d.x.float())
        gcn_n_featd, gcn_g_featd = self.gcn_d(xd, stu_d)

        # ── fusion (dimension unchanged: hidden_dim * 4 = 1024) ──────────────
        joint_emb = torch.cat([protein_out, smiles_out], dim=1)
        joint_stu = torch.cat([gcn_g_featp, gcn_g_featd], dim=-1)
        out = torch.cat([joint_emb, joint_stu], dim=-1)

        pwff = self.pwff_1(out)
        pwff = nn.ReLU()(pwff)
        pwff = self.dropout(pwff)
        pwff = self.pwff_2(pwff)
        out = pwff + out

        fused = self.dropout(self.relu(self.fusion_graph_seq(out)))
        out = self.kan_head(fused)

        return out, fused  # fused: [B, hidden_dim*2] for DeltaHead use

    def forward_pair(self, data_wt, drug_wt, prot_wt, data_mut, drug_mut, prot_mut):
        """
        Forward pass for a (WT, Mut) cliff pair.
        Returns:
            pred_wt  : [B] scalar affinity prediction for WT
            pred_mut : [B] scalar affinity prediction for Mut
            delta_pred : [B] predicted Δaffinity from DeltaHead
        """
        pred_wt,  fused_wt  = self.forward(data_wt,  drug_wt,  prot_wt)
        pred_mut, fused_mut = self.forward(data_mut, drug_mut, prot_mut)

        if self.use_delta_head:
            # Rebuild the protein graph node representation exactly as in forward().
            x_aa_mut = self.proj_aa(self.one_hot_embed(prot_mut.native_x.long()))
            prot_mut.x = prot_mut.x.to(torch.float32)
            x_esm_mut = self.proj_esm(prot_mut.x)
            if self.graph_mut_inject:
                x_mut_mut = self.mut_embed(prot_mut.mut_flag.long())
            else:
                x_mut_mut = torch.zeros(x_aa_mut.size(0), x_aa_mut.size(-1),
                                        device=x_aa_mut.device)
            x_prot_mut = F.relu(x_aa_mut + x_esm_mut + x_mut_mut)

            # Local structural delta between mut and wt protein graphs
            local_delta_emb = self.mut_local_delta(
                x_prot_mut, prot_mut.edge_index,
                prot_mut.local_mask if hasattr(prot_mut, 'local_mask') else
                    torch.zeros(prot_mut.x.size(0), dtype=torch.bool, device=prot_mut.x.device),
                prot_mut.batch, pred_mut.size(0)
            )[1]  # index [1] = local_delta_emb

            diff_feat     = fused_mut - fused_wt           # [B, hidden_dim*2]
            hadamard_feat = fused_mut * fused_wt           # [B, hidden_dim*2]
            delta_pred = self.delta_head(local_delta_emb, diff_feat, hadamard_feat)
        else:
            delta_pred = pred_mut - pred_wt  # fallback: arithmetic diff

        return pred_wt, pred_mut, delta_pred

    def generate_masks(self, adj, adj_sizes, n_heads):
        out = torch.ones(adj.shape[0], adj.shape[1])
        max_size = adj.shape[1]
        if isinstance(adj_sizes, int):
            out[0, adj_sizes:max_size] = 0
        else:
            for e_id, drug_len in enumerate(adj_sizes):
                out[e_id, drug_len:max_size] = 0
        return out.to(device=adj.device, dtype=adj.dtype)


def smiles_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    c_size = mol.GetNumAtoms()
    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    mol_adj = np.zeros((c_size, c_size))
    for e1, e2 in g.edges:
        mol_adj[e1, e2] = 1
    mol_adj += np.matrix(np.eye(mol_adj.shape[0]))
    index_row, index_col = np.where(mol_adj >= 0.5)
    for i, j in zip(index_row, index_col):
        edge_index.append([i, j])
    edge_index = np.array(edge_index)
    return c_size, edge_index


#############################################################################
# Data loading — paths resolved relative to DATA_ROOT
#############################################################################

fixed_csv = os.path.join(DATA_ROOT, f'{dataset_name}_processed_fixed.csv')
raw_csv = os.path.join(DATA_ROOT, f'{dataset_name}_processed.csv')
if args.csv_path:
    csv_path = args.csv_path
elif args.force_raw:
    csv_path = raw_csv
else:
    csv_path = fixed_csv if os.path.exists(fixed_csv) else raw_csv
print(f"Reading data from: {csv_path}")
df = pd.read_csv(csv_path)

smiles = set(df['compound_iso_smiles'])
target = set(df['target_key'])

target_seq = {}
seq_col = 'fixed_target_sequence' if 'fixed_target_sequence' in df.columns else 'target_sequence'
for i in range(len(df)):
    seq = df.loc[i, seq_col]
    if pd.isna(seq):
        seq = df.loc[i, 'target_sequence']
    target_seq[df.loc[i, 'target_key']] = seq

smiles_graph = {}
for sm in smiles:
    _, graph = smiles_to_graph(sm)
    smiles_graph[sm] = graph

target_uniprot_dict = {}
target_process_start = {}
target_process_end = {}
for i in range(len(df)):
    t = df.loc[i, 'target_key']
    if dataset_name == 'kiba':
        uniprot = df.loc[i, 'target_key']
    else:
        uniprot = df.loc[i, 'uniprot']
    target_uniprot_dict[t] = uniprot
    target_process_start[t] = df.loc[i, 'target_sequence_start']
    target_process_end[t] = df.loc[i, 'target_sequence_end']

contact_dir = os.path.join(DATA_ROOT, f'target_contact_map_{dataset_name}')
target_graph = {}


def target_to_graph(target_key, target_sequence, contact_dir, start, end):
    target_edge_index = []
    target_size = len(target_sequence)
    contact_file = os.path.join(contact_dir, target_key + '.npy')
    contact_map = np.load(contact_file)
    contact_map = contact_map[start:end, start:end]
    index_row, index_col = np.where(contact_map > 0.8)
    for i, j in zip(index_row, index_col):
        target_edge_index.append([i, j])
    target_edge_index = np.array(target_edge_index)
    return target_size, target_edge_index


vocab_dir = os.path.join(DATA_ROOT, 'Vocab')
drug_vocab = WordVocab.load_vocab(os.path.join(vocab_dir, 'smiles_vocab.pkl'))
target_vocab = WordVocab.load_vocab(os.path.join(vocab_dir, 'protein_vocab.pkl'))

tar_len = 1000
seq_len = 100

smiles_idx = {}
smiles_emb = {}
smiles_len = {}
for sm in smiles:
    content = []
    flag = 0
    for i in range(len(sm)):
        if flag >= len(sm):
            break
        if (flag + 1 < len(sm)):
            if drug_vocab.stoi.__contains__(sm[flag:flag + 2]):
                content.append(drug_vocab.stoi.get(sm[flag:flag + 2]))
                flag = flag + 2
                continue
        content.append(drug_vocab.stoi.get(sm[flag], drug_vocab.unk_index))
        flag = flag + 1

    if len(content) > seq_len - 2:
        content = content[:seq_len - 2]

    X = [drug_vocab.sos_index] + content + [drug_vocab.eos_index]
    smiles_len[sm] = len(content)
    if seq_len > len(X):
        padding = [drug_vocab.pad_index] * (seq_len - len(X))
        X.extend(padding)

    smiles_emb[sm] = torch.tensor(X)

    if not smiles_idx.__contains__(sm):
        tem = []
        for i, c in enumerate(X):
            if atom_dict.__contains__(c):
                tem.append(i)
        smiles_idx[sm] = tem

target_emb = {}
target_len = {}
for k in target_seq:
    seq = target_seq[k]
    content = []
    flag = 0
    for i in range(len(seq)):
        if flag >= len(seq):
            break
        if (flag + 1 < len(seq)):
            if target_vocab.stoi.__contains__(seq[flag:flag + 2]):
                content.append(target_vocab.stoi.get(seq[flag:flag + 2]))
                flag = flag + 2
                continue
        content.append(target_vocab.stoi.get(seq[flag], target_vocab.unk_index))
        flag = flag + 1

    if len(content) > tar_len - 2:
        content = content[:tar_len - 2]

    X = [target_vocab.sos_index] + content + [target_vocab.eos_index]
    target_len[seq] = len(content)
    if tar_len > len(X):
        padding = [target_vocab.pad_index] * (tar_len - len(X))
        X.extend(padding)
    target_emb[seq] = torch.tensor(X)

print("Building dataset...")
protein_graph_dir = os.path.join(DATA_ROOT, f'graphs_{dataset_name}')
drug_graph_path = os.path.join(DATA_ROOT, f'unimol_compounds_{dataset_name}_512.pt')
mutation_info_path = os.path.join(DATA_ROOT, 'mutation_info.csv')
default_pair_csv = os.path.join(DATA_ROOT, 'cliff_pairs.csv')
train_pair_csv = args.train_pair_csv or default_pair_csv
eval_pair_csv = args.eval_pair_csv or default_pair_csv

DatasetClass = DTADatasetLocal if args.use_local_branch else DTADataset
extra_kwargs = {'k_hops': args.local_k_hops} if args.use_local_branch else {}

dataset = DatasetClass(
    root=DATA_ROOT, path=csv_path,
    smiles_emb=smiles_emb, target_emb=target_emb,
    smiles_idx=smiles_idx, smiles_graph=smiles_graph,
    target_graph=target_graph,
    smiles_len=smiles_len, target_len=target_len,
    protein_graph_dir=protein_graph_dir,
    drug_graph_file=drug_graph_path,
    dataset_name=dataset_name,
    mutation_info_path=mutation_info_path,
    **extra_kwargs,
)

original_dataset_name = dataset_name

output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)
ckpt_name = args.ckpt_base_name or f"checkpoint_{dataset_name}_seed{seed}_{args.exp_name}"
model_file_name = os.path.join(output_dir, f'{ckpt_name}.pt')

print("Building model...")
model = PMHGT(embedding_dim=256, lstm_dim=128, hidden_dim=256, dropout_rate=0.2,
              alpha=0.2, n_heads=8, bilstm_layers=2, protein_vocab=26,
              smile_vocab=45, theta=0.5,
              use_local_branch=args.use_local_branch,
              seq_mut_inject=args.seq_mut_inject,
              graph_mut_inject=args.graph_mut_inject,
              residual_gate_scale=args.residual_gate_scale,
              use_delta_head=args.use_delta_head).to(device)


def apply_trainable_modules(model_obj, module_text: str):
    module_text = (module_text or "all").strip()
    if module_text.lower() == "all":
        for p in model_obj.parameters():
            p.requires_grad = True
        trainable = [n for n, _ in model_obj.named_parameters()]
        return trainable

    wanted = [x.strip() for x in module_text.split(",") if x.strip()]
    for p in model_obj.parameters():
        p.requires_grad = False
    trainable = []
    for name, p in model_obj.named_parameters():
        if any(name.startswith(prefix) for prefix in wanted):
            p.requires_grad = True
            trainable.append(name)
    if len(trainable) == 0:
        raise ValueError(
            f"--trainable_modules='{module_text}' matched 0 parameters. "
            f"Check prefixes in model.named_parameters()."
        )
    return trainable


trainable_param_names = apply_trainable_modules(model, args.trainable_modules)
print(f"Trainable modules setting: {args.trainable_modules}")
print(f"Trainable params: {len(trainable_param_names)} / "
      f"{sum(1 for _ in model.parameters())} tensors")

param_total = sum(p.numel() for p in model.parameters())
param_kan   = sum(p.numel() for p in model.kan_head.parameters())
if args.use_local_branch:
    param_local = sum(p.numel() for p in model.mut_local.parameters())
    param_gate  = sum(p.numel() for p in model.mut_gating.parameters())
    print(f"Total parameters: {param_total:,}  "
          f"(local branch: {param_local:,}, gating: {param_gate:,}, "
          f"kan_head: {param_kan:,})")
else:
    print(f"Total parameters: {param_total:,}  (kan_head: {param_kan:,})")

optimizer = torch.optim.Adam(
    [p for p in model.parameters() if p.requires_grad],
    lr=LR
)
schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, NUM_EPOCHS, eta_min=1e-5, last_epoch=-1)

# ── checkpoint resume ──────────────────────────────────────────
start_epoch = 0
best_mse = 1000
best_epoch = -1
val2 = float('inf')
no_improve_count = 0

resume_path = args.resume
if resume_path is None:
    resume_path = model_file_name if os.path.exists(model_file_name) else None

if resume_path and os.path.exists(resume_path):
    state_path = resume_path.replace('.pt', '_state.pt')
    model.load_state_dict(torch.load(resume_path, map_location=device, weights_only=False))
    if os.path.exists(state_path):
        state = torch.load(state_path, map_location='cpu', weights_only=False)
        start_epoch = state['epoch']
        best_mse = state['best_mse']
        best_epoch = state['best_epoch']
        val2 = state.get('best_test_mse', float('inf'))
        no_improve_count = state.get('no_improve_count', 0)
        try:
            optimizer.load_state_dict(state['optimizer'])
        except ValueError as e:
            print(f"WARNING: optimizer state not loaded due to param-group mismatch "
                  f"(likely from trainable_modules change): {e}")
        schedule.load_state_dict(state['scheduler'])
        print(f"Resumed from epoch {start_epoch}, best_val_mse={best_mse:.4f}")
    else:
        print(f"Loaded model weights from {resume_path} (no optimizer state found)")
else:
    print("Starting from scratch")

# ── data split ──────────────────────────────────────────────────
dataset_df = dataset.data
train_indices = dataset_df[dataset_df['split'] == 'train'].index.tolist()
val_indices = dataset_df[dataset_df['split'] == 'val'].index.tolist()
test_indices = dataset_df[dataset_df['split'] == 'test'].index.tolist()
train_dataset = torch.utils.data.Subset(dataset, train_indices)
val_dataset = torch.utils.data.Subset(dataset, val_indices)
test_dataset = torch.utils.data.Subset(dataset, test_indices)
eval_workers = max(0, args.num_workers // 2)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    persistent_workers=args.num_workers > 0,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=eval_workers,
    persistent_workers=eval_workers > 0,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=eval_workers,
    persistent_workers=eval_workers > 0,
)
print(f"Split: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")

# ── teacher distillation: load pre-computed soft labels ──────────────────────
# teacher_preds_tensor[global_idx] = frozen-teacher prediction for dataset[global_idx].
# NaN entries = teacher could not process that sample (missing graph etc.).
# Generate with: python gen_teacher_preds.py --output teacher_preds_all.npy
# (default split='all' so all global indices are covered)
teacher_preds_tensor = None
# is_cliff_sample_tensor[global_idx] = True if that sample appears in any cliff pair.
# Indexed by global dataset index so KD masking works correctly in the training loop.
is_cliff_sample_tensor = None

if args.teacher_pred_file and os.path.exists(args.teacher_pred_file):
    teacher_preds_np = np.load(args.teacher_pred_file)
    # Sanity check: the .npy must cover every row of dataset_df by global index.
    # A length mismatch means the file was generated from a different dataset split
    # or an older version of the CSV, which would silently mis-index all KD targets.
    assert len(teacher_preds_np) == len(dataset_df), (
        f"[KD] teacher_preds length ({len(teacher_preds_np)}) != "
        f"dataset length ({len(dataset_df)}).  "
        f"Regenerate with: python gen_teacher_preds.py --output <file> "
        f"(default split='all' covers all global indices)."
    )
    teacher_preds_tensor = torch.tensor(teacher_preds_np, dtype=torch.float32)
    n_valid_preds = int((~np.isnan(teacher_preds_np)).sum())
    print(f"Teacher predictions loaded from {args.teacher_pred_file}: "
          f"total={len(teacher_preds_tensor)}, valid(non-NaN)={n_valid_preds}, "
          f"kd_lambda={args.kd_lambda}")

    if args.kd_noncliff_only and os.path.exists(train_pair_csv):
        # Build a boolean tensor indexed by global dataset index.
        # Correct approach: look up each (smiles, target_key) pair in the dataset
        # DataFrame and mark those global indices as cliff samples.
        cliff_df = pd.read_csv(train_pair_csv)
        is_cliff_np = np.zeros(len(dataset_df), dtype=bool)

        # Collect all (compound_iso_smiles, target_key) tuples that appear in
        # any cliff pair (both wt and mut sides).
        cliff_pairs_set = set()
        pair_col_specs = [
            ('drug_smiles', 'wt_target_key'),
            ('drug_smiles', 'mut_target_key'),
            ('smiles_wt', 'target_wt'),
            ('smiles_mut', 'target_mut'),
        ]
        for col_sm, col_tgt in pair_col_specs:
            if col_sm in cliff_df.columns and col_tgt in cliff_df.columns:
                for sm, tgt in zip(cliff_df[col_sm], cliff_df[col_tgt]):
                    cliff_pairs_set.add((sm, tgt))

        # Mark matching rows in dataset_df as cliff samples.
        for row_idx, row in dataset_df.iterrows():
            key = (row.get('compound_iso_smiles', ''), row.get('target_key', ''))
            if key in cliff_pairs_set:
                is_cliff_np[row_idx] = True

        is_cliff_sample_tensor = torch.tensor(is_cliff_np, dtype=torch.bool)
        n_cliff = int(is_cliff_np.sum())
        if n_cliff == 0:
            print("WARNING: KD noncliff-only found 0 cliff samples in dataset_df; "
                  "check cliff_pairs.csv column names / dataset alignment.")
        else:
            print(f"KD noncliff-only: {n_cliff} cliff samples will be excluded from KD loss "
                  f"(out of {len(is_cliff_np)} total)")
elif args.teacher_pred_file:
    print(f"WARNING: teacher_pred_file '{args.teacher_pred_file}' not found, KD disabled.")

# ── cliff pair iterator ─────────────────────────────────────────
cliff_iter = None
if args.use_cliff_loss:
    cliff_pair_ds = CliffPairDataset(train_pair_csv, dataset, split='train')
    if len(cliff_pair_ds) > 0:
        cliff_iter = PairBatchIterator(
            cliff_pair_ds,
            batch_size=args.cliff_batch_size,
            sampling=args.pair_sampling,
            reweight_alpha=args.pair_reweight_alpha,
            seed=args.seed,
            gene_balance_ratio=args.gene_balance_ratio,
            weight_cap=args.pair_weight_cap,
            density_weighting=args.pair_density_weighting,
            density_alpha=args.pair_density_alpha,
            density_bins=args.pair_density_bins,
        )
        print(f"Cliff loss enabled: {len(cliff_pair_ds)} train pairs, "
              f"lambda={args.cliff_lambda}, margin={args.cliff_margin}, "
              f"warmup={args.cliff_warmup_epochs} epochs, "
              f"pair_sampling={args.pair_sampling}, "
              f"gene_balance_ratio={args.gene_balance_ratio}, "
              f"pair_reweight_alpha={args.pair_reweight_alpha}, "
              f"pair_weight_cap={args.pair_weight_cap}, "
              f"pair_density_weighting={args.pair_density_weighting}, "
              f"pair_density_alpha={args.pair_density_alpha}, "
              f"pair_density_bins={args.pair_density_bins}")
    else:
        print("WARNING: No cliff pairs for training, cliff loss disabled")

# ── delta head pair iterator (reuses CliffPairDataset) ─────────
delta_pair_iter = None
if args.use_delta_head:
    _delta_pair_ds = CliffPairDataset(train_pair_csv, dataset, split='train')
    if len(_delta_pair_ds) > 0:
        delta_pair_iter = PairBatchIterator(
            _delta_pair_ds,
            batch_size=args.cliff_batch_size,
            sampling=args.pair_sampling,
            reweight_alpha=args.pair_reweight_alpha,
            seed=args.seed + 17,
            gene_balance_ratio=args.gene_balance_ratio,
            weight_cap=args.pair_weight_cap,
            density_weighting=args.pair_density_weighting,
            density_alpha=args.pair_density_alpha,
            density_bins=args.pair_density_bins,
        )
        print(f"DeltaHead enabled: {len(_delta_pair_ds)} train pairs, "
              f"delta_lambda={args.delta_lambda}, warmup={args.cliff_warmup_epochs} epochs, "
              f"pair_sampling={args.pair_sampling}, "
              f"gene_balance_ratio={args.gene_balance_ratio}, "
              f"pair_reweight_alpha={args.pair_reweight_alpha}, "
              f"pair_weight_cap={args.pair_weight_cap}, "
              f"pair_density_weighting={args.pair_density_weighting}, "
              f"pair_density_alpha={args.pair_density_alpha}, "
              f"pair_density_bins={args.pair_density_bins}")
    else:
        print("WARNING: No cliff pairs found, DeltaHead disabled")


def train_one_epoch(model, train_loader, optimizer, epoch, cliff_iter=None,
                    delta_pair_iter=None):
    model.train()
    loss_fn = torch.nn.MSELoss()
    kd_loss_fn = torch.nn.MSELoss()
    total_mse_loss = 0
    total_kd_loss = 0
    total_kd_valid_samples = 0   # number of samples that actually enter KD loss
    total_cliff_loss = 0
    total_delta_loss = 0
    delta_call_count = 0

    use_cliff = (cliff_iter is not None and epoch >= args.cliff_warmup_epochs)
    if use_cliff and args.cliff_rampup_epochs > 0:
        ramp_progress = min(1.0, (epoch - args.cliff_warmup_epochs) / args.cliff_rampup_epochs)
        cur_lambda = args.cliff_lambda * ramp_progress
    else:
        cur_lambda = args.cliff_lambda if use_cliff else 0.0

    use_kd = (teacher_preds_tensor is not None and args.kd_lambda > 0)
    use_delta = (delta_pair_iter is not None and args.use_delta_head
                 and epoch >= args.cliff_warmup_epochs)

    cliff_call_count = 0
    total_mutant_samples = 0
    total_samples = 0

    used_batches = 0
    for batch_idx, data in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
        if args.max_train_batches > 0 and batch_idx >= args.max_train_batches:
            break
        used_batches += 1
        data = [d.to(device) for d in data]
        data_batch, stru_d, stru_p = data
        optimizer.zero_grad()

        output, _ = model(data_batch, stru_d, stru_p)
        mse_loss = loss_fn(output.float(), data_batch.y.float().to(device))
        total_mse_loss += mse_loss.item()

        loss = mse_loss

        # ── track mutant sample ratio (diagnostic) ───────────────────────────
        # Counts how many samples in this batch actually have a mutant local
        # neighbourhood (local_mask has ≥1 True node).  Printed once per epoch
        # to confirm the mutation pathway is active for the expected fraction.
        if hasattr(stru_p, 'local_mask') and stru_p.local_mask.any():
            bsz = len(data_batch.sm)
            mutant_graph_ids = stru_p.batch[stru_p.local_mask].unique()
            total_mutant_samples += len(mutant_graph_ids)
        total_samples += len(data_batch.sm)

        # ── Knowledge Distillation loss ───────────────────────────────────────
        # teacher_preds_tensor is indexed by global dataset index (sample_idx).
        # Two guards before computing KD loss:
        #   1. NaN mask  – teacher_preds entries are NaN for samples the teacher
        #      could not process (missing graph).  Passing NaN to MSELoss gives
        #      NaN gradients and silently corrupts training.
        #   2. noncliff mask (optional) – when --kd_noncliff_only is set, cliff
        #      samples are allowed to deviate from teacher (teacher fails there
        #      anyway).  Mask is indexed by global dataset index, NOT by string
        #      ID sets, to avoid false positives from shared SMILES / target keys.
        if use_kd and hasattr(data_batch, 'sample_idx'):
            batch_indices = data_batch.sample_idx.view(-1).long().cpu()
            teacher_targets = teacher_preds_tensor[batch_indices.cpu()].to(device)

            # Guard 1: NaN mask
            valid_mask = ~torch.isnan(teacher_targets)

            # Guard 2: noncliff mask (applied only when flag is set and tensor exists)
            if args.kd_noncliff_only and is_cliff_sample_tensor is not None:
                noncliff_mask = ~is_cliff_sample_tensor[batch_indices.cpu()].to(device)
                valid_mask = valid_mask & noncliff_mask

            if valid_mask.any():
                kd_loss = F.mse_loss(
                    output[valid_mask].float(),
                    teacher_targets[valid_mask].float()
                )
                loss = loss + args.kd_lambda * kd_loss
                total_kd_loss += kd_loss.item()
                total_kd_valid_samples += int(valid_mask.sum().item())

        # ── Cliff ranking loss ────────────────────────────────────────────────
        if use_cliff and (batch_idx % args.cliff_freq == 0):
            cliff_batch = cliff_iter.get_batch()
            rank_loss = weighted_cliff_ranking_loss(
                model, cliff_batch, device, margin=args.cliff_margin
            )
            loss = loss + cur_lambda * rank_loss
            total_cliff_loss += rank_loss.item()
            cliff_call_count += 1

        # ── DeltaHead loss ────────────────────────────────────────────────────
        # Sample a (WT, Mut) cliff pair batch, run forward_pair(), compute
        # MSE between predicted Δ and true Δ (= y_mut - y_wt from labels).
        if use_delta and (batch_idx % args.cliff_freq == 0):
            pair_batch = delta_pair_iter.get_batch()
            try:
                # pair_batch: [data_wt, drug_wt, prot_wt,
                #              data_mut, drug_mut, prot_mut, signs, pair_weights]
                d_wt, dr_wt, pr_wt, d_mut, dr_mut, pr_mut, _signs, pair_weights = [
                    x.to(device) for x in pair_batch
                ]
                true_delta = (d_mut.y - d_wt.y).float().to(device)
                _, _, delta_pred = model.forward_pair(
                    d_wt, dr_wt, pr_wt, d_mut, dr_mut, pr_mut
                )
                delta_loss = F.mse_loss(delta_pred.float(), true_delta, reduction='none')
                delta_loss = (delta_loss.view(-1) * pair_weights.view(-1)).mean()
                loss = loss + args.delta_lambda * delta_loss
                total_delta_loss += delta_loss.item()
                delta_call_count += 1
            except Exception as e:
                print(f"WARNING: DeltaHead batch skipped at epoch {epoch}, batch {batch_idx}: "
                      f"{type(e).__name__}: {e}")

        loss.backward()
        optimizer.step()

    n_batches = max(1, used_batches)
    avg_mse   = total_mse_loss / n_batches
    avg_kd    = total_kd_loss / n_batches if use_kd else 0
    avg_cliff = total_cliff_loss / cliff_call_count if cliff_call_count > 0 else 0
    avg_delta = total_delta_loss / delta_call_count if delta_call_count > 0 else 0
    current_lr = optimizer.param_groups[0]['lr']
    mutant_ratio = total_mutant_samples / total_samples if total_samples > 0 else 0
    kd_coverage = total_kd_valid_samples / total_samples if (use_kd and total_samples > 0) else 0
    print(f"  MSE loss: {avg_mse:.4f}  lr: {current_lr:.6f}"
          f"  mutant_ratio={mutant_ratio:.3f} ({total_mutant_samples}/{total_samples})" +
          (f"  KD loss: {avg_kd:.4f} (lambda={args.kd_lambda:.2f},"
           f" kd_coverage={kd_coverage:.3f} [{total_kd_valid_samples}/{total_samples}])"
           if use_kd else "") +
          (f"  Cliff loss: {avg_cliff:.4f} (lambda={cur_lambda:.3f}, calls={cliff_call_count})" if use_cliff else "") +
          (f"  Delta loss: {avg_delta:.4f} (lambda={args.delta_lambda:.2f}, calls={delta_call_count})" if use_delta else ""))


# ── training loop ──────────────────────────────────────────────
if args.eval_only:
    print("eval_only mode: skipping training loop")
for epoch in range(0 if args.eval_only else start_epoch, 0 if args.eval_only else NUM_EPOCHS):
    train_one_epoch(model, train_loader, optimizer, epoch, cliff_iter,
                    delta_pair_iter=delta_pair_iter)

    G, P = predicting(model, val_loader)
    val1 = get_mse(G, P)

    if val1 < best_mse:
        best_mse = val1
        best_epoch = epoch + 1
        no_improve_count = 0
        G_test, P_test = predicting(model, test_loader)
        val2 = get_mse(G_test, P_test)
        torch.save(model.state_dict(), model_file_name)
        print(f'  MSE improved at epoch {best_epoch}; val_mse={best_mse:.4f}, test_mse={val2:.4f}')
    else:
        no_improve_count += 1
        print(f'  val_mse={val1:.4f}, no improvement since epoch {best_epoch} '
              f'({no_improve_count}/{args.patience if args.patience > 0 else "∞"}); '
              f'best_val={best_mse:.4f}, best_test_mse={val2:.4f}')

    torch.save({
        'epoch': epoch + 1,
        'best_mse': best_mse,
        'best_epoch': best_epoch,
        'best_test_mse': val2,
        'no_improve_count': no_improve_count,
        'optimizer': optimizer.state_dict(),
        'scheduler': schedule.state_dict(),
    }, model_file_name.replace('.pt', '_state.pt'))

    schedule.step()

    if args.patience > 0 and no_improve_count >= args.patience:
        print(f"\nEarly stopping triggered at epoch {epoch + 1} "
              f"(no improvement for {args.patience} epochs)")
        break

# ── final evaluation ──────────────────────────────────────────
print(f"\n{'='*60}")
print(f"Training complete. Best epoch: {best_epoch}")
print(f"Loading best model from {model_file_name}")
if not os.path.exists(model_file_name):
    raise FileNotFoundError(
        f"Checkpoint not found: {model_file_name}. "
        f"If using --eval_only, provide --resume <checkpoint.pt> "
        f"or ensure the default checkpoint exists."
    )
model.load_state_dict(
    torch.load(model_file_name, map_location=device, weights_only=False)
)

G_test, P_test = predicting(model, test_loader)
cindex, rm2, mse, pearson, spearman = calculate_metrics_and_return(G_test, P_test, test_loader)
print(f"CI: {cindex:.4f}, RM2: {rm2:.4f}, MSE: {mse:.4f}, Pearson: {pearson:.4f}, Spearman: {spearman:.4f}")

cliff_results = evaluate_cliff(G_test, P_test, dataset_df, eval_pair_csv, split='test')

if args.save_test_predictions:
    test_meta = dataset_df.iloc[test_indices].copy().reset_index(drop=True)
    pred_df = test_meta[['compound_iso_smiles', 'target_key', 'uniprot', 'split', 'affinity']].copy()
    pred_df['prediction'] = P_test
    pred_df['label'] = G_test
    pred_df['seed'] = seed
    pred_path = os.path.join(output_dir, f'{ckpt_name}_test_predictions.csv')
    pred_df.to_csv(pred_path, index=False)
    print(f"Test predictions saved to {pred_path}")

results_path = os.path.join(output_dir, f'{ckpt_name}_results.txt')
with open(results_path, 'w') as f:
    f.write(f"Experiment: {args.exp_name}\n")
    f.write(f"Seed: {seed}\n")
    f.write(f"Best epoch: {best_epoch}\n")
    f.write(f"LocalBranch+Gating: {args.use_local_branch}, K-hops: {args.local_k_hops}\n")
    f.write(f"DeltaHead: {args.use_delta_head}, delta_lambda: {args.delta_lambda}\n")
    f.write(f"CI: {cindex:.4f}, RM2: {rm2:.4f}, MSE: {mse:.4f}, "
            f"Pearson: {pearson:.4f}, Spearman: {spearman:.4f}\n")
    if cliff_results:
        for k, v in cliff_results.items():
            f.write(f"{k}: {v}\n")
    f.write(f"\nArgs: {vars(args)}\n")
print(f"Results saved to {results_path}")
print("KAN head experiment complete.")
