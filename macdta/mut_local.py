"""
Mutation-center local structure branch + mutation-conditioned gating for PMHGT-DTA.

Components
----------
build_local_mask       : BFS over undirected protein contact graph to find k-hop
                         neighborhood of the mutation site.
MutLocalBranch         : 2-layer GIN + gated-attention pooling over the local
                         subgraph.  Wild-type proteins receive a learnable
                         no-mutation token.  (kept for backward compat)
MutLocalDeltaBranch    : Upgraded version of MutLocalBranch.  Produces TWO outputs:
                           local_emb        [B, out_dim]  – Mut local embedding
                           local_delta_emb  [B, out_dim]  – Mut local − WT global ref
                         The WT global reference is derived from a scatter_mean of
                         all protein graph nodes, so no extra WT graph data is
                         needed.  For WT samples (no local_mask), local_delta_emb
                         is a zero vector.
MutCondGating          : Lightweight gating module.  Takes the local embedding
                         produced by MutLocalBranch / MutLocalDeltaBranch and
                         generates two sigmoid gates that multiplicatively modulate
                         the protein sequence representation (protein_out) and the
                         protein graph representation (gcn_g_featp) before fusion.
DeltaHead              : MLP that predicts Δaffinity = affinity_mut − affinity_wt
                         from (local_delta_emb, diff_feat, hadamard_feat).
                         Only instantiated / called during training on cliff pairs.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv
from torch_geometric.utils import softmax as pyg_softmax
from torch_scatter import scatter_mean
from torch.nn import Sequential, Linear, ReLU


def build_local_mask(edge_index, mut_idx, num_nodes, k_hops=2):
    """
    BFS over the **undirected** protein contact graph to find all nodes within
    k hops of the mutation site.

    The graph is treated as undirected: for every directed edge (s→d) in
    edge_index we also traverse (d→s), so the neighbourhood is symmetric and
    independent of edge orientation in the stored contact map.

    Args:
        edge_index : [2, E] LongTensor or None
        mut_idx    : int, 0-based residue index of the mutation site
        num_nodes  : int, total number of residues in the graph
        k_hops     : int, BFS radius

    Returns:
        [num_nodes] BoolTensor, True for nodes in the k-hop neighbourhood
    """
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    if mut_idx < 0 or mut_idx >= num_nodes:
        return mask

    mask[mut_idx] = True
    if edge_index is None or edge_index.numel() == 0:
        return mask

    # Build undirected adjacency list (both directions)
    adj = [[] for _ in range(num_nodes)]
    for s, d in zip(edge_index[0].tolist(), edge_index[1].tolist()):
        if 0 <= s < num_nodes and 0 <= d < num_nodes:
            adj[s].append(d)
            adj[d].append(s)   # explicit undirected expansion

    visited = {mut_idx}
    frontier = [mut_idx]
    for _ in range(k_hops):
        next_frontier = []
        for node in frontier:
            for nbr in adj[node]:
                if nbr not in visited:
                    visited.add(nbr)
                    next_frontier.append(nbr)
        frontier = next_frontier
        if not frontier:
            break

    for node in visited:
        mask[node] = True
    return mask


# ──────────────────────────────────────────────────────────────
# Shared GIN encoder blocks (reused by both branch variants)
# ──────────────────────────────────────────────────────────────

def _make_gin_block(in_dim: int, hidden_dim: int):
    """Return a 2-layer GIN + two LayerNorms."""
    nn1 = Sequential(Linear(in_dim, hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim))
    conv1 = GINConv(nn1)
    ln1 = nn.LayerNorm(hidden_dim)
    nn2 = Sequential(Linear(hidden_dim, hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim))
    conv2 = GINConv(nn2)
    ln2 = nn.LayerNorm(hidden_dim)
    return conv1, ln1, conv2, ln2


class MutLocalBranch(nn.Module):
    """
    Mutation-centered local structure branch (original version, kept for compat).

    For each protein graph in the batch:
      1. Extract the induced subgraph over the k-hop neighbourhood of the
         mutation site (local_mask == True).
      2. Process it through a 2-layer GIN with LayerNorm and residual.
      3. Pool with gated (softmax-weighted) attention to produce a fixed-size
         embedding.

    Wild-type proteins (local_mask all-False) receive a learnable
    no-mutation token so the downstream gating module always has a meaningful
    signal to condition on.
    """

    def __init__(self, in_dim=512, hidden_dim=256, out_dim=256):
        super().__init__()
        self.out_dim = out_dim
        self.conv1, self.ln1, self.conv2, self.ln2 = _make_gin_block(in_dim, hidden_dim)
        self.pool_gate = nn.Linear(hidden_dim, 1)
        self.out_proj = nn.Linear(hidden_dim, out_dim)
        self.no_mut_token = nn.Parameter(torch.zeros(out_dim))
        nn.init.normal_(self.no_mut_token, std=0.02)

    def _pool_local(self, x, edge_index, local_mask, batch, num_graphs, device):
        """Extract local subgraph, run GIN, gated-attention pool. Returns pooled [B, hidden]."""
        no_mut = self.no_mut_token.unsqueeze(0).expand(num_graphs, -1)
        if not local_mask.any():
            return no_mut, torch.zeros(num_graphs, dtype=torch.bool, device=device)

        local_idx = local_mask.nonzero(as_tuple=True)[0]
        local_x = x[local_idx]
        local_batch = batch[local_idx]

        node_remap = torch.full((x.size(0),), -1, dtype=torch.long, device=device)
        node_remap[local_idx] = torch.arange(len(local_idx), device=device)

        src, dst = edge_index[0], edge_index[1]
        edge_keep = local_mask[src] & local_mask[dst]
        if edge_keep.any():
            local_edges = torch.stack(
                [node_remap[src[edge_keep]], node_remap[dst[edge_keep]]], dim=0
            )
        else:
            local_edges = torch.zeros(2, 0, dtype=torch.long, device=device)

        h = F.relu(self.conv1(local_x, local_edges))
        h = self.ln1(h)
        h = h + F.relu(self.conv2(h, local_edges))
        h = self.ln2(h)

        gate = self.pool_gate(h).squeeze(-1)
        gate = pyg_softmax(gate, local_batch, num_nodes=num_graphs)
        weighted = h * gate.unsqueeze(-1)

        pooled = torch.zeros(num_graphs, h.size(-1), device=device)
        pooled.scatter_add_(0, local_batch.unsqueeze(-1).expand_as(weighted), weighted)
        pooled = self.out_proj(pooled)

        has_local = torch.zeros(num_graphs, dtype=torch.bool, device=device)
        has_local[local_batch.unique()] = True

        out = torch.where(has_local.unsqueeze(-1), pooled, no_mut)
        return out, has_local

    def forward(self, x, edge_index, local_mask, batch, num_graphs):
        """
        Args:
            x          : [N, in_dim]  protein node features
            edge_index : [2, E]       protein graph edges
            local_mask : [N]          bool, True for k-hop neighbourhood nodes
            batch      : [N]          graph-membership vector
            num_graphs : int

        Returns:
            [num_graphs, out_dim]  local structural embedding per graph
        """
        device = x.device
        out, _ = self._pool_local(x, edge_index, local_mask, batch, num_graphs, device)
        return out


class MutLocalDeltaBranch(nn.Module):
    """
    Upgraded mutation-centered local structure branch.

    Produces TWO outputs per protein graph in the batch:

      local_emb       [B, out_dim]
          The Mut-local structural embedding (same semantics as MutLocalBranch).
          Wild-type proteins fall back to the learnable no_mut_token.

      local_delta_emb [B, out_dim]
          Approximate structural perturbation signal:
              local_delta_emb = local_emb − wt_ref_emb
          where wt_ref_emb is a scatter_mean of ALL protein graph node features
          projected through a dedicated linear (proj_wt).  For wild-type samples
          (no k-hop neighbourhood), local_delta_emb is set to zeros so the
          DeltaHead receives a neutral signal.

    Interface is backward-compatible with MutLocalBranch for forward():
        forward(...) → (local_emb, local_delta_emb)
    """

    def __init__(self, in_dim=512, hidden_dim=256, out_dim=256):
        super().__init__()
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim

        # --- local (Mut) branch: same as MutLocalBranch ---
        self.conv1, self.ln1, self.conv2, self.ln2 = _make_gin_block(in_dim, hidden_dim)
        self.pool_gate = nn.Linear(hidden_dim, 1)
        self.out_proj = nn.Linear(hidden_dim, out_dim)
        self.no_mut_token = nn.Parameter(torch.zeros(out_dim))
        nn.init.normal_(self.no_mut_token, std=0.02)

        # --- WT global reference branch (new) ---
        # Projects mean-pooled global node features into the same space as local_emb
        self.proj_wt = nn.Linear(in_dim, out_dim)
        self.wt_ln = nn.LayerNorm(out_dim)

    # ------------------------------------------------------------------
    def _pool_local(self, x, edge_index, local_mask, batch, num_graphs, device):
        """
        Returns:
            pooled    [B, out_dim]  — gated attention pool of local nodes
            has_local [B]  bool     — which graphs have a non-empty local mask
        """
        no_mut = self.no_mut_token.unsqueeze(0).expand(num_graphs, -1)
        if not local_mask.any():
            return no_mut, torch.zeros(num_graphs, dtype=torch.bool, device=device)

        local_idx = local_mask.nonzero(as_tuple=True)[0]
        local_x = x[local_idx]
        local_batch = batch[local_idx]

        node_remap = torch.full((x.size(0),), -1, dtype=torch.long, device=device)
        node_remap[local_idx] = torch.arange(len(local_idx), device=device)

        src, dst = edge_index[0], edge_index[1]
        edge_keep = local_mask[src] & local_mask[dst]
        if edge_keep.any():
            local_edges = torch.stack(
                [node_remap[src[edge_keep]], node_remap[dst[edge_keep]]], dim=0
            )
        else:
            local_edges = torch.zeros(2, 0, dtype=torch.long, device=device)

        h = F.relu(self.conv1(local_x, local_edges))
        h = self.ln1(h)
        h = h + F.relu(self.conv2(h, local_edges))
        h = self.ln2(h)

        gate = self.pool_gate(h).squeeze(-1)
        gate = pyg_softmax(gate, local_batch, num_nodes=num_graphs)
        weighted = h * gate.unsqueeze(-1)

        pooled = torch.zeros(num_graphs, h.size(-1), device=device)
        pooled.scatter_add_(0, local_batch.unsqueeze(-1).expand_as(weighted), weighted)
        pooled = self.out_proj(pooled)

        has_local = torch.zeros(num_graphs, dtype=torch.bool, device=device)
        has_local[local_batch.unique()] = True

        out = torch.where(has_local.unsqueeze(-1), pooled, no_mut)
        return out, has_local

    def _global_wt_ref(self, x, batch, num_graphs, device):
        """
        Global mean-pool of all protein nodes → projected WT reference.
        Returns [B, out_dim].
        """
        # scatter_mean over all nodes per graph
        global_mean = scatter_mean(x, batch, dim=0, dim_size=num_graphs)  # [B, in_dim]
        wt_ref = self.wt_ln(self.proj_wt(global_mean))                    # [B, out_dim]
        return wt_ref

    def forward(self, x, edge_index, local_mask, batch, num_graphs):
        """
        Args:
            x          : [N, in_dim]  protein node features (post-projection)
            edge_index : [2, E]
            local_mask : [N] bool
            batch      : [N] graph membership
            num_graphs : int

        Returns:
            local_emb       [B, out_dim]
            local_delta_emb [B, out_dim]  zero for WT samples
        """
        device = x.device

        local_emb, has_local = self._pool_local(
            x, edge_index, local_mask, batch, num_graphs, device
        )
        wt_ref = self._global_wt_ref(x, batch, num_graphs, device)

        # delta = local_emb - wt_ref, zeroed out for WT graphs
        raw_delta = local_emb - wt_ref
        local_delta_emb = torch.where(has_local.unsqueeze(-1), raw_delta,
                                      torch.zeros_like(raw_delta))

        return local_emb, local_delta_emb


class MutCondGating(nn.Module):
    """
    Mutation-conditioned identity-preserving modulation module.

    Takes the local mutation embedding produced by MutLocalBranch /
    MutLocalDeltaBranch and generates two bounded per-dimension deltas that
    are applied as *multiplicative residuals* on top of the backbone:

        protein_out  = protein_out  * (1 + masked_delta_seq)
        gcn_g_featp  = gcn_g_featp  * (1 + masked_delta_graph)

    where masked_delta_* is zero for WT samples (no mutation local mask).

    Design rationale
    ----------------
    The original design ( x' = x * sigmoid(gate) ) had two problems:
      1. sigmoid(fc2(h)) ≈ 0.5 at init (fc2.bias=0 but fc2.weight random),
         so the backbone representations are randomly compressed from batch 1,
         long before the module has learned anything useful.
      2. gate ∈ (0,1) only — no dimension can ever be up-scaled, no identity.

    The current design ( x' = x * (1 + tanh(fc2(h)) * scale) ) instead:
      • fc2.weight and fc2.bias are both zero-initialised → delta = 0 at init
        → (1 + delta) = 1 → strict identity from epoch 0.
      • tanh keeps delta ∈ (-scale, +scale): both attenuation AND amplification.
      • Multiplicative form preserves the "per-dimension modulation" semantic
        and scales the correction proportionally to the activation magnitude,
        unlike pure additive correction which would be magnitude-independent.
      • WT masking in main_warm.py ensures delta is forced to 0 for WT samples
        so only true mutant samples are modulated.

    Architecture
    ------------
    local_emb [B, local_dim]
        → Linear(local_dim, hidden_dim) → ReLU → LayerNorm
        → Linear(hidden_dim, hidden_dim * 2)   ← zero-init weight & bias
        → tanh → * scale
        → split → delta_seq   [B, hidden_dim]
                → delta_graph  [B, hidden_dim]

    Usage (in main_warm.py forward)
    --------------------------------
    delta_seq, delta_graph = self.mut_gating(local_emb)
    mut_mask = graph_has_local.float().unsqueeze(-1)   # 0 for WT, 1 for mutant
    protein_out = protein_out * (1.0 + delta_seq   * mut_mask)
    gcn_g_featp = gcn_g_featp * (1.0 + delta_graph * mut_mask)
    """

    def __init__(self, local_dim: int, hidden_dim: int, scale: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(local_dim, hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.scale = scale
        # Zero-init: at training start the module produces no correction,
        # so the backbone behaves exactly as pretrained PMHGT.
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, local_emb: torch.Tensor):
        """
        Args:
            local_emb : [B, local_dim]

        Returns:
            delta_seq   : [B, hidden_dim]  per-dim modulation delta for protein_out
            delta_graph : [B, hidden_dim]  per-dim modulation delta for gcn_g_featp

        Applied in main_warm.py as:
            protein_out * (1 + delta_seq  * mut_mask)
            gcn_g_featp * (1 + delta_graph * mut_mask)
        where mut_mask=0 for WT, 1 for mutant → strict identity for WT samples.
        """
        h = F.relu(self.fc1(local_emb))
        h = self.ln(h)
        # tanh keeps corrections bounded in [-scale, +scale]; zero-init makes
        # them start at 0 and grow only as mutation patterns are learned.
        deltas = torch.tanh(self.fc2(h)) * self.scale
        delta_seq, delta_graph = deltas.chunk(2, dim=-1)
        return delta_seq, delta_graph


class DeltaHead(nn.Module):
    """
    Explicit affinity-delta prediction head.

    Predicts  Δaffinity = affinity_mut − affinity_wt

    from three complementary signals:
      local_delta_emb  [B, delta_dim]  structural perturbation at mutation site
      diff_feat        [B, hidden_dim] fused_mut − fused_wt  (arithmetic diff)
      hadamard_feat    [B, hidden_dim] fused_mut ⊙ fused_wt  (element product)

    Architecture
    ------------
    cat([local_delta_emb, diff_feat, hadamard_feat])  [B, delta_dim + 2*hidden_dim]
        → Linear → GELU → LayerNorm
        → Linear → GELU → Dropout
        → Linear(hidden_dim, 1) → squeeze → scalar delta_pred [B]

    Only instantiated when --use_delta_head is set.
    Only called during training via model.forward_pair(); never called in eval.
    """

    def __init__(self, delta_dim: int, hidden_dim: int, dropout: float = 0.2):
        super().__init__()
        in_dim = delta_dim + hidden_dim * 2
        mid_dim = hidden_dim

        self.fc1 = nn.Linear(in_dim, mid_dim)
        self.ln1 = nn.LayerNorm(mid_dim)
        self.fc2 = nn.Linear(mid_dim, mid_dim)
        self.drop = nn.Dropout(dropout)
        self.fc3 = nn.Linear(mid_dim, 1)

        nn.init.zeros_(self.fc3.bias)

    def forward(self,
                local_delta_emb: torch.Tensor,
                diff_feat: torch.Tensor,
                hadamard_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            local_delta_emb : [B, delta_dim]
            diff_feat       : [B, hidden_dim]
            hadamard_feat   : [B, hidden_dim]

        Returns:
            delta_pred : [B]   predicted Δaffinity
        """
        x = torch.cat([local_delta_emb, diff_feat, hadamard_feat], dim=-1)
        x = F.gelu(self.fc1(x))
        x = self.ln1(x)
        x = self.drop(F.gelu(self.fc2(x)))
        return self.fc3(x).squeeze(-1)
