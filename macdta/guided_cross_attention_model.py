"""
Guided Cross Attention Model for PMHGT-DTA
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GuidedCrossAttention(nn.Module):
    """
    Guided Cross Attention mechanism for drug-target interaction
    """
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(GuidedCrossAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.scaling = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)
        
    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        """
        Args:
            query: (batch_size, seq_len_q, embed_dim)
            key: (batch_size, seq_len_k, embed_dim)
            value: (batch_size, seq_len_v, embed_dim)
            key_padding_mask: (batch_size, seq_len_k) - True for padding positions
            attn_mask: (seq_len_q, seq_len_k) - additive mask
        """
        batch_size, seq_len_q, _ = query.size()
        seq_len_k = key.size(1)
        
        # Linear projections
        q = self.q_proj(query) * self.scaling
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attn_weights = torch.matmul(q, k.transpose(-2, -1))
        
        # Apply attention mask
        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask
        
        # Apply key padding mask
        if key_padding_mask is not None:
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf')
            )
        
        # Softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        
        # Residual connection and layer norm
        output = self.layer_norm(query + attn_output)
        
        return output, attn_weights


class MultiHeadCrossAttention(nn.Module):
    """
    Multi-head cross attention with feed-forward network
    """
    def __init__(self, embed_dim, num_heads, ff_dim=None, dropout=0.1):
        super(MultiHeadCrossAttention, self).__init__()
        
        if ff_dim is None:
            ff_dim = embed_dim * 4
        
        self.cross_attn = GuidedCrossAttention(embed_dim, num_heads, dropout)
        
        self.ff_net = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        self.layer_norm = nn.LayerNorm(embed_dim)
    
    def forward(self, query, key, value, key_padding_mask=None):
        # Cross attention
        attn_output, attn_weights = self.cross_attn(query, key, value, key_padding_mask)
        
        # Feed-forward network with residual
        output = self.layer_norm(attn_output + self.ff_net(attn_output))
        
        return output, attn_weights


class BidirectionalCrossAttention(nn.Module):
    """
    Bidirectional cross attention between drug and protein
    """
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(BidirectionalCrossAttention, self).__init__()
        
        self.drug_to_protein = GuidedCrossAttention(embed_dim, num_heads, dropout)
        self.protein_to_drug = GuidedCrossAttention(embed_dim, num_heads, dropout)
        
    def forward(self, drug_features, protein_features, drug_mask=None, protein_mask=None):
        """
        Args:
            drug_features: (batch_size, drug_seq_len, embed_dim)
            protein_features: (batch_size, protein_seq_len, embed_dim)
            drug_mask: (batch_size, drug_seq_len)
            protein_mask: (batch_size, protein_seq_len)
        """
        # Drug attends to protein
        drug_enhanced, d2p_attn = self.drug_to_protein(
            drug_features, protein_features, protein_features, protein_mask
        )
        
        # Protein attends to drug
        protein_enhanced, p2d_attn = self.protein_to_drug(
            protein_features, drug_features, drug_features, drug_mask
        )
        
        return drug_enhanced, protein_enhanced, d2p_attn, p2d_attn

