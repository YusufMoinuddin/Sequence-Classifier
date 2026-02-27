# models/transformer_classifier.py
import torch
import torch.nn as nn
from typing import Optional


class DNATransformerClassifier(nn.Module):
    """
    Transformer-based classifier for fixed-length DNA sequences.

    Assumes input is token IDs in [0, vocab_size):
        x: LongTensor of shape (batch_size, seq_len)

    Optional QM features:
        qm_feats: FloatTensor of shape (batch_size, qm_dim)

    Output:
        logits: FloatTensor of shape (batch_size,)
    """
    def __init__(
        self,
        vocab_size: int = 4,     # A,C,G,T
        d_model: int = 128,
        n_heads: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        max_seq_len: int = 8,
        dropout: float = 0.1,
        # --- NEW (QM feature support) ---
        use_qm: bool = False,
        qm_dim: int = 4,         # qchem_style_* columns are 4 features
        qm_embed_dim: int = 32,  # small embedding for QM vector
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # --- NEW ---
        self.use_qm = use_qm
        self.qm_dim = qm_dim
        self.qm_embed_dim = qm_embed_dim

        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # Positional embeddings
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        # Transformer encoder (stacked self-attention + FFN)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,   # (B,L,d_model)
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # Classification head
        self.dropout = nn.Dropout(dropout)

        # --- NEW: QM projection + updated classifier input dim ---
        if self.use_qm:
            self.qm_proj = nn.Sequential(
                nn.Linear(qm_dim, qm_embed_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            fc_in = d_model + qm_embed_dim
        else:
            self.qm_proj = None
            fc_in = d_model

        self.fc_out = nn.Linear(fc_in, 1)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.token_embedding.weight)
        nn.init.xavier_uniform_(self.pos_embedding.weight)
        nn.init.xavier_uniform_(self.fc_out.weight)
        if self.fc_out.bias is not None:
            nn.init.zeros_(self.fc_out.bias)

        # --- NEW: init qm_proj weights if present ---
        if self.qm_proj is not None:
            for m in self.qm_proj:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        qm_feats: Optional[torch.Tensor] = None,   # --- NEW ---
    ) -> torch.Tensor:
        """
        x: LongTensor (B, L)
        attention_mask: optional BoolTensor (B, L), True for real tokens, False for padding.
                        For fixed 8-mer motifs you can ignore this (no padding).

        qm_feats: optional FloatTensor (B, qm_dim). Required if use_qm=True.
        """
        batch_size, seq_len = x.shape
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"seq_len={seq_len} > max_seq_len={self.max_seq_len}. "
                f"Increase max_seq_len in DNATransformerClassifier."
            )

        if self.use_qm:
            if qm_feats is None:
                raise ValueError("Model created with use_qm=True, but qm_feats was None.")
            if qm_feats.dim() != 2 or qm_feats.size(0) != batch_size or qm_feats.size(1) != self.qm_dim:
                raise ValueError(
                    f"qm_feats must have shape (B, {self.qm_dim}). Got {tuple(qm_feats.shape)}."
                )

        # Token embedding
        token_emb = self.token_embedding(x)  # (B,L,d_model)

        # Positional embedding: positions 0..L-1
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)  # (1,L)
        pos_emb = self.pos_embedding(positions)                          # (1,L,d_model)

        h = token_emb + pos_emb                                          # (B,L,d_model)

        src_key_padding_mask = None
        if attention_mask is not None:
            # attention_mask: True=real tokens, False=padding
            # src_key_padding_mask: True=padding
            src_key_padding_mask = ~attention_mask

        # Encode
        h_enc = self.encoder(h, src_key_padding_mask=src_key_padding_mask)  # (B,L,d_model)

        # Global average pooling over sequence length
        pooled = h_enc.mean(dim=1)  # (B,d_model)
        pooled = self.dropout(pooled)

        # --- NEW: fuse QM embedding ---
        if self.use_qm:
            qm_emb = self.qm_proj(qm_feats)               # (B, qm_embed_dim)
            pooled = torch.cat([pooled, qm_emb], dim=-1)  # (B, d_model + qm_embed_dim)

        logits = self.fc_out(pooled).squeeze(-1)  # (B,)
        return logits