"""
Lyra: The main model combining PGC and S4D layers.

Architecture:
1. Input embedding: Linear projection to model dimension
2. PGC layers: Local feature extraction with gated convolutions
3. S4D layers: Global dependencies via state space models
4. Pooling + decoder: Sequence-level prediction

The design aligns with the mathematical structure of epistasis:
- PGC captures local, low-order interactions
- S4D captures global, higher-order polynomial interactions
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Union

from .s4d import S4D
from .pgc import PGC, RMSNorm


class Lyra(nn.Module):
    """
    Lyra: Efficient Subquadratic Architecture for Biological Sequence Modeling.
    
    Combines Projected Gated Convolutions (PGC) for local epistatic interactions
    with Diagonal State Space Models (S4D) for global polynomial approximation.
    
    Args:
        d_input: Input feature dimension (e.g., 20 for amino acids, 4 for nucleotides)
        d_model: Internal model dimension
        d_output: Output dimension (e.g., 1 for regression, num_classes for classification)
        num_pgc_layers: Number of PGC layers (default: 2)
        num_s4_layers: Number of S4D layers (default: 4)
        d_state: State dimension for S4D (controls polynomial degree, default: 64)
        pgc_expansion: Expansion factor for PGC hidden dimension (default: 1.0)
        pgc_kernel_size: Kernel size for PGC convolution (default: 3)
        dropout: Dropout probability (default: 0.2)
        prenorm: Use pre-normalization (default: True)
        pool: Pooling method - 'mean', 'max', or 'first' (default: 'mean')
        
    Example:
        >>> model = Lyra(d_input=20, d_model=64, d_output=1)
        >>> x = torch.randn(32, 100, 20)  # (batch, seq_len, features)
        >>> y = model(x)  # (batch, 1)
    """
    def __init__(
        self,
        d_input: int,
        d_model: int = 64,
        d_output: int = 1,
        num_pgc_layers: int = 2,
        num_s4_layers: int = 4,
        d_state: int = 64,
        pgc_expansion: float = 1.0,
        pgc_kernel_size: int = 3,
        dropout: float = 0.2,
        prenorm: bool = True,
        pool: str = 'mean',
        s4_lr: float = 0.001,
    ):
        super().__init__()
        self.d_model = d_model
        self.prenorm = prenorm
        self.pool = pool
        
        # Input embedding
        self.encoder = nn.Linear(d_input, d_model)
        
        # PGC layers for local feature extraction
        self.pgc_layers = nn.ModuleList([
            PGC(
                d_model=d_model,
                expansion_factor=pgc_expansion,
                kernel_size=pgc_kernel_size,
                dropout=dropout,
            )
            for _ in range(num_pgc_layers)
        ])
        
        # S4D layers for global dependencies
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        for _ in range(num_s4_layers):
            self.s4_layers.append(
                S4D(
                    d_model=d_model,
                    d_state=d_state,
                    dropout=dropout,
                    transposed=True,
                    lr=s4_lr,
                )
            )
            self.norms.append(RMSNorm(d_model))
            self.dropouts.append(nn.Dropout(dropout))
        
        # Output decoder
        self.final_norm = RMSNorm(d_model)
        self.final_dropout = nn.Dropout(dropout)
        self.decoder = nn.Linear(d_model, d_output)
        
    def forward(
        self, 
        x: torch.Tensor, 
        return_embeddings: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, L, d_input)
            return_embeddings: If True, also return sequence embeddings
            
        Returns:
            y: Output predictions of shape (B, d_output)
            embeddings: (optional) Sequence embeddings of shape (B, L, d_model)
        """
        # Encode input: (B, L, d_input) -> (B, L, d_model)
        x = self.encoder(x)
        
        # Apply PGC layers with residual connections
        for pgc_layer in self.pgc_layers:
            x = x + pgc_layer(x)
        
        # Transpose for S4D: (B, L, d_model) -> (B, d_model, L)
        x = x.transpose(-1, -2)
        
        # Apply S4D layers with residual connections
        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            z = x
            if self.prenorm:
                # Pre-normalization (transpose for RMSNorm which expects last dim)
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)
            
            # Apply S4D block
            z = layer(z)
            
            # Dropout on S4D output
            z = dropout(z)
            
            # Residual connection
            x = z + x
            
            if not self.prenorm:
                # Post-normalization
                x = norm(x.transpose(-1, -2)).transpose(-1, -2)
        
        # Transpose back: (B, d_model, L) -> (B, L, d_model)
        x = x.transpose(-1, -2)
        
        # Store embeddings before pooling
        embeddings = x
        
        # Pooling over sequence length
        if self.pool == 'mean':
            x = x.mean(dim=1)
        elif self.pool == 'max':
            x = x.max(dim=1).values
        elif self.pool == 'first':
            x = x[:, 0]
        else:
            raise ValueError(f"Unknown pooling method: {self.pool}")
        
        # Final norm and decode
        x = self.final_norm(x)
        x = self.final_dropout(x)
        x = self.decoder(x)
        
        if return_embeddings:
            return x, embeddings
        return x
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class LyraForTokenClassification(Lyra):
    """
    Lyra variant for token-level classification (e.g., disorder prediction).
    
    Instead of pooling, outputs predictions for each position in the sequence.
    """
    def __init__(self, *args, **kwargs):
        kwargs['pool'] = 'none'  # Disable pooling
        super().__init__(*args, **kwargs)
        # Override decoder to work on full sequence
        self.decoder = nn.Linear(self.d_model, kwargs.get('d_output', 1))
    
    def forward(
        self, 
        x: torch.Tensor,
        return_embeddings: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for token classification.
        
        Args:
            x: Input tensor of shape (B, L, d_input)
            
        Returns:
            y: Output predictions of shape (B, L, d_output)
        """
        # Encode input
        x = self.encoder(x)
        
        # Apply PGC layers
        for pgc_layer in self.pgc_layers:
            x = x + pgc_layer(x)
        
        # Transpose for S4D
        x = x.transpose(-1, -2)
        
        # Apply S4D layers
        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            z = x
            if self.prenorm:
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)
            z = layer(z)
            z = dropout(z)
            x = z + x
            if not self.prenorm:
                x = norm(x.transpose(-1, -2)).transpose(-1, -2)
        
        # Transpose back
        x = x.transpose(-1, -2)
        embeddings = x
        
        # Apply decoder to each position (no pooling)
        x = self.final_norm(x)
        x = self.final_dropout(x)
        x = self.decoder(x)
        
        if return_embeddings:
            return x, embeddings
        return x
