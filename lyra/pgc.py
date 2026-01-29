"""
Projected Gated Convolution (PGC)

Captures local epistatic interactions through:
1. Depthwise convolution for local pattern extraction
2. Linear projection for channel mixing  
3. Hadamard (element-wise) product for gating

The multiplicative gating explicitly encodes second-order interactions.
Stacking PGC layers enables higher-order dependencies.
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    
    More efficient than LayerNorm as it doesn't require mean computation.
    """
    def __init__(self, d_model: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class PGC(nn.Module):
    """
    Projected Gated Convolution layer.
    
    Combines depthwise convolution (local features) with linear projection
    (channel mixing) through multiplicative gating. This explicitly captures
    second-order epistatic interactions between local sequence features and
    cross-channel representations.
    
    Args:
        d_model: Input/output dimension
        expansion_factor: Expansion factor for hidden dimension (default: 1.0)
        kernel_size: Convolution kernel size (default: 3)
        dropout: Dropout probability (default: 0.0)
    """
    def __init__(
        self,
        d_model: int,
        expansion_factor: float = 1.0,
        kernel_size: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.expansion_factor = expansion_factor
        hidden_dim = int(d_model * expansion_factor)
        
        # Depthwise convolution for local pattern extraction
        # groups=d_model means each channel has its own filter
        self.conv = nn.Conv1d(
            d_model, 
            d_model, 
            kernel_size=kernel_size, 
            padding=kernel_size // 2,
            groups=d_model
        )
        
        # Project to 2x hidden dim for gating (x and v branches)
        self.in_proj = nn.Linear(d_model, hidden_dim * 2)
        
        # Normalization layers
        self.in_norm = RMSNorm(hidden_dim * 2)
        self.out_norm = RMSNorm(hidden_dim)
        
        # Project back to model dimension
        self.out_proj = nn.Linear(hidden_dim, d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        Apply PGC layer.
        
        Args:
            u: Input tensor of shape (B, L, D)
            
        Returns:
            Output tensor of shape (B, L, D)
            
        The gating mechanism:
            xv = proj(u)           # Project and split
            x, v = split(xv)       # x for conv, v for gate
            x_conv = conv(x)       # Local features via depthwise conv
            gate = v * x_conv      # Second-order interaction!
            out = proj(gate)       # Project back
        """
        # Project to 2x hidden dim and normalize
        xv = self.in_norm(self.in_proj(u))  # (B, L, 2*hidden)
        
        # Split into two branches
        x, v = xv.chunk(2, dim=-1)  # Each: (B, L, hidden)
        
        # Apply depthwise convolution to x branch
        # Conv1d expects (B, C, L), so transpose
        x_conv = self.conv(x.transpose(-1, -2)).transpose(-1, -2)  # (B, L, hidden)
        
        # Multiplicative gating: this is where 2nd-order interactions happen
        # v captures "what to attend to", x_conv captures "local patterns"
        gate = v * x_conv  # (B, L, hidden)
        
        # Project back and normalize
        out = self.out_norm(self.out_proj(gate))  # (B, L, D)
        out = self.dropout(out)
        
        return out


class PGCStack(nn.Module):
    """
    Stack of PGC layers with residual connections.
    
    Multiple PGC layers enable capturing progressively higher-order
    epistatic interactions.
    """
    def __init__(
        self,
        d_model: int,
        num_layers: int,
        expansion_factor: float = 1.0,
        kernel_size: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            PGC(d_model, expansion_factor, kernel_size, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = x + layer(x)  # Residual connection
        return x
