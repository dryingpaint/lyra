"""
S4D: Diagonal State Space Model

Implements the S4D layer from "On the Parameterization and Initialization of 
Diagonal State Space Models" (Gu et al., 2022).

The key insight is that diagonal SSMs can efficiently approximate polynomials,
making them well-suited for modeling epistatic interactions in biological sequences.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


class DropoutNd(nn.Module):
    """
    Dropout for N-dimensional tensors with optional tied masks across spatial dims.
    """
    def __init__(self, p: float = 0.5, tie: bool = True, transposed: bool = True):
        super().__init__()
        if p < 0 or p >= 1:
            raise ValueError(f"dropout probability must be in [0, 1), got {p}")
        self.p = p
        self.tie = tie
        self.transposed = transposed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.p > 0:
            if not self.transposed:
                x = rearrange(x, 'b ... d -> b d ...')
            # Tie mask across spatial dimensions if requested
            mask_shape = x.shape[:2] + (1,) * (x.ndim - 2) if self.tie else x.shape
            mask = torch.rand(*mask_shape, device=x.device) < (1.0 - self.p)
            x = x * mask * (1.0 / (1 - self.p))
            if not self.transposed:
                x = rearrange(x, 'b d ... -> b ... d')
        return x


class S4DKernel(nn.Module):
    """
    Generates the convolution kernel for S4D.
    
    The kernel is parameterized by:
    - log_dt: log of discretization timestep
    - log_A_real: log of decay rates (real part of A)
    - A_imag: frequencies (imaginary part of A)  
    - C: output mixing coefficients
    
    The initialization places eigenvalues at evenly spaced frequencies,
    corresponding to a discrete Fourier basis - ideal for capturing
    polynomial/epistatic interactions.
    """
    def __init__(
        self, 
        d_model: int, 
        d_state: int = 64, 
        dt_min: float = 0.001, 
        dt_max: float = 0.1, 
        lr: float = None
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        # Initialize discretization timestep (learnable)
        log_dt = torch.rand(d_model) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        
        # Initialize C (output coefficients) - complex valued
        C = torch.randn(d_model, d_state // 2, dtype=torch.cfloat)
        self.C = nn.Parameter(torch.view_as_real(C))
        
        # Initialize A matrix (diagonal, complex)
        # Real part: decay rates (initialized to 0.5)
        # Imag part: frequencies (evenly spaced, 0 to π*(N/2-1))
        log_A_real = torch.log(0.5 * torch.ones(d_model, d_state // 2))
        A_imag = math.pi * repeat(torch.arange(d_state // 2), 'n -> h n', h=d_model)
        
        self._register_param("log_dt", log_dt, lr)
        self._register_param("log_A_real", log_A_real, lr)
        self._register_param("A_imag", A_imag, lr)

    def _register_param(self, name: str, tensor: torch.Tensor, lr: float = None):
        """Register parameter with optional custom learning rate."""
        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))
            optim = {"weight_decay": 0.0}
            if lr is not None:
                optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)

    def forward(self, L: int) -> torch.Tensor:
        """
        Generate convolution kernel of length L.
        
        Args:
            L: Sequence length
            
        Returns:
            Kernel of shape (d_model, L)
        """
        dt = torch.exp(self.log_dt)  # (H,)
        C = torch.view_as_complex(self.C)  # (H, N/2)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag  # (H, N/2)
        
        # Discretize: A_bar = exp(A * dt)
        dtA = A * dt.unsqueeze(-1)  # (H, N/2)
        
        # Compute kernel via Vandermonde matrix
        # K[h, l] = sum_n C[h,n] * A_bar[h,n]^l
        K = dtA.unsqueeze(-1) * torch.arange(L, device=A.device)  # (H, N/2, L)
        C = C * (torch.exp(dtA) - 1.0) / A  # Discretized C
        
        # Sum over hidden state dimension, take real part, multiply by 2 for conjugate pairs
        K = 2 * torch.einsum('hn, hnl -> hl', C, torch.exp(K)).real
        
        return K


class S4D(nn.Module):
    """
    Diagonal State Space Model layer.
    
    Implements efficient sequence modeling via FFT-based convolution with
    a learned kernel. The kernel is parameterized to capture polynomial
    interactions, making it suitable for modeling epistasis.
    
    Args:
        d_model: Model dimension (number of channels)
        d_state: State dimension (controls polynomial degree / epistatic order)
        dropout: Dropout probability
        transposed: If True, expects input shape (B, H, L), else (B, L, H)
        dt_min, dt_max: Range for discretization timestep initialization
        lr: Learning rate for SSM parameters (None uses default)
    """
    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        dropout: float = 0.0,
        transposed: bool = True,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        lr: float = None,
        **kernel_args
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_output = d_model
        self.transposed = transposed
        
        # Skip connection coefficient
        self.D = nn.Parameter(torch.randn(d_model))
        
        # Kernel generator
        self.kernel = S4DKernel(
            d_model, 
            d_state=d_state, 
            dt_min=dt_min, 
            dt_max=dt_max, 
            lr=lr,
            **kernel_args
        )
        
        # Activation and regularization
        self.activation = nn.GELU()
        self.dropout = DropoutNd(dropout) if dropout > 0.0 else nn.Identity()
        
        # Output projection with GLU
        self.output_linear = nn.Sequential(
            nn.Conv1d(d_model, 2 * d_model, kernel_size=1),
            nn.GLU(dim=-2),
        )

    def forward(self, u: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Apply S4D layer.
        
        Args:
            u: Input tensor of shape (B, H, L) if transposed else (B, L, H)
            
        Returns:
            Output tensor of same shape as input
        """
        if not self.transposed:
            u = u.transpose(-1, -2)
        
        L = u.size(-1)
        
        # Generate kernel and perform FFT convolution
        k = self.kernel(L=L)  # (H, L)
        
        # FFT convolution (circular, padded to avoid wraparound)
        k_f = torch.fft.rfft(k, n=2 * L)  # (H, L+1)
        u_f = torch.fft.rfft(u, n=2 * L)  # (B, H, L+1)
        y = torch.fft.irfft(u_f * k_f, n=2 * L)[..., :L]  # (B, H, L)
        
        # Skip connection
        y = y + u * self.D.unsqueeze(-1)
        
        # Activation + dropout + output projection
        y = self.dropout(self.activation(y))
        y = self.output_linear(y)
        
        if not self.transposed:
            y = y.transpose(-1, -2)
        
        return y
