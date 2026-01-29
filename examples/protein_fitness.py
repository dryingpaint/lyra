"""
Example: Protein Fitness Prediction with Lyra

Demonstrates training Lyra on a synthetic protein fitness dataset.
In practice, you would use real datasets like:
- GFP fluorescence (Sarkisyan et al.)
- AAV packaging (Ogden et al.)
- Antibody binding (Mason et al.)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import sys
sys.path.insert(0, '..')
from lyra import Lyra


def one_hot_encode(sequences: list[str], alphabet: str = "ACDEFGHIKLMNPQRSTVWY") -> torch.Tensor:
    """One-hot encode protein sequences."""
    aa_to_idx = {aa: i for i, aa in enumerate(alphabet)}
    encoded = []
    for seq in sequences:
        seq_encoded = torch.zeros(len(seq), len(alphabet))
        for i, aa in enumerate(seq):
            if aa in aa_to_idx:
                seq_encoded[i, aa_to_idx[aa]] = 1.0
        encoded.append(seq_encoded)
    return torch.stack(encoded)


def generate_synthetic_data(
    n_samples: int = 1000,
    seq_length: int = 50,
    n_amino_acids: int = 20,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate synthetic protein fitness data with epistatic effects.
    
    The fitness function has:
    - Additive effects (1st order)
    - Pairwise interactions (2nd order epistasis)
    - Some 3rd order terms
    """
    # Random one-hot encoded sequences
    X = torch.zeros(n_samples, seq_length, n_amino_acids)
    for i in range(n_samples):
        for j in range(seq_length):
            aa = torch.randint(0, n_amino_acids, (1,)).item()
            X[i, j, aa] = 1.0
    
    # Flatten for computing fitness
    X_flat = X.view(n_samples, -1)
    
    # Additive effects (1st order)
    w1 = torch.randn(seq_length * n_amino_acids) * 0.1
    y = X_flat @ w1
    
    # Pairwise epistasis (2nd order) - sparse
    n_pairs = 50
    for _ in range(n_pairs):
        i, j = torch.randint(0, seq_length * n_amino_acids, (2,))
        weight = torch.randn(1).item() * 0.5
        y += weight * X_flat[:, i] * X_flat[:, j]
    
    # Some 3rd order terms
    for _ in range(10):
        i, j, k = torch.randint(0, seq_length * n_amino_acids, (3,))
        weight = torch.randn(1).item() * 0.3
        y += weight * X_flat[:, i] * X_flat[:, j] * X_flat[:, k]
    
    # Add noise
    y += torch.randn(n_samples) * 0.1
    
    return X, y.unsqueeze(-1)


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = criterion(pred, y)
            total_loss += loss.item()
    return total_loss / len(dataloader)


def main():
    # Config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Generate data
    print("Generating synthetic epistatic fitness data...")
    X_train, y_train = generate_synthetic_data(n_samples=2000)
    X_test, y_test = generate_synthetic_data(n_samples=500)
    
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Create model
    model = Lyra(
        d_input=20,           # 20 amino acids
        d_model=64,           # Internal dimension
        d_output=1,           # Fitness score
        num_pgc_layers=2,     # Local interactions
        num_s4_layers=4,      # Global interactions
        d_state=64,           # Polynomial degree
        dropout=0.1,
    ).to(device)
    
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Training setup
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    criterion = nn.MSELoss()
    
    # Train
    print("\nTraining...")
    for epoch in range(50):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        test_loss = evaluate(model, test_loader, criterion, device)
        scheduler.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")
    
    print("\nDone!")
    print(f"Final test MSE: {test_loss:.4f}")


if __name__ == "__main__":
    main()
