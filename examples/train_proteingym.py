"""
Train Lyra on ProteinGym DMS benchmarks.

ProteinGym contains 217 Deep Mutational Scanning assays with ~2.7M mutations.
This script trains Lyra on individual assays and evaluates using Spearman correlation.

Usage:
    python train_proteingym.py --assay GFP_AEQVI_Sarkisyan_2016
    python train_proteingym.py --assay BRCA1_HUMAN_RING  
    python train_proteingym.py --all  # Run all assays
"""

import argparse
import os
import sys
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from scipy.stats import spearmanr
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from lyra import Lyra


# Standard amino acid alphabet
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}


class ProteinDataset(Dataset):
    """Dataset for protein sequences with fitness labels."""
    
    def __init__(self, sequences: list[str], labels: np.ndarray, max_len: int = None):
        self.sequences = sequences
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.max_len = max_len or max(len(s) for s in sequences)
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        # One-hot encode
        encoded = torch.zeros(self.max_len, len(AMINO_ACIDS))
        for i, aa in enumerate(seq[:self.max_len]):
            if aa in AA_TO_IDX:
                encoded[i, AA_TO_IDX[aa]] = 1.0
        return encoded, self.labels[idx]


def download_proteingym(data_dir: Path):
    """Download ProteinGym substitution benchmark."""
    import urllib.request
    import zipfile
    
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # ProteinGym v1.3 substitution benchmark URL
    url = "https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/DMS_ProteinGym_substitutions.zip"
    zip_path = data_dir / "DMS_ProteinGym_substitutions.zip"
    extract_dir = data_dir / "substitutions"
    
    if extract_dir.exists() and any(extract_dir.glob("*.csv")):
        print(f"ProteinGym data already exists at {extract_dir}")
        return extract_dir
    
    print(f"Downloading ProteinGym from {url}...")
    urllib.request.urlretrieve(url, zip_path)
    
    print("Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    
    zip_path.unlink()  # Clean up zip
    print(f"Data extracted to {extract_dir}")
    return extract_dir


def load_assay(data_dir: Path, assay_name: str) -> tuple[pd.DataFrame, str]:
    """Load a single DMS assay from ProteinGym."""
    # Find the CSV file
    csv_files = list(data_dir.glob(f"*{assay_name}*.csv"))
    if not csv_files:
        # Try exact match
        csv_files = list(data_dir.glob(f"{assay_name}.csv"))
    if not csv_files:
        raise ValueError(f"Assay {assay_name} not found. Available: {[f.stem for f in data_dir.glob('*.csv')][:10]}...")
    
    csv_path = csv_files[0]
    df = pd.read_csv(csv_path)
    
    # Get reference sequence from first mutated sequence (approximately)
    # In practice, we use mutated_sequence directly
    return df, csv_path.stem


def train_epoch(model, dataloader, optimizer, criterion, device, scaler=None):
    """Train for one epoch with optional mixed precision."""
    model.train()
    total_loss = 0
    
    for X, y in dataloader:
        X, y = X.to(device), y.to(device).unsqueeze(-1)
        
        optimizer.zero_grad()
        
        if scaler is not None:
            with torch.amp.autocast('cuda'):
                pred = model(X)
                loss = criterion(pred, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate(model, dataloader, device) -> tuple[float, float]:
    """Evaluate model, return (MSE, Spearman correlation)."""
    model.eval()
    all_preds = []
    all_labels = []
    
    for X, y in dataloader:
        X = X.to(device)
        pred = model(X).cpu().numpy().flatten()
        all_preds.extend(pred)
        all_labels.extend(y.numpy().flatten())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    mse = np.mean((all_preds - all_labels) ** 2)
    spearman = spearmanr(all_preds, all_labels).correlation
    
    return mse, spearman


def train_on_assay(
    assay_name: str,
    data_dir: Path,
    output_dir: Path,
    d_model: int = 64,
    d_state: int = 64,
    num_s4_layers: int = 4,
    num_pgc_layers: int = 2,
    batch_size: int = 32,
    epochs: int = 50,
    lr: float = 1e-3,
    device: str = "cuda",
    use_wandb: bool = False,
):
    """Train Lyra on a single ProteinGym assay."""
    
    print(f"\n{'='*60}")
    print(f"Training on: {assay_name}")
    print(f"{'='*60}")
    
    # Load data
    df, full_name = load_assay(data_dir, assay_name)
    print(f"Loaded {len(df)} variants from {full_name}")
    
    # Extract sequences and scores
    sequences = df['mutated_sequence'].tolist()
    scores = df['DMS_score'].values
    
    # Normalize scores
    scores = (scores - scores.mean()) / (scores.std() + 1e-8)
    
    # Get sequence length
    seq_len = len(sequences[0])
    print(f"Sequence length: {seq_len}")
    
    # Train/test split (80/20, maintaining order for reproducibility)
    n_train = int(0.8 * len(sequences))
    indices = np.random.RandomState(42).permutation(len(sequences))
    train_idx, test_idx = indices[:n_train], indices[n_train:]
    
    train_seqs = [sequences[i] for i in train_idx]
    test_seqs = [sequences[i] for i in test_idx]
    train_scores = scores[train_idx]
    test_scores = scores[test_idx]
    
    print(f"Train: {len(train_seqs)}, Test: {len(test_seqs)}")
    
    # Create datasets
    train_dataset = ProteinDataset(train_seqs, train_scores, max_len=seq_len)
    test_dataset = ProteinDataset(test_seqs, test_scores, max_len=seq_len)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)
    
    # Create model
    model = Lyra(
        d_input=20,
        d_model=d_model,
        d_output=1,
        num_pgc_layers=num_pgc_layers,
        num_s4_layers=num_s4_layers,
        d_state=d_state,
        dropout=0.1,
    ).to(device)
    
    n_params = model.count_parameters()
    print(f"Model parameters: {n_params:,}")
    
    # Training setup
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()
    scaler = torch.amp.GradScaler('cuda')
    
    # W&B logging
    if use_wandb:
        import wandb
        wandb.init(
            project="lyra-proteingym",
            name=f"{assay_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                "assay": assay_name,
                "d_model": d_model,
                "d_state": d_state,
                "num_s4_layers": num_s4_layers,
                "num_pgc_layers": num_pgc_layers,
                "batch_size": batch_size,
                "epochs": epochs,
                "lr": lr,
                "n_params": n_params,
                "n_train": len(train_seqs),
                "n_test": len(test_seqs),
                "seq_len": seq_len,
            }
        )
    
    # Training loop
    best_spearman = -1
    best_epoch = 0
    
    pbar = tqdm(range(epochs), desc="Training")
    for epoch in pbar:
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, scaler)
        test_mse, test_spearman = evaluate(model, test_loader, device)
        scheduler.step()
        
        if test_spearman > best_spearman:
            best_spearman = test_spearman
            best_epoch = epoch
            # Save best model
            torch.save(model.state_dict(), output_dir / f"{assay_name}_best.pt")
        
        pbar.set_postfix({
            "loss": f"{train_loss:.4f}",
            "spearman": f"{test_spearman:.4f}",
            "best": f"{best_spearman:.4f}"
        })
        
        if use_wandb:
            wandb.log({
                "train_loss": train_loss,
                "test_mse": test_mse,
                "test_spearman": test_spearman,
                "best_spearman": best_spearman,
                "lr": scheduler.get_last_lr()[0],
                "epoch": epoch,
            })
    
    print(f"\nBest Spearman: {best_spearman:.4f} at epoch {best_epoch}")
    
    if use_wandb:
        wandb.finish()
    
    return {
        "assay": assay_name,
        "best_spearman": best_spearman,
        "best_epoch": best_epoch,
        "n_params": n_params,
        "n_train": len(train_seqs),
        "n_test": len(test_seqs),
        "seq_len": seq_len,
    }


def main():
    parser = argparse.ArgumentParser(description="Train Lyra on ProteinGym")
    parser.add_argument("--assay", type=str, help="Assay name (e.g., GFP_AEQVI_Sarkisyan_2016)")
    parser.add_argument("--all", action="store_true", help="Run all assays")
    parser.add_argument("--data-dir", type=str, default="./data/proteingym", help="Data directory")
    parser.add_argument("--output-dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--d-model", type=int, default=64, help="Model dimension")
    parser.add_argument("--d-state", type=int, default=64, help="State dimension")
    parser.add_argument("--num-s4-layers", type=int, default=4, help="Number of S4D layers")
    parser.add_argument("--num-pgc-layers", type=int, default=2, help="Number of PGC layers")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--wandb", action="store_true", help="Use W&B logging")
    parser.add_argument("--download", action="store_true", help="Download data if not present")
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Download data if needed
    if args.download or not (data_dir / "substitutions").exists():
        substitutions_dir = download_proteingym(data_dir)
    else:
        substitutions_dir = data_dir / "substitutions"
        # Check for nested structure
        if not any(substitutions_dir.glob("*.csv")):
            # Try looking in subdirectories
            for subdir in substitutions_dir.iterdir():
                if subdir.is_dir() and any(subdir.glob("*.csv")):
                    substitutions_dir = subdir
                    break
    
    if args.all:
        # Run all assays
        csv_files = sorted(substitutions_dir.glob("*.csv"))
        print(f"Found {len(csv_files)} assays")
        
        results = []
        for csv_path in csv_files:
            assay_name = csv_path.stem
            try:
                result = train_on_assay(
                    assay_name=assay_name,
                    data_dir=substitutions_dir,
                    output_dir=output_dir,
                    d_model=args.d_model,
                    d_state=args.d_state,
                    num_s4_layers=args.num_s4_layers,
                    num_pgc_layers=args.num_pgc_layers,
                    batch_size=args.batch_size,
                    epochs=args.epochs,
                    lr=args.lr,
                    device=device,
                    use_wandb=args.wandb,
                )
                results.append(result)
            except Exception as e:
                print(f"Error on {assay_name}: {e}")
                results.append({"assay": assay_name, "error": str(e)})
        
        # Save summary
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_dir / "proteingym_results.csv", index=False)
        
        # Print summary
        valid_results = [r for r in results if "best_spearman" in r]
        if valid_results:
            avg_spearman = np.mean([r["best_spearman"] for r in valid_results])
            print(f"\n{'='*60}")
            print(f"Summary: {len(valid_results)} assays completed")
            print(f"Average Spearman: {avg_spearman:.4f}")
            print(f"{'='*60}")
    
    elif args.assay:
        result = train_on_assay(
            assay_name=args.assay,
            data_dir=substitutions_dir,
            output_dir=output_dir,
            d_model=args.d_model,
            d_state=args.d_state,
            num_s4_layers=args.num_s4_layers,
            num_pgc_layers=args.num_pgc_layers,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            device=device,
            use_wandb=args.wandb,
        )
        
        # Save result
        with open(output_dir / f"{args.assay}_result.json", "w") as f:
            json.dump(result, f, indent=2)
    
    else:
        # List available assays
        csv_files = sorted(substitutions_dir.glob("*.csv"))
        print(f"Available assays ({len(csv_files)}):")
        for f in csv_files[:20]:
            print(f"  - {f.stem}")
        if len(csv_files) > 20:
            print(f"  ... and {len(csv_files) - 20} more")
        print("\nUse --assay <name> to train on a specific assay, or --all to run all.")


if __name__ == "__main__":
    main()
