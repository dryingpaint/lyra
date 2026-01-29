"""
Train Lyra on ProteinGym with official 5-fold cross-validation splits.

Uses the official fold_random_5 splits from ProteinGym for proper benchmarking.

Usage:
    python train_proteingym_cv.py --assay GFP_AEQVI_Sarkisyan_2016
    python train_proteingym_cv.py --assay GFP_AEQVI_Sarkisyan_2016 --fold 0
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
        encoded = torch.zeros(self.max_len, len(AMINO_ACIDS))
        for i, aa in enumerate(seq[:self.max_len]):
            if aa in AA_TO_IDX:
                encoded[i, AA_TO_IDX[aa]] = 1.0
        return encoded, self.labels[idx]


def train_epoch(model, dataloader, optimizer, criterion, device, scaler=None):
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
    model.eval()
    all_preds, all_labels = [], []
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


def train_fold(
    df: pd.DataFrame,
    fold: int,
    assay_name: str,
    output_dir: Path,
    d_model: int = 64,
    d_state: int = 64,
    num_s4_layers: int = 4,
    num_pgc_layers: int = 2,
    batch_size: int = 64,
    epochs: int = 50,
    lr: float = 1e-3,
    device: str = "cuda",
):
    """Train on a single fold."""
    
    # Split by fold
    train_df = df[df['fold_random_5'] != fold]
    test_df = df[df['fold_random_5'] == fold]
    
    print(f"  Fold {fold}: Train={len(train_df)}, Test={len(test_df)}")
    
    # Extract sequences and scores
    train_seqs = train_df['mutated_sequence'].tolist()
    test_seqs = test_df['mutated_sequence'].tolist()
    
    # Normalize scores using training set statistics
    train_mean = train_df['DMS_score'].mean()
    train_std = train_df['DMS_score'].std()
    train_scores = (train_df['DMS_score'].values - train_mean) / (train_std + 1e-8)
    test_scores = (test_df['DMS_score'].values - train_mean) / (train_std + 1e-8)
    
    seq_len = len(train_seqs[0])
    
    # Create datasets
    train_dataset = ProteinDataset(train_seqs, train_scores, max_len=seq_len)
    test_dataset = ProteinDataset(test_seqs, test_scores, max_len=seq_len)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                             num_workers=4, pin_memory=True)
    
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
    
    # Training setup
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()
    scaler = torch.amp.GradScaler('cuda')
    
    best_spearman = -1
    best_epoch = 0
    
    pbar = tqdm(range(epochs), desc=f"  Fold {fold}", leave=False)
    for epoch in pbar:
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, scaler)
        test_mse, test_spearman = evaluate(model, test_loader, device)
        scheduler.step()
        
        if test_spearman > best_spearman:
            best_spearman = test_spearman
            best_epoch = epoch
            torch.save(model.state_dict(), output_dir / f"{assay_name}_fold{fold}_best.pt")
        
        pbar.set_postfix({"spearman": f"{test_spearman:.4f}", "best": f"{best_spearman:.4f}"})
    
    return {
        "fold": fold,
        "best_spearman": best_spearman,
        "best_epoch": best_epoch,
        "n_train": len(train_seqs),
        "n_test": len(test_seqs),
    }


def main():
    parser = argparse.ArgumentParser(description="Train Lyra on ProteinGym with official CV splits")
    parser.add_argument("--assay", type=str, required=True, help="Assay name")
    parser.add_argument("--fold", type=int, default=None, help="Single fold to run (0-4), or all if not specified")
    parser.add_argument("--data-dir", type=str, default="./data/proteingym", help="Data directory")
    parser.add_argument("--output-dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--d-state", type=int, default=64)
    parser.add_argument("--num-s4-layers", type=int, default=4)
    parser.add_argument("--num-pgc-layers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load CV folds file
    cv_dir = data_dir / "cv_folds_singles_substitutions"
    cv_file = cv_dir / f"{args.assay}.csv"
    
    if not cv_file.exists():
        print(f"CV file not found: {cv_file}")
        print("Available assays:")
        for f in sorted(cv_dir.glob("*.csv"))[:20]:
            print(f"  {f.stem}")
        return
    
    df = pd.read_csv(cv_file)
    seq_len = len(df['mutated_sequence'].iloc[0])
    
    print(f"\n{'='*60}")
    print(f"Assay: {args.assay}")
    print(f"Total variants: {len(df)}")
    print(f"Sequence length: {seq_len}")
    print(f"{'='*60}\n")
    
    # Run folds
    folds = [args.fold] if args.fold is not None else range(5)
    results = []
    
    for fold in folds:
        result = train_fold(
            df=df,
            fold=fold,
            assay_name=args.assay,
            output_dir=output_dir,
            d_model=args.d_model,
            d_state=args.d_state,
            num_s4_layers=args.num_s4_layers,
            num_pgc_layers=args.num_pgc_layers,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            device=device,
        )
        results.append(result)
        print(f"  Fold {fold}: Spearman = {result['best_spearman']:.4f}")
    
    # Summary
    spearmans = [r['best_spearman'] for r in results]
    mean_spearman = np.mean(spearmans)
    std_spearman = np.std(spearmans)
    
    print(f"\n{'='*60}")
    print(f"Results for {args.assay}")
    print(f"{'='*60}")
    print(f"Mean Spearman: {mean_spearman:.4f} ± {std_spearman:.4f}")
    print(f"Per-fold: {[f'{s:.4f}' for s in spearmans]}")
    
    # Save results
    summary = {
        "assay": args.assay,
        "mean_spearman": mean_spearman,
        "std_spearman": std_spearman,
        "fold_results": results,
        "config": {
            "d_model": args.d_model,
            "d_state": args.d_state,
            "num_s4_layers": args.num_s4_layers,
            "num_pgc_layers": args.num_pgc_layers,
            "epochs": args.epochs,
            "lr": args.lr,
        }
    }
    
    with open(output_dir / f"{args.assay}_cv_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to {output_dir / f'{args.assay}_cv_results.json'}")


if __name__ == "__main__":
    main()
