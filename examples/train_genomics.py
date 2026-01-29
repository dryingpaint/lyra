"""
Train Lyra on Genomics Benchmark tasks.

Tasks include:
- Human enhancers (cohn, ensembl)  
- Human regulatory (ensembl)
- Coding vs intergenics
- Human OCR (open chromatin regions)
- Drosophila enhancers
- Demo datasets (human/mouse)

Usage:
    python train_genomics.py --task human_enhancers_cohn
    python train_genomics.py --all
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from lyra import Lyra

# Nucleotide encoding
NUC_TO_IDX = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}


class GenomicsDataset(Dataset):
    """Dataset for DNA sequences."""
    
    def __init__(self, sequences, labels, max_len=None):
        self.sequences = sequences
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.max_len = max_len or max(len(s) for s in sequences)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx].upper()
        # One-hot encode (5 channels: A, C, G, T, N)
        encoded = torch.zeros(self.max_len, 5)
        for i, nuc in enumerate(seq[:self.max_len]):
            if nuc in NUC_TO_IDX:
                encoded[i, NUC_TO_IDX[nuc]] = 1.0
            else:
                encoded[i, 4] = 1.0  # Unknown -> N
        return encoded, self.labels[idx]


def train_epoch(model, dataloader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss = 0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        if scaler:
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
def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    for X, y in dataloader:
        X = X.to(device)
        pred = model(X).argmax(dim=-1).cpu().numpy()
        all_preds.extend(pred)
        all_labels.extend(y.numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    mcc = matthews_corrcoef(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    return acc, mcc, f1


def load_genomics_benchmark(task_name):
    """Load a task from genomic-benchmarks."""
    from genomic_benchmarks.loc2seq import download_dataset
    from genomic_benchmarks.data_check import is_downloaded
    
    # Download if needed
    if not is_downloaded(task_name):
        download_dataset(task_name)
    
    # Load train/test
    from genomic_benchmarks.loc2seq.with_splits import get_train_test_data
    train_seqs, train_labels, test_seqs, test_labels = get_train_test_data(task_name)
    
    # Convert to lists
    train_seqs = [str(s) for s in train_seqs]
    test_seqs = [str(s) for s in test_seqs]
    train_labels = list(train_labels)
    test_labels = list(test_labels)
    
    n_classes = len(set(train_labels))
    
    return train_seqs, train_labels, test_seqs, test_labels, n_classes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, help="Task name")
    parser.add_argument("--all", action="store_true", help="Run all tasks")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--d-state", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Available tasks
    tasks = [
        "human_enhancers_cohn",
        "human_enhancers_ensembl", 
        "human_ensembl_regulatory",
        "human_nontata_promoters",
        "human_ocr_ensembl",
        "drosophila_enhancers_stark",
        "demo_coding_vs_intergenomic_seqs",
        "demo_human_or_worm",
    ]
    
    if args.all:
        run_tasks = tasks
    elif args.task:
        run_tasks = [args.task]
    else:
        print("Available tasks:")
        for t in tasks:
            print(f"  - {t}")
        return
    
    results = []
    
    for task_name in run_tasks:
        print(f"\n{'='*60}")
        print(f"Task: {task_name}")
        print(f"{'='*60}")
        
        try:
            # Load data
            train_seqs, train_labels, test_seqs, test_labels, n_classes = load_genomics_benchmark(task_name)
            seq_len = len(train_seqs[0])
            
            print(f"Train: {len(train_seqs)}, Test: {len(test_seqs)}")
            print(f"Sequence length: {seq_len}, Classes: {n_classes}")
            
            # Create datasets
            train_dataset = GenomicsDataset(train_seqs, train_labels, max_len=seq_len)
            test_dataset = GenomicsDataset(test_seqs, test_labels, max_len=seq_len)
            
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True)
            
            # Create model
            model = Lyra(
                d_input=5,  # A, C, G, T, N
                d_model=args.d_model,
                d_output=n_classes,
                num_pgc_layers=2,
                num_s4_layers=4,
                d_state=args.d_state,
                dropout=0.1,
            ).to(device)
            
            print(f"Parameters: {model.count_parameters():,}")
            
            # Training
            optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
            criterion = nn.CrossEntropyLoss()
            scaler = torch.amp.GradScaler('cuda')
            
            best_acc = 0
            best_mcc = 0
            
            pbar = tqdm(range(args.epochs), desc="Training")
            for epoch in pbar:
                train_loss = train_epoch(model, train_loader, optimizer, criterion, device, scaler)
                acc, mcc, f1 = evaluate(model, test_loader, device)
                scheduler.step()
                
                if acc > best_acc:
                    best_acc = acc
                    best_mcc = mcc
                
                pbar.set_postfix({"acc": f"{acc:.4f}", "mcc": f"{mcc:.4f}", "best": f"{best_acc:.4f}"})
            
            print(f"\nBest: Accuracy={best_acc:.4f}, MCC={best_mcc:.4f}")
            results.append({
                "task": task_name,
                "accuracy": best_acc,
                "mcc": best_mcc,
                "n_train": len(train_seqs),
                "n_test": len(test_seqs),
                "seq_len": seq_len,
                "n_classes": n_classes,
            })
            
        except Exception as e:
            print(f"Error: {e}")
            results.append({"task": task_name, "error": str(e)})
    
    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    for r in results:
        if "accuracy" in r:
            print(f"{r['task']}: Acc={r['accuracy']:.4f}, MCC={r['mcc']:.4f}")
        else:
            print(f"{r['task']}: ERROR - {r.get('error', 'unknown')}")


if __name__ == "__main__":
    main()
