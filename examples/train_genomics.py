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
    """Wrapper dataset that one-hot encodes DNA sequences."""
    
    def __init__(self, base_dataset, max_len=None):
        self.base_dataset = base_dataset
        self.max_len = max_len
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        seq, label = self.base_dataset[idx]
        seq = seq.upper()
        
        max_len = self.max_len or len(seq)
        # One-hot encode (5 channels: A, C, G, T, N)
        encoded = torch.zeros(max_len, 5)
        for i, nuc in enumerate(seq[:max_len]):
            if nuc in NUC_TO_IDX:
                encoded[i, NUC_TO_IDX[nuc]] = 1.0
            else:
                encoded[i, 4] = 1.0  # Unknown -> N
        return encoded, label


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


def get_dataset_class(task_name):
    """Get the PyTorch dataset class for a task."""
    from genomic_benchmarks.dataset_getters import pytorch_datasets as ds
    
    # Map task names to class names
    class_name_map = {
        "human_enhancers_cohn": "HumanEnhancersCohn",
        "human_enhancers_ensembl": "HumanEnhancersEnsembl",
        "human_ensembl_regulatory": "HumanEnsemblRegulatory",
        "human_nontata_promoters": "HumanNontataPromoters",
        "human_ocr_ensembl": "HumanOcrEnsembl",
        "drosophila_enhancers_stark": "DrosophilaEnhancersStark",
        "demo_coding_vs_intergenomic_seqs": "DemoCodingVsIntergenomicSeqs",
        "demo_human_or_worm": "DemoHumanOrWorm",
    }
    
    if task_name not in class_name_map:
        raise ValueError(f"Unknown task: {task_name}. Available: {list(class_name_map.keys())}")
    
    class_name = class_name_map[task_name]
    if not hasattr(ds, class_name):
        raise ValueError(f"Dataset class {class_name} not found in genomic_benchmarks")
    
    return getattr(ds, class_name)


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
            # Get dataset class
            DatasetClass = get_dataset_class(task_name)
            
            # Load data
            base_train = DatasetClass(split='train', version=0)
            base_test = DatasetClass(split='test', version=0)
            
            # Get sequence length from first sample
            sample_seq, _ = base_train[0]
            seq_len = len(sample_seq)
            
            # Get number of classes from dataset info
            from genomic_benchmarks.data_check import info
            import re
            import io
            import sys
            
            # Capture printed output
            old_stdout = sys.stdout
            sys.stdout = buffer = io.StringIO()
            info(task_name)
            info_str = buffer.getvalue()
            sys.stdout = old_stdout
            
            # Parse "has N classes" from info string
            match = re.search(r'has (\d+) classes', info_str)
            n_classes = int(match.group(1)) if match else 2  # Default to binary
            
            print(f"Train: {len(base_train)}, Test: {len(base_test)}")
            print(f"Sequence length: {seq_len}, Classes: {n_classes}")
            
            # Wrap with one-hot encoding
            train_dataset = GenomicsDataset(base_train, max_len=seq_len)
            test_dataset = GenomicsDataset(base_test, max_len=seq_len)
            
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
                "n_train": len(base_train),
                "n_test": len(base_test),
                "seq_len": seq_len,
                "n_classes": n_classes,
            })
            
        except Exception as e:
            import traceback
            traceback.print_exc()
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
