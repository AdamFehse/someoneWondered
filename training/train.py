"""
Transformer Training Script for Stellar System Generation

Training overview:
- Autoregressive next-token prediction on quantized orbital sequences.
- Loss: cross-entropy over tokens (PAD ignored).
- Optimizer: AdamW (Adam with decoupled weight decay), default betas/eps.

Run on Colab with GPU for faster training.
"""

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
import numpy as np
from pathlib import Path
import logging
import argparse
import json

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.app.ml.transformer import create_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExoplanetDataset(Dataset):
    """Dataset for transformer training"""

    def __init__(self, sequences, max_seq_len=64):
        self.sequences = sequences
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]

        # Ensure length
        if len(seq) < self.max_seq_len:
            seq = np.pad(seq, (0, self.max_seq_len - len(seq)), constant_values=1)
        else:
            seq = seq[:self.max_seq_len]

        return torch.LongTensor(seq)


def train_epoch(model, dataloader, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0

    for batch_idx, sequences in enumerate(dataloader):
        sequences = sequences.to(device)

        # Forward pass: predict next token
        logits = model(sequences)

        # Compute loss on all positions
        # Shift targets by 1 (predict next token)
        targets = sequences[:, 1:].contiguous()
        logits = logits[:, :-1, :].contiguous()

        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=1  # Ignore PAD token
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

        if (batch_idx + 1) % 10 == 0:
            logger.info(f"  Batch {batch_idx+1}/{len(dataloader)}: loss={loss.item():.4f}")

    return total_loss / len(dataloader)


def eval_epoch(model, dataloader, device):
    """Evaluate for one epoch"""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for sequences in dataloader:
            sequences = sequences.to(device)

            logits = model(sequences)
            targets = sequences[:, 1:].contiguous()
            logits = logits[:, :-1, :].contiguous()

            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=1
            )

            total_loss += loss.item()

    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser(description='Train transformer for stellar systems')
    parser.add_argument('--data', type=str, required=True, help='Path to training data')
    parser.add_argument('--output', type=str, default='models/transformer_v1.pt', help='Output path')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--max-samples', type=int, default=0, help='Limit dataset size (0 = all)')
    parser.add_argument('--testing-mode', action='store_true', help='Use fast defaults for quick iteration')
    parser.add_argument('--profile', choices=['test1', 'test2', 'full'], default='full', help='Preset training profile')

    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.testing_mode and args.profile == 'full':
        args.profile = 'test1'

    if args.profile == 'test1':
        args.epochs = 3
        args.batch_size = 8
        if args.max_samples == 0:
            args.max_samples = 2000
    elif args.profile == 'test2':
        args.epochs = 8
        args.batch_size = 16
        if args.max_samples == 0:
            args.max_samples = 8000

    logger.info("=== Transformer Training ===")
    logger.info(f"Data: {args.data}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.lr}")
    if args.profile != 'full':
        logger.info(f"Profile: {args.profile}")
    if args.max_samples:
        logger.info(f"Max samples: {args.max_samples}")
    if args.testing_mode:
        logger.info("Testing mode: enabled")

    # Load data
    logger.info("\nLoading data...")
    data = np.load(args.data)
    sequences = data['sequences']
    if args.max_samples and len(sequences) > args.max_samples:
        sequences = sequences[:args.max_samples]
    logger.info(f"Loaded {len(sequences)} sequences")
    logger.info(f"Sequence shape: {sequences.shape}")

    # 90/10 split train/val to mesure cross‑entropy loss
    split = int(0.9 * len(sequences))
    train_seqs = sequences[:split]
    val_seqs = sequences[split:]

    logger.info(f"Train: {len(train_seqs)}, Val: {len(val_seqs)}")

    # Create datasets
    train_dataset = ExoplanetDataset(train_seqs)
    val_dataset = ExoplanetDataset(val_seqs)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Create model
    logger.info("\nCreating model...")
    model = create_model(vocab_size=256)
    model = model.to(args.device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {total_params:,} parameters")

    # Create optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    logger.info("\nTraining...")
    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        logger.info(f"\nEpoch {epoch+1}/{args.epochs}")

        train_loss = train_epoch(model, train_loader, optimizer, args.device)
        val_loss = eval_epoch(model, val_loader, args.device)

        scheduler.step()

        logger.info(f"Train loss: {train_loss:.4f}")
        logger.info(f"Val loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            logger.info(f"✓ Saving checkpoint (val_loss={val_loss:.4f})")

            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': model.get_config(),
            }

            torch.save(checkpoint, output_path)

    logger.info(f"\n✓ Training complete")
    logger.info(f"Best model saved to: {args.output}")


if __name__ == "__main__":
    main()
