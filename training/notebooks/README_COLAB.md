# Training on Google Colab

This guide walks you through training the transformer model on Google Colab's free GPU.

### 1. Run Preprocessing

```python
%cd training
!python data/preprocessing.py
```

This downloads NASA exoplanet data and creates `training/data/processed/train.npz`

### 2. Train Model

```python
!python train.py \
    --data data/processed/train.npz \
    --output ../models/transformer_v1.pt \
    --epochs 50 \
    --batch-size 32 \
    --device cuda
```

Training takes ~ 5 sec for all Planets on T4 GPU.

**Hyperparameters**:
- `--epochs`: Number of training epochs (default: 50)
- `--batch-size`: Batch size (default: 32, can increase to 64 with GPU memory)
- `--lr`: Learning rate (default: 1e-4)

## Monitoring Training

The training script prints:
- Epoch number
- Batch loss (every 10 batches)
- Train loss (end of epoch)
- Validation loss (end of epoch)

## Resources

- [Colab GPU Documentation](https://research.google.com/colaboratory/faq.html#gpu-availability)
