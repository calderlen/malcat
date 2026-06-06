import argparse
import os
import torch
from torch.utils.data import DataLoader, random_split

from .ingest import LightCurveDataset, collate_lcs

# train one epoch
def train_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.

    for batch in dataloader:

        # move batch to device
        lc = batch['lc'].to(device)
        mask = batch['mask'].to(device)

        optimizer.zero_grad()
        output = model(lc, mask)
        loss = loss_fn(output, lc)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    return total_loss / len(dataloader)

# evaluate on validation set
def validate(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0.

    with torch.no_grad():
        for batch in dataloader:

            # move batch to device
            lc = batch['lc'].to(device)
            mask = batch['mask'].to(device)

            output = model(lc, mask)
            loss = loss_fn(output, lc)
            
            total_loss += loss.item()

    return total_loss / len(dataloader)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('lc_folders', nargs='+')
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = LightCurveDataset(
        args.lc_folders,
        f_ext='.dat2',
    )

    if len(dataset) < 2:
        raise ValueError('Need at least 2 light curves to create a train/validation split.')

    val_len = max(1, round(0.2 * len(dataset)))
    train_len = len(dataset) - val_len
    train_dataset, val_dataset = random_split(
        dataset,
        [train_len, val_len],
        generator=torch.Generator().manual_seed(7),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=8,
        collate_fn=collate_lcs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=8,
        collate_fn=collate_lcs,
    )

    from .model import malcat

    model = malcat().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.MSELoss()

    best_val_loss = float('inf')
    epochs_without_improvement = 0
    os.makedirs('output/checkpoints', exist_ok=True)

    for epoch in range(10):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss = validate(model, val_loader, loss_fn, device)
        print(f'epoch {epoch + 1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(
                {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                },
                'output/checkpoints/best.pt',
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= 5:
                print(f'early stopping after {epoch + 1} epochs')
                break


if __name__ == '__main__':
    main()
