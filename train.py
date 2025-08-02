import torch
import torch.nn as nn

def train_model(model, dataloader, optimizer, device, epochs=5):
    loss_fn = nn.CrossEntropyLoss()
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        size = len(dataloader.dataset)

        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            # Compute prediction error
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backward propagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()

            if batch % 100 == 0:
                current = (batch + 1) * len(X)
                print(f"[Epoch {epoch+1}] Loss: {loss.item():.4f} [{current}/{size}]")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} Finished | Avg Loss: {avg_loss:.4f}")
