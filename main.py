from data_loader import get_data
from model import CNN
from evaluate import test_model
from train import train_model
import torch
import torch.nn as nn

def main():
    # Check for device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the data
    train_loader, val_loader = get_data()
    
    # Define model, optimizer and loss
    model = CNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    # Train model
    history = train_model(model, train_loader, optimizer, device, epochs=5)

    # Test model
    test_model(model, val_loader, loss_fn, device)

if __name__ == '__main__':
    main()