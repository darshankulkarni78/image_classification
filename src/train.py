import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os

def train():
    train_tf = transforms.ToTensor()
    test_tf = transforms.ToTensor()

    train_ds = datasets.ImageFolder("data_final/train", transform=train_tf)
    test_ds = datasets.ImageFolder("data_final/test", transform=test_tf)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(1280, 2)
    model = model.cuda()

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(20):
        model.train()
        for x, y in train_loader:
            x = x.cuda()
            y = y.cuda()
            opt.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()

    correct = 0
    total = 0
    model.eval()

    with torch.no_grad():
        for x, y in test_loader:
            x = x.cuda()
            y = y.cuda()
            out = model(x)
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    acc = correct / total
    print("Test accuracy:", acc)

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/mobilenet_car_truck.pth")

if __name__ == "__main__":
    train()