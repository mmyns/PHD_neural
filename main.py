from torch.utils.data import DataLoader
import OPAL_dataset
import og_network
import torch
from torch import nn
import timm
import imgaug.augmenters as iaa
from torchvision.transforms import RandAugment
aug = RandAugment(2, 5)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')


training_data = OPAL_dataset.CustomImageDataset("/esat/biomeddata/mmaeyens/logs/filesandlabels.csv",
                                                frames = [34],
                                                splits = [0,2,3,4],
                                                labels= ["All"],transform=aug)

test_data = OPAL_dataset.CustomImageDataset("/esat/biomeddata/mmaeyens/logs/filesandlabels.csv",
                                                frames = [34],
                                                splits = [1],
                                                labels= ["All"])

train_dataloader = DataLoader(training_data, batch_size=4, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=4, shuffle=True)

#model = og_network.NeuralNetwork().to(device)
model = timm.create_model('vit_base_patch16_224', pretrained= True,num_classes=2).to(device)

print(model)

loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device,dtype=torch.float32), torch.squeeze(y, dim = 1).to(device)
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device,dtype=torch.float32), torch.squeeze(y,dim = 1).to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1)[:] == y[:]).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

epochs = 30
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")