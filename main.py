from torch.functional import split
from torch.nn.modules.loss import BCEWithLogitsLoss
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR
import OPAL_dataset
import og_network
import resnet_model
import color_dataset
import torch
from torch import nn
import timm
import pandas as pd
import imgaug.augmenters as iaa
from torchvision.transforms import RandAugment
from pytorchsettings import PytorchSettings
import os



settings = PytorchSettings(dropout_rate=0.5,
                        label_smoothing=0.2,
                        selected_labels= ["ABMRh","Cellular"],
                        split_number = 1,
                        epochs = 50,
                        images = "color")  

aug = RandAugment(settings.augment_N, settings.augment_M) 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

df_input = pd.read_csv("/esat/biomeddata/mmaeyens/logs/filesandlabels.csv")
def get_results(model,input_df,dataset,files = settings.images):
    df = input_df[input_df["split"] == settings.split_number]
    full_names = []
    labels = []
    predictions = []
    temp_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    files = list(df[files])
    for name in files:
        splitted = name.split(os.sep)
        splitted2 = splitted[-1].split("]")
        full_names.append(splitted2[0] + "]")
    with torch.no_grad():
        for X, y in temp_dataloader:
            X, y = X.to(device,dtype=torch.float32), torch.squeeze(y,dim = 1).to(device)
            pred = model(X)[0]
            if len(settings.selected_labels) == 1:
                new_pred = torch.nn.Softmax()(pred)[1]
                predictions.extend(list([new_pred.tolist()]))
            else:
                new_pred = torch.nn.Sigmoid()(pred)
                predictions.extend(list([new_pred.tolist()]))


    final_dict = {"file_names":full_names}
    if len(settings.selected_labels) == 1:
        final_dict[settings.selected_labels[0]] = list(df[settings.selected_labels[0]])
        final_dict[settings.selected_labels[0] + "_predictions"] = predictions
    else:
        for n,label_name in enumerate(settings.selected_labels):
            final_dict[label_name] = list(df[label_name])
            final_dict[label_name + "_predictions"] = [x[n] for x in predictions]
    for key in final_dict:
        print(key)
        print(len(final_dict[key]))
    saved_df = pd.DataFrame(
        data=final_dict)
    return saved_df


training_data = color_dataset.CustomImageDataset("/esat/biomeddata/mmaeyens/logs/filesandlabels.csv",
                                                splits = [0,2,3,4],
                                                labels= settings.selected_labels,transform=aug)

test_data = color_dataset.CustomImageDataset("/esat/biomeddata/mmaeyens/logs/filesandlabels.csv",
                                                splits = [settings.split_number],
                                                labels= settings.selected_labels)

train_dataloader = DataLoader(training_data, batch_size=4, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=4, shuffle=True)

#Pick required model


#model = og_network.NeuralNetwork().to(device)
#model = timm.create_model('resnet18', pretrained=True).to(device)
#model = timm.create_model('vit_base_patch16_224', pretrained= True,num_classes=2).to(device)
model = resnet_model.NeuralNetwork().to(device)
settings.model = model.get_model_name()

print(settings)

print(model)
if settings.activation == "Binary" and len(settings.selected_labels) > 1:
    loss_fn = BCEWithLogitsLoss()
    settings.label_smoothing = 0
else:
    loss_fn = nn.CrossEntropyLoss(label_smoothing=settings.label_smoothing)
optimizer = torch.optim.Adam(model.parameters(), lr=settings.lr)
scheduler = ExponentialLR(optimizer, gamma=0.9)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device,dtype=torch.float32), torch.squeeze(y, dim = 1).to(device,dtype=torch.float32)
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
    scheduler.step()
global_loss = 10000
def test(dataloader, model, loss_fn):
    global global_loss
    target_true = 0
    predicted_true = 0
    correct_true = 0
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device,dtype=torch.float32), torch.squeeze(y,dim = 1).to(device,dtype=torch.float32)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
    #         predicted_classes = torch.argmax(pred, dim=1)
    #         target_classes = y
    #         target_true += torch.sum(target_classes == 1).float().item()
    #         predicted_true += torch.sum(predicted_classes == 1).float().item()
    #         correct_true += (predicted_classes[predicted_classes == y[:]] == 1).type(torch.float).sum().item()
    #         correct += (pred.argmax(1)[:] == y[:]).type(torch.float).sum().item()
    # recall = correct_true / target_true 
    # print(correct_true)
    # print(target_true)
    # precision = correct_true / (predicted_true + 0.001)
    # f1_score = 2 * precision * recall / (precision + recall)
    test_loss /= num_batches
    # accuracy = correct/size
    if test_loss < global_loss:
        print("Saving model")
        global_loss = test_loss
        torch.save(model.state_dict(), settings.filename + '_model.pt') 
    print(f" Avg loss: {test_loss:>8f}")

epochs = settings.epochs
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)

print("Done!")
print("Reloading model")
model.load_state_dict(torch.load(settings.filename + '_model.pt'))
final_df = get_results(model,df_input,test_data,settings.images)
final_df.to_csv(f"{settings.filename}_predictions.csv")
settings.save(settings.filename)
