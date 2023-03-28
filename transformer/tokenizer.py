import torch.nn as nn
import torch
from torch.autograd import Variable
import functions
import math
import functions
import numpy as np
import os
from torch.utils.data import DataLoader
import datasetClasses
import torch
import wandb

class tokenizer(nn.Module):
    """
    tokenizer for the transformer model
    """
    def __init__(self):
        super(tokenizer, self).__init__()
        self.flatten = nn.Flatten()
        self.encoder = nn.Sequential(nn.Linear(2500, 2000),
                                    nn.BatchNorm1d(2000),
                                    nn.LayerNorm(2000),
                                    nn.GELU()
                                   # nn.Linear(1000, 1000),
                                   # nn.BatchNorm1d(1000),
                                   # nn.LayerNorm(1000),
                                   # nn.GELU(),
                                   # nn.Linear(1000, 1000),
                                   # nn.BatchNorm1d(1000),
                                   # nn.LayerNorm(1000),
                                   # nn.GELU()
                                              )
        self.decoder = nn.Sequential(nn.Linear(2000, 2500),
                                    nn.BatchNorm1d(2500),
                                    nn.LayerNorm(2500),
                                    nn.GELU(),
                                    nn.Linear(2500, 2500),
                                    nn.Sigmoid()
                                   # nn.BatchNorm1d(2500),
                                   # nn.LayerNorm(2500),
                                   # nn.GELU(),
                                   # nn.Linear(2500, 2500),
                                   # nn.BatchNorm1d(2500),
                                   # nn.LayerNorm(2500),
                                   # nn.GELU(),
                                   # nn.Linear(2500, 2500)
                                                         )

    def forward(self, x):
        x = self.flatten(x)
        x = self.encoder(x)
        x = self.decoder(x)
        x = torch.reshape(x, (x.size(0), 50, 50))

        return x


class tokenizerExp(nn.Module):
    """
    tokenizer for the transformer model
    """
    def __init__(self):
        super(tokenizerExp, self).__init__()

        # activation function
        self.gelu = nn.GELU()
        self.sigmoid = nn.Sigmoid()

        # down
        self.conv1 = nn.Conv2d(1,10, 3)
        self.conv2 = nn.Conv2d(10, 20, 3)
        self.batchNorm1 = nn.BatchNorm2d(20)
        self.conv3 = nn.Conv2d(20, 40, 3)
        self.batchNorm2 = nn.BatchNorm2d(40)
        self.conv4 = nn.Conv2d(40, 60, 3)
        self.batchNorm3 = nn.BatchNorm2d(60)
        self.conv5 = nn.Conv2d(60, 1, 3)
        self.flatten = nn.Flatten(start_dim = 2, end_dim = 3)

        # up
        self.conv6 = nn.ConvTranspose2d(1, 60, 3)
        self.batchNorm4 = nn.BatchNorm2d(60)
        self.conv7 = nn.ConvTranspose2d(60, 40, 3)
        self.batchNorm5 = nn.BatchNorm2d(40)
        self.conv8 = nn.ConvTranspose2d(40, 20, 3)
        self.batchNorm6 = nn.BatchNorm2d(20)
        self.conv9 = nn.ConvTranspose2d(20, 10, 3)
        self.conv10 = nn.ConvTranspose2d(10, 1, 3)

    def encoder(self, x):
        skips = []
        x = x.unsqueeze(dim = 1)

        x = self.conv1(x)
        skips.append(x)
        x = self.gelu(x)
        x = self.conv2(x)
        x = self.batchNorm1(x)
        skips.append(x)
        x = self.gelu(x)
        x = self.conv3(x)
        x = self.batchNorm2(x)
        skips.append(x)
        x = self.gelu(x)
        x = self.conv4(x)
        x = self.batchNorm3(x)
        skips.append(x)
        x = self.gelu(x)
        x = self.conv5(x)
        skips.append(x)
        x = self.sigmoid(x)
        x = self.flatten(x)

        return [x, skips]

    def decoder(self, x, skips):
        x = torch.reshape(x, (x.size(0), 1, 40,40))
        x = x + skips[-1]
        x = self.conv6(x)
        x = self.batchNorm4(x)
        x = x + skips[-2]
        x = self.gelu(x)
        x = self.conv7(x)
        x = self.batchNorm5(x)
        x = x + skips[-3]
        x = self.gelu(x)
        x = self.conv8(x)
        x = self.batchNorm6(x)
        x = x + skips[-4]
        x = self.gelu(x)
        x = self.conv9(x)
        x = x + skips[-5]
        x = self.gelu(x)
        x = self.conv10(x)
        x = self.gelu(x)
        return x

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x[0], x[1])
        x = x.squeeze(dim = 1)
        return x

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            # Convolutional layer with 16 filters, kernel size of 3, stride of 2, and padding of 1
            nn.ReLU(),
            nn.Conv2d(16, 8, 3, stride=2, padding=1),
            # Convolutional layer with 8 filters, kernel size of 3, stride of 2, and padding of 1
            nn.ReLU(),
            nn.Conv2d(8, 4, 3, stride=2, padding=1),
            # Convolutional layer with 4 filters, kernel size of 3, stride of 2, and padding of 1
            nn.ReLU()
        )

        # Decoder layers
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(4, 8, 3, stride=2, padding=1, output_padding=1),
            # Transpose convolutional layer with 8 filters, kernel size of 3, stride of 2, padding of 1, and output padding of 1
            nn.ReLU(),
            nn.ConvTranspose2d(8, 16, 3, stride=2, padding=1, output_padding=1),
            # Transpose convolutional layer with 16 filters, kernel size of 3, stride of 2, padding of 1, and output padding of 1
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            # Transpose convolutional layer with 1 filter, kernel size of 3, stride of 2, padding of 1, and output padding of 1
            nn.Sigmoid()  # Sigmoid activation function
        )

    def forward(self, x):
        x = x.unsqueeze(dim=1)
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def trainTokenizer(model, trainLoader, optimizer, criterion, device, epochs, pathOrigin, WandB):
    """
    trains the tokenizer encoder and decoder

    model: nn.Module object
    trainLoader: DataLoader object
    optimizer: torch.optim object
    criterion: nn.MSEloss()
    device: string
    epochs: int
    pathOrigin: string
    WandB: boolean

    return
    """
    model.train()
    losses = np.ones(len(trainLoader)*epochs)

    if WandB == True:
        wandb.init(
            # set the wandb project where this run will be logged
            project="tokenizer",

            # track hyperparameters and run metadata
            config={
            }
        )

    for i in range(epochs):
        counter = 0
        for inputs in trainLoader:
            inputs = inputs.float().to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                losses[counter] = loss.detach().cpu().item()
                if counter % 1 == 0:
                    print("current loss: ", loss.detach().cpu().item())
                ## log to wandb
                if WandB:
                    wandb.log({"train loss": loss.detach().cpu().item()})

                counter += 1

        # save checkpoint after each epoch
        functions.saveCheckpoint(model, optimizer, pathOrigin + "/" + "models/" + "tokenizer")
        np.savetxt(os.path.join(pathOrigin, "models", "tokenizerRun.csv"), losses, delimiter=",")
    return

"""
device = "cuda"
pathOrigin = "/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code"
model = tokenizerExp().float().to(device)
#model = tokenizer().float().to(device)
#model = ConvAutoencoder().float().to(device)

tkData = datasetClasses.tokenizerData("/home/jonas/datasets/parbati")
trainLoader = DataLoader(tkData, 100, shuffle = True)
optimizer = torch.optim.AdamW(model.parameters(), lr = 0.01, weight_decay=0.01)
criterion = torch.nn.MSELoss()
trainTokenizer(model, trainLoader, optimizer, criterion, device, 5, pathOrigin, True)

"""





