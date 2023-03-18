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
        self.encoder = nn.Sequential(nn.Linear(2500, 1000),
                                    nn.BatchNorm1d(1000),
                                    nn.LayerNorm(1000),
                                    nn.GELU(),
                                    nn.Linear(1000, 1000),
                                    nn.BatchNorm1d(1000),
                                    nn.LayerNorm(1000),
                                    nn.GELU())
        self.decoder = nn.Sequential(nn.Linear(1000, 2500),
                                    nn.BatchNorm1d(2500),
                                    nn.LayerNorm(2500),
                                    nn.GELU(),
                                    nn.Linear(2500, 2500),
                                    nn.ReLU())
    def forward(self, x):
        x = self.flatten(x)
        x = self.encoder(x)
        x = self.decoder(x)
        x = torch.reshape(x, (x.size(0), 50, 50))

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
    losses = np.ones(len(trainLoader))

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
model = tokenizer().float().to(device)
tkData = datasetClasses.tokenizerData(os.path.join(pathOrigin, "datasets", "tokenizer"))
trainLoader = DataLoader(tkData, 500, shuffle = True)
optimizer = torch.optim.AdamW(model.parameters(), lr = 0.0001, weight_decay=0.0001)
criterion = torch.nn.MSELoss()
trainTokenizer(model, trainLoader, optimizer, criterion, device, 5, pathOrigin, True)

"""






