import torch.nn as nn
import torch
from torch.autograd import Variable
import functions
import math

class LSTM(nn.Module):
    def __init__(self, lstmLayersEnc, lstmLayersDec, lstmHiddenSize, lstmInputSize, dropout, device):
        super(LSTM, self).__init__()

        # LSTM layers
        self.device = device
        self.dropout = dropout
        self.lstmLayersEnc = lstmLayersEnc
        self.lstmLayersDec = lstmLayersDec
        self.lstmHiddenSize = lstmHiddenSize
        self.lstmInputSize = lstmInputSize
        self.lstmEncoder = nn.LSTM(input_size=self.lstmInputSize, hidden_size=self.lstmHiddenSize,
                            num_layers=self.lstmLayersEnc, batch_first=True, dropout = self.dropout)

        self.lstmDecoder = nn.LSTM(input_size=self.lstmInputSize, hidden_size=self.lstmHiddenSize,
                            num_layers=self.lstmLayersDec, batch_first=True, dropout = self.dropout)

        self.loss = torch.nn.MSELoss()

    def encoder(self, x):
        """
        encodes the input with LSTM cells

        x: torch.tensor
            (b, s, dim)
        return list of torch.tensor and tuple of torch.tensor and torch.tensor
            output, hidden and cell state
        """

        # init hidden and cell state
        h_0 = Variable(torch.zeros(self.lstmLayersEnc, x.size(0), self.lstmHiddenSize)).to(self.device)  # hidden state
        c_0 = Variable(torch.zeros(self.lstmLayersEnc, x.size(0), self.lstmHiddenSize)).to(self.device)  # internal state

        # Propagate input through LSTM
        output, cellHidden = self.lstmEncoder(x, (h_0, c_0))  # lstm with input, hidden, and internal state

        # get last output across batches
        output = output[:, -1, :].unsqueeze(dim=1)  # nBatch, 1, dimSeq

        return [output, cellHidden]

    def decoder(self, outputEnc, x, reconstruction):
        """
        creates reconstruction loss and predicts into the future

        outputEnc: torch.tensor
        x: torch.tensor
            input for reconstruction loss
        reconstruction: boolean

        return: torch.tensor
            recontruction loss or future predictions based on reconstruction argument
        """

        # get h0, c0, last prediction
        h_0 = outputEnc[1][0]
        c_0 = outputEnc[1][0]
        output = outputEnc[0]

        if reconstruction: # compute reconstruction Loss on the fly
            out = [] # batches across timesteps as list
            # Propagate input through LSTM
            for i in range(4):
                output, (h_0, c_0) = self.lstmDecoder(output, (h_0, c_0))  # lstm with input, hidden, and internal state
                out.append(output)
            pred = torch.cat(out, dim = 1)

            # switch order of sequences in tensor
            reversed = torch.zeros_like(x).to(self.device)
            for i in range(x.size(1)):
                reversed[:, i, :] = x[:, x.size(1)- i - 1, :].clone()


            reconstructionLoss = self.loss(pred, reversed)

            return reconstructionLoss


        if reconstruction == False:
            out = []  # batches across timesteps as list
            # Propagate input through LSTM
            for i in range(4):
                output, (h_0, c_0) = self.lstmDecoder(output, (h_0, c_0))  # lstm with input, hidden, and internal state
                out.append(output)
            pred = torch.cat(out, dim=1)

            return pred

    def forward(self, flattenedInput, y = None, training = None):


        s = self.encoder(flattenedInput)
        recLoss = self.decoder(s, flattenedInput, reconstruction = True)
        output = self.decoder(s, None, reconstruction = False)

        return [output, recLoss]

"""
# test, args: lstmLayersEnc, lstmLayersDec, lstmHiddenSize, lstmInputSize, dropout, device
device = "cuda"
model = LSTM(2,2, 1000, 1000, 0.1, device).to(device)
test = torch.rand(3, 4, 1000).to(device)

print(model(test)[0].size())

"""


