import torch.nn as nn
import torch
from torch.autograd import Variable
import functions
import math

class LSTM(nn.Module):
    def __init__(self, lstmLayers, lstmHiddenSize, lstmInputSize, dropout, device):
        super(LSTM, self).__init__()

        # LSTM layers
        self.device = device
        self.dropout = dropout
        self.lstmLayers = lstmLayers
        self.lstmHiddenSize = lstmHiddenSize
        self.lstmInputSize = lstmInputSize
        self.lstm = nn.LSTM(input_size=self.lstmInputSize, hidden_size=self.lstmHiddenSize,
                            num_layers=self.lstmLayers, batch_first=True, dropout = self.dropout)


    def forward(self, flattenedInput, targets, training):
        """

        Takes in flattened vectors of encoder tensor and processes temporal information

        flattenedInput : torch.tensor (nBatch, sequenceLen, sequenceSize)

        return : torch.tensor (nBatch, sequenceLen, sequenceSize)
            predictions
        """
        if training:
            # LSTM
            h_0 = Variable(torch.zeros(self.lstmLayers, flattenedInput.size(0), self.lstmHiddenSize)).to(self.device)  # hidden state
            c_0 = Variable(torch.zeros(self.lstmLayers, flattenedInput.size(0), self.lstmHiddenSize)).to(self.device)  # internal state

            # Propagate input through LSTM
            output, (hn, cn) = self.lstm(flattenedInput, (h_0, c_0))  # lstm with input, hidden, and internal state

            # get last output across batches
            output = output[:,-1,:].unsqueeze(dim = 1) # nBatch, 1, dimSeq

            #predict future states
            outputs = []

            # train with teacher forcing
            for i in range(flattenedInput.size(1)):
                if i == 0:
                    output, (hn, cn) = self.lstm(output, (hn, cn))
                    outputs.append(output)
                else:
                    outputTeacher = targets[:, i, :].unsqueeze(dim=1)  # take ith element of sequence
                    output, (hn, cn) = self.lstm(outputTeacher, (hn, cn))
                    outputs.append(output)

            # get rid of extra dimension
            outputs = [x.squeeze(dim = 1) for x in outputs]
            outFinal = torch.stack(outputs, dim = 1)

        if training == False:
            # LSTM
            h_0 = Variable(torch.zeros(self.lstmLayers, flattenedInput.size(0), self.lstmHiddenSize)).to(self.device)  # hidden state
            c_0 = Variable(torch.zeros(self.lstmLayers, flattenedInput.size(0), self.lstmHiddenSize)).to(self.device)  # internal state

            # Propagate input through LSTM
            output, (hn, cn) = self.lstm(flattenedInput, (h_0, c_0))  # lstm with input, hidden, and internal state

            # get last output across batches
            output = output[:, -1, :].unsqueeze(dim=1)  # nBatch, 1, dimSeq

            # predict future states
            outputs = []

            # train with teacher forcing
            for i in range(flattenedInput.size(1)):
                output, (hn, cn) = self.lstm(output, (hn, cn))
                outputs.append(output)

            # get rid of extra dimension
            outputs = [x.squeeze(dim=1) for x in outputs]
            outFinal = torch.stack(outputs, dim=1)

        return outFinal

"""
# test
model = LSTM(2, 1000, 1000, 0.1)
assert model.forward(torch.rand(3, 4, 1000), torch.rand(3, 4, 1000), True).size() == \
       torch.Size([3, 4, 1000]) == \
       model.forward(torch.rand(3, 4, 1000), torch.rand(3, 4, 1000), False).size()
"""
