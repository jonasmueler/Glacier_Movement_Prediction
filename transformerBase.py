import torch.nn as nn
import torch
from torch.autograd import Variable
import functions
import math
from torch import nn, Tensor


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, device: str, dropout: float = 0.1, max_len: int = 20):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        self.device = device
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x.permute((1,0, 2), x)
        x = x + self.pe[:x.size(0)].to(self.device)
        x = x.permute((1,0,2), x)
        return self.dropout(x)

class Transformer(nn.Module):
    def __init__(self, hiddenLenc, attLayers, attentionHeads, device, Training=True, predictionInterval=None):
        super(Transformer, self).__init__()

        # global
        self.training = Training
        self.predictionInterval = predictionInterval
        self.device = torch.device(device)
        self.flatten = nn.Flatten(start_dim=0, end_dim=-1)
        self.hiddenLenc = hiddenLenc

        # latent space
        self.positional = PositionalEncoding(self.hiddenLenc, device, 0.1)
        self.attentionLayers = attLayers
        self.attentionHeads = attentionHeads
        self.transformer = nn.Transformer(d_model=self.hiddenLenc, nhead=self.attentionHeads,
                                          num_encoder_layers=self.attentionLayers,
                                          num_decoder_layers=self.attentionLayers,
                                          batch_first=True,
                                          dropout=0.5)

    def get_tgt_mask(self, size):
        """
        Generates target mask for decoder in nn.Transformer

        size: int
            size of embedded images

        returns tensor
            size =  (size,size)
        """
        mask = torch.tril(torch.ones(size, size) == 1)
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf'))
        mask = mask.masked_fill(mask == 1, float(0.0))

        return mask.to(self.device)

    def latentSpace(self, flattenedInput, targets, training):
        """
        recurrent prediction model

        flattenedInput: tensor
            input to transformer
        targets: tensor
            if training == True
        training: boolean
            training or inference

        returns: tensor
            output from the transformer, same shape as input

        """
        if training:  # ~teacher forcing
            # add positional information to input
            flattenedInput = self.positional(flattenedInput)

            # add start tokens to sequences
            helper = torch.zeros(flattenedInput.size(0), 1,  self.hiddenLenc, dtype=torch.float32).to(self.device)
            targets = torch.cat([helper, targets], dim = 1).to(self.device)
            flattenedInput = torch.cat([helper, flattenedInput], dim = 1).to(self.device)

            targetMask = self.get_tgt_mask(targets.size(1))

            out = self.transformer(flattenedInput, targets, tgt_mask=targetMask)
            out = out[:,1: , :]

            return out

        if training == False:  # inference ## add temporal embeddings? ## batchMode fixed
            # add start tokens to sequences
            helper = torch.zeros(flattenedInput.size(0), 1, self.hiddenLenc, dtype=torch.float32).to(self.device)
            yInput = torch.zeros(flattenedInput.size(0), 1, self.hiddenLenc, dtype=torch.float32).to(self.device)
            flattenedInput = torch.cat([helper, self.positional(flattenedInput)], dim=1).to(self.device)

            predictionList = []
            for q in range(self.predictionInterval):
                # get mask
                targetMask = self.get_tgt_mask(yInput.size(1))

                # forward pass
                out = self.transformer(flattenedInput, yInput, tgt_mask=targetMask)
                nextItem = out[:, -1, :].unsqueeze(dim = 1) # last element for all sequences in batch
                predictionList.append(nextItem)
                yInput = torch.cat([yInput, nextItem], dim = 1)

            output = torch.stack(predictionList)

            return output

    def forward(self, x, y, training):
        """
        forward pass

        x: torch.tensor
        y: torch.tensor
        training: boolean
            return latent space loss or just prediction of patches
        returns: torch.tensor

        """

        # latent space
        l = self.latentSpace(x, y, training)

        return l

"""
# test
# args: hiddenLenc, attLayers, attentionHeads, device, Training=True, predictionInterval=None)

model = Transformer(1000, 1,1,"cuda", predictionInterval=4).to("cuda")

print(model.forward(torch.rand(3, 4, 1000).to("cuda"), torch.rand(3, 4, 1000).to("cuda"), training = False))
"""


