import torch.nn as nn
import torch
from torch.autograd import Variable
import functions
import math

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
        self.attentionLayers = attLayers
        self.attentionHeads = attentionHeads
        self.transformer = nn.Transformer(d_model=self.hiddenLenc, nhead=self.attentionHeads,
                                          num_encoder_layers=self.attentionLayers,
                                          num_decoder_layers=self.attentionLayers)

    def positionalEncodings(self, seqLen):
        """
        creates positional encoding matrix for the transformer model based on vasmari et al

        seqLen: int
            length of sequence
        inputLen:
            length of input

        returns: 2d tensor
            constants added for positional encodings
        """
        # create constant 'pe' matrix with values dependant on pos and i
        pe = torch.zeros(seqLen, self.hiddenLenc).to(self.device)
        for pos in range(seqLen):
            for i in range(0, self.hiddenLenc, 2):
                pe[pos, i] = \
                    math.sin(pos / (10000 ** ((2 * i) / self.hiddenLenc)))
                pe[pos, i + 1] = \
                    math.cos(pos / (10000 ** ((2 * (i + 1)) / self.hiddenLenc)))

        pe = pe.unsqueeze(0)

        # make relatively larger
        x = pe * math.sqrt(self.hiddenLenc)
        pe = pe + x

        return pe.to(self.device)

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

            # add start tokens to sequences
            helper = torch.zeros(flattenedInput.size(0), 1,  self.hiddenLenc, dtype=torch.float32).to(self.device)
            targets = torch.cat([helper, targets], dim = 1).to(self.device)
            flattenedInput = torch.cat([helper, flattenedInput], dim = 1).to(self.device)

            # positional information to data
            positionalEmbedding = self.positionalEncodings(flattenedInput.size(1) * 2).squeeze()

            # divide for input and output
            idx = int(flattenedInput.size(1))
            inputMatrix = positionalEmbedding[0:idx, :]

            # for batch
            inputMatrix = torch.stack([inputMatrix for i in range(flattenedInput.size(0))])

            # add positional information
            flattenedInput = flattenedInput + inputMatrix
            flattenedInput = flattenedInput.squeeze(0)

            targetMask = self.get_tgt_mask(targets.size(0))

            out = self.transformer(flattenedInput, targets, tgt_mask=targetMask)
            out = out[:,1: , :]

            # MSE loss for latent space
            loss = nn.MSELoss()(out, targets)

            # free memory
            del targets, targetMask, flattenedInput, targets, inputMatrix, positionalEmbedding, helper

            return [out, loss]

        if training == False:  # inference ## add temporal embeddings?; #### not in batch mode !!!!!
            ### len(targetsT) = self.predictionInterval -> variable prediction length
            # add start token to sequences
            yInput = torch.zeros(1, self.hiddenLenc, dtype=torch.float32).to(self.device)
            helper = torch.zeros(1, self.hiddenLenc, dtype=torch.float32).to(self.device)
            flattenedInput = torch.vstack([helper, flattenedInput])
            predictionList = []
            for q in range(self.predictionInterval):
                # positional information to input
                positionalEmbedding = self.positionalEncodings(flattenedInput.size(0) + (q + 1))
                positionalEmbedding = positionalEmbedding.squeeze()
                flattenedInput = flattenedInput + positionalEmbedding[0:flattenedInput.size(0)]
                yInput = yInput

                # get mask
                targetMask = self.get_tgt_mask(yInput.size(0))

                # forward pass
                out = self.transformer(flattenedInput, yInput, tgt_mask=targetMask)
                nextItem = out[-1]
                predictionList.append(nextItem)
                yInput = torch.vstack([yInput, nextItem]).squeeze()

            # calculate loss
            output = torch.stack(predictionList)

            return output

    def forward(self, d, training):
        """
        forward pass
        training has to be argument of forward method not full class!!

        d: list of tensor
            input and target representations
        training: boolean
            return latent space loss or just prediction of patches
        returns: list of tensor and int
            if training: model predictions and latent space loss
            else: model predictions

        """
        x = d[0]
        targets = d[1]

        # latent space
        l = self.latentSpace(x, targets, training)

        if training:
            l[0] = l[0].unsqueeze(dim = 2) # for loss

            # save memory
            del x, l, targets

            return [l[0], l[1]] # model prediction, latent space loss, reconstruction loss

        elif training == False:

            l[0] = l[0].unsqueeze(dim=2) # for loss

            # save memory
            del x, l, targets

            return l



