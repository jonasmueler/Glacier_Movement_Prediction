import torch.nn as nn
import torch
from torch.autograd import Variable
import functions
import math

class AE_Transformer(nn.Module):
    def __init__(self, encoderIn, hiddenLenc, hiddenLdec, mlpSize, numLayersDateEncoder, sizeDateEncoder,
                 attLayers, attentionHeads, device, Training=True, predictionInterval=None):
        super(AE_Transformer, self).__init__()

        # global
        self.training = Training
        self.predictionInterval = predictionInterval
        self.device = torch.device(device)
        self.flatten = nn.Flatten(start_dim=0, end_dim=-1)
        self.hiddenLenc = hiddenLenc
        self.linearCorrection = nn.Linear(7500, 2500)

        # MLP Encoder
        self.encoderIn = encoderIn
        In = nn.Linear(self.encoderIn, self.hiddenLenc)
        Linear = nn.Linear(self.hiddenLenc, self.hiddenLenc)
        out = nn.Linear(self.hiddenLenc, self.hiddenLenc)
        r = nn.ReLU()
        helper = [In]

        for i in range(mlpSize):
            helper.append(Linear)
            helper.append(r)
        helper.append(out)
        self.MLPlistEnc = nn.ModuleList(helper)


        # hidden space
        self.attentionLayers = attLayers
        self.attentionHeads = attentionHeads
        self.transformer = nn.Transformer(d_model=self.hiddenLenc, nhead=self.attentionHeads,
                                          num_encoder_layers=self.attentionLayers,
                                          num_decoder_layers=self.attentionLayers)

        # MLP Decoder
        self.hiddenLdec = hiddenLdec
        In = nn.Linear(self.hiddenLenc, self.hiddenLdec)
        Linear = nn.Linear(self.hiddenLdec, self.hiddenLdec)
        out = nn.Linear(self.hiddenLdec, 2500)
        r = nn.ReLU()
        helper = [In]

        for i in range(mlpSize):
            helper.append(Linear)
            helper.append(r)
        helper.append(out)
        self.MLPlistDec = nn.ModuleList(helper)

    def encoder(self, x, encDate, targets):
        """
        downscale channel dimensions then flatten and weight with output of date Encoder

        x: tensor
            Input image patches (5,50,50)
        encDate: 1d tensor
            len = output encoder for each input image
        targets: boolean
        return: list of tensor, list of tensor and list of tensor
            [flattened image vectors for latent space, skip connections list for each image in input,
            list indices for 2dmaxunpool in decoder]
        """
        out = []
        for t in range(x.size(0)):
            result = Variable(torch.zeros(x.size(1), self.hiddenLenc)).to(self.device)
            for i in range(x.size(1)):
                if targets:
                    image = x[t, i, :, :]
                    image = image.view(1, 50, 50)
                    s = self.flatten(image)

                if targets == False:
                    image = x[t, i, :, :, :]
                    image = image.view(3, 50, 50)
                    s = self.flatten(image)
                    s = self.linearCorrection(s)

                # MLP
                for layer in self.MLPlistEnc:
                    s = layer(s)
                result[i, :] = s
            out.append(result)
        res = torch.stack(out)
        # save memory
        del s, result, out, image

        return [res]


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

    def latentSpace(self, flattenedInput, targets, targetsT, training):
        """
        gets flattened feature vectors from encoder and feeds them into transformer

        flattenedInput: tensor
            output of encoder
        targets: tensor
            if training == True
        targetsT: list of tensor
            temporal information targets and input
        temporalInfInference:
            temporal information for inference, when no more teacher forcing is used
        training: boolean
            training or inference

        returns: tensor
            output from the transformer, same shape as input

        """
        if training:  # ~teacher forcing
            # add temporal information to targets
            targets = self.encoder(targets, targetsT, targets=True)[0]
            targetsOut = targets.clone().to(self.device)

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

            targetMatrix = positionalEmbedding[idx:flattenedInput.size(1) * 2, :]
            # for batch
            targetMatrix = torch.stack([targetMatrix for i in range(flattenedInput.size(0))])

            # add positional information
            flattenedInput = flattenedInput + inputMatrix
            flattenedInput = flattenedInput.squeeze(0)
            targets = targets + targetMatrix
            targets = targets.squeeze(0)

            targetMask = self.get_tgt_mask(targets.size(0))

            out = self.transformer(flattenedInput, targets, tgt_mask=targetMask)
            out = out[:,1: , :]

            # MSE loss for latent space
            loss = nn.MSELoss()(out, targetsOut)

            # free memory
            del targetsOut, targetMask, flattenedInput, targets, inputMatrix, targetMatrix, positionalEmbedding, helper

            return [out, loss]

        if training == False:  # inference ## add temporal embeddings
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
                yInput = yInput + positionalEmbedding[flattenedInput.size(0):]

                # get mask
                targetMask = self.get_tgt_mask(yInput.size(0))

                # forward pass
                out = self.transformer(flattenedInput, yInput, tgt_mask=targetMask)
                nextItem = out[-1]
                predictionList.append(nextItem)
                yInput = torch.vstack([yInput, nextItem]).squeeze()

            # calculate loss
            output = torch.stack(predictionList)
            targets = self.encoder(targets, targetsT, targets=True)[0]
            loss = nn.MSELoss()(output, targets[0:4, :])

            return [output, loss]

    def decoder(self, latentOutput):
        """
        transposed convolutions to get the original image shape

        latentOutput: tensor
            output of latent space
        return: tensor
            output NDSI image snow maks of shape (5, 50,50) # 5 timepoints
        """
        out = []
        for t in range(latentOutput.size(0)):
            result = torch.zeros((latentOutput.size(1), 50, 50)).to(self.device)
            for i in range(latentOutput.size(1)):
                # memory management

                s = Variable(latentOutput[t, i, :]).to(self.device)

                # MLP
                for layer in self.MLPlistDec:
                    s = layer(s)

                # start deconvolution
                s = torch.reshape(s, (1, 50, 50))


                # save in tensor
                result[i, :, :] = s
            out.append(result)

        outPut = torch.stack(out).to(self.device)
        del s, result, out

        return outPut

    def forward(self, d, training):
        """
        forward pass
        training has to be argument of forward method not full class!!

        d: list of tensor and encoder and decoder date vectors
            input data
        training: boolean
            return latent space loss or just prediction of patches
        returns: list of tensor and int
            if training: model predictions and latent space loss
            else: model predictions

        """

        # get data
        # inpt = torch.stack(d[0][0], dim = 0).to("cuda")
        s = d[0][0].float()
        datesEncoder = d[0][1]
        target = d[1][0].float()
        datesDecoder = d[1][1]

        # encoder
        res = self.encoder(s, datesEncoder, targets=False)
        reconstructionLoss = torch.zeros(1).to(self.device) # no reconstruction loss

        # latent space
        l = self.latentSpace(res[0], target, datesDecoder, training)

        if training:
            # decoder
            s = self.decoder(l[0])
            s = s.unsqueeze(dim = 2) # for loss

            # save memory
            del datesDecoder, datesEncoder, target, res

            return [s, l[1], reconstructionLoss] # model prediction, latent space loss, reconstruction loss

        elif training == False:
            # decoder
            s = self.decoder(l[0])
            s = s.unsqueeze(dim=2) # for loss

            # save memory
            del datesDecoder, datesEncoder, target, res

            return [s, l[1], reconstructionLoss]



