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

        # encoder
        self.flatten = nn.Flatten(start_dim=0, end_dim=-1)
        self.seq = nn.Sequential(
          nn.Linear(encoderIn*3, encoderIn),
          nn.ReLU(),
          nn.Linear(encoderIn,encoderIn),
          nn.ReLU(),
          nn.Linear(encoderIn, encoderIn),
          nn.ReLU()
        )

        ## input embedding
        self.emb = nn.Sequential(
            nn.Linear(encoderIn, encoderIn),
            nn.ReLU(),
            nn.Linear(encoderIn, encoderIn),
            nn.ReLU(),
            nn.Linear(encoderIn, encoderIn),
            nn.ReLU(),
            nn.Linear(encoderIn, encoderIn),
            nn.ReLU()
        )

        # date encoder
        self.hiddenLenc = hiddenLenc
        self.sizeDate = sizeDateEncoder
        self.numLayersDateEncoder = numLayersDateEncoder
        In = nn.Linear(3, self.sizeDate)
        Linear = nn.Linear(self.sizeDate, self.sizeDate)
        Out = nn.Linear(self.sizeDate, self.hiddenLenc)
        r = nn.ReLU()
        soft = nn.Softmax()
        helper = [In]

        for i in range(self.numLayersDateEncoder):
            helper.append(Linear)
            helper.append(r)
        helper.append(Out)
        helper.append(soft)
        self.dateEncoderMLPlist = nn.ModuleList(helper)

        # latent space
        self.attentionLayers = attLayers
        self.attentionHeads = attentionHeads
        self.transformer = nn.Transformer(d_model=self.hiddenLenc, nhead=self.attentionHeads,
                                          num_encoder_layers=self.attentionLayers,
                                          num_decoder_layers=self.attentionLayers)



    def dateEncoder(self, dateVec):
        """
        date encoder uses dates as input and projects onto single scalar number with sigmoid activation in last layer
        dateVec: tensor
            input date as vector [day, month, year]
        return: float
        """

        # MLP
        s = dateVec.to(self.device)
        for layer in self.dateEncoderMLPlist:
            s = layer(s)
        return s

    def encoder(self, x, encDate, targets):
        """
        downscale channel dimensions then flatten and weight with output of date Encoder

        x: tensor
            Input image patches (5,50,50)
        encDate: 1d tensor
            len = output encoder for each input image
        targets: boolean
        return: tensor
        """
        # init memory
        result = Variable(torch.zeros((len(x), self.hiddenLenc))).to(self.device)

        for i in range(len(x)):
            if targets:
                image = x[i, :, :]
                image = image.view(1, 50, 50)
                image = self.flatten(image)

            if targets == False:
                image = x[i, :, :, :]
                image = image.view(3, 50, 50)
                image = self.flatten(image)
                image = self.seq(image)

            image = self.emb(image)

            result[i, :] = image + self.dateEncoder(encDate[i])  # + encoder temporal embedding

        return result


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

        return pe

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
        training: boolean
            training or inference

        returns: tensor
            output from the transformer, same shape as input

        """
        if training:  # ~teacher forcing
            # add temporal information to targets
            targets = self.encoder(targets, targetsT, targets=True)
            targetsOut = targets.clone().to(self.device)
            # add start tokens to sequences
            helper = torch.zeros(1, self.hiddenLenc, dtype=torch.float32).to(self.device)
            targets = torch.vstack([helper, targets]).to(self.device)
            flattenedInput = torch.vstack([helper, flattenedInput]).to(self.device)

            # positional information to data
            positionalEmbedding = self.positionalEncodings(flattenedInput.size(0) * 2)

            # divide for input and output
            idx = int(flattenedInput.size(0))
            inputMatrix = positionalEmbedding[:, 0:idx, :]
            targetMatrix = positionalEmbedding[:, idx:flattenedInput.size(0) * 2, :]

            # add positional information
            flattenedInput = flattenedInput + inputMatrix
            flattenedInput = flattenedInput.squeeze(0)
            targets = targets + targetMatrix
            targets = targets.squeeze(0)

            targetMask = self.get_tgt_mask(targets.size(0))
            out = self.transformer(flattenedInput, targets, tgt_mask=targetMask)
            out = out[1:, :]

            # MSE loss for latent space
            loss = nn.MSELoss()(out, targetsOut)

            # free memory
            del targets, targetsOut, helper, flattenedInput, positionalEmbedding, inputMatrix, targetMatrix

            return [out, loss]

        if training == False:  # inference ## add temporal embeddings
            ### len(targetsT) = self.predictionInterval -> variable prediction length
            # add start token to sequences
            yInput = torch.zeros(1, self.hiddenLenc, dtype=torch.float32)
            helper = torch.zeros(1, self.hiddenLenc, dtype=torch.float32)
            flattenedInput = torch.vstack([helper, flattenedInput])

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
                out = out + self.dateEncoder(targetsT[q])  # add temporal information to predictions
                nextItem = out[-1]
                yInput = torch.vstack([yInput, nextItem]).squeeze()

            return yInput[1:, :]

    def decoder(self, latentOutput):
        """
        transposed convolutions to get the original image shape

        latentOutput: tensor
            output of latent space
        decdate: 1d tensor
            dates for predictions
        skips: list of tensor
            skip connections
        indices: list of tensor
            indices for maxunpool in recosntruction of the image
        return: tensor
            output NDSI image snow maks of shape (5, 50,50) # 5 timepoints
        """

        result = Variable(torch.zeros((latentOutput.size(0), 50, 50))).to(self.device)
        for i in range(latentOutput.size(0)):
            # memory management
            s = Variable(latentOutput[i, :]).to(self.device)

            #
            image = torch.reshape(s, (1, 50, 50))

            # save in tensor
            result[i, :, :] = image
        return result

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

        # latent space
        l = self.latentSpace(res, target, datesDecoder, training)

        if training:
            # decoder
            s = self.decoder(l[0])
            s = s.unsqueeze(dim = 1) # for loss

            # free memory
            del datesEncoder, target, datesDecoder, res

            return [s, l[1], torch.tensor(0)]

        elif training == False:
            # decoder
            s = self.decoder(l[0])
            s = s.unsqueeze(dim=1) # for loss

            return s



