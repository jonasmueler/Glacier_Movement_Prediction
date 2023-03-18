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
        self.CLayer1 = nn.Conv2d(3, 10, (3, 3), 1)
        self.CLayer1Targets = nn.Conv2d(1, 10, (3, 3), 1)
        self.CLayer2 = nn.Conv2d(10, 20, (3, 3), 1)
        self.CLayer3 = nn.Conv2d(20, 40, (3, 3), 1)
        self.CLayer4 = nn.Conv2d(40, 60, (3, 3), 1)
        self.CLayer5 = nn.Conv2d(60, 80, (3, 3), 1)
        self.CLayer6 = nn.Conv2d(80, 100, (3, 3), 1)
        self.CLayer7 = nn.Conv2d(100, 10, (3, 3), 1)
        self.maxPool = nn.MaxPool2d(3, 1, return_indices=True)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten(start_dim=0, end_dim=-1)

        # date encoder
        self.hiddenLenc = hiddenLenc
        self.sizeDate = sizeDateEncoder
        self.numLayersDateEncoder = numLayersDateEncoder
        In = nn.Linear(3, self.sizeDate)
        Linear = nn.Linear(self.sizeDate, self.sizeDate)
        Out = nn.Linear(self.sizeDate, self.hiddenLenc)
        r = nn.ReLU()
        s = nn.Softmax() ## check dimensions! error should be rudimentary
        helper = [In]

        for i in range(self.numLayersDateEncoder):
            helper.append(Linear)
            helper.append(r)
        helper.append(Out)
        helper.append(s)
        self.dateEncoderMLPlist = nn.ModuleList(helper)

        # skip connections
        self.sc1 = nn.Conv2d(50,10, kernel_size = (3,3), padding = 1)
        self.sc2 = nn.Conv2d(100, 20, kernel_size=(3, 3), padding=1)
        self.sc3 = nn.Conv2d(200, 40, kernel_size=(3, 3), padding=1)
        self.sc4 = nn.Conv2d(300, 60, kernel_size=(3, 3), padding=1)
        self.sc5 = nn.Conv2d(400, 80, kernel_size=(3, 3), padding=1)
        self.sc6 = nn.Conv2d(500, 100, kernel_size=(3, 3), padding=1)
        self.sc7 = nn.Conv2d(50, 10, kernel_size=(3, 3), padding=1)

        # hidden space
        self.attentionLayers = attLayers
        self.attentionHeads = attentionHeads
        self.transformer = nn.Transformer(d_model=self.hiddenLenc, nhead=self.attentionHeads,
                                          num_encoder_layers=self.attentionLayers,
                                          num_decoder_layers=self.attentionLayers)

        # decoder
        self.TCLayer1 = nn.ConvTranspose2d(10, 100, (3, 3), 1)
        self.TCLayer2 = nn.ConvTranspose2d(100, 80, (3, 3), 1)
        self.TCLayer3 = nn.ConvTranspose2d(80, 60, (3, 3), 1)
        self.TCLayer4 = nn.ConvTranspose2d(60, 40, (3, 3), 1)
        self.TCLayer5 = nn.ConvTranspose2d(40, 20, (3, 3), 1)
        self.TCLayer6 = nn.ConvTranspose2d(20, 10, (3, 3), 1)
        self.TCLayer7 = nn.ConvTranspose2d(10, 1, (3, 3), 1)
        self.maxUnPool = nn.MaxUnpool2d(3, 1)

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

    def getSkips(self, skips):
        """
        takes skip intermediate perceptual maps from input images, stacks them and extracts
        learned generalized skip conectins for decoder

        skips: list of tensor
            input; output of decoder
        returns:  list of tensor
        """
        skipsOut = []
        for d in range(len(skips[0])):
            helper = [x[d] for x in skips]
            helper = Variable(torch.cat(helper))
            if d == 0:
                res = self.sc1(helper)
                skipsOut.append(res)
            if d == 1:
                res = self.sc2(helper)
                skipsOut.append(res)
            if d == 2:
                res = self.sc3(helper)
                skipsOut.append(res)
            if d == 3:
                res = self.sc4(helper)
                skipsOut.append(res)
            if d == 4:
                res = self.sc5(helper)
                skipsOut.append(res)
            if d == 5:
                res = self.sc6(helper)
                skipsOut.append(res)
            if d == 6:
                res = self.sc7(helper)
                skipsOut.append(res)
        return skipsOut

    def getMaxPoolindices(self, x):

        """
        takes on-hot maps of max-pool operations in the encoder and averages them over all input patches in order to use in decoder
        x: list of list

        returns: list of tensor
            averaged max pool maps
        """
        maps = []
        for i in range(len(x[0])):
            helper = torch.div(x[0][i] + x[1][i] + x[2][i] + x[3][i] + x[4][i], 5, rounding_mode = "floor")
            f = torch.nn.Threshold(5, 1, inplace=True)
            helper = f(helper)
            maps.append(helper)

        return maps



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
        # init memory; idea: safe maxPool indices and skip connections in list of lists for decoder
        result = Variable(torch.zeros((len(x), self.hiddenLenc))).to(self.device)
        poolingIndices = []
        skipConnections = []

        for i in range(len(x)):
            helper = []
            helper1 = []
            if targets:
                image = x[i, :, :]
                image = image.view(1, 50, 50)
                # start convolutions
                s = self.CLayer1Targets(image)
                s, indices = self.maxPool(s)
                s = self.relu(s)
                helper.append(s)
                helper1.append(indices)

            if targets == False:
                image = x[i, :, :, :]
                image = image.view(3, 50, 50)
                # start convolutions
                s = self.CLayer1(image)
                s, indices = self.maxPool(s)
                s = self.relu(s)
                helper.append(s)
                helper1.append(indices)

            s = self.CLayer2(s)
            s, indices = self.maxPool(s)
            s = self.relu(s)
            helper.append(s)
            helper1.append(indices)

            s = self.CLayer3(s)
            s, indices = self.maxPool(s)
            s = self.relu(s)
            helper.append(s)
            helper1.append(indices)

            s = self.CLayer4(s)
            s, indices = self.maxPool(s)
            s = self.relu(s)
            helper.append(s)
            helper1.append(indices)

            s = self.CLayer5(s)
            s, indices = self.maxPool(s)
            s = self.relu(s)
            helper.append(s)
            helper1.append(indices)

            s = self.CLayer6(s)
            s, indices = self.maxPool(s)
            s = self.relu(s)
            helper.append(s)
            helper1.append(indices)

            s = self.CLayer7(s)
            s, indices = self.maxPool(s)
            s = self.relu(s)

            helper.append(s)
            helper1.append(indices)
            s = self.flatten(s)

            result[i, :] = s + self.dateEncoder(encDate[i])  # + encoder temporal embedding

            # save lists for decoder
            skipConnections.append(helper)
            poolingIndices.append(helper1)

        return [result, skipConnections, poolingIndices]


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

    def decoder(self, latentOutput, skips, indices):
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

        result = torch.zeros((latentOutput.size(0), 50, 50)).to(self.device)
        for i in range(latentOutput.size(0)):
            # memory management
            s = Variable(latentOutput[i, :]).to(self.device)

            # start deconvolution; image should be (100, 8, 8)
            image = torch.reshape(s, (10, 22, 22))

            s = image + skips[6]
            s = self.maxUnPool(image, indices[6])
            s = self.TCLayer1(s)
            s = self.relu(s)

            s = s + skips[5]
            s = self.maxUnPool(s, indices[5])
            s = self.TCLayer2(s)
            s = self.relu(s)

            s = s + skips[4]
            s = self.maxUnPool(s, indices[4])
            s = self.TCLayer3(s)
            s = self.relu(s)

            s = s + skips[3]
            s = self.maxUnPool(s, indices[3])
            s = self.TCLayer4(s)
            s = self.relu(s)

            s = s + skips[2]
            s = self.maxUnPool(s, indices[2])
            s = self.TCLayer5(s)
            s = self.relu(s)

            s = s + skips[1]
            s = self.maxUnPool(s, indices[1])
            s = self.TCLayer6(s)
            s = self.relu(s)

            s = s + skips[0]
            s = self.maxUnPool(s, indices[0])
            s = self.TCLayer7(s)
            s = self.relu(s)

            # save in tensor
            result[i, :, :] = s
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
        skipConnections = self.getSkips(res[1])
        poolingInd = self.getMaxPoolindices(res[2])
        reconstruction = self.decoder(res[0], skipConnections, poolingInd)

        # get reconstruction loss of input
        # take last channel -> snow/Ice map
        s = s[:,2, :, :].squeeze()

        reconstructionLoss = nn.MSELoss()(reconstruction, s)


        # latent space
        l = self.latentSpace(res[0], target, datesDecoder, training)

        if training:
            # decoder
            s = self.decoder(l[0], skipConnections, poolingInd)  # output encoder: [result, skipConnections, poolingIndices]
            s = s.unsqueeze(dim = 1) # for loss
            return [s, l[1], reconstructionLoss] # model prediction, latent space loss, reconstruction loss

        elif training == False:
            # decoder
            s = self.decoder(l, skipConnections, poolingInd)  # output encoder: [result, skipConnections, poolingIndices]
            s = s.unsqueeze(dim=1) # for loss

            return s



