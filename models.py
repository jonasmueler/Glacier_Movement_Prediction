import torch.nn as nn
import torch
from torch.autograd import Variable
import functions
import math

class AE_LSTM(nn.Module):
    def __init__(self, hiddenL, mlpSize, numLayersDateEncoder, sizeDateEncoder, lstmLayers, lstmHiddenSize, lstmInputSize):
        super(AE_LSTM, self).__init__()

        # encoder
        self.CLayer1 = nn.Conv2d(3,100,(5,5), 1)
        self.CLayer2 = nn.Conv2d(100, 50,(5,5), 1)
        self.CLayer3 = nn.Conv2d(50, 30,(5,5), 1)
        self.CLayer4 = nn.Conv2d(30, 15,(5,5), 1)
        self.CLayer5 = nn.Conv2d(15, 10,(5,5), 1)
        self.CLayer6 = nn.Conv2d(10, 5,(5,5), 1)
        self.CLayer7 = nn.Conv2d(5, 1,(5,5), 1)
        self.maxPool = nn.MaxPool2d(3, 1)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

        # date encoder
        self.sizeDate = sizeDateEncoder
        self.numLayersDateEncoder = numLayersDateEncoder
        self.hiddenL = hiddenL
        In = nn.Linear(3, self.sizeDate)
        Linear = nn.Linear(self.sizeDate, self.sizeDate)
        Out = nn.Linear(self.sizeDate, self.hiddenL)
        r = nn.ReLU()
        s = nn.Softmax()
        helper = [In]

        for i in range(self.numLayersDateEncoder):
            helper.append(Linear)
            helper.append(r)
        helper.append(Out)
        helper.append(s)
        self.dateEncoderMLPlist = nn.ModuleList(helper)

        # LSTM layers
        self.lstmLayers = lstmLayers
        self.lstmHiddenSize = lstmHiddenSize
        self.lstmInputSize = lstmInputSize
        self.lstm = nn.LSTM(input_size=self.lstmInputSize, hidden_size=self.lstmHiddenSize,
                            num_layers=self.lstmLayers, batch_first=True)

        # MLP
        In = nn.Linear(self.hiddenL, self.hiddenL)
        Linear = nn.Linear(self.hiddenL, self.hiddenL)
        r = nn.ReLU()
        helper = [In]

        for i in range(mlpSize):
            helper.append(Linear)
            helper.append(r)
        self.MLPlist = nn.ModuleList(helper)

        # decoder
        self.TCLayer1 = nn.ConvTranspose2d(1, 2, (5, 5), 1)
        self.TCLayer2 = nn.ConvTranspose2d(2, 4, (5, 5), 1)
        self.TCLayer3 = nn.ConvTranspose2d(4, 6, (5, 5), 1)
        self.TCLayer4 = nn.ConvTranspose2d(6, 6, (5, 5), 1)
        self.TCLayer5 = nn.ConvTranspose2d(6, 4, (5, 5), 1)
        self.TCLayer6 = nn.ConvTranspose2d(4, 2, (5, 5), 1)
        self.TCLayer7 = nn.ConvTranspose2d(2, 1, (5, 5), 1)
        #self.maxPool = nn.MaxPool2d(3, 1)

    def dateEncoder(self, dateVec):
        """
        date encoder uses dates as input and projects onto single scalar number with sigmoid activation in last layer
        dateVec: np.array
            input date as vector [day, month, year]
        return: float
        """

        # MLP
        s = dateVec
        for layer in self.dateEncoderMLPlist:
            s = layer(s)
        return s

    def encoder(self, x, encDate):
        """
        downscale channel dimensions then flatten and weight with output of date Encoder

        x: list of tensors ## change !!
            Input image patches
        encDate: 1d tensor
            len = output encoder for each input image
        return: 2d tensor
            flattened image vectors for latent space
        """
        # init memory
        result = torch.zeros((len(x), self.hiddenL))
        for i in range(len(x)): # add max pooling
            image = x[i][-3:,:,:].float()
            s = self.CLayer1(image)
            s = self.relu(s)
            #s = self.maxPool(s)

            s = self.CLayer2(s)
            s = self.relu(s)
            #s = self.maxPool(s)

            s = self.CLayer3(s)
            s = self.relu(s)
            #s = self.maxPool(s)

            s = self.CLayer4(s)
            s = self.relu(s)
            #s = self.maxPool(s)

            s = self.CLayer5(s)
            s = self.relu(s)
            #s = self.maxPool(s)

            s = self.CLayer6(s)
            s = self.relu(s)
            #s = self.maxPool(s)

            s = self.CLayer7(s)
            s = self.relu(s)
            s = self.flatten(s)

            # MLP
            for layer in self.MLPlist:
                s = layer(s)

            result[i,:] = s * self.dateEncoder(encDate[i])

        return result

    def latentSpace(self, flattenedInput):
        """
        
        Takes in flattened vectors of encoder tensor and processes temporal information

        flattenedInput : torch.tensor (sequenceLen, sequenceSize)

        return : torch.tensor (sequenceLen, sequenceSize)
            predictions for decoder
        """

        #LSTM
        h_0 = Variable(torch.zeros(self.lstmLayers, self.lstmHiddenSize))  # hidden state
        c_0 = Variable(torch.zeros(self.lstmLayers, self.lstmHiddenSize))  # internal state

        # Propagate input through LSTM
        output, _ = self.lstm(flattenedInput, (h_0, c_0))  # lstm with input, hidden, and internal state

        # MLP
        for i in range(output.size(0)):
            s = output[i,:].clone()
            for layer in self.MLPlist:
                s = layer(s)
            output[i, :] = s

        return output

    def decoder(self, latentOutput, decDate):
        """
        transposed convolutions to get the original image shape

        latentOutput: tensor
            output of latent space
        decdate: 1d tensor
            dates for predictions
        return: tensor
            output NDSI image of shape (1, 50,50)
        """
        result = torch.zeros((latentOutput.size(0), 50, 50))
        for i in range(latentOutput.size(0)):  # add max pooling with indices !!
            image = latentOutput[i] * self.dateEncoder(decDate[i])
            image = torch.reshape(image, (1, 22, 22))
            s = self.TCLayer1(image)
            s = self.relu(s)
            # s = self.maxPool(s)

            s = self.TCLayer2(s)
            s = self.relu(s)
            # s = self.maxPool(s)

            s = self.TCLayer3(s)
            s = self.relu(s)
            # s = self.maxPool(s)

            s = self.TCLayer4(s)
            s = self.relu(s)
            # s = self.maxPool(s)

            s = self.TCLayer5(s)
            s = self.relu(s)
            # s = self.maxPool(s)

            s = self.TCLayer6(s)
            s = self.relu(s)
            # s = self.maxPool(s)

            s = self.TCLayer7(s)
            s = self.relu(s)
            result[i, :, :] = s

        return result

    def forward(self,d):
        """
        forward pass
        d: list of tensor and encoder and decoder date vectors
            input data
        returns: tensor
            dims = (seqLen, x,y)
        """

        # get data
        s = d[0]
        datesEncoder = d[1]
        datesDecoder = d[2]

        # encoder
        s = self.encoder(s, datesEncoder)

        # latent space
        s = self.latentSpace(s)

        # decoder
        s = self.decoder(s, datesDecoder)

        return(s)


class AE_Attention(nn.Module):
    def __init__(self, encoderIn, hiddenLenc, hiddenLdec, mlpSize, numLayersDateEncoder, sizeDateEncoder, attLayers):
        super(VAE_Attention, self).__init__()

        # encoder
        self.CLayer1 = nn.Conv2d(3, 50, (5, 5), 1)
        self.CLayer2 = nn.Conv2d(50, 100, (5, 5), 1)
        self.CLayer3 = nn.Conv2d(100, 200, (5, 5), 1)
        self.CLayer4 = nn.Conv2d(200, 300, (5, 5), 1)
        self.CLayer5 = nn.Conv2d(300, 400, (5, 5), 1)
        self.CLayer6 = nn.Conv2d(400, 500, (5, 5), 1)
        self.CLayer7 = nn.Conv2d(500, 100, (5, 5), 1)
        self.maxPool = nn.MaxPool2d(3, 1)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten(start_dim=0, end_dim=-1)

        # date encoder
        self.sizeDate = sizeDateEncoder
        self.numLayersDateEncoder = numLayersDateEncoder
        In = nn.Linear(3, self.sizeDate)
        Linear = nn.Linear(self.sizeDate, self.sizeDate)
        Out = nn.Linear(self.sizeDate, 100)
        r = nn.ReLU()
        s = nn.Softmax()
        helper = [In]

        for i in range(self.numLayersDateEncoder):
            helper.append(Linear)
            helper.append(r)
        helper.append(Out)
        helper.append(s)
        self.dateEncoderMLPlist = nn.ModuleList(helper)

        # Attention layers
        self.attLayers = attLayers
        self.attention = nn.MultiheadAttention(100, self.attLayers)

        # MLP Encoder
        self.encoderIn = encoderIn
        self.hiddenLenc = hiddenLenc
        In = nn.Linear(self.encoderIn, self.hiddenLenc)
        Linear = nn.Linear(self.hiddenLenc, self.hiddenLenc)
        out = nn.Linear(self.hiddenLenc, 100)
        r = nn.ReLU()
        helper = [In]

        for i in range(mlpSize):
            helper.append(Linear)
            helper.append(r)
        helper.append(out)
        self.MLPlistEnc = nn.ModuleList(helper)

        # MLP Decoder

        self.hiddenLdec = hiddenLdec
        In = nn.Linear(100, self.hiddenLdec)
        Linear = nn.Linear(self.hiddenLdec, self.hiddenLdec)
        out = nn.Linear(self.hiddenLdec, 484)
        r = nn.ReLU()
        helper = [In]

        for i in range(mlpSize):
            helper.append(Linear)
            helper.append(r)
        helper.append(out)
        self.MLPlistDec = nn.ModuleList(helper)

        # decoder
        self.TCLayer1 = nn.ConvTranspose2d(1, 500, (5, 5), 1)
        self.TCLayer2 = nn.ConvTranspose2d(500, 400, (5, 5), 1)
        self.TCLayer3 = nn.ConvTranspose2d(400, 300, (5, 5), 1)
        self.TCLayer4 = nn.ConvTranspose2d(300, 200, (5, 5), 1)
        self.TCLayer5 = nn.ConvTranspose2d(200, 100, (5, 5), 1)
        self.TCLayer6 = nn.ConvTranspose2d(100, 50, (5, 5), 1)
        self.TCLayer7 = nn.ConvTranspose2d(50, 1, (5, 5), 1)
        # self.maxPool = nn.MaxPool2d(3, 1)

    def dateEncoder(self, dateVec):
        """
        date encoder uses dates as input and projects onto single scalar number with sigmoid activation in last layer
        dateVec: np.array
            input date as vector [day, month, year]
        return: float
        """

        # MLP
        s = dateVec
        for layer in self.dateEncoderMLPlist:
            s = layer(s)
        return s

    def encoder(self, x, encDate):
        """
        downscale channel dimensions then flatten and weight with output of date Encoder

        x: list of tensors ## change !!
            Input image patches
        encDate: 1d tensor
            len = output encoder for each input image
        return: 2d tensor
            flattened image vectors for latent space
        """
        # init memory
        result = torch.zeros((len(x), 100))
        skipConnections = []
        for i in range(len(x)):  # add max pooling
            helper = []
            image = x[i][-3:, :, :].float()
            s = self.CLayer1(image)
            s = self.relu(s)
            helper.append(s)
            # s = self.maxPool(s)

            s = self.CLayer2(s)
            s = self.relu(s)
            helper.append(s)
            # s = self.maxPool(s)

            s = self.CLayer3(s)
            s = self.relu(s)
            helper.append(s)
            # s = self.maxPool(s)

            s = self.CLayer4(s)
            s = self.relu(s)
            helper.append(s)
            # s = self.maxPool(s)

            s = self.CLayer5(s)
            s = self.relu(s)
            helper.append(s)
            # s = self.maxPool(s)

            s = self.CLayer6(s)
            s = self.relu(s)
            helper.append(s)
            # s = self.maxPool(s)

            s = self.CLayer7(s)
            s = self.relu(s)
            helper.append(s)
            s = self.flatten(s)
            # MLP
            for layer in self.MLPlistEnc:
                s = layer(s)

            result[i, :] = s * self.dateEncoder(encDate[i])
            skipConnections.append(helper)
        return [result, skipConnections]

    def latentSpace(self, flattenedInput):
        """

        Takes in flattened vectors of encoder tensor and processes temporal information

        flattenedInput : torch.tensor (sequenceLen, sequenceSize)

        return : torch.tensor (sequenceLen, sequenceSize)
            predictions for decoder
        """

        # attention
        output, attn_output_weights = self.attention(flattenedInput, flattenedInput, flattenedInput)


        return output

    def decoder(self, latentOutput, decDate, skips):
        """
        transposed convolutions to get the original image shape

        latentOutput: tensor
            output of latent space
        decdate: 1d tensor
            dates for predictions
        return: tensor
            output NDSI image of shape (1, 50,50)
        """
        result = torch.zeros((latentOutput.size(0), 50, 50))
        for i in range(latentOutput.size(0)):  # add max pooling with indices !!
            image = latentOutput[i, :].clone() * self.dateEncoder(decDate[i])
            # MLP
            for x in range(latentOutput.size(0)):
                s = image.clone()
                for layer in self.MLPlistDec:
                    s = layer(s)

            image = torch.reshape(s, (1, 22, 22))
            s = self.TCLayer1(image)
            s = self.relu(s)
            s = s + skips[i][5]
            # s = self.maxPool(s)

            s = self.TCLayer2(s)
            s = self.relu(s)
            s = s + skips[i][4]
            # s = self.maxPool(s)

            s = self.TCLayer3(s)
            s = self.relu(s)
            s = s + skips[i][3]
            # s = self.maxPool(s)

            s = self.TCLayer4(s)
            s = self.relu(s)
            s = s + skips[i][2]
            # s = self.maxPool(s)

            s = self.TCLayer5(s)
            s = self.relu(s)
            s = s + skips[i][1]
            # s = self.maxPool(s)

            s = self.TCLayer6(s)
            s = self.relu(s)
            s = s + skips[i][0]
            # s = self.maxPool(s)

            s = self.TCLayer7(s)
            s = self.relu(s)
            result[i, :, :] = s

        return result

    def forward(self, d):
        """
        forward pass
        d: list of tensor and encoder and decoder date vectors
            input data
        returns: tensor
            dims = (seqLen, x,y)
        """

        # get data
        s = d[0]
        datesEncoder = d[1]
        datesDecoder = d[2]

        # encoder
        res = self.encoder(s, datesEncoder)

        # latent space
        s = self.latentSpace(res[0])

        # decoder
        s = self.decoder(s, datesDecoder, res[1])

        return (s)


class AE_Transformer(nn.Module):
    def __init__(self, encoderIn, hiddenLenc, hiddenLdec, mlpSize, numLayersDateEncoder, sizeDateEncoder,
                 attLayers, attentionHeads, Training=True, predictionInterval=None):
        super(AE_Transformer, self).__init__()

        # global
        self.training = Training
        self.predictionInterval = predictionInterval

        # encoder
        self.CLayer1 = nn.Conv2d(3, 10, (3, 3), 1)
        self.CLayer1Targets = nn.Conv2d(1, 10, (3, 3), 1)
        self.CLayer2 = nn.Conv2d(10, 20, (3, 3), 1)
        self.CLayer3 = nn.Conv2d(20, 40, (3, 3), 1)
        self.CLayer4 = nn.Conv2d(40, 60, (3, 3), 1)
        self.CLayer5 = nn.Conv2d(60, 80, (3, 3), 1)
        self.CLayer6 = nn.Conv2d(80, 100, (3, 3), 1)
        self.CLayer7 = nn.Conv2d(100, 5, (3, 3), 1)
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
        s = nn.Softmax()
        helper = [In]

        for i in range(self.numLayersDateEncoder):
            helper.append(Linear)
            helper.append(r)
        helper.append(Out)
        helper.append(s)
        self.dateEncoderMLPlist = nn.ModuleList(helper)

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

        # skip connections
        self.sc1 = nn.Conv2d(50,10, kernel_size = (3,3), padding = 1)
        self.sc2 = nn.Conv2d(100, 20, kernel_size=(3, 3), padding=1)
        self.sc3 = nn.Conv2d(200, 40, kernel_size=(3, 3), padding=1)
        self.sc4 = nn.Conv2d(300, 60, kernel_size=(3, 3), padding=1)
        self.sc5 = nn.Conv2d(400, 80, kernel_size=(3, 3), padding=1)
        self.sc6 = nn.Conv2d(500, 100, kernel_size=(3, 3), padding=1)
        self.sc7 = nn.Conv2d(25, 5, kernel_size=(3, 3), padding=1)



        # hidden space
        self.attentionLayers = attLayers
        self.attentionHeads = attentionHeads
        self.transformer = nn.Transformer(d_model=self.hiddenLenc, nhead=self.attentionHeads,
                                          num_encoder_layers=self.attentionLayers,
                                          num_decoder_layers=self.attentionLayers)

        # MLP Decoder
        self.hiddenLdec = hiddenLdec
        In = nn.Linear(self.hiddenLdec, self.hiddenLdec)
        Linear = nn.Linear(self.hiddenLdec, self.hiddenLdec)
        out = nn.Linear(self.hiddenLdec, self.hiddenLdec)
        r = nn.ReLU()
        helper = [In]

        for i in range(mlpSize):
            helper.append(Linear)
            helper.append(r)
        helper.append(out)
        self.MLPlistDec = nn.ModuleList(helper)

        # decoder
        self.TCLayer1 = nn.ConvTranspose2d(5, 100, (3, 3), 1)
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
        s = dateVec
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
            helper = torch.cat(helper)
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
        result = torch.zeros((len(x), self.hiddenLenc))
        poolingIndices = []
        skipConnections = []

        for i in range(len(x)):  # add max pooling
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

            # MLP
            for layer in self.MLPlistEnc:
                s = layer(s)
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
        pe = torch.zeros(seqLen, self.hiddenLenc)
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

        return mask

    def latentSpace(self, flattenedInput, targets, targetsT, training):
        """
        gets flattened feature vectors from encoder and feeds them into transformer

        flattenedInput: tensor
            output of encoder
        targets: tensor
            if training == True
        targetsT: list of tensor
            temporal information of images
        training: boolean
            training or inference

        returns: tensor
            output from the transformer, same shape as input

        """
        if training:  # ~teacher forcing
            # add start token to sequences
            helper = torch.zeros(1, self.hiddenLenc, dtype=torch.float32)
            flattenedInput = torch.vstack([helper, flattenedInput])
            targets = self.encoder(targets, targetsT, targets=True)[0]  # add temporal information
            targetsOut = targets.clone()  # for latentspace loss
            targets = torch.vstack([helper, targets])

            # positional information to input
            positionalEmbedding = self.positionalEncodings(flattenedInput.size(0) * 2)

            # divide for input and output
            idx = int(flattenedInput.size(0))
            inputMatrix = positionalEmbedding[:, 0:idx, :]
            targetMatrix = positionalEmbedding[:, idx:flattenedInput.size(0) * 2, :]

            flattenedInput = flattenedInput + inputMatrix
            flattenedInput = flattenedInput.squeeze(0)

            # add positional information
            targets = targets + targetMatrix
            targets = targets.squeeze(0)

            targetMask = self.get_tgt_mask(targets.size(0))
            out = self.transformer(flattenedInput, targets, tgt_mask=targetMask)
            out = out[1:, :]

            # MSE loss for latent space
            loss = nn.MSELoss()(out, targetsOut)

            return [out, loss]

        if training == False:  # inference
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
                out = out + self.dateEncoder(targetsT[q])  # add temporal information
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

        result = torch.zeros((latentOutput.size(0), 50, 50))
        for i in range(latentOutput.size(0)):
            image = latentOutput[i, :].clone()

            # MLP
            s = image.clone()
            for layer in self.MLPlistDec:
                s = layer(s)

            # start deconvolution; image should be (100, 8, 8)
            image = torch.reshape(s, (5, 22, 22))

            s = image + skips[6]
            s = self.maxUnPool(image, indices[i][6])
            s = self.TCLayer1(s)
            s = self.relu(s)

            s = s + skips[5]
            s = self.maxUnPool(s, indices[i][5])
            s = self.TCLayer2(s)
            s = self.relu(s)

            s = s + skips[4]
            s = self.maxUnPool(s, indices[i][4])
            s = self.TCLayer3(s)
            s = self.relu(s)

            s = s + skips[3]
            s = self.maxUnPool(s, indices[i][3])
            s = self.TCLayer4(s)
            s = self.relu(s)

            s = s + skips[2]
            s = self.maxUnPool(s, indices[i][2])
            s = self.TCLayer5(s)
            s = self.relu(s)

            s = s + skips[1]
            s = self.maxUnPool(s, indices[i][1])
            s = self.TCLayer6(s)
            s = self.relu(s)

            s = s + skips[0]
            s = self.maxUnPool(s, indices[i][0])
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

        # latent space
        l = self.latentSpace(res[0], target, datesDecoder, training)

        if training:
            # decoder
            skipConnections = self.getSkips(res[1])
            s = self.decoder(l[0], skipConnections, res[2])  # output encoder: [result, skipConnections, poolingIndices]

            return [s, l[1]]

        elif training == False:
            # decoder
            skipConnections = self.getSkips(res[1])
            s = self.decoder(l, skipConnections, res[2])  # output encoder: [result, skipConnections, poolingIndices]

            return s


# test forward pass
#test = [[torch.rand(5, 3, 50, 50, requires_grad=True),[torch.rand(1,3, requires_grad=True), torch.rand(1,3, requires_grad=True),torch.rand(1,3, requires_grad=True),torch.rand(1,3, requires_grad=True),torch.rand(1,3, requires_grad=True)]],
 #       [torch.rand(5, 1, 50, 50, requires_grad=True),[torch.rand(1,3, requires_grad=True), torch.rand(1,3, requires_grad=True),torch.rand(1,3, requires_grad=True),torch.rand(1,3, requires_grad=True),torch.rand(1,3, requires_grad=True)]]]


#model = AE_Transformer(2420,2420, 2420, 3, 1, 1000, 6,4, True, None)
#r = model.forward(test, True)
#print(r)

#total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)



#print(total_params)
#r = model.positionalEncodings(10)



#encoderIn, hiddenLenc, hiddenLdec, mlpSize, numLayersDateEncoder, sizeDateEncoder,
#                     attLayers, Training = True, predictionInterval = None

