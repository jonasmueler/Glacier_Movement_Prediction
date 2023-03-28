from torch import nn
import torch


device = "cuda"
class ConvLSTM(nn.Module):
    def __init__(self, input_channel, num_filter, b_h_w, kernel_size, stride=1, padding=1):
        super().__init__()
        self._conv = nn.Conv2d(in_channels=input_channel + num_filter,
                               out_channels=num_filter *4,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding)
        self._batch_size, self._state_height, self._state_width = b_h_w
        # if using requires_grad flag, torch.save will not save parameters in deed although it may be updated every epoch.
        # Howerver, if you use declare an optimizer like Adam(model.parameters()),
        # parameters will not be updated forever.
        self.Wci = nn.Parameter(torch.zeros(1, num_filter, self._state_height, self._state_width)).to(device)
        self.Wcf = nn.Parameter(torch.zeros(1, num_filter, self._state_height, self._state_width)).to(device)
        self.Wco = nn.Parameter(torch.zeros(1, num_filter, self._state_height, self._state_width)).to(device)
        self._input_channel = input_channel
        self._num_filter = num_filter

    # inputs and states should not be all none
    # inputs: S*B*C*H*W
    def forward(self, inputs=None, states=None): # hardcoded

        if states is None:
            c = torch.zeros((inputs.size(1), self._num_filter, self._state_height,
                                  self._state_width), dtype=torch.float).to(device)
            h = torch.zeros((inputs.size(1), self._num_filter, self._state_height,
                             self._state_width), dtype=torch.float).to(device)
        else:
            h, c = states
        seq_len = inputs.size(dim = 0)
        outputs = []
        for index in range(seq_len):
            # initial inputs
            if inputs is None:
                x = torch.zeros((h.size(0), self._input_channel, self._state_height,
                                      self._state_width), dtype=torch.float).to(device)
            else:
                x = inputs[index, ...]

            cat_x = torch.cat([x, h], dim=1)
            conv_x = self._conv(cat_x)

            i, f, tmp_c, o = torch.chunk(conv_x, 4, dim=1)

            i = torch.sigmoid(i+self.Wci*c)
            f = torch.sigmoid(f+self.Wcf*c)
            c = f*c + i*torch.tanh(tmp_c)
            o = torch.sigmoid(o+self.Wco*c)
            h = o*torch.tanh(c)
            outputs.append(h)
        return torch.stack(outputs), (h, c)

class ConvLSTMFuturePredictor(nn.Module):
    def __init__(self, input_channel, num_filter, b_h_w, kernel_size, future_horizon = 4, stride=1, padding=1):
        super().__init__()
        self.future_horizon = future_horizon
        self.convLSTM1 = ConvLSTM(input_channel, num_filter, b_h_w, kernel_size, stride, padding)
        self.convLSTM2 = ConvLSTM(num_filter, num_filter, b_h_w, kernel_size, stride, padding)
        self.convLSTM3 = ConvLSTM(num_filter, num_filter, b_h_w, kernel_size, stride, padding)
        self.convLSTM4 = ConvLSTM(num_filter, num_filter, b_h_w, kernel_size, stride, padding)
        self.conv1 = nn.Conv2d(20,1, 3, 1, 1)


    def encoder(self, x):
        x = self.convLSTM1(x)
        lastOutput = x[0]
        lastHidden = x[1][0]
        lastCell = x[1][1]

        x = self.convLSTM2(lastOutput, (lastHidden, lastCell))
        lastOutput = x[0]
        lastHidden = x[1][0]
        lastCell = x[1][1]

        x = self.convLSTM3(lastOutput, (lastHidden, lastCell))
        lastOutput = x[0]
        lastHidden = x[1][0]
        lastCell = x[1][1]

        x = self.convLSTM4(lastOutput, (lastHidden, lastCell))

        return x

    def decoder(self, x):
        out = []

        for i in range(x.size(0)):
            s = self.conv1(x[i, :, :, :])
            out.append(s)
        out = torch.stack(out)
        out = torch.permute(out, (1,0,2,3,4)).squeeze()
        return out

    def forward(self, x):
        x = self.encoder(x)
        lastOutput = x[0][-1].unsqueeze(dim = 0)
        lastHidden = x[1][0]
        lastCell = x[1][1]

        # start predicting
        predictions = []
        for i in range(self.future_horizon):
            s = self.convLSTM4(lastOutput, (lastHidden, lastCell))

            # save predictions
            predictions.append(s[0][-1])

            # update states
            lastOutput = torch.stack(predictions, dim = 0)
            lastHidden = s[1][0]
            lastCell = s[1][1]

        lastOutput = self.decoder(lastOutput)

        return lastOutput



"""
model = ConvLSTMFuturePredictor(1, 20, (20, 50, 50), 3).to(device)
#model = ConvLSTM(1, 60, (20, 50, 50), 3).to(device)

test = torch.rand(4,20, 1, 50, 50).to(device)
print(model(test).size())

"""