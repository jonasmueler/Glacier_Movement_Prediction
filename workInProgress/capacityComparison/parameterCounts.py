from torchinfo import summary
import LSTM
import lstmAttention
import ConvLSTM
import functions
from unet_model import UNet
import torch
# memory overflow bug fix
#torch.backends.cudnn.enabled = True
#torch.backends.cudnn.benchmark = True


device = "cuda"
model = lstmAttention.model = lstmAttention.LSTM(3,3, 2500, 2500, 0.1, 5,  device).to(device)
#model = ConvLSTM.ConvLSTMPredictor([64, 64, 24, 24, 64, 24]).to(device)
#model = LSTM.LSTM(3,3, 2500, 2500, 0.1, device).to(device)
#model = UNet(1,1).to(device)

batch_size = 100
print(summary(model, input_size=(batch_size, 4, 50, 50)))


#def count_parameters(model):
#    return sum(p.numel() for p in model.parameters() if p.requires_grad)

#print(count_parameters(model))
