import torch
import torch.nn as nn
from visionTransformerUtils import VisionTransformer



class visionFuturePrediction(nn.Module):
    def __init__(self, predictionInterval, image_size, patch_size, 
    num_layers, num_heads, hidden_dim, mlp_dim, dropout, attention_dropout, num_classes):
        super(visionFuturePrediction, self).__init__()
        
        """
        image_size: int,
        patch_size: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        num_classes: int = 1000,
        """
        
        self.transformer = VisionTransformer(image_size, patch_size, 
        num_layers, num_heads, hidden_dim, mlp_dim, dropout, attention_dropout, num_classes)
        self.predictionInterval = predictionInterval

    def forward(self, x, y, training):
        if training == True:
            # take last input
            x = x.unsqueeze(dim = 2)
            #x = torch.cat([x, x, x], dim = 2)
            y = y.unsqueeze(dim=2)
            #y = torch.cat([y, y, y], dim=2)

            s = x[:,-1,: ,: , :] # last input
            out = []
            for i in range(self.predictionInterval):

                s = self.transformer(s)
                s = torch.reshape(s, (s.size(0),1,50,50))
                #s = torch.cat([s, s, s], dim = 1)
                out.append(s)
                s = y[:,i,: ,: , :]
            out = torch.stack(out, dim = 1)[:,:,:,:,:].squeeze()
        if training == False:
            # take last input
            x = x.unsqueeze(dim=2)
            #x = torch.cat([x, x, x], dim=2)

            s = x[:, -1, :, :, :]  # last input
            out = []
            for i in range(self.predictionInterval):
                s = self.transformer(s)
                s = torch.reshape(s, (s.size(0), 1, 50, 50))
                #s = torch.cat([s, s, s], dim=1)
                out.append(s)
            out = torch.stack(out, dim=1)[:, :, :, :, :].squeeze()
        return out

"""
model = visionFuturePrediction(4,50,5, 1, 1, 10, 10, 0.1, 0.1, num_classes = 2500)
print(model(torch.rand(20,4,50,50), torch.rand(20,4,50,50), training = True).size())

model = VisionTransformer(
        image_size=50,
        patch_size=10,
        num_layers=1,
        num_heads=1,
        hidden_dim=1,
        mlp_dim=1
    )
print(model(torch.rand(20,3,50,50)).size())
"""
