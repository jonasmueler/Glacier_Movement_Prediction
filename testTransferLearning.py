import transferLearning
import models
import torch

model = models.AE_Transformer(2420,2420, 2420, 3, 1, 1000, 6,1, True, None)

# test with 200*200 image
testInput = torch.rand(5, 3, 200, 200, requires_grad=True)
testTargets = torch.rand(5, 3, 200, 200, requires_grad=True)


