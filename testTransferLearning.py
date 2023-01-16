import transferLearning
import models
import torch

# create model
test = [[torch.rand(5, 3, 50, 50, requires_grad=True),[torch.rand(1,3, requires_grad=True), torch.rand(1,3, requires_grad=True),torch.rand(1,3, requires_grad=True),torch.rand(1,3, requires_grad=True),torch.rand(1,3, requires_grad=True)]],
        [torch.rand(5, 1, 50, 50, requires_grad=True),[torch.rand(1,3, requires_grad=True), torch.rand(1,3, requires_grad=True),torch.rand(1,3, requires_grad=True),torch.rand(1,3, requires_grad=True),torch.rand(1,3, requires_grad=True)]]]


model = models.AE_Transformer(2420,2420, 2420, 3, 1, 1000, 6,1, True, None)
r = model.forward(test, True)
print(r)


