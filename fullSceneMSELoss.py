import transferLearning
import models
import torch

model = models.AE_Transformer(2420,2420, 2420, 3, 1, 1000, 1,1, True, None)
# test with 200*200 image
testInput = torch.rand(1, 3, 200, 200, requires_grad=True)
testTargets = torch.rand(5, 1, 200, 200, requires_grad=True)

testInputDates = [torch.rand(1,3, requires_grad=True), torch.rand(1,3, requires_grad=True), torch.rand(1,3, requires_grad=True),
                    torch.rand(1,3, requires_grad=True), torch.rand(1,3, requires_grad=True)]

testTargetDates = [torch.rand(1,3, requires_grad=True), torch.rand(1,3, requires_grad=True), torch.rand(1,3, requires_grad=True),
                    torch.rand(1,3, requires_grad=True), torch.rand(1,3, requires_grad=True)]

## final test of functions in

res = transferLearning.fullSceneLoss(testInput, testInputDates, testTargets, testTargetDates, model)

print(res)

