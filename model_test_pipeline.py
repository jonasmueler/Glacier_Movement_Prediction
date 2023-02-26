import functions
import TransformerNoEmbedding
import torch
import torch.optim as optim
import os
import matplotlib.pyplot as plt

pathOrigin = "/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code"
device = "cpu"

model = TransformerNoEmbedding.AE_Transformer(484,484,484, 1, 1, 480, 11, 22,torch.device(device), True, 5)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# Print model's state_dict
print(model.CLayer1.weight)

path = "/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code/models/transformerPatchesSmall/transformerPatchesSmall"
checkpoint = torch.load(path)

model.load_state_dict(checkpoint['state_dict'])
optimizer.load_state_dict(checkpoint['optimizer'])

print(model.CLayer1.weight)

pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

#print(pytorch_total_params)

## get predictions

os.chdir(pathOrigin + "/datasets")
sceneDataHelheim = functions.openData("trainDataFullScenes")
"""

testInput = torch.rand(5, 3, 50, 50, requires_grad=True)
testTargets = torch.rand(5, 1, 50, 50, requires_grad=True)

testInputDates = torch.stack([torch.rand(1,3, requires_grad=True), torch.rand(1,3, requires_grad=True), torch.rand(1,3, requires_grad=True),
                    torch.rand(1,3, requires_grad=True), torch.rand(1,3, requires_grad=True)])

testTargetDates = torch.stack([torch.rand(1,3, requires_grad=True), torch.rand(1,3, requires_grad=True), torch.rand(1,3, requires_grad=True),
                    torch.rand(1,3, requires_grad=True), torch.rand(1,3, requires_grad=True)])


sceneDataHelheim = [[[testInput, testInputDates], [testTargets, testTargetDates]]]

"""
for i in range(20):
    functions.inferenceScenes(model,
                                sceneDataHelheim[i],
                                50,
                                50,
                                (1,300,300),
                                "Helheim",
                                str(i),
                                "transformerScenesSmall",
                                device,
                                plot = True,
                                safe = True)

"""
def plotPatches(model, data, path, plot):
    
    plots patches and targets and saves on harddrive

    model: nn.model object
    data: list of list of tensor and tensor, and list of tensor and tensor
    path: str
    plot: boolean

    
    model.eval()

    # predictions
    forward = model.forward(data, training=True)
    predictions = forward[0]

    # put into list
    predList = []
    targetList = []
    for i in range(5):
        pred = predictions[i].detach().cpu().numpy().squeeze()
        predList.append(pred)

        targ = data[1][0][i].detach().cpu().numpy().squeeze()
        targetList.append(targ)

    plotData = predList + targetList

    # plot
    fig, axs = plt.subplots(2, 5, figsize=(30, 10), constrained_layout=True)
    for i in range(10):
        row = i // 5
        col = i % 5
        axs[row, col].imshow(functions.minmaxScaler(plotData[i]), cmap='gray')
        axs[row, col].axis('off')
        if i == 0:
            axs[row, col].set_title("Predictions", fontdict = {"fontsize" : 26})
        if i == 5:
            axs[row, col].set_title("Targets", fontdict = {"fontsize" : 26})

    # save on harddrive
    p = path + "/patchPredictionPlot.pdf"
    plt.savefig(p, dpi = 1000)


    # Show the plot
    if plot:
        plt.show()



plotPatches(model,
            sceneDataHelheim[0],
            "/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code/plots",
            True)
"""

