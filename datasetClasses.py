import os
from torch.utils.data import Dataset
from torch import is_tensor
import functions

class glaciers(Dataset):
    def __init__(self, path, mode):
        """
        dataset class for train loop
        path: str
            path to image and target folder
        """
        self.path = path
        self.mode = mode

        # get list of all image paths in directory
        images = os.listdir(os.path.join(self.path, "images"))
        paths = [os.path.join(os.path.join(self.path, "images"), item) for item in images]
        self.images = paths

        targets = os.listdir(os.path.join(self.path, "targets"))
        paths = [os.path.join(os.path.join(self.path, "targets"), item) for item in targets]
        self.targets = paths

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """
        returns datum for training

        idx: int
            index to datum
        returns: torch.tensor
                image and targets
        """
        if is_tensor(idx):
            idx = idx.tolist()

        # get data in tensor format
        inpt = functions.openData(self.images[idx])
        target = functions.openData(self.targets[idx])

        return inpt, target

class tokenizerData(Dataset):
    def __init__(self, path):
        """
        dataset class for train loop
        path: str
            path to image and target folder
        """
        self.path = path
        self.mode = mode

        # get list of all image paths in directory
        images = os.listdir(self.path)
        paths = [os.path.join(self.path, item) for item in images]
        self.images = paths

    def __len__(self):
        _, _, files = next(os.walk(os.path.join(self.path, "images"))))
        fileCount = len(files)
        return fileCount

    def __getitem__(self, idx):
        """
        returns datum for training

        idx: int
            index to datum
        returns: torch.tensor
                image and targets
        """
        if is_tensor(idx):
            idx = idx.tolist()

        # get data in tensor format
        inpt = functions.openData(self.images[idx])

        return inpt
