import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as transforms
import torchvision.io as tv_io
from torchvision.models import VGG16_Weights

import glob
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

weights = VGG16_Weights.DEFAULT
pre_trans = weights.transforms()

DATA_LABELS = ["freshapples", "freshbanana", "freshoranges", "rottenapples", "rottenbanana", "rottenoranges"] 

class MyDataset(Dataset):
    def __init__(self, data_dir):
        self.imgs = []
        self.labels = []
        
        for l_idx, label in enumerate(DATA_LABELS):
            data_paths = glob.glob(data_dir + label + '/*.png', recursive=True)
            for path in data_paths:
                img = tv_io.read_image(path, tv_io.ImageReadMode.RGB)
                self.imgs.append(pre_trans(img).to(device))
                self.labels.append(torch.tensor(l_idx).to(device))


    def __getitem__(self, idx):
        img = self.imgs[idx]
        label = self.labels[idx]
        return img, label

    def __len__(self):
        return len(self.imgs)

valid_path = "data/fruits/valid/"
valid_data = MyDataset(valid_path)
valid_loader = DataLoader(valid_data, batch_size=32)
valid_N = len(valid_loader.dataset)

def get_batch_accuracy(output, y, N):
    pred = output.argmax(dim=1, keepdim=True)
    correct = pred.eq(y.view_as(pred)).sum().item()
    return correct / N

def validate(model):
    accuracy = 0

    model.eval()
    with torch.no_grad():
        for x, y in valid_loader:
            output = model(x)
            accuracy += get_batch_accuracy(output, y, valid_N)
    print('Accuracy: {:.4f}'.format(accuracy))
    return accuracy

def run_assessment(model):
    print('Evaluating model to obtain average accuracy...\n')
    average = validate(model)
    print('\nAccuracy required to pass the assessment is 0.92 or greater.')
    print('Your average accuracy is {:5.4f}.\n'.format(average))
    
    if average >= .92:
        open('/dli/assessment_results/PASSED', 'w+')
        print('Congratulations! You passed the assessment!\nSee instructions below to generate a certificate.')
    else:
        print('Your accuracy is not yet high enough to pass the assessment, please continue trying.')
