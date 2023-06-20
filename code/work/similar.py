import os
from PIL import Image
from torchvision import transforms
import torch
from tqdm import tqdm
from torchvision import models
import json
import numpy as np

# needed input dimensions for the CNN
inputDim = (224,224)
# directories :\Users\telmi\Documents\dhh23\EarlyModernReuse\early_modern_data-main\data\all_images\cropped\illustration
inputDir = "C:/Users/telmi/Documents/dhh23/EarlyModernReuse/early_modern_data-main/data/all_images/cropped/illustration" #"/scratch/project_2005488/DHH23/early_modern_samples/similarity"
inputDirCNN = "C:/Users/telmi/Documents/dhh23/EarlyModernReuse/data"

os.makedirs(inputDirCNN, exist_ok = True)

transformationForCNNInput = transforms.Compose([transforms.Resize(inputDim)])

for imageName in os.listdir(inputDir):
    I = Image.open(os.path.join(inputDir, imageName))
    newI = transformationForCNNInput(I)

    # copy the rotation information metadata from original image and save, else your transformed images may be rotated
    # exif = I.info['exif']
    newI.save(os.path.join(inputDirCNN, imageName))
    
    newI.close()
    I.close()