#!/usr/bin/env python
# coding: utf-8

'''
This work is inspired blog post of Maciej D. Korzec https://towardsdatascience.com/recommending-similar-images-using-pytorch-da019282770c
Some imports, which were needed to run the code on the Puhti supercomputer
Install these via pip if you don't have them already
'''


#import sys
#!{sys.executable} -m pip install torchvision
#!{sys.executable} -m pip install tqdm
#!{sys.executable} -m pip install numpy
#!{sys.executable} -m pip install pandas


# Imports
import os
import torch
import pandas as pd
import numpy as np
import pickle
import csv
import argparse
import random
from PIL import Image
from torchvision import transforms, models
from tqdm import tqdm
from numpy.testing import assert_almost_equal

# Add arguments for the command line



parser = argparse.ArgumentParser(description="Analyze images how similar they are, and write the results to a .csv file")
parser.add_argument("--inputpath", help="Relative path to the folder of the images", default="")
parser.add_argument("--outputpath", help="Relative path to where the results will be stored", default="")
parser.add_argument("--method", help="Use GPU or CPU for computing", default="cpu")
parser.add_argument("--cutoff", help="How similar images will be stored", default=0.94, type=float, choices=range(0,1))
parser.add_argument("--amount", help="How many similar images will be stored", default=50, type=int)
args = parser.parse_args()


print("Converting images to feature vectors:")
img_dir="../res/"
os.makedirs(img_dir, exist_ok = True)
for image in tqdm(os.listdir(img_dir)):
    I = Image.open(os.path.join(img_dir, image)).convert("RGB")
    #vec = images_to_vectors.get_vector(I)
    #allVectors[image] = vec
    I.close() 
RESULTS_FILE_NAME = 'datanew.csv'
SIMILAR_NAMES_PATH = 'similar_names.pkl'
SIMILAR_VALUES_PATH = 'similar_values.pkl'
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# Needed input dimensions for the CNN
# PyTorch's documentation suggests resolution of at least 224 x 224
input_dimensions = (224,224)

# Directory, from where the images to be analyzed are taken
# Change accordingly to your needs and folder structure
input_directory = args.inputpath
#input_directory = "<folder for all images you want to analyze> "
#SEARCH_DIR = "<folder for images to be checked>"
#DB_DIR = "folder where all data structures are stored to avoid re-calculation"
#MODEL_TYPE = ["resnet18", "resnet50"] # Supported pre-trained models
#THRESHOLD = 0.95

# Output directory for the similar images
# Change accordingly to your needs and folder structure
if args.outputpath:
    output_dir = args.outputpath
    RESULTS_FILE_NAME = str(output_dir) + RESULTS_FILE_NAME
    SIMILAR_NAMES_PATH = str(output_dir) + SIMILAR_NAMES_PATH
    SIMILAR_VALUES_PATH = str(output_dir) + SIMILAR_VALUES_PATH
else:
    output_dir = "/"

os.makedirs(output_dir, exist_ok = True)

#transformationForCNNInput = transforms.Compose([transforms.Resize(input_dimensions)])
# create transforms, Resnet expects the images are in format 256x256 pixels

transformationForCNNInput = transforms.Compose([
        transforms.Resize((256, 256)),
       # transforms.ToTensor()              
    ])
# This will take reasonably large amount of time.
# Could be investigated, if can be made faster
#for imageName in os.listdir(input_directory):
#    I = Image.open(os.path.join(input_directory, imageName))
#    newI = transformationForCNNInput(I)

    # Copy the rotation information metadata from original image and save, else your transformed images may be rotated
#    newI.save(os.path.join(output_dir, imageName))
    
#    newI.close()
#    I.close()

# The class for the resnet
class images_to_vectors_resnet50():
    def __init__(self):
        
        # Get the command line argumnet for CPU or GPU
        method = args.method

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
        self.numberFeatures = 2048
        
        #Also other models available, see more at PyTorch's documentation
        self.modelName = "resnet-50"
        self.model, self.featureLayer = self.getFeatureLayer()
        self.model = self.model.to(self.device)
        self.model.eval()
        self.toTensor = transforms.ToTensor()
        
        # These values are suggested by PyTorch's documentation
        # normalize the resized images as expected by resnet18
        # [0.485, 0.456, 0.406] --> normalized mean value of ImageNet, [0.229, 0.224, 0.225] std of ImageNet
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
    def get_vector(self, img):
        image = self.normalize(self.toTensor(img)).unsqueeze(0).to(self.device)
        embedding = torch.zeros(1, self.numberFeatures, 1, 1)

        def copyData(m, i, o):
            embedding.copy_(o.data)

        h = self.featureLayer.register_forward_hook(copyData)
        self.model(image)
        h.remove()

        return embedding.numpy()[0, :, 0, 0]

    def getFeatureLayer(self):
        cnnModel = models.resnet50(weights="DEFAULT")
        layer = cnnModel._modules.get('avgpool')
        self.layer_output_size = 2048
        return cnnModel, layer
        

# generate vectors for all the images in the set
images_to_vectors = images_to_vectors_resnet50() 

allVectors = {}
print("Converting images to feature vectors:")
img_dir="../res/"
for image in tqdm(os.listdir(img_dir)):
    I = Image.open(os.path.join(img_dir, image)).convert("RGB")
    vec = images_to_vectors.get_vector(I)
    allVectors[image] = vec
    I.close() 

# now let us define a function that calculates the cosine similarity entries in the similarity matrix
def get_similarity_matrix(vectors):
    v = np.array(list(vectors.values())).T
    sim = np.inner(v.T, v.T) / ((np.linalg.norm(v, axis=0).reshape(-1,1)) * ((np.linalg.norm(v, axis=0).reshape(-1,1)).T))
    keys = list(vectors.keys())
    matrix = pd.DataFrame(sim, columns = keys, index = keys)
    return matrix
        
similarity_matrix = get_similarity_matrix(allVectors)

# Amount of similar images to be stored
k = args.amount

similar_names = pd.DataFrame(index = similarity_matrix.index, columns = range(k))
similar_values = pd.DataFrame(index = similarity_matrix.index, columns = range(k))

print("Counting similarity values for images:")
for j in tqdm(range(similarity_matrix.shape[0])):
    kSimilar = similarity_matrix.iloc[j, :].sort_values(ascending = False).head(k)
    similar_names.iloc[j, :] = list(kSimilar.index)
    similar_values.iloc[j, :] = kSimilar.values
similarNames_path = SIMILAR_NAMES_PATH
similarValues_path = SIMILAR_VALUES_PATH
similar_names.to_pickle(similarNames_path)
similar_values.to_pickle(similarValues_path)

# open a file, where you stored the pickled data
file = open(similarNames_path, 'rb')
sim_names = pickle.load(file)
file.close()

file = open(similarValues_path, 'rb')
sim_values = pickle.load(file)
file.close()

def get_similar_images(image, sim_names, simVals):
    cutoff_value = args.cutoff
    filtered = simVals[simVals > cutoff_value]
    if image in set(sim_names.index):
        imgs = list(sim_names.loc[image, :])
        vals = list(filtered.loc[image, :])
        if image in imgs:
            assert_almost_equal(max(vals), 1, decimal = 5)
            imgs.remove(image)
            vals.remove(max(vals))
        return imgs, vals
    else:
        print("'{}' Unknown image".format(image))

# Writes a .csv-file of the similar images and their similarity scores
# Appends the lines, so the file has to be cleared/deleted after each run
def write_image_data_to_csv(list_of_dicts):
    with open(RESULTS_FILE_NAME, 'a') as csvfile:
        field_names = ["original_image", "similar_image", "similarity_score"]
        writer = csv.DictWriter(csvfile, fieldnames=field_names)
        writer.writeheader()
        for i in list_of_dicts:
            writer.writerow(i)

# take three examples from the provided image set and plot
folder_path = args.inputpath

# Number of files/images, which will be sampled
num_files = 1000
all_files = os.listdir(folder_path)

# Shuffle the list of files randomly
random.shuffle(all_files)
selected_files = all_files#[:num_files]

results = list()

# Get similar images
for image in selected_files:
    imgs, vals = get_similar_images(image, sim_names, sim_values)
    for x in range(0, len(imgs)):
        if pd.isna(vals[x]):
            continue
        Dict = {"original_image" : image, "similar_image" : imgs[x], "similarity_score" : vals[x]}
        results.append(Dict)

write_image_data_to_csv(results)
print("Similarities computed")