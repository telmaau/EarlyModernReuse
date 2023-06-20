import json
import numpy as np
import pandas as pd
import pickle
from PIL import Image

import matplotlib.pyplot as plt
from numpy.testing import assert_almost_equal

import random  
import os



# plotting functions

def setAxes(ax, image, query = False, **kwargs):
    value = kwargs.get("value", None)
    year=kwargs.get("year", None)
    if year:
        image_year = image+", "+str(year)
    else:
        image_year = image
    if query:
        ax.set_xlabel("Original Image\n{0}".format(image_year), fontsize = 6)
    else:
        ax.set_xlabel("Similarity value {1:1.3f}\n{0}".format( image_year,  value), fontsize = 6)
    ax.set_xticks([])
    ax.set_yticks([])

def plotImages(img_id, inputDir,newprints_nodups,numCol=5):
    original_year = list(newprints_nodups[newprints_nodups["original_image"]==img_id]["publication_year"])
    reuse_year = list(newprints_nodups[newprints_nodups["original_image"]==img_id]["publication_year_2"])
    simImages = list(newprints_nodups[newprints_nodups["original_image"]==img_id]["similar_image"])
    simValues = list(newprints_nodups[newprints_nodups["original_image"]==img_id]["similarity_score"])
    numRow= int(len(simImages)/5) +1
    #if len(simImages) % 5 != 0:
    #    numRow=numRow +1 # add an extra row to show all images
        
    height= 2*numRow
    fig = plt.figure(figsize=(10, height))
    ax=[]
    
    # plot original image
    img = Image.open(os.path.join(inputDir, img_id))
    ax = fig.add_subplot(numRow, numCol, 1)
    setAxes(ax, img_id, query=True, year=original_year[0])
    img = img.convert('RGB')
    plt.imshow(img)
    img.close()
    
    # plot similar images
    n=2
    for i,v,y in zip(simImages,simValues,reuse_year):
        ax=[]
        img = Image.open(os.path.join(inputDir, i))
        ax.append(fig.add_subplot(numRow, numCol, n))
        
        setAxes(ax[-1],i, value = float(v), year =y)
        img = img.convert('RGB')
        plt.imshow(img)
        img.close()
        n+=1
    
    plt.show()