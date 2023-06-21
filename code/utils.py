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

def get_similar_images(image, sim_names, simVals, cutoff=0.85):
    cutoff_value = cutoff
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
        
    
def getSim2df(pic_id_list, simNames, simValues, cutoff=0.85, **kwargs):
    no_matches=[]
    results = list()
    for image in pic_id_list:
        p_id=image.split("_")[0]
        if image in simValues.index:
            imgs, vals = get_similar_images(image, simNames, simValues, cutoff)
            for x in range(0, len(imgs)):
                if pd.isna(vals[x]):
                    continue
                Dict = {"original_image" : image, "similar_image" : imgs[x], "similarity_score" : vals[x], "page_id":p_id, "page_id_2":imgs[x].split("_")[0]}
                results.append(Dict)
        else:
            no_matches.append(image)
            
    # save into df
    reuse = pd.DataFrame(results)
    #if meta:
    #    reuse=reuse.merge(meta)
    #    reuse=reuse.merge(meta, left_on="page_id_2", right_on="page_id", suffixes=["","_2"])
    return(reuse, no_matches)