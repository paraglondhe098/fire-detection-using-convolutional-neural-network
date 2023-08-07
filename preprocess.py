import numpy as np
import pandas as pd
import cv2 as cv
import os
import random

def joinpath(img_name_list,PATH):
    imgpath=[]
    for imgname in img_name_list:
        imgpath.append(os.path.join(PATH,imgname))
    return imgpath
def add_binary_labels(list1,list2):
    a=[]
    b=[]
    for i in list1:
        a.append([i,0])
    for i in list2:
        b.append([i,1])
    return a,b
def create_dataframe(PATH):
    PATH1, PATH0 = os.listdir(PATH)
    PATH1 = os.path.join(PATH, PATH1)
    PATH0 = os.path.join(PATH, PATH0)

    fire_images = os.listdir(PATH1)
    none_fire_images = os.listdir(PATH0)
    fire_images = joinpath(fire_images, PATH1)
    none_fire_images = joinpath(none_fire_images, PATH0)
    nfire, fire = add_binary_labels(none_fire_images, fire_images)
    combined = nfire + fire
    random.shuffle(combined)
    data = pd.DataFrame(combined, columns=["impath", "Label"])
    return (data)
def create_input_data(dataframe):
    x = [] #Images
    y = [] #Labels
    for impath,label in dataframe.values :
        try :
            image = cv.imread(impath)
            image = cv.cvtColor(image,cv.COLOR_BGR2RGB)
            image = image/255
            image = cv.resize(image,(196,196))
            x.append(image)
            y.append(label)
        except cv.error :
            pass
    return np.array(x),np.array(y)

if __name__ == '__main__':
    PATH = "Data/Test_Data"
    data = create_dataframe(PATH)
    x,y = create_input_data(data)
    cv.imshow("ii",x[0])
    cv.waitKey(0)
    print(y[0])