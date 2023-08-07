import cv2 as cv
import tensorflow as tf
import numpy as np
import os

def preprocess(impath):
    try:
        image = cv.imread(impath)
        image = cv.cvtColor(image,cv.COLOR_BGR2RGB)
        image = image/255
        image = cv.resize(image, (196, 196))
        return image
    except cv.error:
        print("Invalid Image !")
        image = np.zeros((196, 196 ,3),dtype='uint8')
        return image
def predict(impath,model):
    imarray = []
    imarray.append(preprocess(impath))
    x = np.array(imarray)
    return np.round(model.predict(x),3)
def predictstr(impath,model):
    prediction = predict(impath,model)
    result = prediction[0][0]
    if 0 <= result < 0.5 :
        return "Not Fire"
    else:
        return "Fire"

if __name__ == '__main__':
    modelslist = os.listdir('Models')
    modelpath = os.path.join('Models',modelslist[-1])
    model = tf.keras.models.load_model(modelpath)
    PATH = 'Test_images/'
    imglist = os.listdir(PATH)
    imname = imglist[2]  #<-----(Enter image name (With Extension) here to predict)---------
    impath = os.path.join(PATH,imname)
    img = cv.imread(impath)
    try :
        cv.putText(img, predictstr(impath, model), (20, 30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), thickness=2)
        cv.imshow('Result', img)
    except cv.error :
        img = np.zeros((500, 500, 3), dtype='uint8')
        cv.putText(img, "INVALID IMAGE", (125, 250), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), thickness=2)
        cv.imshow('Result', img)
    cv.waitKey(0)
    # print(predictstr(impath,model))