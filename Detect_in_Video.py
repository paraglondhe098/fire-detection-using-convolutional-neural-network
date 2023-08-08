import cv2 as cv
import tensorflow as tf
import numpy as np
import os

def preprocess(image):
    try:
        image = cv.cvtColor(image,cv.COLOR_BGR2RGB)
        image = image/255
        image = cv.resize(image, (196, 196))
        return image
    except cv.error:
        print("Invalid Image !")
        image = np.zeros((196, 196 ,3),dtype='uint8')
        return image

def predict(image,model):
    imarray = []
    imarray.append(preprocess(image))
    x = np.array(imarray)
    return np.round(model.predict(x),3)
def predictstr(image,model):
    prediction = predict(image,model)
    result = prediction[0][0]
    if 0 <= result < 0.5 :
        return "Not Fire"
    else:
        return "Fire"

if __name__ == '__main__':
    modelslist = os.listdir('Models')
    modelpath = os.path.join('Models',modelslist[-1])
    model = tf.keras.models.load_model(modelpath)
    capture = cv.VideoCapture(0) #<---- You can add video path here
    while True:
        try:
            istrue, frame = capture.read()
            cv.putText(frame, predictstr(frame, model), (20, 30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0),
                       thickness=2)
            cv.imshow('Result', frame)
            if cv.waitKey(20) & 0xFF == ord('d'):
                break
        except cv.error:
            break
    capture.release()
    cv.destroyAllWindows()

    # print(predictstr(impath,model))