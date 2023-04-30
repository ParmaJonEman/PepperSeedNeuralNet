import cv2
import numpy as np
import torchvision.transforms as transforms
import os
import tensorflow as tf
from keras.utils import to_categorical

def convertMatsToTensors(listOfMats):
    transform = transforms.ToTensor()
    tensorList = []
    tensorNumpy = np.zeros((len(listOfMats), 500, 500, 3))
    for i in range(len(listOfMats)):
        tensor = transform(listOfMats[i].swapaxes(1, 2))
        tensorList.append(tensor)
        tensorNumpy[i] = tensor
    return tensorNumpy


def openEveryImageInFolder(foldername, label, data, labels):
    for filename in os.listdir(os.getcwd() + "/" + foldername):
        seed = cv2.imread(foldername + "/" + filename)
        data.append(seed)
        labels.append(label)


def setupTrainingData():
    data = []
    labels = []
    openEveryImageInFolder('bancropped', 7, data, labels)
    openEveryImageInFolder('habcropped', 1, data, labels)
    data = convertMatsToTensors(data)
    labels = np.array(labels)
    return data, labels

if __name__ == '__main__':
    testData, testLabels = setupTrainingData()

    print("Testing data size: " + str(len(testData)))

    testLabels = to_categorical(testLabels, num_classes=9)

    model = tf.keras.models.load_model('vgg16HighDefUnfrozen.h5')
    #model = tf.keras.models.load_model('Densenet201FineTuned50Unfrozen.h5')
    test_loss, test_acc = model.evaluate(testData, testLabels, verbose=2)
    print('\nTest accuracy:', test_acc)

    prediction = model.predict(np.array(testData))
    print("Prediction Results: ")
    for i in range(len(testData)):
        output = str(np.argmax(testLabels[i]))
        output += "," + str(np.argmax(prediction[i]))
        output += "," + str(prediction[i][np.argmax(prediction[i])])

        print(output)