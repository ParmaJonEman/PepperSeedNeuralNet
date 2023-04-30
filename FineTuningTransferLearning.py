import cv2
import numpy as np
import torchvision.transforms as transforms
import os
import keras
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.applications import DenseNet201
from keras import layers, models
from keras.callbacks import EarlyStopping

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
    openEveryImageInFolder('highHungarian', 0, data, labels)
    openEveryImageInFolder('highHabanero', 1, data, labels)
    openEveryImageInFolder('highCayenne', 2, data, labels)
    openEveryImageInFolder('highSeranno', 3, data, labels)
    openEveryImageInFolder('highPoblano', 4, data, labels)
    openEveryImageInFolder('highJalapeno', 5, data, labels)
    openEveryImageInFolder('highCherry', 6, data, labels)
    openEveryImageInFolder('highBanana', 7, data, labels)
    data = convertMatsToTensors(data)
    labels = np.array(labels)
    return train_test_split(data, labels, test_size=.2, random_state=42)

if __name__ == '__main__':
    trainData, testData, trainLabels, testLabels = setupTrainingData()

    print("Training data size: " + str(len(trainData)))

    trainLabels = to_categorical(trainLabels, num_classes=9)
    testLabels = to_categorical(testLabels, num_classes=9)

    ## Load model
    base_model = DenseNet201(weights="imagenet", include_top=False, input_shape=trainData[0].shape)
	## Change 50 to whatever number of layers you want to unfreeze
    for layer in base_model.layers[:-50]:
        layer.trainable = False

    base_model.summary()

    flatten_layer = layers.Flatten()
    prediction_layer = layers.Dense(8, activation='softmax')

    model = models.Sequential([
        base_model,
        flatten_layer,
        prediction_layer
    ])

    model.compile(
         optimizer=keras.optimizers.Adam(1e-5),
         loss='categorical_crossentropy',
         metrics=['accuracy'],
     )
    
    es = EarlyStopping(monitor='val_accuracy', mode='max', patience=5, restore_best_weights=True)
    
    model.fit(trainData, trainLabels, epochs=50, validation_split=0.2, batch_size=32, callbacks=[es])
    model.save('densenet201.h5')
    # Comment out the above code and uncomment the following line if you just want to load a saved model
    # model = tf.keras.models.load_model('densenet201.h5')
    test_loss, test_acc = model.evaluate(testData,  testLabels, verbose=2)
    print('\nTest accuracy:', test_acc)

    prediction = model.predict(np.array(testData))
    print("Prediction Results: ")
    for i in range(len(testData)):
        output = str(np.argmax(testLabels[i]))
        output += "," + str(np.argmax(prediction[i]))
        output += "," + str(prediction[i][np.argmax(prediction[i])])

        print(output)