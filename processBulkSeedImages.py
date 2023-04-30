import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import os


def cleanupSeedImage(filename, foldername):
    print(filename)
    bulkSeedImage = cv2.imread(filename)

    bulkSeedImageBlur = cv2.medianBlur(bulkSeedImage, 15)

    plt.hist(bulkSeedImageBlur[:, :, 2].ravel(), 256, [0, 255])
    plt.show()

    bulkSeedImageBlur = cv2.copyMakeBorder(bulkSeedImageBlur, 100, 100, 100, 100, cv2.BORDER_CONSTANT)
    bulkSeedImage = cv2.copyMakeBorder(bulkSeedImage, 100, 100, 100, 100, cv2.BORDER_CONSTANT)

    t, threshold = cv2.threshold(bulkSeedImageBlur[:, :, 2], 110, 255, cv2.THRESH_BINARY)
    cv2.imwrite("threshold.jpg", threshold)
    print("morphin time")
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (24 * 1 + 1, 24 * 1 + 1), (1, 1))

    threshold = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, element, iterations=2)
    print("morphin time over")
    cv2.imwrite("thresholdMorph.jpg", threshold)
    cv2.waitKey()

    threshold = cv2.medianBlur(threshold, 5)

    bulkSeedImage[threshold == 0] = 0
    bulkSeedImage[threshold != 0] = bulkSeedImage[threshold != 0]

    print("starting canny edge detection")
    cannyOutput = cv2.Canny(threshold, 50, 150, 3)

    print("starting contour finding")
    contours, _ = cv2.findContours(cannyOutput, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mu = [None]*len(contours)
    for i in range(len(contours)):
        mu[i] = cv2.moments(contours[i])

    mc = [None]*len(contours)
    for i in range(len(contours)):
        # add 1e-5 to avoid division by zero
        mc[i] = (mu[i]['m10'] / (mu[i]['m00'] + 1e-5), mu[i]['m01'] / (mu[i]['m00'] + 1e-5))

    seeds = []
    maskColor = (255, 255, 255)

    print("starting seed extraction")
    for i in range(len(contours)):
        blankForIndividualSeeds = np.zeros(bulkSeedImage.shape, dtype=np.uint8)
        cv2.drawContours(blankForIndividualSeeds, contours, i, maskColor, -1)
        seed = bulkSeedImage.copy()
        seed[blankForIndividualSeeds == 0] = 0
        seed[blankForIndividualSeeds != 0] = seed[blankForIndividualSeeds != 0]

        x, y, w, h = cv2.boundingRect(contours[i])
        seed = seed[y:y + h, x:x + w]
        if seed.shape[0] > 200 and seed.shape[1] > 200 and seed.shape[0] < 500 and seed.shape[1] < 500:
            print(str(i))
            horPad = 500 - seed.shape[1]
            vertPad = 500 - seed.shape[0]
            rightPad = math.floor(horPad/2)
            leftPad = math.ceil(horPad/2)
            topPad = math.ceil(vertPad/2)
            bottomPad = math.floor(vertPad/2)

            seed = cv2.copyMakeBorder(seed, topPad, bottomPad, leftPad, rightPad, cv2.BORDER_CONSTANT)
            seeds.append(seed)
    try:
        os.mkdir(foldername)
    except OSError as error:
        print(error)
        print("This is probably not a big deal, it just means we're overwriting the files in that folder")
    for i in range(len(seeds)):
        file = foldername + "/" + str(i) + ".jpg"
        cv2.imwrite(file, seeds[i])

if __name__ == '__main__':
    cleanupSeedImage("highdef/bananablackcrop.jpg", "bancropped")
    cleanupSeedImage("highdef/habanerocropped.jpg", "habcropped")
    # cleanupSeedImage("highdef/bananablack.jpg", "highBanana")
    # cleanupSeedImage("highdef/bellblack.jpg", "highBell")
    # cleanupSeedImage("highdef/cayenneblack.jpg", "highCayenne")
    # cleanupSeedImage("highdef/cherryblack.jpg", "highCherry")
    # cleanupSeedImage("highdef/habaneroblack.jpg", "highHabanero")
    # cleanupSeedImage("highdef/hungarianblack.jpg", "highHungarian")
    # cleanupSeedImage("highdef/jalapenoblack.jpg", "highJalapeno")
    # cleanupSeedImage("highdef/poblanoblack.jpg", "highPoblano")
    # cleanupSeedImage("highdef/serannoblack.jpg", "highSeranno")





