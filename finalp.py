import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')
# Import packages
import numpy as np
import pandas as pd
import cv2
from sklearn import neighbors
import gc; gc.enable()
import time

# Parameters
trainline = 10
testline = 10
size = 30 # Size of Bag of Words vocabulary
features = 1600 # for SIFT and ORB
method = "ORB" # Either "SIFT" or "ORB"

start = time.time()
#Read image--------------------------------------------------------------
def TrainImage(animal, i):
    path = "/Users/FFFFFan/Downloads/train/" + animal + "." + str(i) + ".jpg"
    image = cv2.imread(path, 0)
    return image

def TestImage(i):
    path = "/Users/FFFFFan/Downloads/tes/" + str(i) + ".jpg"
    image = cv2.imread(path, 0)
    return image

#Detect and descript Features----------------------------------------------------------
bow = cv2.BOWKMeansTrainer(size)

detector = None
if method == "ORB":
    detector = cv2.ORB(features)
elif method == "SIFT":
    detector = cv2.SIFT(features)
else:
    print "Please set method to either SIFT or ORB "

print "Extracting dog/cat features for BOW model"
def extract(animal, bow):
    # the function is to extract dog/cat features for BOW model
    for i in range(trainline):
        image = TrainImage(animal, i)
        if image is None:
            continue
        kp, des = detector.detectAndCompute(image, None)
        if des is None:
               continue
        bow.add(np.float32(des))
    return(bow)

bow = extract("cat", bow)
bow = extract("dog", bow)
#Generate vocabulary---------------------------------------------------------
vocabulary = bow.cluster()
Train = []
Y = []
Test = []

print "Generate vocabulary"
matcher = None
if method == "SIFT":
    matcher = cv2.BFMatcher(normType = cv2.NORM_L1)
else:
    vocabulary = np.uint8(vocabulary)
    matcher = cv2.BFMatcher(normType = cv2.NORM_HAMMING)

extractor = cv2.DescriptorExtractor_create(method)
ext = cv2.BOWImgDescriptorExtractor(extractor, matcher)
ext.setVocabulary(vocabulary)

print "Create predictor and response variables"
def xy(animal, Train, Y):
    for i in range(trainline):
        img = TrainImage(animal, i)
        if img is None:
            continue
        kp, des = detector.detectAndCompute(img, None)
        if des is None:
           continue
        bowDes = ext.compute(img, kp, des)
        Train.append(bowDes.flatten())
        if animal == "cat":
           Y.append(0)
        else:
           Y.append(1)
    return (Train, Y)

result = xy("cat", Train, Y)

result = xy("dog", result[0], result[1])

Train = result[0]
Y = result[1]

for i in range(testline):
    img = TestImage(i)
    if img is None:
        continue
    kp, des = detector.detectAndCompute(img, None)
    if des is None:
        XTest.append(np.zeros(size))
        continue
    bowDes = ext.compute(img, kp, des)
    Test.append(bowDes.flatten())

Train = np.asarray(Train)
Y = np.asarray(Y)
Test = np.asarray(Test)

#Fit the model and predict--------------------------------------------------------------------

def metrics(real, pred):
    t = pd.crosstab(real, pred, rownames=['real'], colnames=['preds'])
    print t
    
    
    tp = np.double(t[1][1])
    tn = np.double(t[0][0])
    fp = np.double(t[1][0])
    fn = np.double(t[0][1])
    mcr  = round((fp+fn)/t.sum().sum(), 2)
    print "Misclassification rate:", mcr

print "KNN on Training model"
knn= neighbors.KNeighborsClassifier(3)
knn.fit(Train,Y)

print "Prediction on training data"
Z = knn.predict(Train)
metrics(Y, Z)

print "Prediction on test data"
ZTest = knn.predict(Test)
print ZTest

#Time-------------------------------------------------------------------

m, s = divmod((time.time() - start), 60)
print "Time taken to run:", int(m)*60+round(s,3), "seconds"

