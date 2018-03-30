from collections import defaultdict
from mnist import MNIST
import numpy as np
import sys
import os
import math
import skimage
from skimage import feature

path = str(sys.argv[1])


def lecture_mnist(path):

    mdata = MNIST(path)
    DataApp, LabelApp = mdata.load_training()
    DataTest, LabelTest = mdata.load_testing()
    
    # DataApp et DataTest sont des listes contenant elles-memes des listes, ces dernieres
    # etant les donnees images
    # LabelApp et LabelTest sont des type array contenant la classe des donnees de meme indice
    # dans DataApp et DataTest 		

    return np.array(DataApp+DataTest) , np.array(LabelApp+LabelTest)


def decoupage_donnees(X,Y):

    dataLength = len(X)
    trainingLength = int(0.5 * dataLength)

    s = np.arange(dataLength)

    np.random.shuffle(s)

    Xapp, Xtest = np.split(X[s], [trainingLength])
    Yapp, Ytest = np.split(Y[s], [trainingLength])

    return Xapp, Yapp, Xtest, Ytest


def kppv_distances(Xtest, Xapp):        
        
        appN = np.tile(np.linalg.norm(Xapp, axis=1)**2,(len(Xtest),1))
        testN = np.tile(np.linalg.norm(Xtest, axis=1)**2,(len(Xapp),1))
        
        prod = 2*np.matmul(Xtest, Xapp.transpose())
        
        return appN - prod + testN.transpose()

#def kppv_distances(Xtest, Xapp):
#
#    norm = lambda x,y,z : np.matmul(x, y.transpose())
#
#    return norm(Xapp,Xapp,0) + norm(Xtest,Xtest,1).transpose() - 2*np.matmul(Xapp,Xtest.transpose()).transpose()


def kppv_predict(Dist, Yapp, K):
        
        Ypred = np.arange(len(Dist))
        
        for l in range(len(Dist)):
                Indices = np.argsort(Dist[l,:])
                
                YappKeepKClosest = Yapp[Indices[0:K]]
                
                ClassCounts = np.bincount(YappKeepKClosest)
                
                MajorityClass =  np.argmax(ClassCounts)
                
                Ypred[l] = MajorityClass

        print(Ypred, Ypred.shape())


        bestKIndex = Dist.argsort()[:,:K]
        bestKClasses = Yapp[bestKIndex]

        #print(Dist,Dist[0,bestKIndex],Yapp,Yapp[bestKIndex])
        beef =[]

        for bests in bestKClasses:
                d = defaultdict(int)
                for i in bests:
                        d[i] += 1

                result = max(d.items(), key=lambda x: x[1])
                beef.append(result[0])

        print(np.asarray(beef), np.asarray(beef).shape())

        return Ypred

def evaluation_classifieur(Ytest, Ypred):
        Count = np.count_nonzero(Ytest==Ypred)
        Accuracy = Count / len(Ytest)
        return Accuracy

MnistX, MnistY = lecture_mnist(path)

print(" ")
print("X and Y shapes")
print(np.shape(MnistX))
print(np.shape(MnistY))
print(" ")

MnistXapp, MnistYapp, MnistXtest, MnistYtest = decoupage_donnees(MnistX, MnistY)

print("Mnist app and test shapes")
print(np.shape(MnistXapp))
print(np.shape(MnistYapp))
print(np.shape(MnistXtest))
print(np.shape(MnistYtest))
print(" ")

MnistXappReduit = MnistXapp[0:1000,:]
MnistYappReduit = MnistYapp[0:1000]
MnistXtestReduit = MnistXtest[:,:]
MnistYtestReduit = MnistYtest[:]

MnistDist = kppv_distances(MnistXtestReduit, MnistXappReduit)

KMax = 5

for K in range(1,KMax):

	MnistPredictions = kppv_predict(MnistDist, MnistYappReduit, K)

	Accuracy = evaluation_classifieur(MnistYtestReduit, MnistPredictions)
	print("K = ", K, "Accuracy : ", Accuracy)
print(" ")


def ExtractFeatures(X,n,m,c):
        HOG = X
        LBP = X
        
        for i in range(0,len(X)):
                image = X[i,:].reshape(n,m)
                P = 5
                R = 2.2
                HOG[i,0:81] = skimage.feature.hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3)).flatten()
                HOG[i,81:] = np.zeros((1,703))
                LBP[i,:] = skimage.feature.local_binary_pattern(image, P, R, method='default').flatten()
        
        return HOG, LBP
        
K = 5

MnistHOGapp, MnistLBPapp = ExtractFeatures(MnistXappReduit,28,28,1)
MnistHOGtest, MnistLBPtest = ExtractFeatures(MnistXtestReduit,28,28,1)

# HOG Features

MnistHOGDist = kppv_distances(MnistHOGtest, MnistHOGapp)
MnistHOGPred = kppv_predict(MnistHOGDist, MnistYappReduit, K)
print(MnistHOGPred)
MnistHOGAccuracy = evaluation_classifieur(MnistYtestReduit, MnistHOGPred)

print(MnistHOGAccuracy)

MnistLBPDist = kppv_distances(MnistLBPtest, MnistLBPapp)
MnistLBPPred = kppv_predict(MnistLBPDist, MnistYapp, K)
MnistLBPAccuracy = evaluation_classifieur(MnistYtest, MnistLBPPred)

print(MnistLBPAccuracy)


# Cross-Validation

def create_folds(X, Y, n):
        X_list = []
        Y_list = []
        
        for i in range (1,n):
                X_list.append(X[i:,:]) # Change !
                Y_list.append(Y[i:])
        
        #return X_listapp, Y_listapp, X_listtest, Y_listtest
        return X_list, Y_list, X_list, Y_list

def kppv_distances_forEachFold(X_listtest, X_listapp):
        Dist_list = []
        numFolds = len(X_listtest)
        
        for i in range(0,numFolds):
                Dist_list.append(kppv_distances(X_listtest[i], X_listapp[i]))
        
        return Dist_list

def kppv_predict_forEachFold(Dist_list, Y_listapp, K):
        Y_pred_list = []
        numFolds = len(Y_listapp)
        
        for i in range(0,numFolds):
                Y_pred_list.append(kppv_predict(Dist_list[i], Y_listapp[i], K))
        
        return Y_pred_list

def evaluation_CV(Y_listtest, Y_pred_list):
        Accuracy = 0
        numFolds = len(Y_listtest)
        
        for i in range(0,numFolds):
                Accuracy += evaluation_classifieur(Y_listtest[i], Ypred[i])
        
        Accuracy = Accuracy / numFolds
        
        return Accuracy

n = 10
K = 5

MnistX_listapp, MnistY_listapp, MnistX_listtest, MnistY_listtest = create_folds(MnistX[0:1000,:],MnistY[0:1000], n)
MnistDist_list = kppv_distances_forEachFold(MnistX_listtest, MnistX_listapp)
MnistY_pred_list = kppv_predict_forEachFold(MnistDist_list, MnistY_listapp, K)
MnistCVAccuracy = evaluation_CV(MnistY_listtest, MnistY_pred_list)

print(MnistCVAccuracy)
