from collections import defaultdict
from mnist import MNIST
import numpy as np
import sys
import os
import math
import skimage
from skimage import feature as skft
import pickle

mnistPath = str(sys.argv[1])
cifarPath = str(sys.argv[2])

def lecture_mnist(path):

    mdata = MNIST(path)
    DataApp, LabelApp = mdata.load_training()
    DataTest, LabelTest = mdata.load_testing()
    
    return np.array(DataApp+DataTest) , np.array(LabelApp+LabelTest)


def lecture_cifar(path):
    X, Y = None, None

    for file in os.listdir(path):

        with open(os.path.join(path, file), 'rb') as cifile:
            dict = pickle.load(cifile, encoding='latin1')

            X = X + dict['data'] if type(X)==None else dict['data']
            Y = Y + dict['labels'] if type(Y)==None else dict['labels']

    return X,Y


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


def kppv_predict(Dist, Yapp, K):

    bestKIndex = Dist.argsort()[:,:K]
    bestKClasses = Yapp[bestKIndex]
    YPred =[]

    for bests in bestKClasses:
        d = defaultdict(int)
        for i in bests:
            d[i] += 1
        result = max(d.items(), key=lambda x: x[1])
        YPred.append(result[0])

    return YPred
        

evaluation_classifieur=lambda test,pred: np.count_nonzero(test==pred)/len(test)


def ExtractFeatures(X,n,m,c):
    HOG, LBP = np.empty(0), np.empty(0)
    
    for i in range(0,len(X)):
        im = X[i,:].reshape(n,m,c)

        im = np.mean(im, axis = 2)
        R = 1
        P = R*8

        HOGi = skft.hog(im)[np.newaxis,:]
        LBPi = skft.local_binary_pattern(im, P, R).flatten()[np.newaxis,:]

        HOG = np.concatenate((HOG,HOGi)) if HOG.size else HOGi
        LBP = np.concatenate((LBP,LBPi)) if LBP.size else LBPi

    return HOG, LBP

# Cross-Validation

create_folds = lambda X,Y,n : (np.split(X,n), np.split(Y,n))

def kppv_pred_Xfold(X_list, Y_list, K):
        Pred_list = []
        numFolds = len(X_list)
        
        for i in range(0,numFolds):
                X_applist = [i for i in X_list]
                Y_applist = [i for i in Y_list]

                X_test = X_list[i]

                X_applist.pop(i)
                Y_applist.pop(i)

                X_app = np.concatenate(X_applist)
                Y_app = np.concatenate(Y_applist)

                distI = kppv_distances(X_test, X_app)

                Pred_list.append(kppv_predict(distI, Y_app, K))

        return Pred_list


def evaluation_Xfold(Y_list, Pred_list):
        Acc = 0
        numFolds = len(Y_list)
        
        for i in range(0,numFolds):
                Acc += evaluation_classifieur(Y_list[i], Pred_list[i])
        
        return round(Acc/numFolds,3)



if __name__=='__main__':

    mnistX, mnistY = lecture_mnist(mnistPath)
    cifarX, cifarY = lecture_cifar(cifarPath)
    
    mnistXapp, mnistYapp, mnistXtest, mnistYtest = decoupage_donnees(mnistX, mnistY)
    
    mnistXappT = mnistXapp[0:500,:]
    mnistYappT = mnistYapp[0:500]

    mnistXtestT = mnistXtest[0:500,:]
    mnistYtestT = mnistYtest[0:500]   

    mnistDist = kppv_distances(mnistXtestT, mnistXappT)
    
    #Testing with different number of neighbours
    for K in range(1,8):
    
        mnistPred = kppv_predict(mnistDist, mnistYappT, K)
        mnistAcc = evaluation_classifieur(mnistYtestT, mnistPred)

        print(K,"-nn :")
        print(mnistAcc)

        mnistHOGapp, mnistLBPapp = ExtractFeatures(mnistXappT,28,28,1)
        mnistHOGtest, mnistLBPtest = ExtractFeatures(mnistXtestT,28,28,1)

        mnistHOGDist = kppv_distances(mnistHOGtest, mnistHOGapp)
        mnistHOGPred = kppv_predict(mnistHOGDist, mnistYappT, K)
        mnistHOGAcc = evaluation_classifieur(mnistYtestT, mnistHOGPred)
        
        print(mnistHOGAcc,"(HOG)")
        
        mnistLBPDist = kppv_distances(mnistLBPtest, mnistLBPapp)
        mnistLBPPred = kppv_predict(mnistLBPDist, mnistYappT, K)
        mnistLBPAcc = evaluation_classifieur(mnistYtestT, mnistLBPPred)
        
        print(mnistLBPAcc,"(LBP)")


    #Cross Validation

    Nfolds = 10
    Knn = 5
    print("Testing",Nfolds,"-fold X Validation on",K,"-nn:")

    mnistX_list, mnistY_list = create_folds(mnistX[0:500,:],mnistY[0:500], Nfolds)
    mnistPred_list = kppv_pred_Xfold(mnistX_list, mnistY_list, Knn)
    print(evaluation_Xfold(mnistY_list, mnistPred_list))
