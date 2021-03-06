
from collections import defaultdict
import sys
import numpy as np
from mnist import MNIST 
#from scipy.spatial import distance

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


def kppv_distances(Xapp, Xtest):

    norm = lambda x,y,z : np.matrix(np.sum(np.matmul(x, y.transpose()), axis=z))

    return norm(Xapp,Xapp,0) + norm(Xtest,Xtest,1).transpose() - 2*np.matmul(Xapp,Xtest.transpose()).transpose()


def kppv_predict(Dist, Yapp, K):

    print(type(Dist), type(Yapp))

    bestKIndex = Dist.argsort()[:,:K]
    bestKClasses = Yapp[bestKIndex]

    print(Dist,Dist[0,bestKIndex],Yapp,Yapp[bestKIndex])

    for bests in bestKClasses:
        d = defaultdict(int)
        for i in bests:
            d[i] += 1

        result = max(d.iteritems(), key=lambda x: x[1])
        print result[0]

    return 0

if __name__ == "__main__":

    X, Y = lecture_mnist(path)
    a, b, c, d = decoupage_donnees(X[0:20], Y[0:20])

    dists = kppv_distances(a,c)

    kppv_predict(dists, b, 3)
