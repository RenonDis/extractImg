
from collections import defaultdict
import sys
import numpy as np
from mnist import MNIST 
#from scipy.spatial import distance

#path = str(sys.argv[1])

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

    np.random.seed(1)
    N, D_in, D_h, D_out = 30, 2, 10, 3

    # Creation d'une matrice d'entree X et de sortie Y avec des valeurs aleatoires
    X = np.random.random((N, D_in))
    Y = np.random.random((N, D_out))

    # Initialisation aleatoire des poids du reseau
    W1 = 2 * np.random.random((D_in, D_h)) - 1
    b1 = np.zeros((1,D_h))
    W2 = 2 * np.random.random((D_h, D_out)) - 1
    b2 = np.zeros((1,D_out))

    ####################################################
    # Passe avant : calcul de la sortie predite Y_pred #
    ####################################################
    I1 = X.dot(W1) + b1 # Potentiel d'entree de la couche cachee
    O1 = 1/(1+np.exp(-I1)) # Sortie de la couche cachee (fonction d'activation de type sigmoide)
    I2 = O1.dot(W2) + b2 # Potentiel d'entree de la couche de sortie
    O2 = 1/(1+np.exp(-I2)) # Sortie de la couche de sortie (fonction d'activation de type sigmoide)
    Y_pred = O2 # Les valeurs predites sont les sorties de la couche de sortie
    
    ########################################################
    # Calcul et affichage de la fonction perte de type MSE #
    ########################################################
    loss = np.square(Y_pred - Y).sum() / 2
    print(loss)
