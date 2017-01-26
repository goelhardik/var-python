import sys, os, pickle
import numpy as np
import scipy.linalg as slin
import matplotlib.pyplot as plt
import pdb

class VAR(object):
    def __init__(self, x_train, p=1):
        '''
        x_train : Training data : A list of arrays of shape (D, T)
        p : Order of VAR
        '''
        self.x_train = x_train
        self.p = p
        self.prepare_data()

    def prepare_data(self):
        p = self.p
        nshifts = p+1
        maxshift = p
        shifts = np.arange(p+1)
        yy = [[] for i in range(nshifts)]
        for ii in range(len(self.x_train)):
            # make data stationary?
            # self.x_train[ii] = self.stationarize(self.x_train[ii])
            y = self.x_train[ii]
            # normalize?
            ynorm = y
            # ynorm = self.minmax(y)
            #ynorm = self.zscore(y)
            T = ynorm.shape[1]
            start = maxshift
            endd = T
            for i in range(nshifts):
                yy[i].append(ynorm[:, start-shifts[i] : endd-shifts[i]])

        for i in range(nshifts):
            yy[i] = np.hstack(yy[i])

        # now yy is a list of arrays
        # contains target array as 1st array
        # each successive array is of one more time lag
        self.yy = yy

    def stationarize(self, x):
        return np.diff(x)

    def minmax(self, y):
        ymin = np.min(y, axis=1, keepdims=True)
        ymax = np.max(y, axis=1, keepdims=True)
        ynorm = (y-ymin) / (ymax-ymin)
        return ynorm

    def zscore(self, y):
        ymean = np.mean(y, axis=1, keepdims=True)
        ystd = np.std(y, axis=1, keepdims=True)
        ynorm = (y-ymean) / ystd
        return ynorm

    def fit(self):
        yy = self.yy
        p = self.p
        nshifts = p+1
        num_outputs = yy[0].shape[0]
        T = yy[0].shape[1]
        chunks = 100
        ind = np.arange(0, T, 100)
        if (ind[-1] != T):
            ind = np.hstack([ind, [T]])
        # training
        R = np.zeros((1,0))
        for i in range(len(ind)-1):
            start = ind[i]
            endd = ind[i+1]
            M1 = []
            for j in range(1, nshifts):
                tmp1 = yy[j]
                M1.append(tmp1[:, start:endd].T)
            M1 = np.hstack(M1)

            # takes kronecker product, makes X (100xD, T)
            X = np.kron(np.eye(num_outputs), M1)

            Y = yy[0]
            Y = Y[:, start:endd]
            # makes Y (100xD, 1)
            Y = Y.reshape(-1, 1)

            #q, r = np.linalg.qr(np.hstack([X, Y]))
            if (R.any()):
                r = slin.qr(np.vstack([R, np.hstack([X, Y])]), mode='r')
            else:
                r = slin.qr(np.hstack([X, Y]), mode='r')
            R = r[0]
            R = R[:min(R.shape), :]

        pdb.set_trace()
        M = R[:-1, :-1]
        by = R[:-1, -1]
        beta = np.dot(np.linalg.inv(np.dot(M.T, M)), np.dot(M.T, by))
        ind = np.arange((nshifts-1)*num_outputs*num_outputs)
        A = beta[ind]
        A = A.reshape(num_outputs, (nshifts-1)*num_outputs)
        print('A ', A.shape)
        AA = [[] for i in range(nshifts-1)]

        N = (nshifts-1)*num_outputs
        for i in range(nshifts-1):
            k = np.arange(i, N, nshifts-1)
            AA[i] = A[k, :]

        self.AA = AA

    def test(self, flight):
        p = self.p
        maxshift = p
        nshifts = p+1
        shifts = np.arange(p+1)
        num_outputs = flight.shape[0]
        # make data stationary?
        # flight = self.stationarize(flight)
        y = flight
        # normalize?
        ynorm = y
        # ynorm = self.minmax(y)
        #ynorm = self.zscore(y)
        yy = [[] for i in range(nshifts)]
        T = y.shape[1]

        start = maxshift
        endd = T
        for i in range(nshifts):
            yy[i].append(ynorm[:, start-shifts[i] : endd-shifts[i]])
        for i in range(nshifts):
            yy[i] = np.hstack(yy[i])

        out = yy[0]
        out_est = np.zeros_like(out)
        # 1-step prediction
        for i in range(nshifts-1):
            out_est += np.dot(self.AA[i], yy[i+1])

        #plt.plot(out[0])
        #plt.plot(out_est[0])
        #plt.show()
        err = out - out_est
        rmse = np.sqrt(np.mean(np.sum(np.square(err), axis=1)))
        return rmse

    def save(self):
        np.save('var_weights', self.AA)

    def load(self, filename='var_weights.npy'):
        self.AA = np.load(filename)
        print(self.AA.shape)

def trainVAR(var):
    print("Training VAR...")
    var.fit()
    var.save()
    return var

def testVAR(var, x_test):
    rmses = []
    for flight in x_test:
        rmses.append(var.test(flight))

    return rmses

p = int(sys.argv[1])
path = sys.argv[2]
files = os.listdir(path)
x_train = []
for afile in files:
    with open(os.path.join(path, afile), 'rb') as f:
        data = pickle.load(f)
        x_train.append(data)

var = VAR(x_train, p)
var = trainVAR(var)
var.load()

path = sys.argv[3]
files = os.listdir(path)
x_test = []
for afile in files:
    with open(os.path.join(path, afile), 'rb') as f:
        data = pickle.load(f)
        x_test.append(data)


rmses = testVAR(var, x_test)
print(rmses)
print("Mean RMSE ", np.mean(rmses))
