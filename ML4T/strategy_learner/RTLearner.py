"""
A simple wrapper for Random Tree Learner
"""

import numpy as np
import random as rd

class RTLearner(object):

    def __init__(self, leaf_size, verbose = False):
        self.leaf_size = leaf_size
        self.verbose = verbose
        pass

    def author(self):
        return 'dnguyen333' # replace tb34 with your Georgia Tech username

    def addEvidence(self,dataX,dataY):
        """
        @summary: Add training data to learner
        @param dataX: X values of data to add
        @param dataY: the Y training values
        """
        self.tree = self.build_tree(dataX, dataY)

    def build_tree(self, dataX, dataY):

        'if data only has 1 row'
        if dataX.shape[0] == 1:
            return np.array([[-1, dataY[0], np.NAN, np.NAN]])
        elif len(np.unique(dataY)) == 1: # if data is all the same, leaf it
            return np.array([[-1, dataY[0], np.NAN, np.NAN]])
        elif dataX.shape[0] <= self.leaf_size: #if the data pts in brance less than leaf_sz)
            return np.array([[-1, np.mean(dataY), np.NAN, np.NAN]])
        else:
            # generate the random list of index
            rdList =  rd.sample(xrange(0, dataX.shape[1]), dataX.shape[1])

            f_index = rdList[0]
            count = 1
            # move to the next index if all the values of the feature is the same
            while len(np.unique(dataX[:,f_index])) == 1 and count <= dataX.shape[1]:
                f_index = rdList[count]
                count += 1


            splitVal = np.median(dataX[:,f_index])
            data = np.column_stack((dataX, dataY))

            # print (data < splitVal).all()

            # leaf a feature if all the values of the feature is the same
            # also leaf a feature if all the values is smaller than splitVal
            if len(np.unique(data[:,f_index])) ==1 \
                or len(data[data[:,f_index] > splitVal]) == 0 :
                return np.array([[-1,np.mean(dataY),np.NAN, np.NAN]])
            else:
                '''
                if data.shape[0] < 5:
                    print "data size: ", data.shape
                    print "splitVal: ", splitVal
                    print "f_index, data : ", f_index, data[:, f_index]
                '''

                lData = data[data[:,f_index] <= splitVal]
                rData = data[data[:,f_index] > splitVal]
                left_tree = self.build_tree(lData[:,:-1],lData[:,-1])
                right_tree = self.build_tree(rData[:,:-1],rData[:,-1])

                root = np.array([f_index, splitVal, 1, left_tree.shape[0]+1])
                return np.vstack((root, left_tree, right_tree))


    def get_label(self, arr, tRow=0):
        f_index = int(self.tree[tRow][0])
        if f_index == -1:
            return self.tree[tRow][1]
        if arr[f_index] <= self.tree[tRow][1]:
            return self.get_label(arr, tRow + int(self.tree[tRow][2]))
        else:
            return self.get_label(arr, tRow + int(self.tree[tRow][3]))


    def query(self,dataX):
        result = []
        for arr in dataX:
            result.append(self.get_label(arr))
        return result

if __name__=="__main__":
    print "the secret clue is 'zzyzx'"
