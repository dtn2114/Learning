"""
BagLearner Classes
"""

import numpy as np

class BagLearner(object):

    def __init__(self,learner, bags, kwargs=None, boost = False, verbose = False):
        self.verbose = verbose
        self.learner = learner
        self.bags = bags
        self.boost = boost
        self.kwargs = kwargs

        learners = []
        #kwargs = {'k' : 10}
        for i in range(0,bags):
            learners.append(learner(**kwargs))

        self.learners = learners


    def author(self):
        return 'dnguyen333' # replace tb34 with your Georgia Tech username

    def addEvidence(self,dataX,dataY):
        n = dataX.shape[0]

        for lnr in self.learners:
            rows = np.random.randint(n, size = len(dataY))
            sampleX = dataX[rows,:]
            sampleY = dataY[rows]
            lnr.addEvidence(sampleX, sampleY)

    def query(self,points):
        result = np.zeros(points.shape[0])
        for lnr in self.learners:
            result += lnr.query(points)
        result = result/self.bags
        return result

if __name__=="__main__":
    print "the secret clue is 'zzyzx'"
