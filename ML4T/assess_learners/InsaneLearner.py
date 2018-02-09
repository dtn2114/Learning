import numpy as np, BagLearner as bl, LinRegLearner as lrl
class InsaneLearner(object):
    def __init__(self, verbose = False): self.verbose = verbose
    def author(self): return 'dnguyen333, InsaneLearner' # replace tb34 with your Georgia Tech username
    def addEvidence(self,dataX,dataY):
        bllearners = []
        for i in range(0,20):
            bllearner = bl.BagLearner(learner=lrl.LinRegLearner, kwargs={}, bags = 20, boost = False, verbose =False)
            bllearner.addEvidence(dataX, dataY)
            bllearners.append(bllearner)
        self.bllearners = bllearners
    def query(self,points):
        result = np.zeros(points.shape[0])
        for lnr in self.bllearners:
            result += lnr.query(points)
        return result/20
if __name__=="__main__": print "the secret clue is 'zzyzx'"
