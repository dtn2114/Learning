"""
Test a learner.  (c) 2015 Tucker Balch
"""

import numpy as np
import math
import LinRegLearner as lrl
import sys
import DTLearner as dt
import RTLearner as rt
import BagLearner as bl
import InsaneLearner as it
import pandas as pd
from util import get_data, plot_data
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def namestr(obj, namespace):
    return [name for name in namespace if namespace[name] is obj]

def evaluate(learner,trainX, trainY, testX, testY):
    # evaluate in sample
    predY = learner.query(trainX) # get the predictions
    rmse_in = math.sqrt(((trainY - predY) ** 2).sum()/trainY.shape[0])
#    print
#    print "In sample results"
#    print "RMSE: ", rmse_in
    c_in = np.corrcoef(predY, y=trainY)
#    print "corr: ", c_in[0,1]

    # evaluate out of sample
    predY = learner.query(testX) # get the predictions
    rmse_out = math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0])
#    print
#    print "Out of sample results"
#    print "RMSE: ", rmse_out
    c_out = np.corrcoef(predY, y=testY)
#    print "corr: ", c_out[0,1]
    return rmse_in, c_in[0,1], rmse_out, c_out[0,1]

def plot(X, Y, titles="Chart", name="plot", labelY="y"):
    df_temp = pd.concat([X,Y], keys = ["inSample", "OutofSample"], axis=1)
    fig = df_temp.plot(title = titles, fontsize=12)
    fig.legend(["DT","RT"])
    fig.set_xlabel("leaf_size")
    fig.set_ylabel(labelY)
    plt.savefig(name)

def expOneRMSE_DT(trainX, trainY, testX, testY,nam, loop_size=100):
    rmseIn = pd.DataFrame(index=np.arange(0, loop_size), columns = ["rmse"])
    rmseOut = rmseIn.copy()
    for i in range(0,loop_size):
        dtlearner = dt.DTLearner(leaf_size = i, verbose = False)
        dtlearner.addEvidence(trainX, trainY)
        rmse_in, c_in, rmse_out, c_out = evaluate(dtlearner, trainX, trainY, testX, testY)
        rmseIn.loc[i] = [rmse_in]
        rmseOut.loc[i] = [rmse_out]
    plot(rmseIn, rmseOut, titles=nam, name = nam, labelY="rmse")

def expOneRMSE_RT(trainX, trainY, testX, testY,nam, loop_size=100):
    rmseIn = pd.DataFrame(index=np.arange(0, loop_size), columns = ["rmse"])
    rmseOut = rmseIn.copy()
    for i in range(0,loop_size):
        rtlearner = rt.RTLearner(leaf_size = i, verbose = False)
        rtlearner.addEvidence(trainX, trainY)
        rmse_in, c_in, rmse_out, c_out = evaluate(rtlearner, trainX, trainY, testX, testY)
        rmseIn.loc[i] = [rmse_in]
        rmseOut.loc[i] = [rmse_out]
    plot(rmseIn, rmseOut, titles=nam, name =nam, labelY="rmse")

def expOneCorr_DT(trainX, trainY, testX, testY,nam, loop_size=100):
    cIn = pd.DataFrame(index=np.arange(0, loop_size), columns = ["corr"])
    cOut = cIn.copy()
    for i in range(0,loop_size):
        dtlearner = dt.DTLearner(leaf_size = i, verbose = False)
        dtlearner.addEvidence(trainX, trainY)
        rmse_in, c_in, rmse_out, c_out = evaluate(dtlearner, trainX, trainY, testX, testY)
        cIn.loc[i] = [c_in]
        cOut.loc[i] = [c_out]
    plot(cIn, cOut, titles=nam, name = nam, labelY="corr")

def expOneCorr_RT(trainX, trainY, testX, testY,nam, loop_size=100):
    cIn = pd.DataFrame(index=np.arange(0, loop_size), columns = ["corr"])
    cOut = cIn.copy()
    for i in range(0,loop_size):
        rtlearner = rt.RTLearner(leaf_size = i, verbose = False)
        rtlearner.addEvidence(trainX, trainY)
        rmse_in, c_in, rmse_out, c_out = evaluate(rtlearner, trainX, trainY, testX, testY)
        cIn.loc[i] = [c_in]
        cOut.loc[i] = [c_out]
    plot(cIn, cOut, titles=nam, name =nam, labelY="corr")

def expTwoRMSE_DT(trainX, trainY, testX, testY, nam, loop_size=100):
    rmseIn = pd.DataFrame(index=np.arange(0, loop_size), columns = ["rmse"])
    rmseOut = rmseIn.copy()
    for i in range(0,loop_size):
        bllearner = bl.BagLearner(learner = dt.DTLearner, kwargs = {"leaf_size":i}\
                , bags = 20, boost = False, verbose = False)
        bllearner.addEvidence(trainX, trainY)
        rmse_in, c_in, rmse_out, c_out = evaluate(bllearner, trainX, trainY, testX, testY)
        rmseIn.loc[i] = [rmse_in]
        rmseOut.loc[i] = [rmse_out]
    plot(rmseIn, rmseOut, titles=nam, name = nam, labelY="rmse")

def expTwoRMSE_RT(trainX, trainY, testX, testY, nam, loop_size=100):
    rmseIn = pd.DataFrame(index=np.arange(0, loop_size), columns = ["rmse"])
    rmseOut = rmseIn.copy()
    for i in range(0,loop_size):
        bllearner = bl.BagLearner(learner = rt.RTLearner, kwargs = {"leaf_size":i}\
                , bags = 20, boost = False, verbose = False)
        bllearner.addEvidence(trainX, trainY)
        rmse_in, c_in, rmse_out, c_out = evaluate(bllearner, trainX, trainY, testX, testY)
        rmseIn.loc[i] = [rmse_in]
        rmseOut.loc[i] = [rmse_out]
    plot(rmseIn, rmseOut, titles=nam, name = nam, labelY="rmse")

def expTwoCorr_DT(trainX, trainY, testX, testY, nam, loop_size=100):
    cIn = pd.DataFrame(index=np.arange(0, loop_size), columns = ["corr"])
    cOut = cIn.copy()
    for i in range(0,loop_size):
        bllearner = bl.BagLearner(learner = dt.DTLearner, kwargs = {"leaf_size":i}\
                , bags = 20, boost = False, verbose = False)
        bllearner.addEvidence(trainX, trainY)
        rmse_in, c_in, rmse_out, c_out = evaluate(bllearner, trainX, trainY, testX, testY)
        cIn.loc[i] = [c_in]
        cOut.loc[i] = [c_out]
    plot(cIn, cOut, titles=nam, name = nam, labelY="corr")

def expTwoCorr_RT(trainX, trainY, testX, testY, nam, loop_size=100):
    cIn = pd.DataFrame(index=np.arange(0, loop_size), columns = ["corr"])
    cOut = cIn.copy()
    for i in range(0,loop_size):
        bllearner = bl.BagLearner(learner = rt.RTLearner, kwargs = {"leaf_size":i}\
                , bags = 20, boost = False, verbose = False)
        bllearner.addEvidence(trainX, trainY)
        rmse_in, c_in, rmse_out, c_out = evaluate(bllearner, trainX, trainY, testX, testY)
        cIn.loc[i] = [c_in]
        cOut.loc[i] = [c_out]
    plot(cIn, cOut, titles=nam, name = nam, labelY="corr")

def expTwoRMSE(trainX, trainY, testX, testY, nam, loop_size=100):
    rmseDT = pd.DataFrame(index=np.arange(0, loop_size), columns = ["rmse"])
    rmseRT = rmseDT.copy()
    for i in range(0,loop_size):
        bllearner = bl.BagLearner(learner = rt.RTLearner, kwargs = {"leaf_size":i}\
                , bags = 20, boost = False, verbose = False)
        bllearner.addEvidence(trainX, trainY)
        rmse_in, c_in, rmse_out, c_out = evaluate(bllearner, trainX, trainY, testX, testY)
        rmseRT.loc[i] = [rmse_out]
        bllearner1 = bl.BagLearner(learner = dt.DTLearner, kwargs = {"leaf_size":i}\
                , bags = 20, boost = False, verbose = False)
        bllearner1.addEvidence(trainX, trainY)
        rmse_in, c_in, rmse_out, c_out = evaluate(bllearner1, trainX, trainY, testX, testY)
        rmseDT.loc[i] = [rmse_out]
    plot(rmseDT, rmseRT, titles=nam, name = nam, labelY="rmse")

def expTwoCorr(trainX, trainY, testX, testY, nam, loop_size=100):
    corrDT = pd.DataFrame(index=np.arange(0, loop_size), columns = ["corr"])
    corrRT = corrDT.copy()
    for i in range(0,loop_size):
        bllearner = bl.BagLearner(learner = rt.RTLearner, kwargs = {"leaf_size":i}\
                , bags = 20, boost = False, verbose = False)
        bllearner.addEvidence(trainX, trainY)
        corr_in, c_in, rmse_out, c_out = evaluate(bllearner, trainX, trainY, testX, testY)
        corrRT.loc[i] = [c_out]
        bllearner1 = bl.BagLearner(learner = dt.DTLearner, kwargs = {"leaf_size":i}\
                , bags = 20, boost = False, verbose = False)
        bllearner1.addEvidence(trainX, trainY)
        rmse_in, c_in, rmse_out, c_out = evaluate(bllearner1, trainX, trainY, testX, testY)
        corrDT.loc[i] = [c_out]
    plot(corrDT, corrRT, titles=nam, name = nam, labelY="corr")




if __name__=="__main__":
    if len(sys.argv) != 2:
        print "Usage: python testlearner.py <filename>"
        sys.exit(1)
    inf = open(sys.argv[1])
    np.seterr(divide='ignore')
    qn = 2
    data = np.array([map(float,s.strip().split(',')[1:]) for s in inf.readlines()[1:]])
    # compute how much of the data is training and testing
    train_rows = int(0.6* data.shape[0])
    test_rows = data.shape[0] - train_rows


    if qn == 1:
        # separate out training and testing data
        trainX = data[:train_rows,0:-1]
        trainY = data[:train_rows,-1]
        testX = data[train_rows:,0:-1]
        testY = data[train_rows:,-1]
        fn = sys.argv[1][5:-4]
        name_DT = "NonBag_RMSE_DT_InOut_" + fn
        name_RT = "NonBag_RMSE_RT_InOut_" + fn
        nm_cor_DT = "NonBag_Corr_DT_InOUt_" + fn
        nm_cor_RT = "NonBag_Corr_RT_InOUt_" + fn
  #      expOneRMSE_DT(trainX, trainY, testX, testY, name_DT)
  #      expOneRMSE_RT(trainX, trainY, testX, testY, name_RT)
  #      expOneCorr_DT(trainX, trainY, testX, testY, nm_cor_DT)
  #      expOneCorr_RT(trainX, trainY, testX, testY, nm_cor_RT)

    if qn == 2:
        # select rows for question 2 experiment
        train_rows = np.random.randint(data.shape[0], size = train_rows)
        test_rows = np.random.randint(data.shape[0], size = test_rows)
        trainX = data[train_rows,0:-1]
        trainY = data[train_rows,-1]
        testX = data[test_rows,0:-1]
        testY = data[test_rows,-1]
        fn = sys.argv[1][5:-4]
        name_DT = "Bag_RMSE_DT_InOut_" + fn
        name_RT = "Bag_RMSE_RT_InOut_" + fn
        nm_cor_DT = "Bag_Corr_DT_InOUt_" + fn
        nm_cor_RT = "Bag_Corr_RT_InOUt_" + fn
        nm_rmse_DT_RT = "Bag_RMSE_RT_DT_Out_" +fn
        nm_corr_DT_RT = "Bag_Corr_RT_DT_Out_" +fn

 #       expTwoRMSE_DT(trainX, trainY, testX, testY, name_DT)
 #       expTwoRMSE_RT(trainX, trainY, testX, testY, name_RT)
#        expTwoCorr_DT(trainX, trainY, testX, testY, nm_cor_DT)
 #       expTwoCorr_RT(trainX, trainY, testX, testY, nm_cor_RT)
        expTwoRMSE(trainX, trainY, testX, testY, nm_rmse_DT_RT)
 #       expTwoCorr(trainX, trainY, testX, testY, nm_corr_DT_RT)




    '''
    # create a learner and train it
    learner = lrl.LinRegLearner(verbose = True) # create a LinRegLearner
    learner.addEvidence(trainX, trainY) # train it
    print learner.author()

    # create DTLearner and train it
    dtlearner = dt.DTLearner(leaf_size = 50, verbose = False) # create a DTLearner
    dtlearner.addEvidence(trainX, trainY) #train it
    Y = dtlearner.query(testX) #query
    print dtlearner.author()

    # create RTLearner and train it
    rtlearner = rt.RTLearner(leaf_size = 1, verbose = False) # create a RTLearner
    rtlearner.addEvidence(trainX, trainY) #train it
    Y = rtlearner.query(testX) #query
    print rtlearner.author()

    # create BagLearner of 20 DT w/ leaf_sz=1 and train it
    bllearner1 = bl.BagLearner(learner = dt.DTLearner, kwargs = {"leaf_size":1}\
                , bags = 20, boost = False, verbose = False)
    bllearner1.addEvidence(trainX, trainY)
    Y = bllearner1.query(testX)
    print bllearner1.author()

    # create BagLearner of 10 LR and train it
    bllearner2 = bl.BagLearner(learner = lrl.LinRegLearner, kwargs = {}\
                , bags = 10, boost = False, verbose = False)
    bllearner2.addEvidence(trainX, trainY)
    Y = bllearner2.query(testX)
    print bllearner2.author()

    #create InsaneLearner of 20 BagLearners w/ 20 LinRegLearner Instances
    itlearner = it.InsaneLearner(verbose = False) #constructor
    itlearner.addEvidence(trainX, trainY) #training step
    Y = itlearner.query(testX)
    print itlearner.author()

    learners = [dtlearner]
    #learners = [learner, dtlearner, rtlearner, bllearner1, bllearner2, itlearner]

    for learner in learners:
        # evaluating:
        print
        print "Evaluating ", namestr(learner, globals())

        rmse_in, c_in, rmse_out, c_out = evaluate(trainX, trainY, testX, testY)

        # evaluate in sample
        predY = learner.query(trainX) # get the predictions
        rmse = math.sqrt(((trainY - predY) ** 2).sum()/trainY.shape[0])
        print
        print "In sample results"
        print "RMSE: ", rmse
        c = np.corrcoef(predY, y=trainY)
        print "corr: ", c[0,1]

        # evaluate out of sample
        predY = learner.query(testX) # get the predictions
        rmse = math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0])
        print
        print "Out of sample results"
        print "RMSE: ", rmse
        c = np.corrcoef(predY, y=testY)
        print "corr: ", c[0,1]
        '''
