from collections import Counter, defaultdict
import numpy as np
import NGram
import math
from sklearn.metrics.pairwise import cosine_similarity as dist
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier

from gensim.models.word2vec import Word2Vec
import gensim.downloader as api
info = api.info()  # show info about available models/datasets
corpus1 = api.load("text8")  # download the model and return as object ready for use
model1=Word2Vec(corpus1)
# FilePath: "../test/test.txt"

def Get_Vector(Paragraph):
    '''
    Input a paragraph, return log probability calculated on given Bigram
    '''
    p_vec = np.zeros([100,])
    cnt = 1
    for token in Paragraph:
        if token in model1:
            cnt += 1
            p_vec += model1[token]
    return p_vec/float(cnt)

def Predict_Ngram(Inpath = "../test/test.txt", Outpath = "../Output/8_1.csv", Train_Trump = "../train/trump.txt", Train_Obama = "../train/obama.txt"):
    '''
    Input:
        Inpath  : file path for test data
        Outpath : file path for output csv file
        Train_Trump : file path to train Trump's bigram model
        Train_Obama : file path to train Obama's bigram model
    Output:
        Return None, Output should go straight to .csv file
    '''
    f = open(Outpath, 'w')
    f.write('Id,Prediction\n')
    #Preprocess the test set
    Paragraphs_Trump = NGram.corpora_preprocess(Train_Trump)
    P_Vecs_Trump = [Get_Vector(p) for p in Paragraphs_Trump]
    Paragraphs_Obama = NGram.corpora_preprocess(Train_Obama)
    P_Vecs_Obama = [Get_Vector(p) for p in Paragraphs_Obama]
    Paragraphs_test = NGram.corpora_preprocess(Inpath)
    P_Vecs_test = [Get_Vector(p) for p in Paragraphs_test]
    for idx, pvec in enumerate(P_Vecs_test):
        max_cosine = -50
        isTrump = True
        for pv_trump in P_Vecs_Trump:
            max_cosine = max(max_cosine, dist(pvec, pv_trump))
        for pv_obama in P_Vecs_Obama:
            if dist(pvec, pv_obama)>max_cosine:
                isTrump = False
                break
        f.write(str(idx)+',')
        if isTrump:
            f.write('1')
        else:
            f.write('0')
        f.write('\n')

def Preprocess(Train_Trump = "../train/trump.txt", Train_Obama = "../train/obama.txt"):
    Paragraphs_Trump = NGram.corpora_preprocess(Train_Trump)
    P_Vecs_Trump = np.array([Get_Vector(p) for p in Paragraphs_Trump])
    Paragraphs_Obama = NGram.corpora_preprocess(Train_Obama)
    P_Vecs_Obama = np.array([Get_Vector(p) for p in Paragraphs_Obama])
    X = np.concatenate((P_Vecs_Trump,P_Vecs_Obama))
    Y = np.concatenate((np.ones([3100,]),np.zeros([3100,])))
    return X,Y

def Train_SGD(Outpath = '../Output/sgd.csv'):
    clf = SGDClassifier(loss="log", penalty="elasticnet")
    Paragraphs_test = NGram.corpora_preprocess("../test/test.txt")
    P_Vecs_test = np.array([Get_Vector(p) for p in Paragraphs_test[:-1]])
    X,Y = Preprocess()
    clf.fit(X,Y)
    results = clf.predict(P_Vecs_test)
    Write_Result(OutPath, results)

def Train_RandomForest(Outpath = '../Output/forest2.csv', dp =50):
    clf = RandomForestClassifier(max_depth=dp, random_state=0)
    Paragraphs_test = NGram.corpora_preprocess("../test/test.txt")
    P_Vecs_test = np.array([Get_Vector(p) for p in Paragraphs_test[:-1]])
    X,Y = Preprocess()
    clf.fit(X,Y)
    results = clf.predict(P_Vecs_test)
    Write_Result(Outpath, results)

def Write_Result(Outpath, results):
    f = open(Outpath, 'w')
    f.write('Id,Prediction\n')
    for idx,label in enumerate(results):
        f.write(str(idx)+',')
        if label == 1:
            f.write('1')
        else:
            f.write('0')
        f.write('\n')


Predict_Ngram()
