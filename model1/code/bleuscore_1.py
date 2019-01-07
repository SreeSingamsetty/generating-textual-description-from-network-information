# -*- coding: utf-8 -*-
from pickle import load
from numpy import array
from numpy import argmax
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate import bleu
import pandas as pd

chencherry=SmoothingFunction()

testresultfilename="results_nocalc_100d_adadelta_bluescore_new.txt" #enter the testing output file name
dataset="33_absdataset.csv"

testbleuscorefile=testresultfilename[:-4]+"_bluescore.txt"

resultdf=pd.read_csv(testresultfilename,sep='\t',header=None)

print("resultdf shape: ",resultdf.shape)

datasetdf=pd.read_csv(dataset,sep='\t',header=None)
print("dataset shape: ",datasetdf.shape)


with open(testbleuscorefile,'w')as w:
    for name in resultdf.ix[:,0]:
        pred_list = (resultdf.ix[resultdf.ix[:,0].str.match(name), 1].values[0])
        if pred_list==None :
            b_score=0
            pred_string="not predicted"
        else:
            pred_string=pred_list.split(sep=' ')
            obj_list = (datasetdf.ix[datasetdf.ix[:,0].str.match(name), 1].values[0:])
            mystringl1 = [i.split() for i in obj_list]
            if obj_list is None:
                b_score=0
            else:
                b_score=sentence_bleu(mystringl1, pred_string,smoothing_function=chencherry.method1)
        w.write(name+'\t'+str(mystringl1)+'\t'+str(pred_string)+'\t'+str(b_score)+'\n')
