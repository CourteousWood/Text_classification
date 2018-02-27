import array
import collections
import itertools
import operator
import jieba
import sklearn
import sklearn.linear_model as linear_model
import sys
import math
import re
import matplotlib.pyplot as plt


class preprocess:
    def __int__(self):
        self.dict={}
    #将原始数据拆分成训练集和测试集
    def fetch_train_test(self,data_path,test_size=0.2):
        y = []
        text_list = []
        for line in open(data_path,'r',encoding='utf8').readlines():
            #r = '[’!"#$%&\'()*+,-.。：！，/:;<=>?@[\\]^_`{|}~]+'
            label,text = line[:-1].split('\t',1)
            #text=re.sub(r,'',text)
            y.append(int(label))
            text_list.append(list(jieba.cut(text)))
        return sklearn.model_selection.train_test_split(text_list,y,test_size=test_size,random_state=1)

    def fetch_train_test1(self,data_path):
        text_list = []
        text_label=[]
        for line in open(data_path,'r',encoding='gbk').readlines():
            label,text = line[:-1].split(':::',1)
            text_list.append(list(jieba.cut(text)))
            text_label.append(int(label))

        return text_list,text_label

    #创建字典
    def build_dict(self,text_list,min_freq=2):
        freq_dict = collections.Counter(itertools.chain(*text_list))
        freq_list = sorted(freq_dict.items(),key=operator.itemgetter(1),reverse=True)
        words,_ = zip(*filter(lambda wc:wc[1]>=min_freq,freq_list))
        self.dict=dict(zip(words, range(len(words))))
        return self.dict

    #抽取特征
    def text2vect(self,text_list):
        X = []
        P = []

        for key in self.dict:                   #计算字典中每个词在所有样本中出现的次数
            x = 0
            for text in text_list:
                if  key in text:
                    x+=1
            P.append(x)

        for text in text_list:
            vect = array.array('f', [0]*len(self.dict))
            for word in text:
                if word not in self.dict:
                    continue
                vect[self.dict[word]] = (text.count(word)/len(text))*math.log(len(text_list)/(P[self.dict[word]]+1))   #计算TF-IDF值
            X.append(vect)
        return X

    def writeToTxt(self,list_name,file_path):
        try:
            fp = open(file_path,"w+")
            for item in list_name:
                fp.write(str(item)+'\n')
            fp.close()
        except IOError:
            print("fail to open file")
    #模型评估
    def evaluate(self,model,X,y):
        accuracy = model.score(X, y)
        fpr,tpr,thresholds = sklearn.metrics.roc_curve(y, model.predict_proba(X)[:, 1], pos_label=1)
        return accuracy, sklearn.metrics.auc(fpr, tpr)
    def test_result(self,model,X,Y):
        pass;