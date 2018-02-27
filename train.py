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
from utils import preprocess


from functools import wraps
from datetime import datetime


def log_time_delta(func):
    @wraps(func)
    def deco(*args, **args2):
        start = datetime.now()
        res = func(*args, **args2)
        end = datetime.now()
        delta = end - start
        print("这个程序运行了多长时间:", delta)
        return res
    return deco



def lr(prepro):
    prepro=prepro
    X_train, X_test, y_train, y_test = prepro.fetch_train_test("datas/train.txt")
    word2id = prepro.build_dict(X_train, min_freq=2)
    X_train = prepro.text2vect(X_train)
    X_test = prepro.text2vect(X_test)
    lr = linear_model.LogisticRegression(penalty='l2', C=40,random_state=1)
    lr.fit(X_train, y_train)
    # step 5. 模型评估
    accuracy, auc = prepro.evaluate(lr, X_train, y_train)
    sys.stdout.write("训练集正确率：%.4f%%\n" % (accuracy * 100))
    sys.stdout.write("训练集AUC值：%.6f\n" % (auc))

    accuracy1, auc1 = prepro.evaluate(lr, X_test, y_test)
    sys.stdout.write("测试集正确率：%.4f%%\n" % (accuracy1 * 100))
    sys.stdout.write("测试AUC值：%.6f\n" % (auc1))
    return lr
def pred(prepro,model):

    X,Y= prepro.fetch_train_test1("datas/test_result.txt")
    X_test = prepro.text2vect(X)
    accuracy, auc=prepro.evaluate(model,X_test,Y)
    sys.stdout.write("测试集正确率：%.4f%%\n" % (accuracy * 100))
    sys.stdout.write("测试AUC值：%.6f\n" % (auc))


@log_time_delta
def mul_NB(prepro):
    X_train, X_test, y_train, y_test = prepro.fetch_train_test("./datas/train.txt", test_size=0.2)
    word2id = prepro.build_dict(X_train, min_freq=2)
    X_train = prepro.text2vect(X_train)
    X_test = prepro.text2vect(X_test)

    import numpy as np
    from sklearn.naive_bayes import MultinomialNB

    train_score = []
    test_score = []
    alphas = np.logspace(-5, 1, num=10)
    for alpha in alphas:
        bayes = MultinomialNB(alpha=alpha)
        bayes.fit(X_train, y_train)
        train_score.append(bayes.score(X_train, y_train))
        test_score.append(bayes.score(X_test, y_test))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(alphas, train_score, label='train score')
    ax.plot(alphas, test_score, label='test score')
    ax.set_xlabel("alpha")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.0)
    ax.set_title("MultinomialNB")
    ax.set_xscale('log')
    plt.show()


@log_time_delta
def mul_NB1(prepro):
    X_train, X_test, y_train, y_test = prepro.fetch_train_test("./datas/train.txt", test_size=0.2)
    word2id = prepro.build_dict(X_train, min_freq=2)
    X_train = prepro.text2vect(X_train)
    X_test = prepro.text2vect(X_test)

    import numpy as np
    from sklearn.naive_bayes import MultinomialNB


    bayes = MultinomialNB(alpha=0.1)
    bayes.fit(X_train, y_train)

    accuracy, auc = prepro.evaluate(bayes, X_test, y_test)
    sys.stdout.write("训练集正确率：%.4f%%\n" % (accuracy * 100))
    sys.stdout.write("训练集AUC值：%.6f\n" % (auc))
    return bayes

@log_time_delta
def mul_NB2(prepro):
    X_train, X_test, y_train, y_test = prepro.fetch_train_test("./datas/train.txt", test_size=0.2)
    word2id = prepro.build_dict(X_train, min_freq=2)
    X_train = prepro.text2vect(X_train)
    X_test = prepro.text2vect(X_test)

    import numpy as np
    from sklearn.naive_bayes import MultinomialNB

    alpha_can = np.logspace(-5, 1, num=10)
    from sklearn.model_selection import GridSearchCV
    model=MultinomialNB()
    NB_model = GridSearchCV(model, param_grid={'alpha': alpha_can}, cv=5)
    NB_model.fit(X_train, y_train)

    print('The parameters of the best model are: ')
    print(NB_model.best_params_)

    accuracy, auc = prepro.evaluate(NB_model, X_test, y_test)
    sys.stdout.write("训练集正确率：%.4f%%\n" % (accuracy * 100))
    sys.stdout.write("训练集AUC值：%.6f\n" % (auc))
    return NB_model

@log_time_delta
def Decision_Tree(prepro,test_size=0.2,min_freq=2, max_freq=500,criterion ='entropy',splitter='best',max_depth=200,
    min_samples_split=2,min_samples_leaf=2):
    '''分类决策树'''
    from sklearn.tree import DecisionTreeClassifier
    from sklearn import tree


    X_train, X_test, y_train, y_test = prepro.fetch_train_test("./datas/train.txt", test_size=0.2)

    word2id = prepro.build_dict(X_train, min_freq=min_freq)
    X_train = prepro.text2vect(X_train)
    X_test = prepro.text2vect(X_test)
    DecisionTree = DecisionTreeClassifier(criterion=criterion,
                                          splitter=splitter,max_depth=200)
    alpha_can = [100,150,200,300]
    from sklearn.model_selection import GridSearchCV
    DecisionTree = GridSearchCV(DecisionTree, param_grid={'max_depth': alpha_can}, cv=5)


    result = DecisionTree.fit(X_train, y_train)

    accuracy, auc = prepro.evaluate(DecisionTree, X_test, y_test)
    print("当训练集正确率：%.4f%%" % (accuracy * 100))
    print("训练集AUC值：%.6f" % (auc))
    return DecisionTree
@log_time_delta
def GBDT(prepro):
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.metrics import accuracy_score
    X_train, X_test, y_train, y_test = prepro.fetch_train_test("./datas/train.txt", test_size=0.2)
    word2id = prepro.build_dict(X_train, min_freq=2)
    X_train = prepro.text2vect(X_train)
    X_test = prepro.text2vect(X_test)
    # 随机森林分类
    rf = RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_split=5, oob_score=True)
    rf.fit(X_train, y_train)
    print(u'OOB Score=%.5f',rf.oob_score)
    y_train_pred = rf.predict(X_train)
    y_test_pred = rf.predict(X_test)
    print(u'随机森林训练集准确率：', accuracy_score(y_train, y_train_pred))
    print(u'随机森林测试集准确率：', accuracy_score(y_test, y_test_pred))

    # dbdt 分类
    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=2)
    gb.fit(X_train, y_train)
    y_train_pred = gb.predict(X_train)
    y_test_pred = gb.predict(X_test)
    print(u'GBDT训练集准确率：', accuracy_score(y_train, y_train_pred))
    print(u'GBDT测试集准确率：', accuracy_score(y_test, y_test_pred))
# 逻辑回归的 预测.
if __name__ == '__main__':


    dict={"lr":0, "mul_Bayes":1,"de_tree":2,"GBDT" :3,"rnn":4}
    choose="GBDT"

    # 逻辑回归的解决问题, 当c 1 的时候 准确率只有 92左右,当C变成了40
    if(dict[choose]==0):
        prepro = preprocess()
        lr=lr(prepro);
        sys.stdout.write("测试结果如下了 :")
        pred(prepro,lr)

    # 多分类贝叶斯:
    if(dict[choose]==1):
        prepro1 = preprocess()
        NB1=mul_NB1(prepro1)
        pred(prepro1, NB1)
        sys.stdout.write("第二个贝叶斯概率\n")
        NB2=mul_NB2(prepro1)
        pred(prepro1, NB2)


    if(dict[choose]==2):
        prepro2=preprocess()
        DTree=Decision_Tree(prepro2)
        pred(prepro2,DTree)

    if(dict[choose]==3):
        prepro3=preprocess()
        DTree=GBDT(prepro3)
        pred(prepro3,DTree)
