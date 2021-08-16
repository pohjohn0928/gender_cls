import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report
from opencc import OpenCC
import jieba
from datahelper import make_csv,readTestCSV

from tensorflow.python.keras.layers import InputLayer,Lambda,Dense
import bert
import tensorflow.keras as keras
from transformers import AutoTokenizer
import os

import pickle
import time


class PassiveAggressiveCls:
    def getStopwordZh(self):
        file_name = '../stopword/stopwords-zh.txt'
        cc = OpenCC('s2t')
        file = open(file_name, encoding='utf-8')
        chinese_stopword = []
        for word in file:
            word = word.replace('\n', '')
            word = cc.convert(word)
            chinese_stopword.append(word)
        return chinese_stopword

    def fit(self):
        x_train, y_train = readTestCSV('train_datas.csv')
        chinese_stopword = self.getStopwordZh()
        for i in range(len(x_train)):
            x_train[i] = ' '.join(jieba.lcut(x_train[i]))

        tfidf_vectorizer = TfidfVectorizer(stop_words=chinese_stopword, max_df=0.7)
        tfidf_train = tfidf_vectorizer.fit_transform(x_train)

        pac=PassiveAggressiveClassifier(max_iter=50)
        pac.fit(tfidf_train,y_train)
        with open('../PassiveAggressiveModel/PassiveAggressiveModel.pickle', 'wb') as f:
            pickle.dump(pac, f)
        with open('../PassiveAggressiveModel/tfidf.pickle', 'wb') as f:
            pickle.dump(tfidf_vectorizer, f)

    def evaluate(self):
        x_test, y_test = readTestCSV('test_datas.csv')

        start = time.time()
        for j in range(len(x_test)):
            x_test[j] = ' '.join(jieba.lcut(x_test[j]))
        pac = pickle.load(open('../PassiveAggressiveModel/PassiveAggressiveModel.pickle', 'rb'))
        tfidf_vectorizer = pickle.load(open('../PassiveAggressiveModel/tfidf.pickle', 'rb'))

        tfidf_test = tfidf_vectorizer.transform(x_test)
        y_pred = pac.predict(tfidf_test)
        end = time.time()
        print(f'Tfidf + pas : Total Spend {end - start} sec')

        score = accuracy_score(y_test, y_pred)
        print(f'Accuracy: {round(score * 100, 2)}%')
        report = classification_report(y_test, y_pred)
        print(report)


pac_model = PassiveAggressiveCls()
pac_model.evaluate()


class AlbertFB:
# Albert feature base
    def tokenizeData(self,data):
        tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
        tokens = tokenizer.batch_encode_plus(
            data,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='tf')
        return np.array(tokens['input_ids'])

    def createBertModle(self):
        model_name = "albert_tiny"
        model_dir = bert.fetch_brightmart_albert_model(model_name, "../.bert_models")
        model_ckpt = os.path.join(model_dir, "albert_model.ckpt")
        bert_params = bert.params_from_pretrained_ckpt(model_dir)
        l_bert = bert.BertModelLayer.from_params(bert_params, name="bert")
        l_bert.trainable = True

        model = keras.models.Sequential()
        model.add(InputLayer(input_shape=512, ))
        model.add(l_bert)
        model.add(Lambda(lambda x: x[:, 0, :]))

        model.build(input_shape=(None, 512))
        bert.load_albert_weights(l_bert, model_ckpt)
        return model

    def predict(self,contents,labels):
        model = self.createBertModle()
        input_id_test = self.tokenizeData(contents)
        output_test_contents = model.predict(input_id_test)
        pac = pickle.load(open('../albertFeatureBase/feature_base_pas.pickle', 'rb'))
        y_pred=pac.predict(output_test_contents)
        score=accuracy_score(labels,y_pred)
        print(f'Accuracy: {round(score*100,2)}%')
        report = classification_report(labels, y_pred)
        print(report)

    def evaluate(self):
        df_test = pd.read_csv('test_datas.csv')
        test_labels = df_test.label.values
        test_contents = df_test.content.values
        test_contents = list(test_contents)

        start = time.time()
        self.predict(test_contents,test_labels)
        end = time.time()
        print(f'albert feature base : Total Spend {end - start} sec')


