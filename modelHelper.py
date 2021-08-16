# -*- coding: utf-8 -*-
import pickle
import time

import bert
import os

import jieba
import tensorflow.keras as keras
import tensorflow as tf
from opencc import OpenCC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.python.keras.layers import InputLayer,Lambda,Dense,Dropout
from transformers import AutoTokenizer, TFBertForSequenceClassification, BertTokenizer
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping

from datahelper import readTestCSV
import ssl
from xgboost import XGBClassifier

ssl._create_default_https_context = ssl._create_unverified_context


class audientModel:
    def __init__(self):
        self.max_length = 512

    def tokenizeData(self, data):
        tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
        tokens = tokenizer.batch_encode_plus(
            data,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='tf')
        return np.array(tokens['input_ids'])


class XgboostCls:
    def getStopwordZh(self):
        file_name = 'stopword/stopwords-zh.txt'
        cc = OpenCC('s2t')
        file = open(file_name, encoding='utf-8')
        chinese_stopword = []
        for word in file:
            word = word.replace('\n', '')
            word = cc.convert(word)
            chinese_stopword.append(word)
        return chinese_stopword

    def fit(self,x_train, y_train):
        chinese_stopword = self.getStopwordZh()
        for i in range(len(x_train)):
            x_train[i] = ' '.join(jieba.lcut(x_train[i]))

        tfidf_vectorizer = TfidfVectorizer(stop_words=chinese_stopword, max_df=0.7)
        tfidf_train = tfidf_vectorizer.fit_transform(x_train)

        xgbc = XGBClassifier(use_label_encoder=False)
        xgbc.fit(tfidf_train, y_train)
        with open('xgboost/xgboostModel.pickle', 'wb') as f:
            pickle.dump(xgbc, f)
        with open('xgboost/tfidf.pickle', 'wb') as f:
            pickle.dump(tfidf_vectorizer, f)

    def load(self):
        self.model = pickle.load(open('models/xgboost/xgboostModel.pickle', 'rb'))
        self.tfidf_vectorizer = pickle.load(open('models/xgboost/tfidf.pickle', 'rb'))

    def predict(self,contents):
        self.load()
        for i in range(len(contents)):
            contents[i] = ' '.join(jieba.lcut(contents[i]))
        tfidf_test = self.tfidf_vectorizer.transform(contents)
        y_pred = self.model.predict(tfidf_test)
        # y_pred_prob = self.model.predict_proba(tfidf_test)
        return y_pred

    def evaluate(self):
        x_test,y_test = readTestCSV('test_data/test_datas.csv')

        start = time.time()
        for j in range(len(x_test)):
            x_test[j] = ' '.join(jieba.lcut(x_test[j]))

        self.load()

        tfidf_test = self.tfidf_vectorizer.transform(x_test)
        y_pred = self.model.predict(tfidf_test)
        end = time.time()
        print(f'Tfidf + xgboost : Total Spend {end - start} sec')

        score = accuracy_score(y_test, y_pred)
        print(f'Accuracy: {round(score * 100, 2)}%')
        report = classification_report(y_test, y_pred)
        print(report)

    def train(self, train_data_path):
        x_train, y_train = readTestCSV(train_data_path)
        self.fit(x_train, y_train)



class PassiveAggressiveCls:
    def getStopwordZh(self):
        file_name = 'stopword/stopwords-zh.txt'
        cc = OpenCC('s2t')
        file = open(file_name, encoding='utf-8')
        chinese_stopword = []
        for word in file:
            word = word.replace('\n', '')
            word = cc.convert(word)
            chinese_stopword.append(word)
        return chinese_stopword

    def fit(self,x_train, y_train):
        chinese_stopword = self.getStopwordZh()
        for i in range(len(x_train)):
            x_train[i] = ' '.join(jieba.lcut(x_train[i]))

        tfidf_vectorizer = TfidfVectorizer(stop_words=chinese_stopword, max_df=0.7)
        tfidf_train = tfidf_vectorizer.fit_transform(x_train)

        pac = PassiveAggressiveClassifier(max_iter=50)
        pac.fit(tfidf_train,y_train)
        with open('models/PassiveAggressiveModel/PassiveAggressiveModel.pickle', 'wb') as f:
            pickle.dump(pac, f)
        with open('models/PassiveAggressiveModel/tfidf.pickle', 'wb') as f:
            pickle.dump(tfidf_vectorizer, f)

    def predict(self, model, contents):
        self.load()
        for i in range(len(contents)):
            contents[i] = ' '.join(jieba.lcut(contents[i]))
        tfidf_test = self.tfidf_vectorizer.transform(contents)
        y_pred = model.predict(tfidf_test)
        return y_pred

    def load(self):
        # self.pac = pickle.load(open('PassiveAggressiveModel/PassiveAggressiveModel.pickle', 'rb'))
        self.tfidf_vectorizer = pickle.load(open('models/PassiveAggressiveModel/tfidf.pickle', 'rb'))

    def evaluate(self):
        x_test, y_test = readTestCSV('test_data/test_datas.csv')
        start = time.time()
        for j in range(len(x_test)):
            x_test[j] = ' '.join(jieba.lcut(x_test[j]))
        pac = pickle.load(open('models/PassiveAggressiveModel/PassiveAggressiveModel.pickle', 'rb'))
        tfidf_vectorizer = pickle.load(open('models/PassiveAggressiveModel/tfidf.pickle', 'rb'))

        tfidf_test = tfidf_vectorizer.transform(x_test)
        y_pred = pac.predict(tfidf_test)

        end = time.time()
        print(f'Tfidf + pas : Total Spend {end - start} sec')

        score = accuracy_score(y_test, y_pred)
        print(f'Accuracy: {round(score * 100, 2)}%')
        report = classification_report(y_test, y_pred)
        print(report)

    def partial_fit(self,model,content,label):
        self.load()
        content = ' '.join(jieba.lcut(content))
        content = self.tfidf_vectorizer.transform([content])
        model.partial_fit(content,[label])
        return model.predict(content),model

    def train(self, train_data_path):
        x_train, y_train = readTestCSV(train_data_path)
        self.fit(x_train, y_train)

class AlbertModelFT(audientModel):
    def createModle(self):
        model_name = "albert_tiny"
        model_dir = bert.fetch_brightmart_albert_model(model_name, ".albert_models")
        model_ckpt = os.path.join(model_dir, "albert_model.ckpt")
        bert_params = bert.params_from_pretrained_ckpt(model_dir)
        l_bert = bert.BertModelLayer.from_params(bert_params, name="bert")
        l_bert.trainable = True

        model = keras.models.Sequential()
        model.add(InputLayer(input_shape=self.max_length, ))
        model.add(l_bert)
        model.add(Lambda(lambda x: x[:, 0, :]))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        model.build(input_shape=(None, self.max_length))
        bert.load_albert_weights(l_bert, model_ckpt)
        return model

    def fit(self, x_train, y_train, x_test, y_test):
        x_train = self.tokenizeData(x_train)
        x_test = self.tokenizeData(x_test)

        y_train = np.array(y_train)
        y_test = np.array(y_test)

        model = self.createModle()

        earlystop_callback = EarlyStopping(
            monitor='val_accuracy',
            min_delta=0.0001,  # 精確度至少提高0.0001
            patience=3)

        checkpoint_path = "models/albertFineTunning/gender_cls.hdf5"
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            save_weights_only=True,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            verbose=1)

        model.compile(loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.Adam(),
                      metrics=['accuracy'])

        model.fit(x=x_train, y=y_train, validation_data=(x_test, y_test), epochs=10, batch_size=32, verbose=1,
                  callbacks=[earlystop_callback, model_checkpoint_callback])

    def load(self):
        self.model = self.createModle()
        checkpoint_path = f"models/albertFineTunning/gender_cls.hdf5"
        self.model.load_weights(checkpoint_path)

    def predict(self, contents):
        self.load()
        contents = self.tokenizeData(contents)
        return self.model.predict(contents)

    def train(self, train_data_path, test_data_path):
        x_train, y_train = readTestCSV(train_data_path)
        x_test, y_test = readTestCSV(test_data_path)
        self.fit(x_train, y_train, x_test, y_test)


class AlbertModelFB(audientModel):
    def createBertModle(self):
        model_name = "albert_tiny"
        model_dir = bert.fetch_brightmart_albert_model(model_name, ".albert_models")
        model_ckpt = os.path.join(model_dir, "albert_model.ckpt")
        bert_params = bert.params_from_pretrained_ckpt(model_dir)
        l_bert = bert.BertModelLayer.from_params(bert_params, name="bert")
        l_bert.trainable = False

        model = keras.models.Sequential()
        model.add(InputLayer(input_shape=self.max_length, ))
        model.add(l_bert)
        model.add(Lambda(lambda x: x[:, 0, :]))

        model.build(input_shape=(None, self.max_length))
        bert.load_albert_weights(l_bert, model_ckpt)
        return model

    def fit(self, x_train, y_train):
        x_train = self.tokenizeData(x_train)
        y_train = np.array(y_train)

        albert_model = self.createBertModle()

        x_train = albert_model(x_train)

        pac = PassiveAggressiveClassifier(max_iter=50)
        pac.fit(x_train, y_train)
        with open('PassiveAggressiveModel.pickle', 'wb') as f:
            pickle.dump(pac, f)

    def predict(self,contents):
        model = self.createBertModle()
        input_id_test = self.tokenizeData(contents)
        output_test_contents = model.predict(input_id_test)
        pac = pickle.load(open('models/albertFeatureBase/feature_base_pas.pickle', 'rb'))
        y_pred=pac.predict(output_test_contents)
        return y_pred

    def train(self, train_data_path):
        x_train, y_train = readTestCSV(train_data_path)
        self.fit(x_train[:1000], y_train[:1000])


class BertSeqCls:
    def __init__(self):
        self.max_length = 512
        self.model_name = 'bert-base-chinese'
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name, do_lower_case=False)
        self.batch_size = 3

    def convert_example_to_feature(self, data):
        return self.tokenizer.encode_plus(data,
                                          add_special_tokens=True,  # add [CLS], [SEP]
                                          max_length=self.max_length,  # max length of the text that can go to BERT
                                          padding='max_length',  # add [PAD] tokens
                                          return_attention_mask=True,  # add attention mask to not focus on pad tokens
                                          truncation=True,
                                          )

    def map_example_to_dict(self, input_ids, attention_masks, token_type_ids, label):
        return {
                   "input_ids": input_ids,
                   "token_type_ids": token_type_ids,
                   "attention_mask": attention_masks,
               }, label

    def encode_examples(self, contents, labels):
        input_ids_list = []
        token_type_ids_list = []
        attention_mask_list = []
        label_list = []

        for content, label in zip(contents, labels):
            bert_input = self.convert_example_to_feature(content)

            input_ids_list.append(bert_input['input_ids'])
            token_type_ids_list.append(bert_input['token_type_ids'])
            attention_mask_list.append(bert_input['attention_mask'])
            label_list.append([label])

        return tf.data.Dataset.from_tensor_slices(
            (input_ids_list, attention_mask_list, token_type_ids_list, label_list)).map(self.map_example_to_dict)

    def map_test_example_to_dict(self, input_ids, attention_masks, token_type_ids):
        return {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_masks,
        }

    def encode_test_examples(self, contents):
        input_ids_list = []
        token_type_ids_list = []
        attention_mask_list = []

        for content in contents:
            bert_input = self.convert_example_to_feature(content)

            input_ids_list.append(bert_input['input_ids'])
            token_type_ids_list.append(bert_input['token_type_ids'])
            attention_mask_list.append(bert_input['attention_mask'])

        return tf.data.Dataset.from_tensor_slices(
            (input_ids_list, attention_mask_list, token_type_ids_list)).map(self.map_test_example_to_dict)

    def fit(self, ds_train_encoded, ds_test_encoded):
        learning_rate = 2e-5
        number_of_epochs = 10
        model = TFBertForSequenceClassification.from_pretrained(self.model_name)
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        earlystop_callback = EarlyStopping(
            monitor='val_accuracy',
            min_delta=0.0001,  # 精確度至少提高0.0001
            patience=3)

        checkpoint_path = "models/bertSqeCls/gender_cls.hdf5"
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            save_weights_only=True,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            verbose=1)

        model.fit(ds_train_encoded, epochs=number_of_epochs, validation_data=ds_test_encoded,
                  callbacks=[earlystop_callback, model_checkpoint_callback])

    def load(self):
        self.model = TFBertForSequenceClassification.from_pretrained(self.model_name, output_attentions=True)
        checkpoint_path = "models/bertSqeCls/gender_cls.hdf5"
        self.model.load_weights(checkpoint_path)

    def predict(self, test_sentence):
        input_ids_list = []
        token_type_ids_list = []
        attention_mask_list = []

        for i in test_sentence:
          bert_input = self.convert_example_to_feature(i)
          input_ids_list.append(bert_input['input_ids'])
          token_type_ids_list.append(bert_input['token_type_ids'])
          attention_mask_list.append(bert_input['attention_mask'])

        bert_input = tf.data.Dataset.from_tensor_slices((input_ids_list, attention_mask_list, token_type_ids_list)).map(self.map_test_example_to_dict)
        bert_input = bert_input.batch(self.batch_size)

        self.load()
        predicts = self.model.predict(bert_input)[0]

        pred = []
        for pre in predicts:
          pred.append(np.argmax(pre))
        return pred

    def get_attentions_before_fine_tune(self, test_sentence, label=0):
        input_ids_list = []
        token_type_ids_list = []
        attention_mask_list = []
        bert_input = self.convert_example_to_feature(test_sentence)
        input_ids_list.append(bert_input['input_ids'])
        token_type_ids_list.append(bert_input['token_type_ids'])
        attention_mask_list.append(bert_input['attention_mask'])

        input_ids_list = tf.convert_to_tensor(input_ids_list)
        token_type_ids_list = tf.convert_to_tensor(token_type_ids_list)
        attention_mask_list = tf.convert_to_tensor(attention_mask_list)

        inputs = {'input_ids': input_ids_list,'token_type_ids': token_type_ids_list,'attention_mask': attention_mask_list}
        inputs["labels"] = tf.reshape(tf.constant(label), (-1, 1))
        model = TFBertForSequenceClassification.from_pretrained(self.model_name, output_attentions=True)
        return model(inputs).attentions

    def get_attentions_after_fine_tune(self, test_sentence):
        input_ids_list = []
        token_type_ids_list = []
        attention_mask_list = []

        bert_input = self.convert_example_to_feature(test_sentence)
        input_ids_list.append(bert_input['input_ids'])
        token_type_ids_list.append(bert_input['token_type_ids'])
        attention_mask_list.append(bert_input['attention_mask'])

        bert_input = tf.data.Dataset.from_tensor_slices((input_ids_list, attention_mask_list, token_type_ids_list)).map(
            self.map_test_example_to_dict)
        bert_input = bert_input.batch(self.batch_size)
        self.load()
        predicts = self.model.predict(bert_input)
        return predicts.attentions

    def train(self,train_data_path,test_data_path):
        x_train, y_train = readTestCSV(train_data_path)
        x_test, y_test = readTestCSV(test_data_path)
        print(f'Num of train data : {len(x_train)}')
        print(f'Num of test data : {len(x_test)}')

        ds_train_encoded = self.encode_examples(x_train, y_train).batch(self.batch_size)
        ds_test_encoded = self.encode_examples(x_test, y_test).batch(self.batch_size)

        self.fit(ds_train_encoded, ds_test_encoded)
