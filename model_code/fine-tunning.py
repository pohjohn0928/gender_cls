import time
import bert
import os
import tensorflow.keras as keras
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.python.keras.layers import InputLayer,Lambda,Dense,Input
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping

from datahelper import readTestCSV

class AlbertFT:
    def __init__(self):
        self.max_length = 512

    def tokenizeData(self, data):
        tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
        tokens = tokenizer.batch_encode_plus(
            data,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='tf')
        return np.array(tokens['input_ids'])

    def createModle(self):
        model_name = "albert_tiny"
        model_dir = bert.fetch_brightmart_albert_model(model_name, "../.bert_models")
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

    def fit(self, contents, labels):
        contents = self.tokenizeData(contents)
        labels = np.array(labels)
        x_train, x_test, y_train, y_test = train_test_split(contents, labels, test_size=0.2, random_state=1234,
                                                            shuffle=True)
        model = self.createModle()

        earlystop_callback = EarlyStopping(
            monitor='val_accuracy',
            min_delta=0.0001,  # 精確度至少提高0.0001
            patience=3)

        checkpoint_path = "saved_model/gender_cls.hdf5"
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
        checkpoint_path = "../albertFineTunning/gender_cls.hdf5"
        self.model.load_weights(checkpoint_path)

    def predict(self, contents):
        self.load()
        contents = self.tokenizeData(contents)
        return self.model.predict(contents)


# contents,labels = readTestCSV('test_datas.csv')
# model = AlbertModel()
# start = time.time()
# predicts = model.predict(contents)
# end = time.time()
# print(f'albert fine-tunning : Total Spend {end - start} sec')
#
# y_pred = []
# for predict in predicts:
#     y_pred.append(round(predict[0]))
# score=accuracy_score(labels,y_pred)
# print(f'Accuracy: {round(score*100,2)}%')
# report = classification_report(labels, y_pred)
# print(report)
