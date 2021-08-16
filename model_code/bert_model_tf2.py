import pickle
import time

import bert
import os
import keras
from transformers import AutoTokenizer
import numpy as np
import pandas as pd
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score,classification_report


import ssl
ssl._create_default_https_context = ssl._create_unverified_context

class BertModel:
    def CreateBertModel(self):
        model_name = "uncased_L-12_H-768_A-12"
        model_dir = bert.fetch_google_bert_model(model_name, "../.bert_models")
        model_ckpt = os.path.join(model_dir, "bert_model.ckpt")

        bert_params = bert.params_from_pretrained_ckpt(model_dir)
        l_bert = bert.BertModelLayer.from_params(bert_params, name="bert")
        l_bert.trainable = False
        

        model = keras.models.Sequential([
            keras.layers.InputLayer(input_shape=(512,)),
            l_bert,
            keras.layers.Lambda(lambda x: x[:, 0, :]),
        ])
        model.build(input_shape=(None, 128))
        bert.load_bert_weights(l_bert, model_ckpt)
        return model

    def getFeature(self,contents):
        tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
        tokens = tokenizer.batch_encode_plus(
            contents,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='tf')
        id = np.array(tokens['input_ids'])

        model = self.CreateBertModel()
        return model.predict(id)

def PassiveAggressiveCls(features,labels):
    pac = PassiveAggressiveClassifier(max_iter=50)
    pac.fit(features, labels)
    with open('../bertFeatureBase/PassiveAggressiveModel.pickle', 'wb') as f:
        pickle.dump(pac, f)

def evaluate(contents,labels):
    pac = pickle.load(open('../bertFeatureBase/PassiveAggressiveModel.pickle', 'rb'))
    bert_model = BertModel()
    features = bert_model.getFeature(contents)
    y_pred = pac.predict(features)
    score = accuracy_score(labels, y_pred)
    print(f'Accuracy: {round(score * 100, 2)}%')
    report = classification_report(labels, y_pred)
    print(report)


df_test = pd.read_csv('test_datas.csv')
test_labels = df_test.label.values
test_contents = df_test.content.values
test_contents = list(test_contents)

# bert_model = BertModel()
# features = bert_model.getFeature(test_contents)

# PassiveAggressiveCls(features,test_labels)
start = time.time()
print('Time Start')
evaluate(test_contents,test_labels)
end = time.time()
print(f'Total Spend {end - start} sec')



# class BertFB(audientModel):
#     def CreateBertModel(self):
#         model_name = "uncased_L-12_H-768_A-12"
#         model_dir = bert.fetch_google_bert_model(model_name, ".bert_models")
#         model_ckpt = os.path.join(model_dir, "bert_model.ckpt")
#
#         bert_params = bert.params_from_pretrained_ckpt(model_dir)
#         l_bert = bert.BertModelLayer.from_params(bert_params, name="bert")
#         l_bert.trainable = False
#
#         model = keras.models.Sequential([
#             keras.layers.InputLayer(input_shape=(512,)),
#             l_bert,
#             keras.layers.Lambda(lambda x: x[:, 0, :]),
#         ])
#         model.build(input_shape=(None, 128))
#         bert.load_bert_weights(l_bert, model_ckpt)
#         return model
#
#     def predict_web(self,contents):
#         model = self.CreateBertModel()
#         input_id_test = self.tokenizeData(contents)
#         output_test_contents = model.predict(input_id_test)
#         pac = pickle.load(open('bertFeatureBase/PassiveAggressiveModel.pickle', 'rb'))
#         y_pred = pac.predict(output_test_contents)
#         return y_pred[0]

# class BertFT(audientModel):
#     def CreateBertModel(self):
#         model_name = "uncased_L-12_H-768_A-12"
#         model_dir = bert.fetch_google_bert_model(model_name, ".bert_models")
#         model_ckpt = os.path.join(model_dir, "bert_model.ckpt")
#
#         bert_params = bert.params_from_pretrained_ckpt(model_dir)
#         l_bert = bert.BertModelLayer.from_params(bert_params, name="bert")
#         l_bert.trainable = True
#
#         model = keras.models.Sequential([
#             keras.layers.InputLayer(input_shape=(self.max_length,)),
#             l_bert,
#             keras.layers.Lambda(lambda x: x[:, 0, :]),
#             keras.layers.Dense(128,activation='relu'),
#             keras.layers.Dense(64, activation='relu'),
#             keras.layers.Dense(1, activation='sigmoid'),
#         ])
#         model.build(input_shape=(None, self.max_length))
#         bert.load_bert_weights(l_bert, model_ckpt)
#         return model
#
#     def fit(self,contents,labels):
#         contents = self.tokenizeData(contents)
#         labels = np.array(labels)
#         x_test,y_test = readTestCSV('test_datas.csv')
#         x_test = self.tokenizeData(x_test)
#         y_test = np.array(y_test)
#
#         model = self.CreateBertModel()
#
#         earlystop_callback = EarlyStopping(
#             monitor='val_accuracy',
#             min_delta=0.0001,  # 精確度至少提高0.0001
#             patience=3)
#
#         checkpoint_path = "bertFineTunning/gender_cls.hdf5"
#         model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
#             checkpoint_path,
#             save_weights_only=True,
#             monitor='val_accuracy',
#             mode='max',
#             save_best_only=True,
#             verbose=1)
#
#         model.compile(loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.Adam(),
#                       metrics=['accuracy'])
#
#         model.fit(x=contents, y=labels, validation_data=(x_test, y_test), epochs=10, batch_size=10, verbose=1,
#                   callbacks=[earlystop_callback, model_checkpoint_callback])
#
#     def load(self):
#         self.model = self.CreateBertModel()
#         checkpoint_path = f"bertFineTunning/gender_cls.hdf5"
#         self.model.load_weights(checkpoint_path)
#
#     def predict(self, contents):
#         self.load()
#         contents = self.tokenizeData(contents)
#         return self.model.predict(contents)[0][0]