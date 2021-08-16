from transformers import AutoTokenizer, TFAutoModel, TFBertModel, BertForSequenceClassification, BertTokenizer
import tensorflow.keras as keras
from tensorflow.python.keras.layers import InputLayer,Lambda,Dense,Input
from datahelper import readTestCSV
import numpy as np
import tensorflow as tf

class BertFT:
    def __init__(self):
        self.max_length = 512
        self.model_name = 'bert-base-chinese'
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)

    # def tokenizeData(self,data):
    #     tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    #     inputs = tokenizer(data, return_tensors="tf", padding='max_length',
    #     max_length=self.max_length,truncation=True)
    #     return inputs['input_ids']

    def convert_example_to_feature(self, data):
        return self.tokenizer.encode_plus(data,
                                          add_special_tokens=True,  # add [CLS], [SEP]
                                          max_length=self.max_length,  # max length of the text that can go to BERT
                                          padding='max_length',  # add [PAD] tokens
                                          return_attention_mask=True,  # add attention mask to not focus on pad tokens
                                          truncation=True
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

    def createModel(self):
        bert_model = TFBertModel.from_pretrained("bert-base-chinese")
        bert_model.trainable = True

        input_layer = Input(shape=(self.max_length,), dtype='int64')
        bert = bert_model(input_layer)
        pooler_layer = bert.pooler_output
        dense128 = Dense(units=128, activation='relu')(pooler_layer)
        dense64 = Dense(units=64, activation='relu')(dense128)
        output = Dense(units=1, activation='sigmoid')(dense64)
        model = keras.Model(inputs=input_layer, outputs=output)

        model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
        return model

bert_model = BertFT()
model = bert_model.createModel()

train_data_path = '../train_data/train_datas.csv'
x_train,y_train = readTestCSV(train_data_path)

test_data_path = '../test_data/test_datas.csv'
x_test, y_test = readTestCSV(test_data_path)

batch_size = 5
ds_train_encoded = bert_model.encode_examples(x_train, y_train).batch(batch_size)
ds_test_encoded = bert_model.encode_examples(x_test, y_test).batch(batch_size)

model.fit(ds_train_encoded, epochs=10, validation_data=ds_test_encoded)
