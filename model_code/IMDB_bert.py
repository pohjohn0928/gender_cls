import tensorflow as tf
import tensorflow_datasets as tfds
from transformers import BertTokenizer, AutoTokenizer
from transformers import TFBertForSequenceClassification
import numpy as np


class IMDBModel:
    def __init__(self):
        # self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.model_name = 'bert-base-chinese'
        self.max_length = 512
        self.batch_size = 3

    def getDataSet(self):
        (ds_train, ds_test), ds_info = tfds.load('imdb_reviews',
                                                 split=(tfds.Split.TRAIN, tfds.Split.TEST),
                                                 as_supervised=True,
                                                 with_info=True)

        return ds_train, ds_test

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

    def encode_examples(self, ds, limit=-1):
        input_ids_list = []
        token_type_ids_list = []
        attention_mask_list = []
        label_list = []

        if (limit > 0):
            ds = ds.take(limit)

        for review, label in tfds.as_numpy(ds):
            bert_input = self.convert_example_to_feature(review.decode())

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

    def fit(self):
        ds_train, ds_test = self.getDataSet()
        ds_train_encoded = self.encode_examples(ds_train).shuffle(10000).batch(self.batch_size)
        ds_test_encoded = self.encode_examples(ds_test).batch(self.batch_size)

        learning_rate = 2e-5
        number_of_epochs = 1

        model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', output_attentions=True)
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=1e-08)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
        model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
        model.fit(ds_train_encoded, epochs=number_of_epochs, validation_data=ds_test_encoded)
        return model

    def load(self):
        self.model = TFBertForSequenceClassification.from_pretrained(self.model_name, output_attentions=True)
        checkpoint_path = "../models/bertSqeCls/gender_cls.hdf5"
        self.model.load_weights(checkpoint_path)

    def get_attentions_before_fine_tune(self, test_sentence, label):
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

# imdb = IMDBModel()
# text = '小弟我最近有個疑惑很久的問題，上來請教一下各位'
# # attentions_before_fine_tune = imdb.get_attentions_before_fine_tune(text,1)
# attentions_after_fine_tune = imdb.get_attentions_after_fine_tune(text)

# print(attentions_before_fine_tune)
# print('-----------------------------------------------------')
# print(attentions_after_fine_tune)
# print(attentions_after_fine_tune)
# model = imdb.fit()
# inputs = imdb.convert_example_to_feature(["小弟我最近有個疑惑很久的問題，上來請教一下各位"])
# print(inputs)
# inputs["labels"] = tf.reshape(tf.constant(1), (-1, 1))
# outputs = model(inputs)
# print(outputs.attentions[0][0].shape)

# print(pre)

