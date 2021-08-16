# Note

## 保存最好的模型
> 只保存val_acc最好的模型：每次如果效果變好就覆盖之前的權重文件
```python
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint

def creatModel(X,Y):
    model = Sequential()
    model.add(Dense(12, input_dim=8, init='uniform', activation='relu')) 
    model.add(Dense(8, init='uniform', activation='relu')) 
    model.add(Dense(1, init='uniform', activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    filepath = "weights.best.hdf5"
    checkpoint = ModelCheckpoint(filepath, 
                                 monitor='val_accuracy', 
                                 verbose=1, 
                                 save_best_only=True,
                                 mode='max')
    callbacks_list = [checkpoint]
    model.fit(X, Y, validation_split=0.33, 
              epochs=150, batch_size=10,
              callbacks=callbacks_list, verbose=1)

```

## 導入保存的模型
> 保存点只保存權重，網路結構需要預先保存
```python
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
model.add(Dense(12, input_dim=8, init='uniform', activation='relu')) 
model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))

model.load_weights("weights.best.hdf5")
```

## Concatenate
> consider multi-features
```python
import tensorflow.keras as keras
from keras.models import Model
from tensorflow.python.keras.layers import InputLayer,Lambda,Dense,Dropout,concatenate,Input
import bert
import os

model_name = "albert_tiny"
model_dir = bert.fetch_brightmart_albert_model(model_name, ".models")
model_ckpt = os.path.join(model_dir, "albert_model.ckpt")
bert_params = bert.params_from_pretrained_ckpt(model_dir)
l_bert = bert.BertModelLayer.from_params(bert_params, name="bert")
l_bert.trainable = True

bert_model = keras.models.Sequential()
sequence = Input(shape=(512,))
features = Input(shape=(1,))

bert_model.add(l_bert)
bert_model.add(Lambda(lambda x: x[:, 0, :]))
bert_model.add(Dense(128, activation='relu'))
bert_model.add(Dense(64, activation='relu'))
bert_model.add(Dense(1, activation='sigmoid'))

albert = bert_model(sequence)
merged = concatenate([albert, features])

final = Dense(512, activation='relu')(merged)
final = Dense(1, activation='sigmoid')(final)

model = Model(inputs=[sequence, features], outputs=[final])
```

### Passive Aggressive Classifiers
- Use : used for large-scale learning, one of the few ‘online-learning algorithms‘
  1. online machine learning algorithms : the input data comes in sequential order and the machine learning model is updated step-by-step, 
     as opposed to batch learning, where the entire training dataset is used at once(同時)
     - 不斷的從新產生的數據中學習,沒有一個固定的Datasets,適用於continuous stream of data
  2. huge amount of data and it is computationally infeasible to train the entire dataset 
     because of the sheer size of the data.
  3. get a training example -> update the classifier -> throw away the example
- parameters:
1. C : regularization parameter , and denotes the penalization the model will make on an incorrect prediction
2. max_iter : The maximum number of iterations(迭代) the model makes over the training data.
3. tol : The stopping criterion. If it is set to None, the model will stop when (loss > previous_loss  –  tol).By default, it is set to 1e-3.

```python
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

def PACls(X, y):
    # 中文須先斷詞,及 TfidfVectorizer
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 13)
    model = PassiveAggressiveClassifier(C = 0.5, random_state = 5)
    model.fit(X_train, y_train)
    test_pred = model.predict(X_test)
    print(f"Test Set Accuracy : {accuracy_score(y_test, test_pred) * 100} %\n\n")  
    print(f"Classification Report : \n\n{classification_report(y_test, test_pred)}")
```
- 參考：<br>
  [介紹](https://www.geeksforgeeks.org/passive-aggressive-classifiers/) <br>
  [Online Learning](https://www.coursera.org/lecture/machine-learning/online-learning-ABO2q)
  
# Comparison 
1. Training Data shape : 5552
2. Validation Data shape : 1388

| model | acc | precision | recall | f1-score | spend( data / sec) |
|----------|:-----:|------:|------:|------:|------:|
| Albert Fine-Tuning  | 0.88 | 0.88 | 0.88 | 0.88 | 6.622
| Albert Feature Base + pas  | 0.63 | 0.70 | 0.61 | 0.58 | 7.64
| tfidf + pac  | 0.90 | 0.90 | 0.90 | 0.90 | 462.67
| Bert Feature base + pas | 0.75 | 0.75 | 0.75 | 0.75 | 1.00
| tfidf + xgboost | 0.91 | 0.92 | 0.92 | 0.92 | 436.48

## detail

| model   |  |precision |  recall | f1-score | support | Time
|----------|:-----:|------:|------:|------:|------:| ------:|
| Albert Fine-Tuning  |  0 | 0.83 | 0.94 | 0.88 | 662 | 209.6 sec
|   |  1 | 0.93 | 0.82 | 0.87 | 726
|   |  accuracy |  |  | 0.88 | 1388
|   |  macro avg | 0.88 | 0.88 | 0.88 | 1388
|   |  weighted avg | 0.88 | 0.88 | 0.88 | 1388
|  Albert Feature Base + pas  |  0 | 0.80 | 0.30 | 0.43 | 662 | 181.7 sec
|   |  1 | 0.59 | 0.93 | 0.72 | 726
|   |  accuracy |  |  | 0.63 | 1388
|   |  macro avg | 0.70 | 0.61 | 0.58 | 1388
|   |  weighted avg | 0.69 | 0.63 | 0.59 | 1388
|  tfidf + pac  |  0 | 0.88 | 0.92 | 0.90 | 662 | 3.0 sec
|   |  1 | 0.92 | 0.88 | 0.90 | 726
|   |  accuracy |  |  | 0.90 | 1388
|   |  macro avg | 0.90 | 0.90 | 0.90 | 1388
|   |  weighted avg | 0.90 | 0.90 | 0.90 | 1388
|  Bert Feature base + pas  |  0 | 0.72 | 0.78 | 0.75 | 662 | 1394.3 sec
|   |  1 | 0.78 | 0.72 | 0.75 | 726
|   |  accuracy |  |  | 0.75 | 1388
|   |  macro avg | 0.75 | 0.75 | 0.75 | 1388
|   |  weighted avg | 0.75 | 0.75 | 0.75 | 1388
|  Tfidf + xgboost  |  0 | 0.90 | 0.93 | 0.92 | 662 | 3.1 sec
|   |  1 | 0.93 | 0.91 | 0.92 | 726
|   |  accuracy |  |  | 0.92 | 1388
|   |  macro avg | 0.92 | 0.92 | 0.92 | 1388
|   |  weighted avg | 0.92 | 0.92 | 0.92 | 1388


## Xgboost (extreme gradient boosting)
- 以Gradient Boosting 為基礎下去實作
  (每棵樹為互相關連的，希望後面的樹可以修正前面樹犯錯的地方)
    
- ex.分類問題：和隨機森林一樣採用的特徵隨機採樣的技巧
     不會每次都拿每棵樹的特徵作訓練
  
- Parameters:
1. n_estimators: 總共迭代的次數，即決策樹的個數。預設值為100。
2. max_depth: 樹的最大深度，默認值為6。
3. booster: gbtree 樹模型(預設) / gbliner 線性模型
4. learning_rate: 學習速率，預設0.3。
5. gamma: 懲罰項係數，指定節點分裂所需的最小損失函數下降值。


