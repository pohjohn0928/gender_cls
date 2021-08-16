# coding=utf-8
import pickle
from enum import Enum
from typing import Optional

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from api_predict import PacPredict, XgPredict, AlbertFTPredict, AlbertFBPredict, BerSeqClsPredict, BerSeqClsAttentions
from starlette.concurrency import run_in_threadpool


class ModelInfo(BaseModel):
    content: str
    label: Optional[int] = None


class ModelType(str, Enum):
    Xgboost = "xgboost"
    Pac = "pac"
    Albert_FT = "albert_FT"
    Albert_FB = "albert_FB"
    BertSeqCls = 'bertSeqCls'


app = FastAPI(debug=True, title='Model api', description='Predictions')
app.pac_model = pickle.load(open('models/PassiveAggressiveModel/PassiveAggressiveModel.pickle', 'rb'))


@app.get("/", tags=['home page'])
async def read_root():
    return {"Hello": "World"}


@app.post("/models/{model_type}", tags=['models'])  # @app.get("/items/{item_id}")
async def read_item(model_info: ModelInfo, model_type: ModelType = ModelType.Xgboost):
    print(f'Using {model_type}')
    model = None

    if model_type == ModelType.Xgboost:
        model = XgPredict()
    elif model_type == 'pac':
        model = PacPredict(app.pac_model)
    elif model_type == 'albert_FT':
        model = AlbertFTPredict()
    elif model_type == 'albert_FB':
        model = AlbertFBPredict()
    elif model_type == 'bertSeqCls':
        model = BerSeqClsPredict()

    return_dic = await run_in_threadpool(model.predict, model_type=model_type, content=model_info.content,
                                         label=model_info.label)
    return return_dic


class AttentionInfo(BaseModel):
    after_fine_tune: int
    content: str
    word_index: int
    layer: int
    head: int


@app.post("/attention", tags=['bert attention calculation'])
async def read_item(attention_info: AttentionInfo):
    if attention_info.word_index >= len(attention_info.content) + 2 or attention_info.word_index < 0:
        return {'status': 'Error'}
    else:
        attentions = BerSeqClsAttentions()
        if attention_info.after_fine_tune == 0:
            word_attention, word_list = attentions.get_attentions_before_fine_tune(attention_info.content,
                                                                                   attention_info.layer,
                                                                                   attention_info.head,
                                                                                   attention_info.word_index)
        elif attention_info.after_fine_tune == 1:
            word_attention, word_list = attentions.get_attentions_after_fine_tune(attention_info.content,
                                                                                  attention_info.layer,
                                                                                  attention_info.head,
                                                                                  attention_info.word_index)
        else:
            return {'status': 'Error'}

        return_dic = {'Detect Word': word_list[attention_info.word_index]}

        attentions_word_dic = {}
        counter = 0
        for attention, word in zip(word_attention, word_list):
            attentions_word_dic[f'{counter} -> {word}'] = float(attention)
            counter += 1

        attentions_word_dic = dict(sorted(attentions_word_dic.items(), key=lambda item: item[1], reverse=True))
        return_dic['result'] = attentions_word_dic
        return return_dic


if __name__ == '__main__':
    uvicorn.run("fastApi:app", host='127.0.0.1', port=8000)
