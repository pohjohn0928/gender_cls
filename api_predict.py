from modelHelper import PassiveAggressiveCls, XgboostCls, AlbertModelFT, AlbertModelFB, BertSeqCls
from abc import abstractmethod


class ModelPreInterface:
    def receive_info(self, content: str, label: int):
        self.content = content
        self.label = label

    @abstractmethod
    def predict(self, model_type: str, content: str, label: int):
        pass


class PacPredict(ModelPreInterface):
    def __init__(self, model):
        self.pac_model = model

    def predict(self, model_type: str, content: str, label: int):
        model = PassiveAggressiveCls()
        pre = int(model.predict(self.pac_model, [content])[0])
        return_dic = {"model_type": model_type, "contents": content, "predict": pre, 'label': label, 'partial_fit': 0}
        if label == 0 or label == 1:
            if pre == label:
                return return_dic
            else:
                new_pre, self.pac_model = model.partial_fit(self.pac_model, content, label)
                return_dic['partial_fit'] = 1
                return_dic['after_partial_fit_predict'] = int(new_pre)
                return return_dic
        else:
            return return_dic


class XgPredict(ModelPreInterface):
    def predict(self, model_type: str, content: str, label: int) -> dict:
        model = XgboostCls()
        # pre = float(model.predict([content])[0][1])
        pre = int(model.predict([content])[0])
        return_dic = {"model_type": model_type, "contents": content, "predict": pre, 'label': label}
        return return_dic


class AlbertFTPredict(ModelPreInterface):
    def predict(self, model_type: str, content: str, label: int) -> dict:
        model = AlbertModelFT()
        pre = float(model.predict([content])[0][0])
        return_dic = {"model_type": model_type, "contents": content, "predict": pre, 'label': label}
        return return_dic


class AlbertFBPredict(ModelPreInterface):
    def predict(self, model_type: str, content: str, label: int) -> dict:
        model = AlbertModelFB()
        pre = int(model.predict([content])[0])
        return_dic = {"model_type": model_type, "contents": content, "predict": pre, 'label': label}
        return return_dic


class BerSeqClsPredict(ModelPreInterface):
    def predict(self, model_type: str, content: str, label: int) -> dict:
        model = BertSeqCls()
        pre = int(model.predict([content])[0])
        return_dic = {"model_type": model_type, "contents": content, "predict": pre, 'label': label}
        return return_dic


class BerSeqClsAttentions:
    def __init__(self):
        self.model = BertSeqCls()

    def get_attentions_before_fine_tune(self, sentence, layer, head, word_index):
        attentions = self.model.get_attentions_before_fine_tune(sentence)
        word_attention = attentions[layer][0][head][word_index]
        word_list = [i for i in sentence]
        word_list.insert(0, '[CLS]')
        word_list.append('[SEP]')
        word_attention = word_attention[:len(word_list)]
        return word_attention, word_list

    def get_attentions_after_fine_tune(self, sentence, layer, head, word_index):
        attentions = self.model.get_attentions_after_fine_tune(sentence)
        word_attention = attentions[layer][0][head][word_index]
        word_list = [i for i in sentence]
        word_list.insert(0, '[CLS]')
        word_list.append('[SEP]')
        word_attention = word_attention[:len(word_list)]
        return word_attention, word_list

# model = BerSeqClsAttentions()
# model.get_attentions_before_fine_tune('女友自爆有8顆根管裝過牙套的牙齒', 0, 0, 0)
# model.get_attentions_after_fine_tune('女友自爆有8顆根管裝過牙套的牙齒', 0, 0, 0)
