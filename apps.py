import pickle

from flask import Flask, request, render_template
from modelHelper import AlbertModelFT,AlbertModelFB,PassiveAggressiveCls,XgboostCls,BertSeqCls
# from

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False


@app.route('/')
def home():
    return render_template("gender_cls.html")

@app.route('/predictGender', methods=["POST"])
def predictGender():
    content = request.values['content']
    model_type = request.values['model']

    result = None
    if model_type == 'albertFineTunning' :
        print('Using Albert Fine-tunning')
        model = AlbertModelFT()
        result = float(model.predict([content])[0][0])
    elif model_type == 'albertFeatureBase':
        print('Using Albert Feature-base')
        model = AlbertModelFB()
        result = model.predict([content])[0].tolist()
    elif model_type == 'PassiveAggressiveModel':
        print('Using Passive Aggressive Model')
        model = PassiveAggressiveCls()
        pac_model = pickle.load(open('models/PassiveAggressiveModel/PassiveAggressiveModel.pickle', 'rb'))
        result = int(model.predict(pac_model, [content])[0])
    elif model_type == 'xgboost':
        print('Using Xgboost Model')
        model = XgboostCls()
        result = int(model.predict([content])[0])
    elif model_type == 'bertSqeCls':
        print('Using Bert Sequence Clssifier')
        model = BertSeqCls()
        result = int(model.predict([content])[0])
    dic = {'result': result}
    return dic

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
