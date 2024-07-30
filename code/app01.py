from flask import Flask, request, jsonify
import torch
from build_corpus import build_corpus

app = Flask(__name__)

# 加载模型和必要的资源（在启动应用时只加载一次）
device = "cuda:0" if torch.cuda.is_available() else "cpu"
train_data, train_tag, word_2_index, tag_2_index = build_corpus("train", make_vocab=True)
index_2_tag = [i for i in tag_2_index]
model = torch.load('model1000.pt', map_location=device)

def predict(text):
    text_index = [[word_2_index.get(i, word_2_index["<UNK>"]) for i in text] + [word_2_index["<END>"]]]
    text_index = torch.tensor(text_index, dtype=torch.int64, device=device)
    pre = model.test(text_index, [len(text) + 1])
    pre = [index_2_tag[i] for i in pre]
    return [f'{w}-{s}' for w, s in zip(text, pre)]

##采用POST的方式进行调用
@app.route('/api', methods=['POST'])
def api_predict():
    data = request.get_json()
    text = data['text']
    try:
        prediction = predict(text)
        return jsonify(prediction)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8810)
