from flask import Flask, request, jsonify
app = Flask(__name__)

def is_odd(num):
    return num % 2 != 0

@app.route('/api', methods=['POST'])
def api():
    data = request.get_json()
    num = data['number']
    result = is_odd(num)
    return jsonify({'result': '奇数' if result else '偶数'})

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0',port=8810)

