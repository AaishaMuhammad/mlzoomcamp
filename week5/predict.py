''' Code for Question 4 - Running on Flask '''


import pickle
from flask import Flask
from flask import request
from flask import jsonify

model_file = "./model1.bin"
dv_file = "./dv.bin"

with open(model_file, "rb") as file:
    model = pickle.load(file)
with open(dv_file, "rb") as file:
    dv = pickle.load(file)

app = Flask('predict')


@app.route('/predict', methods=['POST'])
def predict():
    cust = request.get_json()

    X = dv.transform([cust])
    y_pred = model.predict_proba(X)[0, 1]

    result = {
        'probability': float(y_pred)
    }
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host="localhost", port=5000)






