import bentoml
from bentoml.io import NumpyNdarray

model_ref = bentoml.sklearn.get("mlzoomcamp_homework:qtzdz3slg6mwwdu5")
model_runner = model_ref.to_runner()
svc = bentoml.Service("classify", runners=[model_runner])

@svc.api(input=NumpyNdarray(), output=NumpyNdarray())
def classify(vector):
    prediction = model_runner.predict.run(vector)
    result = prediction[0]
    
    return result