FROM svizor/zoomcamp-model:3.9.12-slim

RUN pip install pipenv

WORKDIR /app
COPY ["Pipfile", "Pipfile.lock", "./"]
RUN pipenv install --system --deploy

COPY ["predict_dock.py", "./"]
EXPOSE 5000
ENTRYPOINT ["waitress-serve", "--listen=0.0.0.0:5000", "predict_dock:app"]
