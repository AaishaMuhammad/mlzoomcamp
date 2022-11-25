
## Basic imports and setup
import numpy as np
import tflite_runtime.interpreter as tflite

from io import BytesIO
from urllib import request
from PIL import Image



interpret = tflite.Interpreter(model_path='dino-vs-dragon-v2.tflite')
interpret.allocate_tensors()

in_index = interpret.get_input_details()[0]['index']
out_index = interpret.get_output_details()[0]['index']


def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

# img = download_image('https://upload.wikimedia.org/wikipedia/commons/thumb/d/df/Smaug_par_David_Demaret.jpg/1280px-Smaug_par_David_Demaret.jpg')
# image_scaled = prepare_image(img, (150,150))

def preprocess_image(url):
    image = download_image(url)
    image_scaled = prepare_image(image, (150,150))

    x = np.array(image_scaled, dtype='float32')
    x = np.array([x])
    x = x * 1./255

    return x


def predict(url):
    x = preprocess_image(url)

    interpret.set_tensor(in_index, x)
    interpret.invoke()
    pred = interpret.get_tensor(out_index)

    return pred[0].tolist()


def lambda_handler(event, context):
    url = event['url']
    result = predict(url)
    return result



