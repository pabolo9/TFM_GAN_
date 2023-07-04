import os
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, send_file
from keras.models import load_model
import io
import random
import shutil
import torch
from torchvision.utils import save_image
import torch.nn as nn
import torchvision.utils as vutils

app = Flask(__name__)

# Registrar la función 'enumerate' en el entorno de Jinja2
app.jinja_env.globals['enumerate'] = enumerate

# Ruta de la página principal
@app.route('/')
def index():
    return render_template('index.html')

# Ruta para generar y mostrar las imágenes
@app.route('/generate', methods=['POST'])


def generate_image():

    # Obtener el valor del vector de ruido del formulario
    n_images = int(request.form['n_images'])

    gan_model_dir = 'modelos/'
    gan_model = 'DCGAN_gen_best_v3_2.pth'
    generated_images = []

    # Seteamos algunos hiperparametros que necesitaremos
    nz = 100 #tamaño de input del generador
    ngf = 128 #tamaño de input del generador
    nc = 1 #numero de canales (1 porque trabajamos en escala de grises)

    class gan_gen(nn.Module):

        def __init__(self):
            super(gan_gen, self).__init__()
            self.main = nn.Sequential(
                # La entrada será de tamaño nz, en este caso 100 y pasamos por una
                # capa de convolución transpuesta con salida ngf*16=2048
                nn.ConvTranspose2d(     nz, ngf * 16, 4, 1, 0, bias=False),
                nn.BatchNorm2d(ngf * 16),
                nn.ReLU(True),
                # Pasamos por varias capas de convolucón transpuesta, reduciendo así
                # el tamaño. ngf = 128
                # (ngf*16) x 4 x 4
                nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 8),
                nn.ReLU(True),
                # (ngf*8) x 8 x 8
                nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),
                # (ngf*4) x 16 x 16
                nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True),
                # (ngf*2) x 32 x 32
                nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True),
                # (ngf) x 64 x 64
                nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
                nn.Tanh()
                # (nc) x 128 x 128
                # La salida será 1x128x128
            )


        def forward(self, input):
            output = self.main(input)
            return output


    # Obtener una imagen aleatoria del directorio dataset_64x64
    dataset_dir = 'dataset_64x64/'
    dataset_images = os.listdir(dataset_dir)
    random_image_file = random.choice(dataset_images)
    random_image_path = os.path.join(dataset_dir, random_image_file)

    # Copiar la imagen aleatoria al directorio de descarga
    download_dir = 'downloads/'
    os.makedirs(download_dir, exist_ok=True)
    random_image_copy_path = os.path.join(download_dir, 'random_image.jpg')
    shutil.copy(random_image_path, random_image_copy_path)


    # Construimos la url del generador
    generator_path = os.path.join(gan_model_dir, gan_model)

    # Cargamos los pesos del generador desde el archivo .pth
    generator = gan_gen()
    generator.load_state_dict(torch.load(generator_path, map_location=torch.device('cpu')))
    generator.eval()

    # Guardar las imágenes generadas en el directorio de descarga
    generated_image_paths = []
    for i in range(n_images):
        # Generamos una imagen aleatoria
        with torch.no_grad():
            noise = torch.randn(1, 100, 1, 1)
            imagen_generada = generator(noise)

        generated_image_path = os.path.join(download_dir, f'generated_image_{i}.jpg')    
        # Guardamos la imagen generada en un archivo .jpg
        vutils.save_image(imagen_generada, generated_image_path, normalize=True)
        generated_image_paths.append(generated_image_path)

    return render_template('result.html', generated_images=generated_image_paths, random_image=random_image_copy_path)
    
    
# Ruta para descargar las imágenes generadas
@app.route('/download/<path:filename>')
def download(filename):
    return send_file(filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)


