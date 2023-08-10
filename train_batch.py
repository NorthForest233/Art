import utils
from omegaconf import OmegaConf
import sys
import os
from PIL import Image
import numpy as np
import datetime
import shutil
import pandas as pd

def load_image(path):
    image = np.array(Image.open(path))
    if image.ndim == 3:
        image = image.transpose(2, 0, 1)
        image = image.reshape(1, *image.shape)
    elif image.ndim == 2:
        image = image.reshape(1, 1, *image.shape)
    image = np.array(image / 255, dtype=np.float32)
    return image


if __name__ == '__main__':
    config_file = './configs/model.yaml'
    config = OmegaConf.load(config_file)

    for detector in os.listdir('./detectors'):
        sys.path.append(os.path.join('./detectors', detector))

    art = utils.instantiate_from_config(config['model'])

    data = pd.read_csv('./dataset/data.csv', index_col=0, header=0, sep=',')
    for i, row in enumerate(data.itertuples(index=False)):
        if 0 <= i and i < 10:
            content_image = load_image(os.path.join('./dataset/contents', row.content))
            style_image = load_image(os.path.join('./dataset/styles', row.style))
            mask = load_image(os.path.join('./dataset/masks', row.mask))
            exp_name = 'SC_TRACER_{i}'
            output_dir = './results'
            num_epochs = 20
            output_freq = 1
            
            art.set_images(content_image, style_image, mask)
            if exp_name:
                output_dir = os.path.join(output_dir, f'{exp_name}_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')
            else:
                output_dir = os.path.join(output_dir, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
            os.path.exists(output_dir) or os.mkdir(output_dir)
            shutil.copy(config_file, os.path.join(output_dir, os.path.basename(config_file)))
            art.train(num_epochs=num_epochs,
                            learning_rate=config['model']['learning_rate'],
                            output_dir=output_dir, output_freq=output_freq)
