import utils
from omegaconf import OmegaConf
import sys
import os
from PIL import Image
import numpy as np
import datetime
import shutil
import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as transforms
import matplotlib.pyplot as plt

def load_image(path):
    image = np.array(Image.open(path))
    if image.ndim == 3:
        image = image.transpose(2, 0, 1)
        image = image.reshape(1, *image.shape)
    elif image.ndim == 2:
        image = image.reshape(1, 1, *image.shape)
    image = torch.tensor(image / 255, dtype=torch.float32)
    return image


if __name__ == '__main__':
    parser = utils.get_parser()
    args = parser.parse_args()

    config = OmegaConf.load(args.config)

    for detector in os.listdir('./detectors'):
        sys.path.append(os.path.join('./detectors', detector))

    art = utils.instantiate_from_config(config['model'])
    content_image = load_image('./images/dog.jpg')
    style_image = load_image('./images/starry_night.jpg')
    mask = load_image('./images/dog_mask.png')
    exp_name = None
    output_dir = './results'
    num_epochs = 20
    output_freq = 1
    
    art.set_images(content_image, style_image, mask)
    if exp_name:
        output_dir = os.path.join(output_dir, f'{exp_name}_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')
    else:
        output_dir = os.path.join(output_dir, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.path.exists(output_dir) or os.mkdir(output_dir)
    shutil.copy(args.config, os.path.join(output_dir, os.path.basename(args.config)))
    art.train(num_epochs=num_epochs,
                    learning_rate=config['model']['learning_rate'],
                    output_dir=output_dir, output_freq=output_freq)
