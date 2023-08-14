import utils
from omegaconf import OmegaConf
import sys
import os
from PIL import Image
import numpy as np
import datetime
import shutil
import pandas as pd
import torch


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
    parser.add_argument(
        'dataset',
        type=str,
        help='path to dataset',
    )
    parser.add_argument(
        '--begin',
        type=int,
        default=None,
        help='begin index',
    )
    parser.add_argument(
        '--end',
        type=int,
        default=None,
        help='end index',
    )
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    print(args.begin)

    for detector in os.listdir('./detectors'):
        sys.path.append(os.path.join('./detectors', detector))

    art = utils.instantiate_from_config(config['model'])

    data = pd.read_csv(os.path.join(args.dataset, 'dataset.csv'), index_col=0, header=0, sep=',')
    begin = max(args.begin, 0) if args.begin else 0
    end = min(args.end, len(data)) if args.end else len(data)
    for i in range(begin, end):
        print(i)
        row = data.iloc[i]
        content_image = load_image(os.path.join(args.dataset, 'contents', row.content))
        style_image = load_image(os.path.join(args.dataset, 'styles', row.style))
        mask = load_image(os.path.join(args.dataset, 'masks', row.mask))
        exp_name = f'SC_TRACER_{i}'
        output_dir = './results'
        num_epochs = 20
        output_freq = 1

        art.set_images(content_image, style_image, mask)
        if exp_name:
            output_dir = os.path.join(
                output_dir, f'{exp_name}_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')
        else:
            output_dir = os.path.join(output_dir, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        os.path.exists(output_dir) or os.mkdir(output_dir)
        shutil.copy(args.config, os.path.join(output_dir, os.path.basename(args.config)))
        art.train(num_epochs=num_epochs,
                    learning_rate=config['model']['learning_rate'],
                    output_dir=output_dir, output_freq=output_freq)
