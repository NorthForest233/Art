import utils
from omegaconf import OmegaConf
import sys
import os
from PIL import Image
import numpy as np
import datetime
import shutil

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
    parser = utils.get_parser()
    parser.add_argument(
        'content',
        type=str,
        help='path to content image',
    )
    parser.add_argument(
        'style',
        type=str,
        help='path to style image',
    )
    parser.add_argument(
        'mask',
        type=str,
        help='path to mask image',
    )
    parser.add_argument(
        '--exp_name',
        type=None,
        help='name of the experiment',
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=50,
        help='number of epochs to train',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./results',
        help='directory for synthesized images',
    )
    parser.add_argument(
        '--output_freq',
        type=int,
        default=5,
        help='frequency of saving synthesized images',
    )
    args = parser.parse_args()

    config = OmegaConf.load(args.config)

    for detector in os.listdir('./detectors'):
        sys.path.append(os.path.join('./detectors', detector))

    art = utils.instantiate_from_config(config['model'])
    content_image = load_image(args.content)
    style_image = load_image(args.style)
    mask = load_image(args.mask)
    # content_image = load_image('./images/bicycle.jpg')
    # style_image = load_image('./images/house.jpg')
    # mask = load_image('./images/bicycle_mask.png')
    art.set_images(content_image, style_image, mask)
    if args.exp_name:
        output_dir = os.path.join(args.output_dir, f'{args.exp_name}_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')
    else:
        output_dir = os.path.join(args.output_dir, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.path.exists(output_dir) or os.mkdir(output_dir)
    shutil.copy(args.config, os.path.join(output_dir, os.path.basename(args.config)))
    art.train(num_epochs=args.num_epochs,
                    learning_rate=config['model']['learning_rate'],
                    output_dir=output_dir, output_freq=args.output_freq)
