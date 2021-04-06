# __author__ = 'Vasudev Gupta'

import yaml
import argparse
import wandb

from modeling_utils.cgan import cGAN
from modeling_utils.models import Generator, Discriminator
from data_utils.dataloader import make_dataset

TRAIN_3D = True

if __name__ == '__main__':

    # logger.setLevel(logging.INFO)
    # file_handler = logging.FileHandler('logs/results.log')
    # formt = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
    # file_handler.setFormatter(formt)
    # logger.addHandler(file_handler)

    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, default='config.yaml', help='filename for configuration')
    parser.add_argument('--model_name', type=str, default='cgan_model', help='model no')
    parser.add_argument('--save_model', type=bool, default=False, help='whether to save model')
    parser.add_argument('--restore_model', type=bool, default=False, help='whether to restore saved weights')

    args = parser.parse_args()

    config = yaml.safe_load(open(args.config, 'r'))
    print(config)

    train_dataset, val_dataset = make_dataset(config, mode=("3d" if TRAIN_3D else "2d"))
    val_conditions, val_real_data = val_dataset

    wandb.init(project='temp_distribution_prediction', config=config, name=args.model_name + " 5 inputs")
    config = wandb.config

    generator = Generator()
    discriminator = Discriminator()

    model = cGAN(generator=generator, discriminator=discriminator)
    model.icompile(config)

    final_metric, grads = model.train(train_dataset, val_conditions, val_real_data, config,
                                      restore_model=args.restore_model, save_model=args.save_model)

    print(final_metric)
    model.save_checkpoints()
