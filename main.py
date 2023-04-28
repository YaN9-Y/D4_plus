import os
import cv2
import random
import numpy as np
import torch
import argparse
from shutil import copyfile
from src.config import Config
from src.D4 import D4


def main(mode=None):
    r"""starts the model

    Args:
        mode (int): 1: train, 2: test, 3: eval, reads from config file if not specified
    """

    config = load_config(mode)


    # cuda visble devices
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in config.GPU)


    # init device
    if torch.cuda.is_available():
        config.DEVICE = torch.device("cuda")
        torch.backends.cudnn.benchmark = True   # cudnn auto-tuner
    else:
        config.DEVICE = torch.device("cpu")



    # set cv2 running threads to 1 (prevents deadlocks with pytorch dataloader)
    cv2.setNumThreads(0)


    # initialize random seed
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)
    np.random.seed(config.SEED)
    random.seed(config.SEED)



    # build the model and initialize
    model = D4(config)
    model.load()


    # model training
    if config.MODE == 1:
        config.print()
        print('\nstart training...\n')
        model.train()

    # model test
    elif config.MODE == 2:
        print('\nstart testing...\n')
        model.test()

    # eval mode
    else:
        print('\nstart eval...\n')
        model.eval()


def load_config(mode=None):
    r"""loads model config

    Args:
        mode (int): 1: train, 2: test, 3: eval, reads from config file if not specified
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '--checkpoints', type=str, default='./checkpoints', help='model checkpoints path (default: ./checkpoints)')
    parser.add_argument('--model', type=int, choices=[1, 2, 3, 4, 5, 6, 7], help='1: edge model, 2: inpaint model, 3: edge-inpaint model, 4: joint model')

    # test mode
    if mode == 2:
        parser.add_argument('--input', type=str, help='path to the input images directory or an input image')
        parser.add_argument('--output', type=str, help='path to the output directory')
        parser.add_argument('--crop', type=bool)
        parser.add_argument('--crop_size', type=int, nargs=2)

    args = parser.parse_args()
    config_path = os.path.join(args.path, 'config.yml')

    # create checkpoints path if doesn't exist
    if not os.path.exists(args.path):
        os.makedirs(args.path)

    # copy config template if doesn't exist
    if not os.path.exists(config_path):
        copyfile('./config.yml.example', config_path)

    # load config file
    config = Config(config_path)

    # train mode
    if mode == 1:
        config.MODE = 1
        if args.model:
            config.MODEL = args.model

        if config.DATASET == 'ITS':
            config.USE_DC_A = 0
            config.USE_GUIDED_FILTER = 0
            config.USE_DCP = 0
        else:
            config.USE_DC_A = 1
            config.USE_GUIDED_FILTER = 0
            config.USE_DCP = 1

    # test mode
    elif mode == 2:
        config.MODE = 2
        config.MODEL = args.model if args.model is not None else 3
        config.INPUT_SIZE = 0

        if args.input is not None:
            config.TEST_FLIST = args.input

        if args.output is not None:
            config.RESULTS = args.output

        if args.crop is not None and args.crop_size is not None:
            config.CROP = args.crop
            config.CROP_SIZE = args.crop_size
        
        if config.DATASET == 'SOTS-indoor':
            config.USE_DC_A = 0
            config.USE_GUIDED_FILTER = 0
            config.IS_REAL_MODEL = 0
        elif config.DATASET == 'Foggycityscape':
            config.USE_DC_A = 1
            config.USE_GUIDED_FILTER = 0
            config.IS_REAL_MODEL = 0
        elif config.DATASET == 'SOTS-outdoor':
            config.USE_DC_A = 1
            config.USE_GUIDED_FILTER = 1
            config.IS_REAL_MODEL = 0
        elif config.DATASET == 'IHAZE':
            config.USE_DC_A = 1
            config.USE_GUIDED_FILTER = 1
            config.IS_REAL_MODEL = 0
        elif config.DATASET == 'REAL' or config.DATASET == 'RTTS':
            config.USE_DC_A = 1
            config.USE_GUIDED_FILTER = 1
            config.GF_R = 2 
            config.IS_REAL_MODEL = 1


    return config


if __name__ == "__main__":
    main()
