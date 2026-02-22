import os
import torch
import numpy as np
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
import yaml
import argparse
import time
import math
import sys
import pdb
import warnings
import matplotlib.pyplot as plt
import logging
from datetime import datetime
import random

from model.initial_model import init_network, Model
from trainer import train_net, get_data

warnings.filterwarnings("ignore", category=UserWarning)
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)


def get_parser():
    parser = argparse.ArgumentParser(description='Medical Image Segmentation')
    parser.add_argument('--model', type=str, default='', help='model name')
    parser.add_argument('--base_folder', default='', type=str, help='batch size')
    parser.add_argument('--batch_size', default = 16, type=int, help='batch size')
    parser.add_argument('--log_path', type=str, default='./log/', help='log path')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--optimizer', type=str, default='adamw')
    parser.add_argument('--base_lr', type=float, default=5e-4)
    parser.add_argument('--betas', type=list, default=[0.9, 0.999])
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--max_epochs', type=int, default=300)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--pretrained_weights', type=str, help='pretrained weights', default=None)

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = get_parser()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    print(f"Using device: {device}")

    args.log_path = args.log_path + args.model + '/exp_' + datetime.now().strftime("%Y%m%d-%H_%M_%S")
    assert os.path.exists(args.log_path) == False
    os.makedirs(args.log_path)

    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s',
                        datefmt='%H:%M:%S',
                        handlers=[
                            logging.FileHandler(args.log_path + '/train_log.log'),
                            logging.StreamHandler(sys.stdout)
                        ])
    logger = logging.getLogger()
    writer = SummaryWriter(args.log_path)

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        torch.backends.cudnn.benchmark = False

    train_loader, val_loader = get_data(base_folder=args.base_folder,
                                        batch_size=args.batch_size, return_test=False, debug=False)

    image_encoder, prompt_encoder, mask_decoder, mask_decoder1 = init_network(device=device)

    model = Model(
        image_encoder=image_encoder,
        prompt_encoder=prompt_encoder,
        mask_decoder=mask_decoder,
        mask_decoder1 = mask_decoder1
    ).to(device)

    for name, param in model.named_parameters():
        print(name, param.requires_grad)


    train_net(model, train_loader, val_loader, device, args, logger=logger, writer=writer)


