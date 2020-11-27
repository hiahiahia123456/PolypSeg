import argparse
import os

parser = argparse.ArgumentParser()

"-------------------data option--------------------------"
parser.add_argument('--train_path', type=str, default='/data1/yinzijin/EndoScene/TrainDataset', help='path to train dataset')
parser.add_argument('--test_path', type=str, default='/data1/yinzijin/EndoScene/TestDataset', help='path to test dataset')
parser.add_argument('--valid_path', type=str, default='/data1/yinzijin/EndoScene/ValidationDataset', help='path to validation dataset')

parser.add_argument('--train_save', type=str, default=None)
parser.add_argument('--test_save', type=str, default=None)

"-------------------training option-----------------------"
parser.add_argument('--epoch', type=int, default=20, help='epoch number')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--batchsize', type=int, default=16, help='training batch size')
parser.add_argument('--trainsize', type=int, default=352, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=50, help='every n epochs decay learning rate')
parse.add_argument('--ckpt_period', type=int, default=5, help='how often the checkpoint is saved')
parse.add_argument('--use_gpu', type=bool, default=True, help='whether to use GPU')

config = parse.parse_args()
