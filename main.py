
import argparse

from train import main

def get_args():
    parser = argparse.ArgumentParser()
    
    # project
    parser.add_argument('name', type=str, help='name of the experiment')

    # data
    parser.add_argument('--image-size', default=256, type=int, help='The image size')
    parser.add_argument('--batch-size', '-b', default=32, type=int, help='Batch size')

    # model
    parser.add_argument('--depth', '-d', default=6, type=int, help='Depth of U2Net (Generator)')
    parser.add_argument('--src-channels', '-sc', default=1, type=int, help='Number of channels of the source image')
    parser.add_argument('--dst-channels', '-dc', default=1, type=int, help='Number of channels of the destination image')
    parser.add_argument('--channels', '-c', default=32, type=int, help='Chennel width multiplier')
    parser.add_argument('--use-sn-g', '-sng', action='store_true', help='Use spectral normalization in generator')
    parser.add_argument('--use-sn-d', '-snd', action='store_true', help='Use spectral normalization in discriminator')
    parser.add_argument('--norm-g', '-ng', default='bn', choices=['bn', 'in'], help='Normalization layers for generator')
    parser.add_argument('--norm-d', '-nb', default='bn', choices=['bn', 'in'], help='Normalization layers for discriminator')
    parser.add_argument('--act-g', '-ag', default='relu', choices=['relu', 'lrelu'], help='Activation funciton for Generator')
    parser.add_argument('--act-d', '-ad', default='lrelu', choices=['relu', 'lrelu'], help='Activation function for Discriminator')
    parser.add_argument('--up-mode', '-um', default='bilinear', type=str, help='Upsampling mode')
    parser.add_argument('--down-mode', '-dm', default='avg', choices=['max', 'avg'], help='Downsampling mode')
    parser.add_argument('--rgb-output', '-rgb', action='store_true', help='Tanh activation on output')
    parser.add_argument('--num-d-layers', '-dl', default=3, type=int, help='Number of layers in Discriminator')
    parser.add_argument('--init', '-i', default='normal', choices=['normal', 'xavier'], help='Initualization method')

    # training
    parser.add_argument('--learning-rate-g', '-lrg', default=0.0002, type=float, help='Learning rate for generator')
    parser.add_argument('--learning-rate-d', '-lrd', default=0.0002, type=float, help='Learning rate for Discriminator')
    parser.add_argument('--beta1', '-b1', default=0.5, type=float, help='Beta1 parameter for optimizer')
    parser.add_argument('--beta2', '-b2', default=0.999, type=float, help='Beta2 parameter for optimizer')
    parser.add_argument('--max-iter', '-m', default=-1, type=int, help='Max iterations')
    parser.add_argument('--l1-lambda', '-l1l', default=100, type=float, help='Lambda for L1 loss')
    parser.add_argument('--use-amp', '-a', action='store_true', help='Use AMP')
    parser.add_argument('--result-folder', '-r', default='result', type=str, help='Folder for saving the results')
    parser.add_argument('--save-interval', '-s', default=100, type=int, help='Interval for saving model parameters')

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    main(args)
    