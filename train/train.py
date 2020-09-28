import glob
import os
import argparse
import torch
from test.parse_config import parse_cfg

def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--configfile",
        help="path to config file",
        default='/home/cobot/YOLO/config/train_yolo.cfg',
        type=str,
    )

    args = parser.parse_args()
    cfgs = parse_cfg(args.configfile)

    torch.cuda.set_device(cfgs["gpus"])

    train_darknet(cfgs)


def train_darknet(cfgs):
    this_dir = os.path.dirname(os.path.abspath(__file__))

    abspath = this_dir + '/../yolo/data'
    outfile = os.path.join(abspath, ".train.data")
    outfile = auto_gen_datafile(cfgs["train_dir"],
                                cfgs["val_dir"],
                                cfgs["classes"],
                                cfgs["names"],
                                cfgs["weight_out"],
                                abspath,
                                outfile)
    if isinstance(cfgs['gpus'], int):
        cfgs['gpus'] = [cfgs['gpus']]
    gpus = ','.join(map(str, cfgs['gpus']))
    os.system("pwd")
    command = this_dir + '/../yolo/darknet/darknet detector train ' + outfile + ' ' + cfgs['cfgpath'] + ' ' + cfgs['init_weights'] #+ ' -show_imgs'
    if gpus:
        command = command + ' -gpus ' + gpus
    print(command)
    os.system(command)


def auto_gen_datafile(train_dir, val_dir, classes, names, backup, abspath, outfile=".train.data"):
    """
    :param train_dir: folder
    :param val_dir: folder
    :param classes: int
    :param names: list of cls name. e.g:[car, guazi, ma]
    :param backup: folder
    :return:
    """

    train_list = glob.glob(os.path.join(train_dir, '*.jpg')) + glob.glob(os.path.join(train_dir, '*.png'))
    val_list = glob.glob(os.path.join(val_dir, '*.jpg'))+glob.glob(os.path.join(val_dir, '*.png'))

    if not train_list:
        raise FileNotFoundError("No data file found in {0}, only support for .jpg, .png data format".format(train_dir))

    open(os.path.join(abspath, '.train.txt'), 'w').writelines([elem + '\n' for elem in train_list])
    open(os.path.join(abspath, '.val.txt'), 'w').writelines([elem + '\n' for elem in val_list])
    open(os.path.join(abspath, '.cls.name'), 'w').writelines([str(elem) + '\n' for elem in names])
    with open(outfile, 'w') as f:
        f.write('classes = '+str(classes)+'\n')
        f.write('train  = %s\n' % os.path.join(abspath, ".train.txt"))
        f.write('valid  = %s\n' % os.path.join(abspath, ".val.txt"))
        f.write('names = %s\n' % os.path.join(abspath, ".cls.name"))
        f.write('backup = ' + backup)
    return outfile


if __name__=='__main__':
    main()