import os
import re
from argparse import Namespace
digit = re.compile(r'^[-+]?[-0-9]\d*\.\d*|[-+]?\.?[0-9]\d*$')
floatexp = re.compile(r'^[-+]?[-0-9]\d*e\d*|[-+]?\.?[0-9]\d*$')

def parse_cfg(cfg):
    bools = ['True', 'False']
    cfgstr = cfg
    if os.path.isfile(cfgstr):
        cfgstr = open(cfgstr).read()
    items = cfgstr.split('\n')
    options = {}
    for item in items:
        if '=' not in item or item.strip().startswith('#'):
            continue
        key, val = item.replace(' ', '').split('#')[0].split('=')

        if ',' in val:
            val = val.split(',')
            if digit.match(val[0]):
                options[key] = list(map(lambda x: int(x) if str.isnumeric(x) else float(x), val))
            else:
                options[key] = val
        elif str.isnumeric(val):
            options[key] = int(val)
        elif digit.match(val) or floatexp.match(val):
            options[key] = float(val)
        elif val in bools:
            options[key] = 'True' == val
        else:
            options[key] = val
    return options


def parse_model_config(cfgfile):
    """Parses the yolo-v3 layer configuration file and returns module definitions"""
    if cfgfile.endswith('cfg'):
        cfgfile = open(cfgfile, 'r')
        cfgfile = cfgfile.read()
    lines = cfgfile.split('\n')
    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.rstrip().lstrip() for x in lines] # get rid of fringe whitespaces
    module_defs = []
    for line in lines:
        line = line.strip()
        if line == '' or line[0] == '#':
            continue
        if line.startswith('['): # This marks the start of a new block
            module_defs.append({})
            module_defs[-1]['type'] = line[1:-1].rstrip()
            if module_defs[-1]['type'] == 'convolutional':
                module_defs[-1]['batch_normalize'] = 0
        else:
            key, value = line.split("=")
            value = value.strip()
            module_defs[-1][key.rstrip()] = value.split('#')[0].strip()

    return module_defs

def parse_data_config(path):
    """Parses the data configuration file"""
    options = dict()
    options['gpus'] = '0,1,2,3'
    options['num_workers'] = '10'
    with open(path, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.strip()
        if line == '' or line.startswith('#'):
            continue
        key, value = line.split('=')
        options[key.strip()] = value.strip()
    return options
