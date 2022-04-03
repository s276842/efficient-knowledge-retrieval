import configparser

import torch
import argparse
import vectorized_knowledge
import zeroshot

def setup_args():
    """Setup inference arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataroot", type=str, required=True)
    parser.add_argument("--dataset", type=str, default='val', choices=['test', 'train', 'val'])
    parser.add_argument("--device", type=str, default='cpu')
    parser.add_argument("--method", type=str, required=True, choices=['zeroshot'])
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument('-c', "--conf", type=str, required=True)


    parser.add_argument("--log_file_path", type=str, required=True)
    parser.add_argument("--label_file_path", type=str, required=True)

    args = parser.parse_args()
    args.load(args.config_path)

    return args


if __name__ == '__main__':
    args = setup_args()

    config = configparser.ConfigParser()
    config.read(args.conf)

    method = args.method

    if method == 'vectorize_knowledge':
        vectorized_knowledge.main(args.dataroot, args.device, args.output_path, config)
    elif method == 'zeroshot':
        zeroshot.main(args.dataroot, args.dataset, args.device, args.ouput_path, config)

