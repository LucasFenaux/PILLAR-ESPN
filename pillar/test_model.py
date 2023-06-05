import torch
from train import main


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Classification Testing", add_help=add_help)
    parser.add_argument("--model-file", dest="model_file", default=None, type=str)

    return parser


def main_wrapper(file_args):
    model_file = file_args.model_file
    checkpoint = torch.load(file_args.model_file, map_location="cpu")
    args = checkpoint["args"]
    if hasattr(args, "rank"):
        del args.rank
        args.distributed = False
    args.test_only = True
    args.resume = model_file
    main(args)


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    main_wrapper(args)

