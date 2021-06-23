import argparse
import json


def start():
    """Pull two parameters from stdin; stage and parameters."""
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", nargs=1, type=str)
    parser.add_argument("parameters", nargs=1, type=json.loads)
    args = parser.parse_args()

    # Return tuple in the following format: (stage, dataset, parameters)
    return args.stage, args.parameters
