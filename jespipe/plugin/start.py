import argparse


def start():
    """Pull three parameters from stdin; stage, dataset, and parameters."""
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", nargs=1)
    parser.add_argument("dataset", nargs=1)
    parser.add_argument("parameters", nargs=1)
    args = parser.parse_args()

    # Return tuple in the following format: (stage, dataset, parameters)
    return args.stage, args.dataset, args.parameters
