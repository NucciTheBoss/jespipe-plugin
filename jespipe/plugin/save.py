import joblib
import os
import json


def pickle(file_path, name, object):
    """Quickly save Python objects as a pickle.
    
    Keyword arguments:
    file_path -- file path to save the pickle.
    name -- file name to use for the pickle.
    object -- Python object to convert to a pickle."""
    if os.path.exists(file_path):
        joblib.dump(object, file_path + "/{}.pkl".format(name))

    else:
        raise OSError("Desired file path {} not found.".format(file_path))


def compress_dataframe(file_path, name, dataset):
    """Save a pandas DataFrame as a .csv.gz file.
    
    Keyword arguments:
    file_path -- file path to save the pickle.
    name -- file name to use for the pickle.
    dataset -- Pandas DataFrame to convert to a pickle."""
    if os.path.exists(file_path):
        dataset.to_csv(file_path + "/{}.csv.gz".format(name), index=False, header=None, compression="gzip")

    else:
        raise OSError("Desired file path {} not found.".format(file_path))


def dictionary(file_path, name, dictdata):
    """Save a Python dictionary to a .json file.
    
    Keyword arguments:
    file_path -- file path to save the pickle.
    name -- file name to use for the pickle.
    dictdata -- Python dictionary to save as JSON data."""
    if os.path.exists(file_path):
        fout = open(file_path + "/{}.json".format(name), "wt")
        fout.write(json.dumps(dictdata))
        fout.close()

    else:
        raise OSError("Desired file path {} not found.".format(file_path))


def text(file_path, name, textdata):
    """Save unstructured text data as a .txt file.
    
    Keyword arguments:
    file_path -- file path to save the pickle.
    name -- file name to use for the pickle.
    dictdata -- Python dictionary to save as JSON data."""
    if os.path.exists(file_path):
        fout = open(file_path + "/{}.txt".format(name), "wt")
        fout.write(textdata)
        fout.close()

    else:
        raise OSError("Desired file path {} not found.".format(file_path))
