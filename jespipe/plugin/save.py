import joblib
import os
import json


def pickle(file_path, name, object):
    """Quickly save Python objects as a pickle.
    
    Keyword arguments:
    file_path -- file path to save the pickle.
    name -- file name to use for the pickle.
    object -- Python object to convert to a pickle."""
    if os.path.exists(file_path) is False:
        os.makedirs(file_path)

    joblib.dump(object, file_path + "/{}".format(name))


def compress_dataframe(file_path, name, dataset):
    """Save a pandas DataFrame as a .csv.gz file.
    
    Keyword arguments:
    file_path -- file path to save the pandas DataFrame.
    name -- file name to use for the pandas DataFrame.
    dataset -- pandas DataFrame to convert to a pickle."""
    if os.path.exists(file_path) is False:
        os.makedirs(file_path)

    dataset.to_csv(file_path + "/{}".format(name), index=False, header=None, compression="gzip")

def dictionary(file_path, name, dictdata):
    """Save a Python dictionary to a .json file.
    
    Keyword arguments:
    file_path -- file path to save the dictionary data.
    name -- file name to use for the dictionary data.
    dictdata -- Python dictionary to save as JSON data."""
    if os.path.exists(file_path) is False:
        os.makedirs(file_path)

    fout = open(file_path + "/{}".format(name), "wt"); fout.write(json.dumps(dictdata)); fout.close()

def text(file_path, name, textdata):
    """Save unstructured text data as a .txt file.
    
    Keyword arguments:
    file_path -- file path to save the unstructured text data.
    name -- file name to use for the unstructured text data.
    textdata -- text to save as unstructured text data."""
    if os.path.exists(file_path) is False:
        os.makedirs(file_path)

    fout = open(file_path + "/{}".format(name), "wt"); fout.write(textdata); fout.close()
