import joblib
import os
import json


def features(file_path, object):
    """Quickly save model test features as a pickle.
    
    Keyword arguments:
    file_path -- file path to save the test features.
    object -- test features to convert to a pickle."""
    if os.path.exists(file_path) is False:
        os.makedirs(file_path)

    joblib.dump(object, file_path + "/test_features.pkl")


def labels(file_path, object):
    """Quickly save model test labels as pickle.
    
    Keyword arguments:
    file_path -- file path to save the test labels.
    object -- test labels to convert to a pickle."""
    if os.path.exists(file_path) is False:
        os.makedirs(file_path)

    joblib.dump(object, file_path + "/test_labels.pkl")


def adver_example(file_path, min_change, object):
    """Save adversarial examples as a pickle.
    
    Keyword arguments:
    file_path -- file path to save the adversarial examples.
    min_change -- the minimum allowed change for the attack.
    object -- attack to convert to a pickle."""
    if os.path.exists(file_path) is False:
        os.makedirs(file_path)

    joblib.dump(object, file_path + "/{}.pkl".format(min_change))


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
