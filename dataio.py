# -*- coding: utf-8 -*-
"""
Input/Output utils
"""
import pandas as pd
import pickle

def write_output(prediction, prediction_label, identifiers, identifiers_label, output_filename = "./output.csv", verbose = True):
    """
    Write output
    """
    df = pd.DataFrame({prediction_label: prediction, identifiers_label: identifiers}).set_index(identifiers_label)
    df.to_csv(output_filename)
    if verbose:
        print("Wrote data into %s" % output_filename)

        
def write_model(model, model_filename = "./model.pickle", verbose = True):
    """
    Write model using pickle
    """
    with open(model_filename, "wb") as fd:
        pickle.dump(model, fd)
    if verbose:
        print("Wrote model into %s" % model_filename)


def load_model(model_filename):
    """
    Load model using pickle
    """
    with open(model_filename, "rb") as fd:
        model = pickle.load(fd)
    return model
