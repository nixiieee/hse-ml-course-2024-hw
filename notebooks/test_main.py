import os
import pandas as pd
import numpy as np
import pytest
import io
import pickle
import sys
from sklearn.pipeline import Pipeline
from main import train_model, predict_model

TEST_DATA_PATH = 'test_small.csv'
TRAIN_DATA_PATH = 'train.csv'
MODEL_PATH = 'test_model.pkl'
MODEL_PREDICT_PATH = 'model.pkl'

def is_float(element: any) -> bool:
    if element is None: 
        return False
    try:
        float(element)
        return True
    except ValueError:
        return False

# TRAIN TESTS

def test_train_model_non_existing_train_file():
    with pytest.raises(OSError):
        train_model('non_existing_train_file.csv', TEST_DATA_PATH, None, MODEL_PATH)
    

@pytest.mark.usefixtures('cleanup_model')
def test_train_model_test_file():
    captured_output = io.StringIO()
    sys.stdout = captured_output

    train_model(TRAIN_DATA_PATH, TEST_DATA_PATH, None, MODEL_PATH)

    with open(MODEL_PATH, 'rb') as file:
        loaded_model = pickle.load(file)

    assert isinstance(loaded_model, Pipeline)
    assert hasattr(loaded_model, 'steps')

    output = captured_output.getvalue()
    sys.stdout = sys.__stdout__
    assert is_float(output.strip())

@pytest.mark.usefixtures('cleanup_model')
def test_train_model_split_correct():
    captured_output = io.StringIO()
    sys.stdout = captured_output

    train_model(TRAIN_DATA_PATH, None, 0.2, MODEL_PATH)

    with open(MODEL_PATH, 'rb') as file:
        loaded_model = pickle.load(file)

    assert isinstance(loaded_model, Pipeline)
    assert hasattr(loaded_model, 'steps')

    output = captured_output.getvalue()
    sys.stdout = sys.__stdout__
    assert is_float(output.strip())


def test_train_model_split_incorrect_type():
    with pytest.raises(TypeError):
        train_model(TRAIN_DATA_PATH, None, "aboba", MODEL_PATH)


def test_train_model_split_incorrect_value():
    with pytest.raises(ValueError):
        train_model(TRAIN_DATA_PATH, None, 2, MODEL_PATH)

# PREDICT TESTS 

def test_predict_model_non_existing_test_file():
    with pytest.raises(OSError):
        predict_model('non_existing_file.csv', MODEL_PATH)


def test_predict_model_non_existing_model_file():
    with pytest.raises(FileNotFoundError):
        predict_model(TEST_DATA_PATH, 'non_existing_model.pkl')


def test_predict_model_file():
    captured_output = io.StringIO()
    sys.stdout = captured_output

    predict_model(TEST_DATA_PATH, MODEL_PREDICT_PATH)

    output = captured_output.getvalue()
    sys.stdout = sys.__stdout__

    test_file = pd.read_csv('../data/' + TEST_DATA_PATH)
    assert (len(output.strip().split('\n')) == test_file.shape[0])


@pytest.fixture
def cleanup_model():
    yield
    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)
