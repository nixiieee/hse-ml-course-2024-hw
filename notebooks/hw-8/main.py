import click
import pickle
from sklearn.pipeline import Pipeline

path_to_model = 'model_prog.pickle'

def predict_model(data):
    with open(data, 'r') as file:
        X = file.readlines()
    with open(path_to_model, 'rb') as file:
        pipeline = pickle.load(file)
    y_pred = pipeline.predict(X)
    return y_pred[0]

@click.command()
@click.argument('name')
def main(*args, **kwargs):
    print(predict_model(kwargs['name']))

if __name__ == '__main__':
    main()