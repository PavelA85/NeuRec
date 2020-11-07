import warnings

import numpy

warnings.simplefilter(action='ignore', category=FutureWarning)

import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from reckit import Configurator
from importlib.util import find_spec
from importlib import import_module
from reckit import typeassert
import os
import numpy as np
import random

import pprint
import matplotlib.pyplot as plt
from reckit import timer

pp = pprint.PrettyPrinter(indent=4)

import json
from datetime import datetime

import pandas as pd


def _set_random_seed(seed=2020):
    np.random.seed(seed)
    random.seed(seed)

    try:
        import tensorflow as tf
        tf.set_random_seed(seed)
        print("set tensorflow seed")
    except:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        print("set pytorch seed")
    except:
        pass


@typeassert(recommender=str, platform=str)
def find_recommender(recommender, platform="pytorch"):
    model_dirs = set(os.listdir("model"))
    model_dirs.remove("base")

    module = None
    if platform == "pytorch":
        platforms = ["pytorch", "tensorflow"]
    elif platform == "tensorflow":
        platforms = ["tensorflow", "pytorch"]
    else:
        raise ValueError(f"unrecognized platform: '{platform}'.")

    for platform in platforms:
        if module is not None:
            break
        for tdir in model_dirs:
            spec_path = ".".join(["model", tdir, platform, recommender])
            if find_spec(spec_path):
                module = import_module(spec_path)
                break

    if module is None:
        raise ImportError(f"Recommender: {recommender} not found")

    if hasattr(module, recommender):
        Recommender = getattr(module, recommender)
    else:
        raise ImportError(f"Import {recommender} failed from {module.__file__}!")
    return Recommender


def prepare_metrics(config):
    return list(x + '@' + str(y)
                for x in config.metric
                for y in config.top_k)


def prepare_results(results, metrics):
    result_dict = {value: [] for value in metrics}

    for epoch, result in enumerate(results):
        for metric in result:
            for item, value in enumerate(metric):
                result_dict[metrics[item]].append(value)
    return result_dict


def myplot(result, metrics, recommender, path):
    for m in metrics:
        plt.plot(list(range(len(result[recommender][m]))), result[recommender][m], label=m)

    plt.xlabel('epoch')
    plt.ylabel('value')
    plt.title(recommender)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='metrics', title_fontsize='xx-large')
    fig = plt.gcf()

    if len(path) > 0:
        fig.savefig(os.path.join(path, datetime.now().strftime('%Y%m%d%H%M%S') + '_' + recommender + '.png'),
                    bbox_inches='tight')
    plt.show()


@timer
def evaluate(algorithm):
    config = Configurator()
    config.add_config("NeuRec.ini", section="NeuRec")
    # config.parse_cmd()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config["gpu_id"])
    _set_random_seed(config["seed"])
    config.sections['NeuRec:[NeuRec]']['recommender'] = algorithm
    Recommender = find_recommender(config.recommender, platform=config.platform)
    model_cfg = os.path.join("conf", algorithm + ".ini")
    config.add_config(model_cfg, section="hyperparameters", used_as_summary=True)
    config.sections[algorithm + ":[hyperparameters]"]['epochs'] = '1'
    print('doing {0} {1}'.format(algorithm, config.summarize()))
    recommender = Recommender(config)
    recommender.epochs = 500
    result = recommender.train_model()
    metrics = prepare_metrics(config)
    result = prepare_results(result, metrics)

    return {algorithm: result}, metrics


def main():
    path = 'C:/Projects/NeuRec/results/'
    # os.mkdir(path)

    results = []
    general = ['MF', 'CDAE', 'LightGCN', 'MultVAE', 'NGCF']  ##',FISM']
    sequential = ['Caser', 'FPMC', 'HGN', 'TransRec']
    for algorithm in [*general, *sequential]:
        result, metrics = evaluate(algorithm)
        myplot(result, metrics, algorithm, path)
        results.append(result)

    json_path = os.path.join(path, datetime.now().strftime('%Y%m%d%H%M%S') + '_results.json')
    # pd.Series(results).to_json(orient='values', ath_or_buf=json_path)

    pp.pprint(results)

    with open(json_path, 'w',
              encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=1, cls=MyEncoder)

    for result in results:
        for metric in metrics:
            algorithm, val = next(iter(result.items()))
            label = algorithm + '_' + metric
            plt.plot(list(range(len(val[metric]))), val[metric], label=label)

    plt.xlabel('epoch')
    plt.ylabel('value')
    plt.title('Single plot for all metrics and algo')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='metrics', title_fontsize='xx-large')
    plt.ylim(ymin=0, ymax=1)
    plt.xlim(xmin=0, xmax=1)
    fig = plt.gcf()
    if len(path) > 0:
        fig.savefig(os.path.join(path, datetime.now().strftime('%Y%m%d%H%M%S') + '_all' + '.png'), bbox_inches='tight')
    plt.show()

    # with open(json_path) as json_file:
    #     data = json.load(json_file)
    #     pp.pprint(data)


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.integer):
            return int(obj)
        elif isinstance(obj, numpy.floating):
            return float(obj)
        elif isinstance(obj, numpy.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


if __name__ == "__main__":
    main()
