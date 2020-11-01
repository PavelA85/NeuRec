import warnings

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
        plt.plot(list(range(len(result[m]))), result[m], label=m)

    plt.xlabel('epoch')
    plt.ylabel('value')
    plt.title(recommender)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='metrics', title_fontsize='xx-large')
    fig = plt.gcf()

    if len(path) > 0:
        fig.savefig(os.path.join(path, recommender + '.png'), bbox_inches='tight')
    plt.show()


@timer
def evaluate(algorithm, path):
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
    myplot(result, metrics, algorithm, path)

    return {algorithm: result}, metrics


def main():
    from datetime import datetime
    path = 'C:/Projects/NeuRec/results/' + datetime.now().strftime('%Y%m%d%H%M%S')
    os.mkdir(path)

    results = []
    general = ['MF', 'CDAE', 'LightGCN', 'MultVAE', 'NGCF']  ##',FISM']
    sequential = ['Caser', 'FPMC', 'HGN', 'TransRec']
    for algo in [*general, *sequential]:
        result, metrics = evaluate(algo, path)
        results.append(result)

    #pp.pprint(results)

    for result in results:
        for metric in metrics:
            algo, val = next(iter(result.items()))
            label = algo + '_' + metric
            plt.plot(list(range(len(val[metric]))), val[metric], label=label)

    plt.xlabel('epoch')
    plt.ylabel('value')
    plt.title('Single plot for all metrics and algo')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='metrics', title_fontsize='xx-large')
    fig = plt.gcf()
    if len(path) > 0:
        fig.savefig(os.path.join(path, 'all' + '.png'), bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()
