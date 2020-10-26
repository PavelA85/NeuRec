import warnings

from matplotlib import pylab

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


def mymain():
    config = Configurator()
    config.add_config("NeuRec.ini", section="NeuRec")
    config.parse_cmd()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config["gpu_id"])
    _set_random_seed(config["seed"])
    Recommender = find_recommender(config.recommender, platform=config.platform)

    model_cfg = os.path.join("conf", config.recommender + ".ini")
    config.add_config(model_cfg, section="hyperparameters", used_as_summary=True)

    recommender = Recommender(config)
    results = recommender.train_model()

    ######################################
    import pprint
    pp = pprint.PrettyPrinter(indent=4)
    myresult = {x + '@' + str(y): [] * config.epochs for x in config.metric for y in config.top_k}
    metrics = list(x + '@' + str(y) for x in config.metric for y in config.top_k)
    for epoch, r in enumerate(results):
        for metric_arr in r:
            for itenN, value in enumerate(metric_arr):
                # print("{} {} {}".format(epoch, itenN, value))
                myresult[metrics[itenN]].append(value)

    pp.pprint(myresult)
    ######################################
    import matplotlib.pyplot as plt
    for m in metrics:
        plt.plot(list(range(config.epochs)), myresult[m], label=m)

    plt.xlabel('epoch')
    plt.ylabel('value')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='metrics', title_fontsize='xx-large')
    fig = plt.gcf()

    fig.savefig('books_read1.png', bbox_inches='tight')
    plt.show()
    ######################################
    return results, {x + str(y): [] * config.epochs for x in config.metric for y in config.top_k}


def myplot(results, config, path):
    myresult = {x + '@' + str(y): [] * config.epochs for x in config.metric for y in config.top_k}
    metrics = list(x + '@' + str(y) for x in config.metric for y in config.top_k)
    for epoch, r in enumerate(results):
        for metric_arr in r:
            for itemN, value in enumerate(metric_arr):
                myresult[metrics[itemN]].append(value)

    #pp.pprint(myresult)
    ######################################
    import matplotlib.pyplot as plt
    for m in metrics:
        plt.plot(list(range(config.epochs)), myresult[m], label=m)

    plt.xlabel('epoch')
    plt.ylabel('value')
    plt.title(config.recommender)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='metrics', title_fontsize='xx-large')
    fig = plt.gcf()

    fig.savefig(os.path.join(path, config.recommender + '.png'), bbox_inches='tight')
    plt.show()


def main():
    from datetime import datetime
    rez = 'C:/Projects/NeuRec/results/' + datetime.now().strftime('%Y%m%d%H%M%S')
    os.mkdir(rez)

    for rec in ['MF', 'Caser', 'CDAE',
                #'FISM',
                'FPMC',
                #'HGN',
                'LightGCN',  'MultVAE', 'NGCF', 'TransRec']:
        config = Configurator()
        config.add_config("NeuRec.ini", section="NeuRec")
        config.parse_cmd()
        os.environ['CUDA_VISIBLE_DEVICES'] = str(config["gpu_id"])
        _set_random_seed(config["seed"])
        config.sections['NeuRec:[NeuRec]']['recommender'] = rec
        Recommender = find_recommender(config.recommender, platform=config.platform)

        model_cfg = os.path.join("conf", rec + ".ini")
        config.add_config(model_cfg, section="hyperparameters", used_as_summary=True)

        config.sections[rec + ":[hyperparameters]"]['epochs'] = '20'

        print('doing {0} {1}'.format(rec, config.summarize()))

        recommender = Recommender(config)
        results = recommender.train_model()
        myplot(results, config, rez)


if __name__ == "__main__":
    main()
