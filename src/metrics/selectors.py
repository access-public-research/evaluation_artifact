import numpy as np


def select_overall(val_metrics):
    return int(np.argmax(val_metrics["overall_acc"]))


def select_proxy(val_metrics):
    return int(np.argmax(val_metrics["proxy_worst_acc"]))


def select_tailmoderated(val_metrics, lam=0.5):
    scores = val_metrics["overall_acc"] - lam * val_metrics["proxy_worst_loss"]
    return int(np.argmax(scores))


def select_hybrid(val_metrics, topk_frac=0.1):
    k = max(1, int(len(val_metrics["overall_acc"]) * topk_frac))
    topk = np.argsort(val_metrics["overall_acc"])[-k:]
    best = topk[np.argmax(val_metrics["proxy_worst_acc"][topk])]
    return int(best)
