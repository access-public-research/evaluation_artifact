from .selectors import select_hybrid, select_proxy, select_tailmoderated


def route_selector(val_metrics, snr, snr_threshold=1.5, lam=0.5, topk_frac=0.1):
    if snr >= snr_threshold:
        return select_proxy(val_metrics)
    return select_tailmoderated(val_metrics, lam=lam)
