import copy
import yaml


def _deep_merge(a, b):
    out = copy.deepcopy(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _interpolate_paths(cfg):
    root = cfg.get("project", {}).get("root")
    if not root:
        return cfg

    def _walk(obj):
        if isinstance(obj, dict):
            return {k: _walk(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_walk(v) for v in obj]
        if isinstance(obj, str):
            return obj.replace("${project.root}", str(root))
        return obj

    return _walk(cfg)


def load_config(base_path, dataset_path=None, regime_path=None):
    """Load base config and optionally overlay dataset/regime configs."""
    with open(base_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if dataset_path:
        with open(dataset_path, "r", encoding="utf-8") as f:
            cfg = _deep_merge(cfg, {"dataset": yaml.safe_load(f)})
    if regime_path:
        with open(regime_path, "r", encoding="utf-8") as f:
            cfg = _deep_merge(cfg, {"regime": yaml.safe_load(f)})
    return _interpolate_paths(cfg)
