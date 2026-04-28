import numpy as np


def random_hash_partition(embeddings, num_bits, seed=0, num_cells=None, return_matrix=False):
    rng = np.random.default_rng(seed)
    d = embeddings.shape[1]
    R = rng.standard_normal(size=(int(num_bits), int(d)))
    bits = (embeddings @ R.T) > 0
    ids = bits.astype(np.int32)
    cell_ids = np.zeros(len(embeddings), dtype=np.int32)
    for i in range(int(num_bits)):
        cell_ids |= (ids[:, i] << i)
    if num_cells is not None:
        cell_ids = cell_ids % int(num_cells)
    if return_matrix:
        return cell_ids, R
    return cell_ids
