"""
Functions to compute distances and similarities between embeddings.
"""

import warnings
from typing import TYPE_CHECKING

import numpy as np

from .one_to_n_scores import OneToNScores
from .scores import Scores

if TYPE_CHECKING:  # pragma: no cover
    import torch


def l2_squared_matrix(x, y):
    """Calculates all pairwise squared Euclidean distances between x and y.

    If x, y have more than 2 dimensions, then all but the last two dimensions have to
    be the same. E.g. inputs of shape (U, N, D), (U, M, D) gives an output of shape
    (U, N, M).

    Args:
        x, y: Arrays of shapes (N, D) and (M, D).

    Returns:
        Matrix of shape (N, M).
    """
    x_norm = np.einsum("...i,...i->...", x, x)[..., None]  # (U, N, 1)
    y_norm = np.einsum("...i,...i->...", y, y)[..., None, :]  # (U, 1, M)
    d = np.matmul(x, np.swapaxes(y, -1, -2))  # (U, N, M)
    d *= -2
    d += x_norm
    d += y_norm
    np.clip(d, a_min=0.0, a_max=None, out=d)  # Distances should be non-negative
    return d


def l2_matrix(x, y):
    """Calculates all pairwise Euclidean distances between x and y.

    If x, y have more than 2 dimensions, then all but the last two dimensions have to
    be the same. E.g. inputs of shape (U, N, D), (U, M, D) gives an output of shape
    (U, N, M).

    Args:
        x, y: Arrays of shapes (N, D) and (M, D).
        use_torch: If True, use PyTorch with GPU acceleration for computation.

    Returns:
        Matrix of shape (N, M).
    """
    return np.sqrt(l2_squared_matrix(x, y))


def cosine_matrix(x, y):
    """Calculates all pairwise squared cosine distances between x and y.

    If x, y have more than 2 dimensions, then all but the last two dimensions have to
    be the same. E.g. inputs of shape (U, N, D), (U, M, D) gives an output of shape
    (U, N, M).

    Args:
        x, y: Arrays of shapes (N, D) and (M, D).

    Returns:
        Matrix of shape (N, M).
    """
    # x_norm = np.linalg.norm(x, axis=-1, keepdims=True)  # (U, N, 1)
    # y_norm = np.linalg.norm(y, axis=-1, keepdims=True)  # (U, M, 1)
    # y_norm = np.swapaxes(y_norm, -1, -2)  # (U, 1, M)
    x_norm = np.einsum("...i,...i->...", x, x)[..., None]  # (U, N, 1)
    y_norm = np.einsum("...i,...i->...", y, y)[..., None, :]  # (U, 1, M)
    np.sqrt(x_norm, out=x_norm)
    np.sqrt(y_norm, out=y_norm)
    np.clip(x_norm, a_min=1e-10, a_max=None, out=x_norm)
    np.clip(y_norm, a_min=1e-10, a_max=None, out=y_norm)

    d = np.matmul(x, np.swapaxes(y, -1, -2))  # (U, N, M)
    d /= x_norm
    d /= y_norm
    d *= -1
    d += 1
    return d


def embedding_distances(
    emb: np.ndarray,
    labels: np.ndarray,
    dist: str = "l2_squared",
    pos_limit: int | float | None = None,
    neg_limit: int | float | None = None,
    batch_size: int | None = 1e8,
    return_indices: bool = False,
    use_torch: bool = False,
    torch_dtype: "torch.dtype | str" = "float32",
) -> Scores | tuple[Scores, np.ndarray, np.ndarray]:
    """Compute pairwise distances between embeddings, split into positive and negative
    pairs based on labels.

    For each pair of embeddings (i, j) with i < j, compute the distance and classify
    it as positive (same label) or negative (different label). The resulting distances
    are returned as a ``Scores`` object.

    When ``pos_limit`` or ``neg_limit`` are specified, the number of positive or
    negative pairs is limited. If the limit is an integer >= 1, it specifies the maximum
    number of pairs. If the limit is a float < 1, it specifies the fraction of pairs to
    keep. The actual number of pairs before limiting is stored in ``scores.nb_all_pos``
    and ``scores.nb_all_neg``.

    Args:
        emb: Array of shape ``(N, D)`` containing ``N`` embeddings of dimension ``D``.
        labels: Array of shape ``(N,)`` containing the label for each embedding.
        dist: Distance metric to use. One of ``"l2_squared"``, ``"l2"`` or ``"cosine"``.
        pos_limit: Maximum number (or fraction) of positive pairs to return.
        neg_limit: Maximum number (or fraction) of negative pairs to return.
        batch_size: Process embeddings in batches to limit memory usage. If ``None``,
            all pairs are computed at once.
        return_indices: If True, also return the embedding indices for each pair.
            Only supported with ``use_torch=False``.
        use_torch: Whether to use PyTorch for distance computation.
        torch_dtype: Data type to use for PyTorch computations. We can use "bfloat16"
            on supported GPUs for faster computation and reduced memory usage.

    Returns:
        A ``Scores`` object with positive and negative distance arrays. If
        ``return_indices`` is True, returns a tuple ``(scores, pos_idx, neg_idx)``
        where ``pos_idx`` and ``neg_idx`` are arrays of shape ``(K, 2)`` containing
        the embedding indices for each pair, sorted by distance.
    """
    emb = np.asarray(emb)
    if np.issubdtype(emb.dtype, np.integer):
        emb = emb.astype(np.float32)
    labels = np.asarray(labels)
    n = len(emb)

    # Count total pos/neg pairs from labels alone (no distance computation needed)
    _, counts = np.unique(labels, return_counts=True)
    nb_all_pos = np.sum(counts * (counts - 1) // 2, dtype=int)
    nb_all_neg = n * (n - 1) // 2 - nb_all_pos

    # Convert fraction limits to absolute counts
    if pos_limit is not None:
        if isinstance(pos_limit, float) and pos_limit < 1:
            pos_limit = int(nb_all_pos * pos_limit)
        pos_limit = int(pos_limit)

    if neg_limit is not None:
        if isinstance(neg_limit, float) and neg_limit < 1:
            neg_limit = int(nb_all_neg * neg_limit)
        neg_limit = int(neg_limit)

    # Determine batch size in terms of rows of the distance matrix
    if batch_size is None:
        row_batch_size = n
    else:
        row_batch_size = max(1, int(batch_size) // n)

    if use_torch:
        compute_fn = _embedding_distances_torch
        kwargs = {"torch_dtype": _get_torch_dtype(torch_dtype)}
    else:
        compute_fn = _embedding_distances_numpy
        kwargs = {}

    pos_dists, neg_dists, pos_idx, neg_idx = compute_fn(
        emb_a=emb,
        emb_b=emb,
        labels_a=labels,
        labels_b=labels,
        upper_triag_only=True,
        dist=dist,
        pos_limit=pos_limit,
        neg_limit=neg_limit,
        row_batch_size=row_batch_size,
        return_indices=return_indices,
        **kwargs,
    )

    nb_easy_pos = nb_all_pos - len(pos_dists)
    nb_easy_neg = nb_all_neg - len(neg_dists)

    scores = Scores(
        pos_dists,
        neg_dists,
        nb_easy_pos=nb_easy_pos,
        nb_easy_neg=nb_easy_neg,
        score_class="neg",  # For distances, lower is better.
        equal_class="neg",
    )

    if return_indices:
        # Sort indices to match the sorted order of distances in Scores
        pos_order = np.argsort(pos_dists)
        pos_idx = pos_idx[pos_order]
        neg_order = np.argsort(neg_dists)
        neg_idx = neg_idx[neg_order]
        return scores, pos_idx, neg_idx
    else:
        return scores


def cross_embedding_distances(
    emb_a: np.ndarray,
    emb_b: np.ndarray,
    labels_a: np.ndarray,
    labels_b: np.ndarray,
    dist: str = "l2_squared",
    pos_limit: int | float | None = None,
    neg_limit: int | float | None = None,
    batch_size: int | None = 1e8,
    return_indices: bool = False,
    use_torch: bool = False,
    torch_dtype: "torch.dtype | str" = "float32",
) -> Scores | tuple[Scores, np.ndarray, np.ndarray]:
    """Compute pairwise distances between two set of embeddings, split into positive and
    negative pairs based on labels.

    For each pair of embeddings (i, j), compute the distance and classify it as positive
    (same label) or negative (different label). The resulting distances are returned as
    a ``Scores`` object.

    When ``pos_limit`` or ``neg_limit`` are specified, the number of positive or
    negative pairs is limited. If the limit is an integer >= 1, it specifies the maximum
    number of pairs. If the limit is a float < 1, it specifies the fraction of pairs to
    keep. The actual number of pairs before limiting is stored in ``scores.nb_all_pos``
    and ``scores.nb_all_neg``.

    Args:
        emb_a: Array of shape ``(N, D)`` containing ``N`` embeddings of dim ``D``.
        emb_b: Array of shape ``(M, D)`` containing ``M`` embeddings of dim ``D``.
        labels_a: Array of shape ``(N,)`` containing the label for ``emb_a``.
        labels_b: Array of shape ``(M,)`` containing the label for ``emb_b``.
        dist: Distance metric to use. One of ``"l2_squared"``, ``"l2"`` or ``"cosine"``.
        pos_limit: Maximum number (or fraction) of positive pairs to return.
        neg_limit: Maximum number (or fraction) of negative pairs to return.
        batch_size: Process embeddings in batches to limit memory usage. If ``None``,
            all pairs are computed at once.
        return_indices: If True, also return the embedding indices for each pair.
            Only supported with ``use_torch=False``.
        use_torch: Whether to use PyTorch for distance computation.
        torch_dtype: Data type to use for PyTorch computations. We can use "bfloat16"
            on supported GPUs for faster computation and reduced memory usage.

    Returns:
        A ``Scores`` object with positive and negative distance arrays. If
        ``return_indices`` is True, returns a tuple ``(scores, pos_idx, neg_idx)``
        where ``pos_idx`` and ``neg_idx`` are arrays of shape ``(K, 2)`` containing
        the embedding indices for each pair, sorted by distance.
    """
    emb_a = np.asarray(emb_a)
    emb_b = np.asarray(emb_b)
    if np.issubdtype(emb_a.dtype, np.integer):
        emb_a = emb_a.astype(np.float32)
    if np.issubdtype(emb_b.dtype, np.integer):
        emb_b = emb_b.astype(np.float32)
    labels_a = np.asarray(labels_a)
    labels_b = np.asarray(labels_b)
    n = len(emb_a)
    m = len(emb_b)

    # Count total pos/neg pairs from labels alone (no distance computation needed)
    unique_a, counts_a = np.unique(labels_a, return_counts=True)
    unique_b, counts_b = np.unique(labels_b, return_counts=True)
    _, idx_a, idx_b = np.intersect1d(unique_a, unique_b, return_indices=True)
    nb_all_pos = np.dot(counts_a[idx_a], counts_b[idx_b])
    nb_all_neg = n * m - nb_all_pos

    # Convert fraction limits to absolute counts
    if pos_limit is not None:
        if isinstance(pos_limit, float) and pos_limit < 1:
            pos_limit = int(nb_all_pos * pos_limit)
        pos_limit = int(pos_limit)

    if neg_limit is not None:
        if isinstance(neg_limit, float) and neg_limit < 1:
            neg_limit = int(nb_all_neg * neg_limit)
        neg_limit = int(neg_limit)

    # Determine batch size in terms of rows of the distance matrix
    if batch_size is None:
        row_batch_size = n
    else:
        row_batch_size = max(1, int(batch_size) // m)

    if use_torch:
        compute_fn = _embedding_distances_torch
        kwargs = {"torch_dtype": _get_torch_dtype(torch_dtype)}
    else:
        compute_fn = _embedding_distances_numpy
        kwargs = {}

    pos_dists, neg_dists, pos_idx, neg_idx = compute_fn(
        emb_a=emb_a,
        emb_b=emb_b,
        labels_a=labels_a,
        labels_b=labels_b,
        dist=dist,
        upper_triag_only=False,
        pos_limit=pos_limit,
        neg_limit=neg_limit,
        row_batch_size=row_batch_size,
        return_indices=return_indices,
        **kwargs,
    )

    nb_easy_pos = nb_all_pos - len(pos_dists)
    nb_easy_neg = nb_all_neg - len(neg_dists)

    scores = Scores(
        pos_dists,
        neg_dists,
        nb_easy_pos=nb_easy_pos,
        nb_easy_neg=nb_easy_neg,
        score_class="neg",  # For distances, lower is better.
        equal_class="neg",
    )

    if return_indices:
        # Sort indices to match the sorted order of distances in Scores
        pos_order = np.argsort(pos_dists)
        pos_idx = pos_idx[pos_order]
        neg_order = np.argsort(neg_dists)
        neg_idx = neg_idx[neg_order]
        return scores, pos_idx, neg_idx
    else:
        return scores


def _embedding_distances_numpy(
    emb_a: np.ndarray,
    emb_b: np.ndarray,
    labels_a: np.ndarray,
    labels_b: np.ndarray,
    dist: str,
    upper_triag_only: bool,
    pos_limit: int | None,
    neg_limit: int | None,
    row_batch_size: int,
    return_indices: bool,
):
    dist_fn_dict = {
        "l2_squared": l2_squared_matrix,
        "l2": l2_matrix,
        "cosine": cosine_matrix,
    }
    if dist not in dist_fn_dict:
        raise ValueError(f"Unknown distance: {dist}")
    dist_fn = dist_fn_dict[dist]

    n = len(emb_a)
    m = len(emb_b)

    pos_dists = np.array([], dtype=emb_a.dtype)
    neg_dists = np.array([], dtype=emb_a.dtype)
    pos_idx = np.empty((0, 2), dtype=np.intp)
    neg_idx = np.empty((0, 2), dtype=np.intp)

    for start in range(0, n, row_batch_size):
        end = min(start + row_batch_size, n)
        emb_col = emb_b[start:] if upper_triag_only else emb_b
        batch_dists = dist_fn(emb_a[start:end], emb_col)  # (end-start, M)

        row_indices = np.arange(start, end)[:, None]  # (B, 1)
        if upper_triag_only:
            # Mask for upper triangle: only pairs (i, j) with i < j
            col_indices = np.arange(start, m)[None, :]  # (1, M)
            upper_mask = row_indices < col_indices  # (B, M)
        else:
            col_indices = np.arange(0, m)[None, :]
            upper_mask = True  # Default mask

        # Label match matrix for this batch
        labels_col = labels_b[None, start:] if upper_triag_only else labels_b[None, :]
        same_label = labels_a[start:end, None] == labels_col  # (B, M)

        pos_mask = upper_mask & same_label
        neg_mask = upper_mask & ~same_label
        pos_batch = batch_dists[pos_mask]
        neg_batch = batch_dists[neg_mask]

        if return_indices:
            # Extract (row, col) index pairs for matched entries
            rows = np.broadcast_to(row_indices, batch_dists.shape)
            cols = np.broadcast_to(col_indices, batch_dists.shape)
            pos_idx_batch = np.column_stack([rows[pos_mask], cols[pos_mask]])
            neg_idx_batch = np.column_stack([rows[neg_mask], cols[neg_mask]])

        # Merge with running buffer and trim to keep only the hardest pairs
        if len(pos_batch) > 0:
            pos_dists = np.concatenate([pos_dists, pos_batch])
            if return_indices:
                pos_idx = np.concatenate([pos_idx, pos_idx_batch])
            if pos_limit is not None and len(pos_dists) > pos_limit:
                # Hardest positive pairs have the largest distances
                keep = np.argpartition(pos_dists, -pos_limit)[-pos_limit:]
                pos_dists = pos_dists[keep]
                if return_indices:
                    pos_idx = pos_idx[keep]

        if len(neg_batch) > 0:
            neg_dists = np.concatenate([neg_dists, neg_batch])
            if return_indices:
                neg_idx = np.concatenate([neg_idx, neg_idx_batch])
            if neg_limit is not None and len(neg_dists) > neg_limit:
                # Hardest negative pairs have the smallest distances
                keep = np.argpartition(neg_dists, neg_limit)[:neg_limit]
                neg_dists = neg_dists[keep]
                if return_indices:
                    neg_idx = neg_idx[keep]

    return pos_dists, neg_dists, pos_idx, neg_idx


def _get_torch_device():  # pragma: no cover
    """Return the best available torch device for accelerated computing.

    Checks for CUDA, then MPS (Apple Silicon), and falls back to CPU.

    Returns:
        torch.device: The best available device.
    """
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _get_torch_dtype(dtype: "torch.dtype | str | None") -> "torch.dtype | None":
    """Function to convert string to torch.type. Useful for config parsing."""
    import torch

    if dtype is None:
        return None
    if isinstance(dtype, torch.dtype):
        return dtype

    if not isinstance(dtype, str):
        raise TypeError(f"dtype must be torch.dtype or str, but got {type(dtype)}")
    dtype = getattr(torch, dtype)
    assert isinstance(dtype, torch.dtype)
    return dtype


def _embedding_distances_torch(
    emb_a: np.ndarray,
    emb_b: np.ndarray,
    labels_a: np.ndarray,
    labels_b: np.ndarray,
    dist: str,
    upper_triag_only: bool,
    pos_limit: int | None,
    neg_limit: int | None,
    row_batch_size: int,
    torch_dtype: "torch.dtype",
    return_indices: bool,
):
    import torch

    def _l2_squared_matrix(x, y, x_n, y_n):
        d = x_n + y_n.mT - 2.0 * x @ y.mT  # (N, M)
        return d

    def _l2_matrix(x, y, x_n, y_n):
        return torch.cdist(x, y, p=2)

    def _cosine_matrix(x, y, x_n, y_n):
        # Note that for normalized vectors, after the post-processing step, we have
        #     _l2_squared(x, y) = 2 * _cosine(x, y)
        return -x @ y.mT  # We add 1 at the post-processing stage

    dist_fn_dict = {
        "l2_squared": _l2_squared_matrix,
        "l2": _l2_matrix,
        "cosine": _cosine_matrix,
    }
    if dist not in dist_fn_dict:
        raise ValueError(f"Unknown distance: {dist}")
    dist_fn = dist_fn_dict[dist]

    input_dtype = emb_a.dtype
    device = _get_torch_device()

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="The given NumPy array is not writable"
        )
        emb_a = torch.as_tensor(emb_a, dtype=torch_dtype, device=device)
        emb_b = torch.as_tensor(emb_b, dtype=torch_dtype, device=device)
    if dist == "cosine":
        emb_a_norm = torch.norm(emb_a, dim=-1, keepdim=True)
        emb_b_norm = torch.norm(emb_b, dim=-1, keepdim=True)
        emb_a = emb_a / torch.clamp(emb_a_norm, min=1e-10)
        emb_b = emb_b / torch.clamp(emb_b_norm, min=1e-10)
    else:
        # Squared norm for l2 distances
        emb_a_norm = (emb_a**2).sum(dim=-1, keepdim=True)  # (N, 1)
        emb_b_norm = (emb_b**2).sum(dim=-1, keepdim=True)  # (N, 1)
    labels_a = torch.as_tensor(labels_a, device=device)
    labels_b = torch.as_tensor(labels_b, device=device)

    n = len(emb_a)
    m = len(emb_b)

    pos_dists = torch.tensor([], dtype=torch_dtype, device=device)
    neg_dists = torch.tensor([], dtype=torch_dtype, device=device)
    pos_idx = torch.empty((0, 2), dtype=torch.long, device=device)
    neg_idx = torch.empty((0, 2), dtype=torch.long, device=device)

    with torch.inference_mode():
        for start in range(0, n, row_batch_size):
            end = min(start + row_batch_size, n)

            # If we want to compute the upper triangle only, then compute pairwise
            # distances only against emb[start:].
            emb_col = emb_b[start:] if upper_triag_only else emb_b
            emb_col_norm = emb_b_norm[start:] if upper_triag_only else emb_b_norm

            batch_dists = dist_fn(
                emb_a[start:end], emb_col, emb_a_norm[start:end], emb_col_norm
            )

            row_indices = torch.arange(start, end, device=device).unsqueeze(1)
            if upper_triag_only:
                # Mask for upper triangle: only pairs (i, j) with i < j
                col_indices = torch.arange(start, m, device=device).unsqueeze(0)
                upper_mask = row_indices < col_indices
            else:
                col_indices = torch.arange(0, m, device=device).unsqueeze(0)
                upper_mask = True

            # Label match matrix for this batch
            labels_col = labels_b[start:] if upper_triag_only else labels_b
            same_label = labels_a[start:end].unsqueeze(1) == labels_col.unsqueeze(0)

            pos_mask = upper_mask & same_label
            neg_mask = upper_mask & ~same_label
            pos_batch = batch_dists[pos_mask]
            neg_batch = batch_dists[neg_mask]

            if return_indices:
                rows = row_indices.expand_as(batch_dists)
                cols = col_indices.expand_as(batch_dists)
                pos_idx_batch = torch.stack([rows[pos_mask], cols[pos_mask]], dim=1)
                neg_idx_batch = torch.stack([rows[neg_mask], cols[neg_mask]], dim=1)

            # Merge with running buffer and trim to keep only the hardest pairs
            if len(pos_batch) > 0:
                pos_dists = torch.cat([pos_dists, pos_batch])
                if return_indices:
                    pos_idx = torch.cat([pos_idx, pos_idx_batch])
                if pos_limit is not None and len(pos_dists) > pos_limit:
                    # Hardest positive pairs have the largest distances
                    pos_dists, keep = torch.topk(pos_dists, pos_limit)
                    if return_indices:
                        pos_idx = pos_idx[keep]

            if len(neg_batch) > 0:
                neg_dists = torch.cat([neg_dists, neg_batch])
                if return_indices:
                    neg_idx = torch.cat([neg_idx, neg_idx_batch])
                if neg_limit is not None and len(neg_dists) > neg_limit:
                    # Hardest negative pairs have the smallest distances
                    neg_dists, keep = torch.topk(neg_dists, neg_limit, largest=False)
                    if return_indices:
                        neg_idx = neg_idx[keep]

    # Convert to numpy at the very end
    pos_dists = pos_dists.cpu().float().numpy().astype(input_dtype)
    neg_dists = neg_dists.cpu().float().numpy().astype(input_dtype)
    pos_idx = pos_idx.cpu().numpy().astype(np.intp)
    neg_idx = neg_idx.cpu().numpy().astype(np.intp)

    # Post-processing for cosine distance
    if dist == "cosine":
        pos_dists += 1
        neg_dists += 1

    return pos_dists, neg_dists, pos_idx, neg_idx


def probe_gallery_distances(
    probe: np.ndarray,
    gallery: np.ndarray,
    probe_labels: np.ndarray,
    gallery_labels: np.ndarray,
    dist: str = "l2_squared",
    return_indices: bool = False,
    return_topk: bool = False,
    rank: int | None = None,
    batch_size: int | None = 1e8,
    use_torch: bool = False,
    torch_dtype: "torch.dtype | str" = "float32",
) -> OneToNScores:
    """Compute distances from probe embeddings to gallery embeddings and return
    one-to-N matching scores.

    For each probe, computes distances to all gallery embeddings and extracts:

    - For non-mated probes (probe label not in gallery): the distance to the
      closest gallery embedding.
    - For mated probes (probe label in gallery): the distance to the closest
      gallery embedding (rank-1), the distance to the closest same-label
      gallery embedding, the rank of that mated embedding, and the label rank
      (number of unique labels with smaller distances).

    Args:
        probe: Array of shape ``(P, D)`` containing probe embeddings.
        gallery: Array of shape ``(G, D)`` containing gallery embeddings.
        probe_labels: Array of shape ``(P,)`` containing the label for each
            probe embedding.
        gallery_labels: Array of shape ``(G,)`` containing the label for each
            gallery embedding.
        dist: Distance metric. One of ``"l2_squared"``, ``"l2"`` or ``"cosine"``.
        batch_size: Maximum number of distances to keep in memory at once.
            Controls the probe batch size as ``batch_size // G``. If ``None``,
            all distances are computed at once.
        use_torch: Whether to use PyTorch for distance computation.
        torch_dtype: Data type for PyTorch computations. We can use "bfloat16"
            on supported GPUs for faster computation and reduced memory usage.

    Returns:
        A ``OneToNScores`` object with the matching scores.
    """
    probe = np.asarray(probe)
    if np.issubdtype(probe.dtype, np.integer):
        probe = probe.astype(np.float32)
    gallery = np.asarray(gallery)
    if np.issubdtype(gallery.dtype, np.integer):
        gallery = gallery.astype(np.float32)
    probe_labels = np.asarray(probe_labels)
    gallery_labels = np.asarray(gallery_labels)

    p = len(probe)
    g = len(gallery)

    if batch_size is None:
        row_batch_size = p
    else:
        row_batch_size = max(1, int(batch_size) // g)

    is_mated = np.isin(probe_labels, gallery_labels)

    return _probe_gallery_distances_numpy(
        probe=probe,
        gallery=gallery,
        probe_labels=probe_labels,
        gallery_labels=gallery_labels,
        is_mated=is_mated,
        dist=dist,
        row_batch_size=row_batch_size,
    )


def _probe_gallery_distances_numpy(
    probe: np.ndarray,
    gallery: np.ndarray,
    probe_labels: np.ndarray,
    gallery_labels: np.ndarray,
    is_mated: np.ndarray,
    dist: str,
    row_batch_size: int,
) -> OneToNScores:
    dist_fn_dict = {
        "l2_squared": l2_squared_matrix,
        "l2": l2_matrix,
        "cosine": cosine_matrix,
    }
    if dist not in dist_fn_dict:
        raise ValueError(f"Unknown distance: {dist}")
    dist_fn = dist_fn_dict[dist]

    p = len(probe)

    # Sort gallery by label so we can use reduceat for per-label min distances
    gallery_sort = np.argsort(gallery_labels)
    gallery = gallery[gallery_sort]
    gallery_labels = gallery_labels[gallery_sort]
    unique_labels, label_starts = np.unique(gallery_labels, return_index=True)

    # Pre-allocate output arrays
    nb_neg = int(np.sum(~is_mated))
    nb_pos = int(np.sum(is_mated))

    neg_rank1 = np.empty(nb_neg, dtype=probe.dtype)
    pos_rank1 = np.empty(nb_pos, dtype=probe.dtype)
    pos_mate = np.empty(nb_pos, dtype=probe.dtype)
    pos_mate_rank = np.empty(nb_pos, dtype=np.intp)
    pos_label_rank = np.empty(nb_pos, dtype=np.intp)
    neg_labels = np.empty(nb_neg, dtype=probe_labels.dtype)
    pos_labels = np.empty(nb_pos, dtype=probe_labels.dtype)

    neg_offset = 0  # Next index to write in neg_rank1/neg_labels
    pos_offset = 0  # Same for pos_... arrays

    for start in range(0, p, row_batch_size):
        end = min(start + row_batch_size, p)
        batch_dists = dist_fn(probe[start:end], gallery)  # (batch, G)
        batch_mated = is_mated[start:end]
        batch_labels = probe_labels[start:end]

        # Non-mated probes: closest gallery distance
        neg_mask = ~batch_mated
        nb_neg_batch = int(neg_mask.sum())
        if nb_neg_batch > 0:
            sl = slice(neg_offset, neg_offset + nb_neg_batch)
            neg_rank1[sl] = batch_dists[neg_mask].min(axis=1)
            neg_labels[sl] = batch_labels[neg_mask]
            neg_offset += nb_neg_batch

        # Mated probes
        nb_pos_batch = int(batch_mated.sum())
        if nb_pos_batch > 0:
            mated_dists = batch_dists[batch_mated]  # (Pm, G)
            mated_probe_labels = batch_labels[batch_mated]  # (Pm,)
            sl = slice(pos_offset, pos_offset + nb_pos_batch)

            # Rank-1: closest gallery item overall
            pos_rank1[sl] = mated_dists.min(axis=1)
            pos_labels[sl] = mated_probe_labels

            # Per-label min distances via reduceat (gallery is label-sorted)
            min_per_label = np.minimum.reduceat(
                mated_dists, label_starts, axis=1
            )  # (Pm, n_unique_labels)

            # Mate distance: look up each probe's own label in min_per_label
            label_idx = np.searchsorted(unique_labels, mated_probe_labels)
            batch_mate = min_per_label[np.arange(nb_pos_batch), label_idx]
            pos_mate[sl] = batch_mate

            # Mate rank: number of gallery items with distance < mate_dist, + 1
            pos_mate_rank[sl] = np.sum(mated_dists < batch_mate[:, None], axis=1) + 1

            # Label rank: number of unique labels with min distance < mate_dist, + 1
            pos_label_rank[sl] = np.sum(min_per_label < batch_mate[:, None], axis=1) + 1

            pos_offset += nb_pos_batch

    return OneToNScores(
        neg_rank1=neg_rank1,
        pos_rank1=pos_rank1,
        pos_mate=pos_mate,
        pos_mate_rank=pos_mate_rank,
        pos_label_rank=pos_label_rank,
        neg_labels=neg_labels,
        pos_labels=pos_labels,
        score_class="neg",
        equal_class="pos",
    )


def _probe_gallery_distances_torch(
    probes: np.ndarray,
    gallery: np.ndarray,
    dist: str,
    rank: int,
    row_batch_size: int,
    return_indices: bool,
    torch_dtype: "torch.dtype",
) -> tuple[np.ndarray, np.ndarray | None]:
    import torch

    def _l2_squared_matrix(x, y, x_n, y_n):
        return x_n + y_n.mT - 2.0 * x @ y.mT

    def _l2_matrix(x, y, x_n, y_n):
        return torch.cdist(x, y, p=2)

    def _cosine_matrix(x, y, x_n, y_n):
        return -x @ y.mT  # We add 1 at the post-processing stage

    dist_fn_dict = {
        "l2_squared": _l2_squared_matrix,
        "l2": _l2_matrix,
        "cosine": _cosine_matrix,
    }
    if dist not in dist_fn_dict:
        raise ValueError(f"Unknown distance: {dist}")
    dist_fn = dist_fn_dict[dist]

    input_dtype = probes.dtype
    device = _get_torch_device()

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="The given NumPy array is not writable"
        )
        probes = torch.as_tensor(probes, dtype=torch_dtype, device=device)
        gallery = torch.as_tensor(gallery, dtype=torch_dtype, device=device)

    if dist == "cosine":
        probe_norm = torch.norm(probes, dim=-1, keepdim=True)
        probes = probes / torch.clamp(probe_norm, min=1e-10)
        gallery_norm = torch.norm(gallery, dim=-1, keepdim=True)
        gallery = gallery / torch.clamp(gallery_norm, min=1e-10)

    # Squared norms for l2_squared; harmless for other metrics
    probe_norms = (probes**2).sum(dim=-1, keepdim=True)
    gallery_norms = (gallery**2).sum(dim=-1, keepdim=True)

    p = len(probes)
    g = len(gallery)
    result = torch.empty((p, rank), dtype=torch_dtype, device=device)
    if return_indices:
        result_idx = torch.empty((p, rank), dtype=torch.long, device=device)
    else:
        result_idx = None

    with torch.inference_mode():
        for start in range(0, p, row_batch_size):
            end = min(start + row_batch_size, p)
            batch_dists = dist_fn(
                probes[start:end], gallery, probe_norms[start:end], gallery_norms
            )

            if rank < g:
                topk = torch.topk(batch_dists, rank, dim=1, largest=False, sorted=True)
                result[start:end] = topk.values
                if return_indices:
                    result_idx[start:end] = topk.indices
            else:
                sorted_result = torch.sort(batch_dists, dim=1)
                result[start:end] = sorted_result.values
                if return_indices:
                    result_idx[start:end] = sorted_result.indices

    result = result.cpu().float().numpy().astype(input_dtype)
    if return_indices:
        result_idx = result_idx.cpu().numpy().astype(np.intp)

    if dist == "cosine":
        result += 1

    return result, result_idx
