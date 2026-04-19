"""
Functions to compute distances and similarities between embeddings.
"""

from typing import TYPE_CHECKING

import numpy as np

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
        emb=emb,
        labels=labels,
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
        pos_dists, neg_dists, nb_easy_pos=nb_easy_pos, nb_easy_neg=nb_easy_neg
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
    emb: np.ndarray,
    labels: np.ndarray,
    dist: str,
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

    n = len(emb)

    pos_dists = np.array([], dtype=emb.dtype)
    neg_dists = np.array([], dtype=emb.dtype)
    pos_idx = np.empty((0, 2), dtype=np.intp)
    neg_idx = np.empty((0, 2), dtype=np.intp)

    for start in range(0, n, row_batch_size):
        end = min(start + row_batch_size, n)
        batch_dists = dist_fn(emb[start:end], emb[start:])  # (end-start, N)

        # Mask for upper triangle: only pairs (i, j) with i < j
        row_indices = np.arange(start, end)[:, None]  # (B, 1)
        col_indices = np.arange(start, n)[None, :]  # (1, N)
        upper_mask = row_indices < col_indices  # (B, N)

        # Label match matrix for this batch
        same_label = labels[start:end, None] == labels[None, start:]  # (B, N)

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
    emb: np.ndarray,
    labels: np.ndarray,
    dist: str,
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

    input_dtype = emb.dtype
    device = _get_torch_device()

    emb = torch.as_tensor(emb, dtype=torch_dtype, device=device)
    if dist == "cosine":
        emb_norm = torch.norm(emb, dim=-1, keepdim=True)
        emb = emb / torch.clamp(emb_norm, min=1e-10)
    else:
        # Squared norm for l2 distances
        emb_norm = (emb**2).sum(dim=-1, keepdim=True)  # (N, 1)
    labels = torch.as_tensor(labels, device=device)
    n = len(emb)

    pos_dists = torch.tensor([], dtype=torch_dtype, device=device)
    neg_dists = torch.tensor([], dtype=torch_dtype, device=device)
    pos_idx = torch.empty((0, 2), dtype=torch.long, device=device)
    neg_idx = torch.empty((0, 2), dtype=torch.long, device=device)

    with torch.inference_mode():
        for start in range(0, n, row_batch_size):
            end = min(start + row_batch_size, n)

            # Compute pairwise distances only against emb[start:] since columns before
            # start are masked out by the upper triangle anyway.
            batch_dists = dist_fn(
                emb[start:end], emb[start:], emb_norm[start:end], emb_norm[start:]
            )

            # Mask for upper triangle: only pairs (i, j) with i < j
            row_indices = torch.arange(start, end, device=device).unsqueeze(1)
            col_indices = torch.arange(start, n, device=device).unsqueeze(0)
            upper_mask = row_indices < col_indices

            # Label match matrix for this batch
            same_label = labels[start:end].unsqueeze(1) == labels[start:].unsqueeze(0)

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
    probes: np.ndarray,
    gallery: np.ndarray,
    dist: str = "l2_squared",
    rank: int | None = None,
    batch_size: int | None = 1e8,
    return_indices: bool = False,
    use_torch: bool = False,
    torch_dtype: "torch.dtype | str" = "float32",
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Compute distances from probe embeddings to the closest gallery embeddings.

    For each probe, computes distances to all gallery embeddings and returns the
    ``rank`` smallest distances, sorted in ascending order.

    Args:
        probes: Array of shape ``(P, D)`` containing probe embeddings.
        gallery: Array of shape ``(G, D)`` containing gallery embeddings.
        dist: Distance metric. One of ``"l2_squared"``, ``"l2"`` or ``"cosine"``.
        rank: Number of closest gallery items to return per probe. If ``None``,
            returns distances to all gallery items (sorted).
        batch_size: Maximum number of distances to keep in memory at once.
            Controls the probe batch size as ``batch_size // G``. If ``None``,
            all distances are computed at once.
        return_indices: If True, also return the gallery indices for the closest
            items.
        use_torch: Whether to use PyTorch for distance computation.
        torch_dtype: Data type for PyTorch computations. We can use "bfloat16"
            on supported GPUs for faster computation and reduced memory usage.

    Returns:
        Array of shape ``(P, rank)`` with sorted distances per probe. If
        ``return_indices`` is True, returns a tuple ``(distances, indices)``
        where ``indices`` has shape ``(P, rank)`` containing gallery indices.
    """
    probes = np.asarray(probes)
    if np.issubdtype(probes.dtype, np.integer):
        probes = probes.astype(np.float32)
    gallery = np.asarray(gallery)
    if np.issubdtype(gallery.dtype, np.integer):
        gallery = gallery.astype(np.float32)

    p = len(probes)
    g = len(gallery)

    if rank is None:
        rank = g
    rank = min(rank, g)

    if batch_size is None:
        row_batch_size = p
    else:
        row_batch_size = max(1, int(batch_size) // g)

    if use_torch:
        compute_fn = _probe_gallery_distances_torch
        kwargs = {"torch_dtype": _get_torch_dtype(torch_dtype)}
    else:
        compute_fn = _probe_gallery_distances_numpy
        kwargs = {}

    result = compute_fn(
        probes=probes,
        gallery=gallery,
        dist=dist,
        rank=rank,
        row_batch_size=row_batch_size,
        return_indices=return_indices,
        **kwargs,
    )
    return result if return_indices else result[0]


def _probe_gallery_distances_numpy(
    probes: np.ndarray,
    gallery: np.ndarray,
    dist: str,
    rank: int,
    row_batch_size: int,
    return_indices: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    dist_fn_dict = {
        "l2_squared": l2_squared_matrix,
        "l2": l2_matrix,
        "cosine": cosine_matrix,
    }
    if dist not in dist_fn_dict:
        raise ValueError(f"Unknown distance: {dist}")
    dist_fn = dist_fn_dict[dist]

    p = len(probes)
    g = len(gallery)
    result = np.empty((p, rank), dtype=probes.dtype)
    if return_indices:
        result_idx = np.empty((p, rank), dtype=np.intp)
    else:
        result_idx = None

    for start in range(0, p, row_batch_size):
        end = min(start + row_batch_size, p)
        batch_dists = dist_fn(probes[start:end], gallery)  # (batch, G)

        if rank < g:
            # Use argpartition for O(G) selection of top-rank elements
            top_idx = np.argpartition(batch_dists, rank, axis=1)[:, :rank]
            top_dists = np.take_along_axis(batch_dists, top_idx, axis=1)
        else:
            top_idx = np.broadcast_to(np.arange(g), batch_dists.shape).copy()
            top_dists = batch_dists

        # Sort the selected elements
        sort_order = np.argsort(top_dists, axis=1)
        result[start:end] = np.take_along_axis(top_dists, sort_order, axis=1)
        if return_indices:
            result_idx[start:end] = np.take_along_axis(top_idx, sort_order, axis=1)

    return result, result_idx


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
