"""Block-local weighted k-means following the public VQVLA OFT exporter."""

import torch


@torch.inference_mode()
def weighted_kmeans(vectors, weights, centers, iterations):
    batches, count, dimension = vectors.shape
    if vectors.shape != weights.shape or dimension != 2 or centers > count or iterations < 1:
        raise ValueError("Invalid vector clustering geometry")
    if not bool(
        torch.isfinite(vectors).all() & torch.isfinite(weights).all() & (weights >= 0).all()
    ):
        raise ValueError("Invalid vector clustering values")
    ids = torch.randint(count, (batches, centers), device=vectors.device)
    means = vectors.gather(1, ids[:, :, None].expand(-1, -1, dimension)).clone()
    assignment = torch.full((batches, count), -1, device=vectors.device, dtype=torch.long)
    weighted = vectors * weights
    norm = (weighted * vectors).sum(-1, keepdim=True)
    for _ in range(iterations):
        distance = (
            norm
            + torch.bmm(weights, means.square().transpose(1, 2))
            - 2 * torch.bmm(weighted, means.transpose(1, 2))
        )
        next_assignment = distance.argmin(-1)
        if torch.equal(assignment, next_assignment):
            assignment = next_assignment
            break
        assignment = next_assignment
        index = assignment[:, :, None].expand(-1, -1, dimension)
        numerator = torch.zeros_like(means).scatter_add_(1, index, weighted)
        denominator = torch.zeros_like(means).scatter_add_(1, index, weights)
        ids = torch.randint(count, (batches, centers), device=vectors.device)
        replacements = vectors.gather(1, ids[:, :, None].expand(-1, -1, dimension))
        means = torch.where(
            (denominator.sum(-1) > 0)[:, :, None],
            numerator / denominator.clamp_min(1e-12),
            replacements,
        )
    # Preserve the official exporter's last-assignment/update ordering.
    return means, assignment


def pack_indices(values, bits):
    if bits not in (6, 8) or values.dtype not in (torch.int64, torch.int32, torch.uint8):
        raise ValueError("Invalid centroid index representation")
    if values.ndim != 2 or not bool(((values >= 0) & (values < (1 << bits))).all()):
        raise ValueError("Invalid centroid indices")
    if bits == 8:
        return values.to(torch.uint8).contiguous()
    if values.shape[1] % 4:
        raise ValueError("Six-bit rows require four-index groups")
    v = values.to(torch.int32).reshape(values.shape[0], -1, 4)
    word = v[:, :, 0] | (v[:, :, 1] << 6) | (v[:, :, 2] << 12) | (v[:, :, 3] << 18)
    return (
        torch.stack([((word >> s) & 255).to(torch.uint8) for s in (0, 8, 16)], -1)
        .flatten(1)
        .contiguous()
    )


def unpack_indices(packed, bits):
    if bits == 8:
        return packed.long()
    if bits != 6 or packed.shape[1] % 3:
        raise ValueError("Invalid packed centroid rows")
    v = packed.int().reshape(packed.shape[0], -1, 3)
    word = v[:, :, 0] | (v[:, :, 1] << 8) | (v[:, :, 2] << 16)
    return torch.stack([(word >> s) & 63 for s in (0, 6, 12, 18)], -1).flatten(1).long()


@torch.inference_mode()
def quantize_weight(weight, hdiag, block, centers, config):
    n, k = weight.shape
    if n % block or k % block or hdiag.shape != (k,):
        raise ValueError("MotionVQ native projection must tile exactly")
    coordinates = [(row, col) for row in range(0, n, block) for col in range(0, k, block)]
    book = []
    indices = torch.empty((n, k // 2), dtype=torch.uint8)
    mean = hdiag.float().mean()
    if not torch.isfinite(mean) or mean <= 0:
        raise ValueError("Missing input-Hessian calibration")
    for start in range(0, len(coordinates), config.block_batch_size):
        batch = coordinates[start : start + config.block_batch_size]
        vectors = torch.stack(
            [weight[r : r + block, c : c + block].float().reshape(-1, 2) for r, c in batch]
        )
        weights = torch.stack(
            [
                ((hdiag[c : c + block].float() + config.hessian_damping * mean) / mean)[None, :]
                .expand(block, -1)
                .reshape(-1, 2)
                for _, c in batch
            ]
        ).to(weight.device)
        centroids, assignments = weighted_kmeans(
            vectors, weights, centers, config.kmeans_iterations
        )
        book.append(centroids.to(dtype=weight.dtype, device="cpu"))
        for (r, c), assignment in zip(batch, assignments):
            indices[r : r + block, c // 2 : (c + block) // 2] = assignment.reshape(
                block, block // 2
            ).to(device="cpu", dtype=torch.uint8)
    bits = 8 if centers == 256 else 6
    return dict(
        codebooks=torch.cat(book),
        indices=pack_indices(indices.T.contiguous(), bits),
        index_bits=bits,
        block_size=block,
        centers=centers,
        shape=(n, k),
        native_dtype=str(weight.dtype),
    )
