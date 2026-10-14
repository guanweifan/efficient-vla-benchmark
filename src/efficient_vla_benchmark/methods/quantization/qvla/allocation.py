"""Official QVLA absolute-cost greedy allocation, not marginal-cost demotion."""

import heapq
import math


def greedy_allocate(proxies, bits, target_avg_bits):
    ordered = sorted(set(bits), reverse=True)
    if not ordered or any(type(b) is not int or b not in (0, 2, 4, 8, 16) for b in ordered):
        raise ValueError("Unsupported candidate bits")
    if not math.isfinite(target_avg_bits) or not ordered[-1] <= target_avg_bits <= ordered[0]:
        raise ValueError("Unreachable bit budget")
    assignments, queue, sequence, total, count = {}, [], 0, 0, 0

    def push(name, row, current):
        nonlocal sequence
        lower = next((b for b in ordered if b < current), None)
        if lower is None or lower not in proxies[name]:
            return
        cost = float(proxies[name][lower][row])
        if not math.isfinite(cost) or cost < 0:
            raise ValueError("Sensitivity costs must be finite and nonnegative")
        heapq.heappush(queue, (cost / (current - lower), sequence, name, row, lower))
        sequence += 1

    # Preserve caller layer insertion order, including the official stable tie-break.
    for name, values in proxies.items():
        size = len(next(iter(values.values())))
        if not size or any(len(v) != size for v in values.values()):
            raise ValueError("Inconsistent channel score arrays")
        assignments[name] = [ordered[0]] * size
        total += ordered[0] * size
        count += size
        for row in range(size):
            push(name, row, ordered[0])
    if not count:
        raise ValueError("Empty allocation")
    while queue and total / count > target_avg_bits:
        _, _, name, row, lower = heapq.heappop(queue)
        previous = assignments[name][row]
        if lower >= previous:
            continue
        assignments[name][row] = lower
        total -= previous - lower
        push(name, row, lower)
    return assignments, {
        "final_avg_bits": total / count,
        "total_channels": count,
        "target_avg_bits": target_avg_bits,
    }
