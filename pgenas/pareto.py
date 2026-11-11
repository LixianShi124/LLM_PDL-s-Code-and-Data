import math

def objectives(item, budget=None):
    error, params, flops = (1 - float(item['accuracy']), int(item['params']), int(item['flops']))
    if not math.isfinite(error) or not 0 <= error <= 1 or params < 1 or (flops < 1):
        raise ValueError('Invalid candidate accuracy/params/FLOPs')
    if budget is None:
        return [error, flops]
    return [error, flops / budget['max_flops']]

def parameter_limit(budget):
    return min(budget['max_params'], budget.get('deploy_params', budget['max_params']))

def dominates(a, b):
    x, y = (objectives(a), objectives(b))
    return all((i <= j for i, j in zip(x, y))) and any((i < j for i, j in zip(x, y)))

def non_dominated(items):
    unique = {r['id']: r for r in items}
    rows = list(unique.values())
    return [a for a in rows if not any((dominates(b, a) for b in rows if b['id'] != a['id']))]

def crowding_distance(items):
    distances = {r['id']: 0.0 for r in items}
    if len(items) <= 2:
        return {r['id']: math.inf for r in items}
    for axis in range(2):
        ordered = sorted(items, key=lambda r: (objectives(r)[axis], r['id']))
        low, high = (objectives(ordered[0])[axis], objectives(ordered[-1])[axis])
        if high == low:
            continue
        distances[ordered[0]['id']] = distances[ordered[-1]['id']] = math.inf
        for i in range(1, len(ordered) - 1):
            gap = objectives(ordered[i + 1])[axis] - objectives(ordered[i - 1])[axis]
            distances[ordered[i]['id']] += gap / (high - low)
    return distances

def update_archive(old, new, capacity):
    if capacity < 1:
        raise ValueError('Archive capacity must be positive')
    archive = non_dominated(list(old) + list(new))
    while len(archive) > capacity:
        if capacity == 1:
            archive = [min(archive, key=lambda r: (-r['accuracy'], r['flops'], r['id']))]
            break
        distances = crowding_distance(archive)
        worst = min(archive, key=lambda r: (distances[r['id']], r['accuracy'], -r['flops'], r['id']))
        archive.remove(worst)
    return sorted(archive, key=lambda r: (-r['accuracy'], r['flops'], r['id']))

def select_parent(archive, mode, rng):
    if not archive:
        raise ValueError('Cannot select a parent from an empty archive')
    if mode == 'exploitation':
        return min(archive, key=lambda r: (-r['accuracy'], r['flops'], r['id']))
    if mode != 'exploration':
        raise ValueError('Unknown parent selection mode')
    pair = rng.sample(archive, min(2, len(archive)))
    distances = crowding_distance(archive)
    return max(pair, key=lambda r: distances[r['id']])

def select_final(archive, budget):
    feasible = [r for r in archive if r['params'] <= budget['deploy_params'] and r['flops'] <= budget['deploy_flops']]
    if not feasible:
        raise ValueError('No candidate satisfies the deployment budgets')
    return min(feasible, key=lambda r: (-r['accuracy'], r['flops'], r['id']))
