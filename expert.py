"""Memoryless waypoint teacher using privileged simulator stem geometry.

Only the tool's existing protected-stem envelope is planned here. This is not
whole-arm collision planning or perception. No policy action is overridden.
"""
import numpy as np

TIP_OFFSET = np.array([0.085, 0.004, 0.])
ROUTE_MARGIN = 0.007  # stem radius + 4 mm for tracking/orientation error


def blocked_positions(positions, roots, tops, angle, radius=ROUTE_MARGIN):
    """Vectorized segment/convex-envelope clipping at horizontal tool poses."""
    positions = np.atleast_2d(positions)
    starts = roots[None, :, :] - positions[:, None, :] + TIP_OFFSET
    delta = tops - roots
    slope = np.sin(max(angle, 0))
    normals = np.array([[-1, 0, 0], [1, 0, 0], [0, 0, -1],
                        [0, 0, 1], [0, -1, 0], [-slope, 1, 0]])
    bounds = np.array([-0.02, 0.155, 0.018, 0.018, 0.01, 0.01]) + radius
    offset = bounds - starts @ normals.T
    rate = delta @ normals.T
    parallel = np.abs(rate) < 1e-12
    ratio = offset / np.where(parallel, 1., rate)
    low = np.maximum(0., np.max(np.where(rate < -1e-12, ratio, -np.inf), axis=-1))
    high = np.minimum(1., np.min(np.where(rate > 1e-12, ratio, np.inf), axis=-1))
    misses_parallel = np.any(parallel & (offset < 0), axis=-1)
    return np.any((low <= high) & ~misses_parallel, axis=1)


def path_clear(path, roots, tops, angle, radius=ROUTE_MARGIN):
    samples = []
    for a, b in zip(path[:-1], path[1:]):
        # 2 mm sampling plus the 4 mm extra margin covers translation between samples.
        n = max(2, int(np.ceil(np.linalg.norm(b - a) / 0.002)) + 1)
        samples.append(np.linspace(a, b, n))
    return not blocked_positions(np.concatenate(samples), roots, tops, angle, radius).any()


def waypoint_action(env):
    from environment import CUT_ANGLE, MAX_DELTA, OPEN_ANGLE

    tip = env.tip
    protected = np.arange(3) != env.target_id
    roots = env.roots[protected]
    tops = env.stems[protected] + env.upper_length * env.stem_direction
    angle = float(env.data.qpos[env.scissor_q])
    cut = env.cut[env.target_id]
    goal = env.home if cut else env.target

    # Align outside the plants before entry and withdraw before changing lanes.
    # Replanning a shortest diagonal every step can graze the safety margin as
    # the arm is still tracking the previous direction at a corner.
    paths = [[tip, goal]]
    for x in (0.38, 0.33, 0.29):
        paths.append([tip, np.array([x, tip[1], tip[2]]),
                      np.array([x, goal[1], goal[2]]), goal])
    # Choose opening AND route together: the leading blade tip sweeps a wider
    # envelope during entry than it occupies at the final cutting pose.
    candidates = []
    for opening in ((0.,) if cut else (1., 0.7, 0.45, 0.34)):
        planned_angle = opening * OPEN_ANGLE
        if blocked_positions(tip, roots, tops, max(angle, planned_angle))[0]:
            continue
        valid = [p for p in paths if path_clear(p, roots, tops, planned_angle,
                 radius=0.015 if len(p) == 2 and np.linalg.norm(goal - tip) > 0.02 else ROUTE_MARGIN)]
        if valid:
            for path in valid:
                length = sum(np.linalg.norm(b - a) for a, b in zip(path[:-1], path[1:]))
                candidates.append((length + 0.002 * (1 - opening), opening, path))
    if not candidates:
        # Closed recovery starts can be too close to a neighbour to reopen.
        # Back out with closed blades, then reconsider opening on the next step.
        escape = np.array([min(0.38, tip[0] - 0.06), tip[1], tip[2]])
        if (not cut and escape[0] >= 0.28
                and path_clear([tip, escape], roots, tops, max(angle, 0.))):
            env.expert_status = "recover_closed"
            delta = (escape - tip) * 0.45 / MAX_DELTA
            delta /= max(1., np.max(np.abs(delta)))
            return np.r_[delta, 0.].astype(np.float32)
        env.expert_status = "blocked_route"
        return np.array([0., 0., 0., np.clip(angle / OPEN_ANGLE, 0, 1)], dtype=np.float32)
    _, opening, path = min(candidates, key=lambda item: item[0])
    if not cut and np.linalg.norm(goal - tip) < 0.0025:
        # A missed closure (or a closed recovery start) needs a fresh closing edge.
        if angle <= CUT_ANGLE:
            env.expert_status = "reopen"
        else:
            opening = 0.
            env.expert_status = "close"
        return np.r_[np.clip((goal - tip) * 0.45 / MAX_DELTA, -1, 1), opening].astype(np.float32)
    if angle > opening * OPEN_ANGLE + 0.01:
        env.expert_status = "narrow_before_move"
        return np.array([0., 0., 0., opening], dtype=np.float32)
    waypoint = next((p for p in path[1:] if np.linalg.norm(p - tip) > 0.003), goal)
    env.expert_status = ("return" if cut else "approach") if len(path) == 2 else ("retreat" if cut else "detour")
    # Correct transverse tracking error independently, then check the actual
    # bounded command segment (which may differ from the long planned line).
    delta = np.clip((waypoint - tip) * 0.45 / MAX_DELTA, -1, 1)
    for scale in (1., 0.5, 0.25, 0.1):
        if path_clear([tip, tip + scale * delta * MAX_DELTA], roots, tops,
                      max(angle, opening * OPEN_ANGLE)):
            return np.r_[scale * delta, opening].astype(np.float32)
    env.expert_status = "blocked_command"
    return np.array([0., 0., 0., np.clip(angle / OPEN_ANGLE, 0, 1)], dtype=np.float32)
