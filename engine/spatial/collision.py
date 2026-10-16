"""engine/spatial/collision.py
Dual-axis continuous collision resolver with substepping, 2D swept bounds,
and time-of-impact (TOI) ordering to prevent tunneling and preserve smooth wall sliding.
"""
from __future__ import annotations
import math
from typing import List, Tuple, Union
from engine.spatial.aabb import AABB, Vector2


def move_and_resolve_walls(
    entity_aabb: AABB,
    vel: Vector2,
    dt: float,
    walls: List[Union[AABB, Tuple[float, float, float, float]]]
) -> Tuple[Vector2, Vector2]:
    """Moves entity and resolves collisions against solid walls.

    Uses continuous substepping, 2D swept bounding boxes, and time-of-impact (TOI)
    axis ordering to guarantee zero tunneling across high velocities while maintaining
    uninterrupted orthogonal and angled wall sliding.
    Returns (new_pos, actual_vel).
    """
    wall_boxes = [w if isinstance(w, AABB) else AABB(w[0], w[1], w[2], w[3]) for w in walls]
    if dt <= 0.0:
        return Vector2(entity_aabb.x, entity_aabb.y), Vector2(0.0, 0.0)

    # 1. Determine substepping count based on displacement and entity bounds
    max_step = min(entity_aabb.width, entity_aabb.height) * 0.5
    displacement = math.hypot(vel.x * dt, vel.y * dt)
    substeps = max(1, math.ceil(displacement / max_step))
    sub_dt = dt / substeps

    curr_x = entity_aabb.x
    curr_y = entity_aabb.y
    width = entity_aabb.width
    height = entity_aabb.height

    for _ in range(substeps):
        target_x = curr_x + vel.x * sub_dt
        target_y = curr_y + vel.y * sub_dt

        # 2D swept bounding box for this substep
        min_x = min(curr_x, target_x)
        min_y = min(curr_y, target_y)
        span_w = width + abs(vel.x * sub_dt)
        span_h = height + abs(vel.y * sub_dt)
        swept_2d = AABB(min_x, min_y, span_w, span_h)

        colliding_walls = [w for w in wall_boxes if swept_2d.intersects(w)]

        if not colliding_walls:
            curr_x = target_x
            curr_y = target_y
            continue

        new_x = target_x
        new_y = target_y

        # Compute Time of Impact (TOI) along X and Y
        x_hits: List[Tuple[float, float]] = []
        y_hits: List[Tuple[float, float]] = []

        for wall in colliding_walls:
            # X-axis TOI
            if vel.x > 0 and (curr_x + width) <= wall.left:
                tx = (wall.left - (curr_x + width)) / (vel.x * sub_dt)
                if 0.0 <= tx <= 1.0:
                    x_hits.append((tx, wall.left - width))
            elif vel.x < 0 and curr_x >= wall.right:
                tx = (wall.right - curr_x) / (vel.x * sub_dt)
                if 0.0 <= tx <= 1.0:
                    x_hits.append((tx, wall.right))

            # Y-axis TOI
            if vel.y > 0 and (curr_y + height) <= wall.top:
                ty = (wall.top - (curr_y + height)) / (vel.y * sub_dt)
                if 0.0 <= ty <= 1.0:
                    y_hits.append((ty, wall.top - height))
            elif vel.y < 0 and curr_y >= wall.bottom:
                ty = (wall.bottom - curr_y) / (vel.y * sub_dt)
                if 0.0 <= ty <= 1.0:
                    y_hits.append((ty, wall.bottom))

        # Resolve collisions based on earliest TOI
        if x_hits and y_hits:
            earliest_tx = min(h[0] for h in x_hits)
            earliest_ty = min(h[0] for h in y_hits)
            if abs(earliest_tx - earliest_ty) < 1e-5:
                # Simultaneous corner impact: clamp both axes to corner boundary
                matching_x = [h[1] for h in x_hits if abs(h[0] - earliest_tx) < 1e-5]
                matching_y = [h[1] for h in y_hits if abs(h[0] - earliest_ty) < 1e-5]
                new_x = min(matching_x) if vel.x > 0 else max(matching_x)
                new_y = min(matching_y) if vel.y > 0 else max(matching_y)
            elif earliest_tx < earliest_ty:
                # X collides first: clamp X, then check Y sliding at new X
                matching_x = [h[1] for h in x_hits if abs(h[0] - earliest_tx) < 1e-5]
                new_x = min(matching_x) if vel.x > 0 else max(matching_x)
                box_at_new_x = AABB(new_x, min_y, width, span_h)
                for wall in colliding_walls:
                    if box_at_new_x.intersects(wall):
                        if vel.y > 0 and (curr_y + height) <= wall.top:
                            new_y = min(new_y, wall.top - height)
                        elif vel.y < 0 and curr_y >= wall.bottom:
                            new_y = max(new_y, wall.bottom)
            else:
                # Y collides first: clamp Y, then check X sliding at new Y
                matching_y = [h[1] for h in y_hits if abs(h[0] - earliest_ty) < 1e-5]
                new_y = min(matching_y) if vel.y > 0 else max(matching_y)
                box_at_new_y = AABB(min_x, new_y, span_w, height)
                for wall in colliding_walls:
                    if box_at_new_y.intersects(wall):
                        if vel.x > 0 and (curr_x + width) <= wall.left:
                            new_x = min(new_x, wall.left - width)
                        elif vel.x < 0 and curr_x >= wall.right:
                            new_x = max(new_x, wall.right)
        elif x_hits:
            earliest_tx = min(h[0] for h in x_hits)
            matching_x = [h[1] for h in x_hits if abs(h[0] - earliest_tx) < 1e-5]
            new_x = min(matching_x) if vel.x > 0 else max(matching_x)
            box_at_new_x = AABB(new_x, min_y, width, span_h)
            for wall in colliding_walls:
                if box_at_new_x.intersects(wall):
                    if vel.y > 0 and (curr_y + height) <= wall.top:
                        new_y = min(new_y, wall.top - height)
                    elif vel.y < 0 and curr_y >= wall.bottom:
                        new_y = max(new_y, wall.bottom)
        elif y_hits:
            earliest_ty = min(h[0] for h in y_hits)
            matching_y = [h[1] for h in y_hits if abs(h[0] - earliest_ty) < 1e-5]
            new_y = min(matching_y) if vel.y > 0 else max(matching_y)
            box_at_new_y = AABB(min_x, new_y, span_w, height)
            for wall in colliding_walls:
                if box_at_new_y.intersects(wall):
                    if vel.x > 0 and (curr_x + width) <= wall.left:
                        new_x = min(new_x, wall.left - width)
                    elif vel.x < 0 and curr_x >= wall.right:
                        new_x = max(new_x, wall.right)
        else:
            # Fallback overlap clamp
            test_box = AABB(target_x, target_y, width, height)
            for wall in colliding_walls:
                if test_box.intersects(wall):
                    if vel.x > 0 and curr_x + width <= wall.left:
                        new_x = min(new_x, wall.left - width)
                    elif vel.x < 0 and curr_x >= wall.right:
                        new_x = max(new_x, wall.right)
                    if vel.y > 0 and curr_y + height <= wall.top:
                        new_y = min(new_y, wall.top - height)
                    elif vel.y < 0 and curr_y >= wall.bottom:
                        new_y = max(new_y, wall.bottom)

        curr_x = new_x
        curr_y = new_y

    actual_vx = (curr_x - entity_aabb.x) / dt if dt > 0 else 0.0
    actual_vy = (curr_y - entity_aabb.y) / dt if dt > 0 else 0.0
    return Vector2(curr_x, curr_y), Vector2(actual_vx, actual_vy)
