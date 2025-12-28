import numpy as np
from src.entity import Entity


class World:
    def __init__(self, width=800, height=600, num_entities=200):
        self.width = width
        self.height = height
        self.entities = [
            Entity(id=i, x=np.random.uniform(0, width), y=np.random.uniform(0, height))
            for i in range(num_entities)
        ]
        self.food_value = 20.0  # Energy gained per pellet
        self.num_entities = num_entities

        # Keep food as a numpy array for speed
        # Shape: (N, 2)
        self.food = np.empty((0, 2))
        self.spawn_food(count=50)

    def spawn_food(self, count=1):
        # Generate random positions
        new_food = np.random.uniform(0, [self.width, self.height], size=(count, 2))
        if self.food.shape[0] == 0:
            self.food = new_food
        else:
            self.food = np.vstack([self.food, new_food])

    def get_sensory_input(self, entity, all_head_positions):
        """
        Calculates 7 inputs for the brain:
        1. Angle to nearest food
        2. Distance to nearest food
        3. Wall Ahead
        4. Wall Left
        5. Wall Right
        6. Angle to nearest Enemy Snake (NEW)
        7. Distance to nearest Enemy Snake (NEW)

        all_head_positions: Numpy array (N, 2) of all snake heads.
        """
        head_x, head_y = entity.body[0]
        head_pos = np.array([head_x, head_y])

        # 1 & 2: Nearest Food
        if self.food.shape[0] == 0:
            food_angle_input = 0.0
            food_dist_input = 1.0
        else:
            deltas = self.food - head_pos
            dists_sq = np.sum(deltas**2, axis=1)
            min_idx = np.argmin(dists_sq)
            min_dist = np.sqrt(dists_sq[min_idx])

            dx, dy = deltas[min_idx]
            angle_to_target = np.arctan2(dy, dx)

            rel_angle = angle_to_target - entity.angle
            rel_angle = (rel_angle + np.pi) % (2 * np.pi) - np.pi

            food_angle_input = rel_angle / np.pi
            food_dist_input = min(min_dist / 500.0, 1.0)

        # 3, 4, 5: Walls (Raycasting)
        def get_wall_dist(angle_offset):
            ray_angle = entity.angle + angle_offset
            cost = np.cos(ray_angle)
            sint = np.sin(ray_angle)

            if cost > 0:
                dx = (self.width - head_x) / cost
            elif cost < 0:
                dx = (0 - head_x) / cost
            else:
                dx = 1000.0

            if sint > 0:
                dy = (self.height - head_y) / sint
            elif sint < 0:
                dy = (0 - head_y) / sint
            else:
                dy = 1000.0

            return min(min(dx, dy) / 500.0, 1.0)

        dist_ahead = get_wall_dist(0)
        dist_left = get_wall_dist(-np.pi / 4)
        dist_right = get_wall_dist(np.pi / 4)

        # 6 & 7: Nearest Enemy Snake
        # all_head_positions contains ALL heads (including self).
        # We need to find closest that isn't me.
        # Assuming entities have unique IDs or we just filter by distance > 0.

        # Calculate distances to all heads
        deltas = all_head_positions - head_pos
        dists_sq = np.sum(deltas**2, axis=1)

        # Mask self (dist 0)
        # Note: if multiple snakes exact overlap, this might mask them too. Rare.
        # Set self distance to infinity to ignore
        dists_sq[dists_sq < 0.001] = float("inf")

        if np.all(np.isinf(dists_sq)):
            # No other snakes
            enemy_angle_input = 0.0
            enemy_dist_input = 1.0
        else:
            min_idx = np.argmin(dists_sq)
            min_dist = np.sqrt(dists_sq[min_idx])

            dx, dy = deltas[min_idx]
            angle_to_target = np.arctan2(dy, dx)

            rel_angle = angle_to_target - entity.angle
            rel_angle = (rel_angle + np.pi) % (2 * np.pi) - np.pi

            enemy_angle_input = rel_angle / np.pi
            enemy_dist_input = min(min_dist / 500.0, 1.0)

        return np.array(
            [
                food_angle_input,
                food_dist_input,
                dist_ahead,
                dist_left,
                dist_right,
                enemy_angle_input,
                enemy_dist_input,
            ]
        )

    def get_sensory_input_batch(self, heads, angles, food):
        """
        Vectorized sensory input calculation.
        heads: (N, 2)
        angles: (N,)
        food: (M, 2)
        Returns: (N, 7) inputs
        """
        N = heads.shape[0]
        inputs = np.zeros((N, 7))

        # --- 1 & 2: Nearest Food ---
        if food.shape[0] > 0:
            # (N, M, 2) = (N, 1, 2) - (1, M, 2)
            deltas = food[None, :, :] - heads[:, None, :]
            dists_sq = np.sum(deltas**2, axis=2)  # (N, M)
            min_indices = np.argmin(dists_sq, axis=1)  # (N,)
            min_dists = np.sqrt(dists_sq[np.arange(N), min_indices])  # (N,)

            # Vectors to nearest food
            nearest_food = food[min_indices]  # (N, 2)
            rel_vecs = nearest_food - heads  # (N, 2)
            coeffs = np.arctan2(rel_vecs[:, 1], rel_vecs[:, 0])  # (N,)

            # Relative Angle
            rel_angles = coeffs - angles
            rel_angles = (rel_angles + np.pi) % (2 * np.pi) - np.pi

            inputs[:, 0] = rel_angles / np.pi
            inputs[:, 1] = np.minimum(min_dists / 500.0, 1.0)
        else:
            inputs[:, 0] = 0.0
            inputs[:, 1] = 1.0

        # --- 3, 4, 5: Walls (Raycasting) ---
        # Wall Dist function vectorized
        # offsets: (3,) -> [0, -pi/4, pi/4]
        # target_angles: (N, 3)
        offsets = np.array([0, -np.pi / 4, np.pi / 4])
        target_angles = angles[:, None] + offsets[None, :]

        costs = np.cos(target_angles)  # (N, 3)
        sints = np.sin(target_angles)  # (N, 3)

        # X Dists
        # If cost > 0: (W - x) / cost. If < 0: (0 - x) / cost.
        # We can use np.where

        # heads_x: (N, 1) broadcast to (N, 3)
        hx = heads[:, 0][:, None]
        hy = heads[:, 1][:, None]

        dist_x = np.full((N, 3), 1000.0)
        mask_pos_x = costs > 0
        mask_neg_x = costs < 0

        # Handle division by zero warning safely by ignoring where cost is 0?
        # Numpy handles inf usually, but let's be safe.
        epsilon = 1e-9
        costs_safe = np.where(np.abs(costs) < epsilon, epsilon, costs)

        dist_x = np.where(mask_pos_x, (self.width - hx) / costs_safe, dist_x)
        dist_x = np.where(mask_neg_x, (0 - hx) / costs_safe, dist_x)

        # Y Dists
        sints_safe = np.where(np.abs(sints) < epsilon, epsilon, sints)
        dist_y = np.full((N, 3), 1000.0)
        mask_pos_y = sints > 0
        mask_neg_y = sints < 0

        dist_y = np.where(mask_pos_y, (self.height - hy) / sints_safe, dist_y)
        dist_y = np.where(mask_neg_y, (0 - hy) / sints_safe, dist_y)

        final_dists = np.minimum(dist_x, dist_y)
        inputs[:, 2:5] = np.minimum(final_dists / 500.0, 1.0)

        # --- 6 & 7: Nearest Enemy Snake ---
        # (N, N, 2)
        deltas = heads[None, :, :] - heads[:, None, :]  # N x N x 2
        dists_sq = np.sum(deltas**2, axis=2)  # N x N

        # Mask self (diagonal)
        np.fill_diagonal(dists_sq, np.inf)

        if N > 1:
            min_indices = np.argmin(dists_sq, axis=1)
            min_dists = np.sqrt(dists_sq[np.arange(N), min_indices])

            # Angle
            nearest_enemy_deltas = deltas[np.arange(N), min_indices]  # (N, 2)
            coeffs = np.arctan2(nearest_enemy_deltas[:, 1], nearest_enemy_deltas[:, 0])

            rel_angles = coeffs - angles
            rel_angles = (rel_angles + np.pi) % (2 * np.pi) - np.pi

            inputs[:, 5] = rel_angles / np.pi
            inputs[:, 6] = np.minimum(min_dists / 500.0, 1.0)
        else:
            inputs[:, 5] = 0.0
            inputs[:, 6] = 1.0

        return inputs

    def update(self):
        # Cache all head positions for sensing
        # Filter alive only
        alive_entities = [e for e in self.entities if e.alive]
        if not alive_entities:
            return  # Extinction

        # Extract heads
        # heads = np.array([e.body[0] for e in alive_entities])
        # Map back index to entity? "alive_entities" list matches "heads" array index.

        # new_borns = []
        # eaten_indices = set()

        # Laser storage for visualization (cleared every frame)
        # self.active_lasers = []

        if self.food.shape[0] == 0:
            self.food = new_food
        else:
            self.food = np.vstack([self.food, new_food])

    def get_sensory_input_batch(self, heads, angles, food):
        """
        Vectorized sensory input calculation.
        heads: (N, 2)
        angles: (N,)
        food: (M, 2)
        Returns: (N, 7) inputs
        """
        N = heads.shape[0]
        inputs = np.zeros((N, 7))

        # --- 1 & 2: Nearest Food ---
        if food.shape[0] > 0:
            # (N, M, 2) = (N, 1, 2) - (1, M, 2)
            deltas = food[None, :, :] - heads[:, None, :]
            dists_sq = np.sum(deltas**2, axis=2)  # (N, M)
            min_indices = np.argmin(dists_sq, axis=1)  # (N,)
            min_dists = np.sqrt(dists_sq[np.arange(N), min_indices])  # (N,)

            # Vectors to nearest food
            nearest_food = food[min_indices]  # (N, 2)
            rel_vecs = nearest_food - heads  # (N, 2)
            coeffs = np.arctan2(rel_vecs[:, 1], rel_vecs[:, 0])  # (N,)

            # Relative Angle
            rel_angles = coeffs - angles
            rel_angles = (rel_angles + np.pi) % (2 * np.pi) - np.pi

            inputs[:, 0] = rel_angles / np.pi
            inputs[:, 1] = np.minimum(min_dists / 500.0, 1.0)
        else:
            inputs[:, 0] = 0.0
            inputs[:, 1] = 1.0

        # --- 3, 4, 5: Walls (Raycasting) ---
        # Wall Dist function vectorized
        # offsets: (3,) -> [0, -pi/4, pi/4]
        # target_angles: (N, 3)
        offsets = np.array([0, -np.pi / 4, np.pi / 4])
        target_angles = angles[:, None] + offsets[None, :]

        costs = np.cos(target_angles)  # (N, 3)
        sints = np.sin(target_angles)  # (N, 3)

        # X Dists
        # If cost > 0: (W - x) / cost. If < 0: (0 - x) / cost.
        # We can use np.where

        # heads_x: (N, 1) broadcast to (N, 3)
        hx = heads[:, 0][:, None]
        hy = heads[:, 1][:, None]

        dist_x = np.full((N, 3), 1000.0)
        mask_pos_x = costs > 0
        mask_neg_x = costs < 0

        # Handle division by zero warning safely by ignoring where cost is 0?
        # Numpy handles inf usually, but let's be safe.
        epsilon = 1e-9
        costs_safe = np.where(np.abs(costs) < epsilon, epsilon, costs)

        dist_x = np.where(mask_pos_x, (self.width - hx) / costs_safe, dist_x)
        dist_x = np.where(mask_neg_x, (0 - hx) / costs_safe, dist_x)

        # Y Dists
        sints_safe = np.where(np.abs(sints) < epsilon, epsilon, sints)
        dist_y = np.full((N, 3), 1000.0)
        mask_pos_y = sints > 0
        mask_neg_y = sints < 0

        dist_y = np.where(mask_pos_y, (self.height - hy) / sints_safe, dist_y)
        dist_y = np.where(mask_neg_y, (0 - hy) / sints_safe, dist_y)

        final_dists = np.minimum(dist_x, dist_y)
        inputs[:, 2:5] = np.minimum(final_dists / 500.0, 1.0)

        # --- 6 & 7: Nearest Enemy Snake ---
        # (N, N, 2)
        deltas = heads[None, :, :] - heads[:, None, :]  # N x N x 2
        dists_sq = np.sum(deltas**2, axis=2)  # N x N

        # Mask self (diagonal)
        np.fill_diagonal(dists_sq, np.inf)

        if N > 1:
            min_indices = np.argmin(dists_sq, axis=1)
            min_dists = np.sqrt(dists_sq[np.arange(N), min_indices])

            # Angle
            nearest_enemy_deltas = deltas[np.arange(N), min_indices]  # (N, 2)
            coeffs = np.arctan2(nearest_enemy_deltas[:, 1], nearest_enemy_deltas[:, 0])

            rel_angles = coeffs - angles
            rel_angles = (rel_angles + np.pi) % (2 * np.pi) - np.pi

            inputs[:, 5] = rel_angles / np.pi
            inputs[:, 6] = np.minimum(min_dists / 500.0, 1.0)
        else:
            inputs[:, 5] = 0.0
            inputs[:, 6] = 1.0

        return inputs

    def update(self):
        alive_entities = [e for e in self.entities if e.alive]
        if not alive_entities:
            return

        N = len(alive_entities)

        # --- GATHER ---
        heads = np.array([e.body[0] for e in alive_entities])  # (N, 2)
        angles = np.array([e.angle for e in alive_entities])  # (N,)

        # Weights Gathering
        # We need to stack them.
        # W1: (N, 7, 12), B1: (N, 12), W2: (N, 12, 4), B2: (N, 4)
        # This allocation might be the expensive part now, but let's see.
        w1_list, b1_list, w2_list, b2_list = [], [], [], []
        for e in alive_entities:
            p = e.brain.get_params()
            w1_list.append(p[0])
            b1_list.append(p[1])
            w2_list.append(p[2])
            b2_list.append(p[3])

        W1 = np.stack(w1_list)
        B1 = np.stack(b1_list)
        W2 = np.stack(w2_list)
        B2 = np.stack(b2_list)

        # --- SENSE ---
        inputs = self.get_sensory_input_batch(heads, angles, self.food)  # (N, 7)

        # --- THINK (Vectorized Forward Pass) ---
        # Layer 1
        hidden = np.tanh(np.einsum("ni,nij->nj", inputs, W1) + B1)

        # Layer 2
        final = np.einsum("nj,njk->nk", hidden, W2) + B2

        # Softmax
        exps = np.exp(final - np.max(final, axis=1, keepdims=True))
        probs = exps / np.sum(exps, axis=1, keepdims=True)

        # --- ACT ---
        steering_actions = np.argmax(probs[:, :3], axis=1)  # (N,)
        shooting_triggers = probs[:, 3] > 0.5  # (N,)

        # --- SCATTER / PHYSICS ---
        # Update Angles
        turn_speed = 0.2
        # 0: Left, 1: Right
        turn_left_mask = steering_actions == 0
        turn_right_mask = steering_actions == 1

        angles[turn_left_mask] -= turn_speed
        angles[turn_right_mask] += turn_speed

        speed = 2.0
        dx = np.cos(angles) * speed
        dy = np.sin(angles) * speed

        new_heads_x = heads[:, 0] + dx
        new_heads_y = heads[:, 1] + dy

        # --- APPLY BACK TO OBJECTS & GAME RULES ---
        # This part handles the "stateful" stuff like body history, food, collisions
        # We loop again, but 'thinking' is done.

        new_borns = []
        eaten_indices = set()
        self.active_lasers = []
        self.shots_fired_this_frame = (
            np.sum(shooting_triggers) if len(alive_entities) > 0 else 0
        )

        for i, entity in enumerate(alive_entities):
            # 1. Update Angle & Move
            entity.angle = angles[i]

            # Manual move logic (duplicate of entity.move but using calculated head)
            # entity.move() does: cos/sin update, insert head, pop tail.
            # We already computed new head.
            entity.body.insert(0, [new_heads_x[i], new_heads_y[i]])
            entity.body.pop()

            # 2. Shooting
            entity.is_shooting = False
            if entity.laser_cooldown > 0:
                entity.laser_cooldown -= 1
            if entity.birth_cooldown > 0:
                entity.birth_cooldown -= 1

            if shooting_triggers[i]:
                if entity.laser_cooldown <= 0:
                    entity.is_shooting = True
                    entity.laser_cooldown = 10
                    entity.energy -= 5.0

            if entity.is_shooting:
                # Laser Physics (Keep this loop-based or vector? Loop is fine for sparse events)
                # ... same laser logic as before ...
                range_ = 100.0
                lx = np.cos(entity.angle)
                ly = np.sin(entity.angle)
                start_p = np.array(entity.body[0])
                end_p = start_p + np.array([lx, ly]) * range_

                self.active_lasers.append((start_p, end_p))

                v = heads - start_p
                proj = v[:, 0] * lx + v[:, 1] * ly
                perp_dist_sq = (v[:, 0] - proj * lx) ** 2 + (v[:, 1] - proj * ly) ** 2

                hits = (
                    (proj > 0)
                    & (proj < range_)
                    & (perp_dist_sq < 400.0)
                    & (proj > 10.0)
                )
                hit_indices = np.where(hits)[0]
                for h_idx in hit_indices:
                    alive_entities[h_idx].energy -= 20.0

            # 3. Update & Birth
            entity.update()

            child = entity.attempt_reproduction()
            if child:
                child.id = self.num_entities + 1
                self.num_entities += 1
                new_borns.append(child)

            # 4. Bounds
            hx, hy = entity.body[0]
            if hx < 0 or hx > self.width or hy < 0 or hy > self.height:
                entity.alive = False
                continue

            # 5. Eat
            if self.food.shape[0] > 0:
                # Check distance to food (we already calced this in Batch Input? No, that was ArgMin)
                # Re-calc local dist to check collision < 100
                # Simple check:
                deltas_f = self.food - np.array([hx, hy])
                dists_f_sq = np.sum(deltas_f**2, axis=1)
                min_f_idx = np.argmin(dists_f_sq)
                if dists_f_sq[min_f_idx] < 100.0:
                    if min_f_idx not in eaten_indices:
                        eaten_indices.add(min_f_idx)
                        entity.energy += self.food_value
                        entity.grow()

        # Food Cleanup
        if eaten_indices:
            keep_mask = np.ones(self.food.shape[0], dtype=bool)
            keep_mask[list(eaten_indices)] = False
            self.food = self.food[keep_mask]
            self.spawn_food(len(eaten_indices))

        if new_borns:
            self.entities.extend(new_borns)
