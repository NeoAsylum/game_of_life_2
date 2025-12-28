import numpy as np
from src.brain import Brain


class Entity:
    def __init__(self, id, x=0, y=0):
        self.id = id
        # Body is a list of [x, y] coordinates. Head is at index 0.
        self.body = [[float(x), float(y)]]
        self.energy = 100.0
        # Heading is a unit vector [dx, dy] or angle. We'll use angle in radians for steering.
        self.angle = np.random.uniform(0, 2 * np.pi)
        self.speed = 2.0  # Fixed speed
        self.brain = Brain()  # Uses new defaults (7 in, 4 out)
        self.alive = True
        self.age = 0
        self.decay_rate = 0.5  # Energy lost per frame

        # New Mechanics
        self.birth_cooldown = 0
        self.is_shooting = False
        self.laser_cooldown = 0

    def decide_direction(self, sensory_inputs):
        """
        Uses brain to decide steering and shooting.
        sensory_inputs: array of size 7
        """
        output = self.brain.forward(sensory_inputs)

        # Steering: Argmax of first 3 outputs
        # 0: Turn Left, 1: Turn Right, 2: Straight
        steering_action = np.argmax(output[:3])

        turn_speed = 0.2  # Radians per frame

        if steering_action == 0:
            self.angle -= turn_speed
        elif steering_action == 1:
            self.angle += turn_speed
        # else: maintain heading

        if len(output) > 3:
            # If the 4th neuron is the strongest signal, trigger shot.
            if output[3] > 0.5:
                self.is_shooting = True

        # Update cooldowns
        if self.birth_cooldown > 0:
            self.birth_cooldown -= 1
        if self.laser_cooldown > 0:
            self.laser_cooldown -= 1

        if self.is_shooting:
            if self.laser_cooldown > 0:
                self.is_shooting = False  # Cannot shoot yet
            else:
                self.laser_cooldown = 10  # Fire rate limit
                self.energy -= 5.0  # Cost to shoot

    def move(self):
        """
        Updates head position based on angle and speed.
        """

        dx = np.cos(self.angle) * self.speed
        dy = np.sin(self.angle) * self.speed

        new_head_x = self.body[0][0] + dx
        new_head_y = self.body[0][1] + dy

        # Insert new head position
        self.body.insert(0, [new_head_x, new_head_y])
        self.body.pop()

    def grow(self):
        """Adds a segment at the tail position (duplicate of last)"""
        if self.body:
            self.body.append(self.body[-1][:])  # Copy last segment

    def attempt_reproduction(self):
        """
        Returns a new Entity if birth is successful, else None.
        """
        if len(self.body) > 12 and self.birth_cooldown == 0:
            # Create Child
            # Child starts at tail position
            tail_x, tail_y = self.body[-1]
            child = Entity(id=-1, x=tail_x, y=tail_y)  # ID assigned by world

            # Inherit Brain
            child.brain.w1 = self.brain.w1.copy()
            child.brain.b1 = self.brain.b1.copy()
            child.brain.w2 = self.brain.w2.copy()
            child.brain.b2 = self.brain.b2.copy()
            child.brain.mutate(magnitude=0.1)

            # Cost to Parent: Lose 6 segments
            for _ in range(6):
                if len(self.body) > 1:
                    self.body.pop()

            self.birth_cooldown = 50  # Frames
            return child
        return None

    def update(self):
        """
        Per frame update: Decay energy, check starvation.
        """
        self.age += 1
        self.energy -= self.decay_rate

        if self.energy <= 0:
            self.shrink()

    def shrink(self):
        """
        Handle starvation logic.
        """
        if len(self.body) > 1:
            self.body.pop()  # Remove tail
            self.energy = 50.0  # Reset energy
        else:
            self.alive = False  # Die if only head remains
