import numpy as np
import copy
import pickle
from src.world import World
from src.entity import Entity


class Simulation:
    def __init__(self, width=800, height=600, population_size=200):
        self.world = World(width, height, num_entities=population_size)
        self.population_size = population_size
        self.stats = {
            "gen_count": 0,  # Total deaths/respawns
            "max_len": 1,
            "avg_len": 1.0,
            "best_age": 0,
        }

    def step(self):
        """
        Advances the simulation by one frame.
        """
        self.world.update()

        # Handle Deaths (Just cleanup)
        # World marks them as !alive.
        # We need to remove them from world.entities list.
        # But simulation also tracks stats.

        alive_entities = []
        for e in self.world.entities:
            if e.alive:
                alive_entities.append(e)
            else:
                # Dead
                self.stats["gen_count"] += 1
                if e.age > self.stats["best_age"]:
                    self.stats["best_age"] = e.age
                if len(e.body) > self.stats["max_len"]:
                    self.stats["max_len"] = len(e.body)

        self.world.entities = alive_entities

        # Extinction Safety NET
        # If population drops too low, spawn randoms to restart life
        if len(self.world.entities) < 10:
            # Spawn 10 randoms
            for _ in range(10):
                self.world.entities.append(
                    Entity(
                        id=self.stats["gen_count"],
                        x=np.random.uniform(0, self.world.width),
                        y=np.random.uniform(0, self.world.height),
                    )
                )

        # Update Live Stats
        # Doing this every frame might be overkill, do it on demand or every N frames?
        # Simulation step() just advances. Stats are updated.

    def get_metrics(self):
        """
        Returns a dict of current metrics for logging/plotting.
        """
        alive_entities = [e for e in self.world.entities if e.alive]
        alive_count = len(alive_entities)

        # Calculate avg length
        current_lengths = [len(e.body) for e in alive_entities]
        avg_len = sum(current_lengths) / alive_count if alive_count > 0 else 0
        max_len = max(current_lengths) if current_lengths else 0

        # Update records
        self.stats["max_len"] = max(self.stats["max_len"], max_len)

        # Calculate aggressiveness (shots fired / snake / frame)
        # We need the world to track shots fired this frame.
        shots = getattr(self.world, "shots_fired_this_frame", 0)
        if alive_count > 0:
            aggressiveness = shots / alive_count
        else:
            aggressiveness = 0.0

        return {
            # Keys for CSV Logging
            "AliveCount": alive_count,
            "AvgLength": avg_len,
            "Aggressiveness": aggressiveness,
            # Keys for Visualizer Overlay (Legacy support)
            "generation": self.stats["gen_count"],
            "pop_size": alive_count,
            "max_len_record": self.stats["max_len"],
            "current_avg_len": avg_len,
            "best_age_record": self.stats["best_age"],
        }

    def save_state(self, filename="checkpoint.pkl"):
        """Saves the current simulation state to a file."""
        try:
            with open(filename, "wb") as f:
                pickle.dump(self, f)
            print(f"Simulation saved to {filename}")
        except Exception as e:
            print(f"Failed to save simulation: {e}")

    @staticmethod
    def load_state(filename="checkpoint.pkl"):
        """Loads a simulation state from a file and returns the Simulation object."""
        try:
            with open(filename, "rb") as f:
                sim = pickle.load(f)
            print(f"Simulation loaded from {filename}")
            return sim
        except Exception as e:
            print(f"Failed to load simulation: {e}")
            return None
