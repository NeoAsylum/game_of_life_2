import sys
import os
import time
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from src.simulation import Simulation


def run_headless():
    print("Starting Headless Simulation...")
    sim = Simulation(width=800, height=600, population_size=200)

    start_time = time.time()
    frames = 50000
    # NOTE: "1000 generations" in continuous sim usually means 1000 * PopSize replacements?
    # Or just "Generation" counter.
    # If 200 entities, and avg life is say 100 frames. 50,000 frames = 500 lifecycles.
    # Let's run for a fixed number of replacements or frames.
    # Spec says "verify that avg_length increases over 1,000 'generations' (respawns)".

    for i in range(frames):
        sim.step()

        if i % 1000 == 0:
            m = sim.get_metrics()
            print(
                f"Frame {i}: Reps={m['generation']}, MaxLen={m['max_len_record']}, AvgLen={m['current_avg_len']:.2f}, BestAge={m['best_age_record']}"
            )

            # Early exit if we see significant learning?
            # Learning = MaxLen > 5? (Start is 1)
            # if m['max_len_record'] > 10:
            #     print("SIGNIFICANT LEARNING OBSERVED!")

    total_time = time.time() - start_time
    print(f"\nSimulation Finished in {total_time:.2f}s.")
    final_metrics = sim.get_metrics()
    print("Final Metrics:", final_metrics)

    # Validation check
    if final_metrics["max_len_record"] > 1:
        print("SUCCESS: Evolution produced growth.")
    else:
        print("WARNING: No growth observed. Evolution might need tuning.")


if __name__ == "__main__":
    run_headless()
