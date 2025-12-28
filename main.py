import sys
import os
import pygame
import time
import csv
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Ensure src is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from src.simulation import Simulation
from src.visualizer import Visualizer


def plot_metrics(log_file):
    try:
        # Load Data
        df = pd.read_csv(log_file)
        if df.empty:
            print("No data to plot.")
            return

        # Create Subplots
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=("Population & Growth", "Aggressiveness"),
            specs=[[{"secondary_y": True}], [{"secondary_y": False}]],
        )

        # Plot 1: Alive Count (Primary Y)
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["AliveCount"],
                name="Alive Count",
                line=dict(color="#ff4d4d"),
            ),
            row=1,
            col=1,
        )

        # Plot 1: Avg Length (Secondary Y)
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["AvgLength"],
                name="Avg Length",
                line=dict(color="#4dffdb"),
            ),
            row=1,
            col=1,
            secondary_y=True,
        )

        # Plot 2: Aggressiveness
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["Aggressiveness"],
                name="Aggressiveness",
                line=dict(color="#d64dff"),
            ),
            row=2,
            col=1,
        )

        # Layout styling
        fig.update_layout(
            title_text="Neural Snake Ecosystem Metrics",
            template="plotly_dark",
            hovermode="x unified",
            height=800,
        )

        # Axis labels
        fig.update_yaxes(title_text="Count", row=1, col=1, secondary_y=False)
        fig.update_yaxes(title_text="Length", row=1, col=1, secondary_y=True)
        fig.update_yaxes(title_text="Shots/Frame", row=2, col=1)
        fig.update_xaxes(title_text="Simulation Steps (x60)", row=2, col=1)

        fig.show()

    except Exception as e:
        print(f"Error plotting: {e}")


def main():
    print("Initializing Neural Snake Ecosystem...")
    width, height = 1000, 800
    sim = Simulation(width=width, height=height, population_size=200)
    vis = Visualizer(sim, width=width, height=height)

    clock = pygame.time.Clock()
    running = True
    rendering_enabled = True

    # Logging Setup
    log_file = "simulation_log.csv"

    # Check if we need to rewrite header (if file exists but old format)
    needs_header = True
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            line = f.readline()
            if "Aggressiveness" in line:
                needs_header = False
            else:
                print("Old log format detected. Overwriting log file.")
                needs_header = True

    if needs_header:
        with open(log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "Timestamp",
                    "Generation",
                    "AliveCount",
                    "AvgLength",
                    "MaxLength",
                    "Aggressiveness",
                ]
            )

    last_flush_time = time.time()
    log_buffer = []
    total_steps = 0

    print("Bindings:")
    print("  ESC: Quit")
    print("  SPACE (Hold): Speed up 10x")
    print("  H: Toggle Headless Mode (Fast Training)")
    print("  P: Plot Metrics (Interactive)")
    print("  S: Save Simulation State")
    print("  L: Load Simulation State")

    try:
        while running:
            # Input Handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    if event.key == pygame.K_h:
                        rendering_enabled = not rendering_enabled
                        print(f"Rendering Enabled: {rendering_enabled}")
                    if event.key == pygame.K_p:
                        print("Generating Plot...")
                        # Flush before plotting so we see latest data
                        if log_buffer:
                            with open(log_file, "a", newline="") as f:
                                writer = csv.writer(f)
                                writer.writerows(log_buffer)
                            log_buffer = []
                        plot_metrics(log_file)
                    if event.key == pygame.K_s:
                        sim.save_state()
                    if event.key == pygame.K_l:
                        loaded_sim = Simulation.load_state()
                        if loaded_sim:
                            sim = loaded_sim
                            vis.sim = sim

            keys = pygame.key.get_pressed()

            # Simulation Logic
            steps = 1
            if not rendering_enabled:
                steps = 50
            elif keys[pygame.K_SPACE]:
                steps = 10

            for _ in range(steps):
                sim.step()
                total_steps += 1

                # 1. Collect Data (Every 60 steps ~ 1 game second)
                if total_steps % 60 == 0:
                    metrics = sim.get_metrics()
                    current_time = time.time()
                    timestamp_str = time.strftime(
                        "%Y-%m-%d %H:%M:%S", time.localtime(current_time)
                    )
                    log_buffer.append(
                        [
                            timestamp_str,
                            metrics["generation"],
                            metrics["AliveCount"],
                            f"{metrics['AvgLength']:.2f}",
                            metrics["max_len_record"],
                            f"{metrics['Aggressiveness']:.4f}",
                        ]
                    )

            # Draw
            if rendering_enabled:
                vis.draw()
                if not keys[pygame.K_SPACE]:
                    clock.tick(60)

            # 2. Flush to Disk (Every 60 seconds WALL CLOCK)
            current_time = time.time()
            if current_time - last_flush_time >= 60.0:
                if log_buffer:
                    with open(log_file, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerows(log_buffer)
                    print(f"Flushed {len(log_buffer)} log entries to CSV.")
                    log_buffer = []
                last_flush_time = current_time

    finally:
        # Flush on exit
        if log_buffer:
            print("Saving remaining logs...")
            with open(log_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(log_buffer)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
