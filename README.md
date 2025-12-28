# Neural Snake Ecosystem (Vectorized)

A high-performance evolutionary simulation where hundreds of neural-network-controlled "Snakes" compete for survival. They navigate a 2D world, consume food, fight with lasers, and evolve over generations using a genetic algorithm.

## 🚀 Key Features
*   **Vectorized Engine**: All physics, vision, and neural network calculations are batched using `NumPy`, allowing for thousands of simulation steps per second.
*   **Neural Brains**: Each snake has a unique brain (7 Inputs -> 12 Hidden -> 4 Outputs) that evolves via natural selection.
*   **Combat Mechanics**: Snakes can shoot lasers to destroy rivals. Aggression is emergent and tracked.
*   **Real-Time Visualization**: Watch the ecosystem live or run in "Headless Mode" for rapid training.
*   **Data Analysis**: Built-in CSV logging and Matplotlib plotting tools to track population health and behavior.

---

## Installation

1.  **Install uv** (if not already installed):
    ```bash
    pip install uv
    ```

2.  **Clone the repository**:
    ```bash
    git clone https://github.com/NeoAsylum/game_of_life_2.git
    cd game_of_life_2
    ```

3.  **Install Dependencies**:
    ```bash
    uv sync
    ```

## Usage

Run the simulation using `uv run`:

```bash
uv run main.py
```

Or activate the virtual environment manually:

*   **Windows**: `.venv\Scripts\activate`
*   **Mac/Linux**: `source .venv/bin/activate`

Then run:
```bash
python main.py
```

### Keyboard Shortcuts
| Key | Action | Description |
| :--- | :--- | :--- |
| **ESC** | Quit | Safely closes the simulation and flushes logs. |
| **SPACE** | Speed Up | Hold to run at 10x visual speed. |
| **H** | **Headless Toggle** | Toggles rendering on/off. **Crucial for fast training.** Cycles between 60 FPS (Rendered) and ~5000+ Steps/Sec (Headless). |
| **P** | **Plot Metrics** | Pauses sim and opens a Matplotlib graph showing Population, Length, and Aggression over time. |

---

## 🧠 Intelligence & Mechanics

### 1. The Entity (Snake)
*   **Energy**: Constantly decays. Restored by eating green food pellets.
*   **Death**: Occurs upon wall collision or starvation (Energy = 0).
*   **Reproduction**: When a snake dies, it is instantly replaced by a "child" of a top-performing survivor.

### 2. The Brain (Neural Network)
The brain is a simple feed-forward network with `tanh` activation.

*   **7 Sensory Inputs**:
    1.  Angle to Nearest Food.
    2.  Distance to Nearest Food.
    3.  Wall Distance (Center Ray).
    4.  Wall Distance (Left 45° Ray).
    5.  Wall Distance (Right 45° Ray).
    6.  Angle to Nearest Enemy Snake.
    7.  Distance to Nearest Enemy Snake.

*   **4 Action Outputs**:
    1.  Turn Left.
    2.  Turn Right.
    3.  Go Straight.
    4.  **Shoot Laser** (Trigger > 0.5 probability).

### 3. Aggression (Lasers)
*   Snakes evolve the ability to shoot.
*   **Cost**: Shooting consumes energy.
*   **Effect**: A laser hit kills another snake instantly.
*   **Metric**: "Aggressiveness" is logged as the average number of shots fired per snake per frame.

---

## 📊 Analytics

### Logging
The simulation automatically logs stats to `simulation_log.csv` every **60 simulation steps** (approx. 1 game-second). It tracks:
*   Generation Count
*   Alive Population
*   Average & Max Length
*   Aggressiveness (Violence Index)

### Plotting
Press **`P`** at any time to generate interactive graphs:
1.  **Top Graph**: Population Size & Average Length (Green/Red lines).
2.  **Bottom Graph**: Aggressiveness (Purple line).

---

## ⚡ Performance Note
This project uses a **Gather-Compute-Scatter** architecture. Instead of looping through objects, we flatten all entity data into generic NumPy arrays, process them with optimized linear algebra (e.g., `np.einsum`), and then scatter the results back. This is why "Headless Mode" ('H') is massively faster than the visual mode.
