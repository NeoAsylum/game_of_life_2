import pygame
import numpy as np


class Visualizer:
    def __init__(self, simulation, width=800, height=600):
        self.sim = simulation
        self.width = width
        self.height = height

        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Neural Snake Ecosystem")
        self.font = pygame.font.SysFont("Arial", 16)

    def handle_events(self):
        """
        Handles Pygame events.
        Returns False if the user tries to quit, True otherwise.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
        return True

    def draw(self):
        # 1. Background - Dark Blue/Black for contrast
        self.screen.fill((10, 10, 15))

        # 2. Draw Food - Neon Cyan/Green
        # Layered circles for a slight "glow" effect
        for fx, fy in self.sim.world.food:
            # Outer glow
            pygame.draw.circle(self.screen, (0, 100, 100), (int(fx), int(fy)), 5)
            # Core
            pygame.draw.circle(self.screen, (0, 255, 200), (int(fx), int(fy)), 3)

        # 3. Draw Entities - Snakes
        for entity in self.sim.world.entities:
            if not entity.alive:
                continue

            # Color Logic:
            # High Energy -> Cyan/White
            # Low Energy (<20) -> Red/Orange Warning
            if entity.energy < 20:
                # Starving: Reddish
                color = (255, 50, 50)
                head_color = (255, 100, 100)
            else:
                # Healthy: Teal/Electric Blue
                # Vary slightly by ID for visuals? No, uniformity is clean.
                color = (0, 200, 255)
                head_color = (200, 240, 255)

            # Draw Body - Anti-aliased lines for smoothness
            if len(entity.body) > 1:
                points = [(int(x), int(y)) for x, y in entity.body]
                # Thicker line background for visibility?
                # pygame.draw.lines(self.screen, (0, 0, 0), False, points, 4) # Outline
                pygame.draw.lines(self.screen, color, False, points, 2)

            # Draw Head
            hx, hy = entity.body[0]
            pygame.draw.circle(self.screen, head_color, (int(hx), int(hy)), 3)

        # 4. Draw Lasers - Bright Pink/Magenta beams
        if hasattr(self.sim.world, "active_lasers"):
            for start, end in self.sim.world.active_lasers:
                # Glow line (thicker, darker)
                pygame.draw.line(
                    self.screen, (150, 0, 150), start.astype(int), end.astype(int), 4
                )
                # Core line (thin, white/bright)
                pygame.draw.line(
                    self.screen, (255, 100, 255), start.astype(int), end.astype(int), 1
                )

        # 5. Draw Stats Overlay
        stats = self.sim.get_metrics()

        # Info Block
        lines = [
            f"Generation:   {stats['generation']}",
            f"Population:   {stats['pop_size']} / {self.sim.population_size}",
            f"Avg Length:   {stats['current_avg_len']:.2f}",
            f"Max Length:   {stats['max_len_record']}",
            f"Best Age:     {stats['best_age_record']}",
            "",  # Spacer
            "CONTROLS:",
            "  [SPACE]  Hold to Speed Up (10x)",
            "  [P]      Plot Metrics (Interactive)",
            "  [S]      Save State",
            "  [L]      Load State",
            "  [H]      Toggle Headless",
            "  [ESC]    Quit",
        ]

        # Draw a semi-transparent panel for text
        panel_width = 240
        panel_height = len(lines) * 20 + 20
        s = pygame.Surface((panel_width, panel_height))
        s.set_alpha(180)  # Alpha level
        s.fill((0, 0, 0))  # Black background
        self.screen.blit(s, (10, 10))

        # Render Text
        x_offset = 20
        y_offset = 20
        for line in lines:
            if "CONTROLS" in line:  # Header highlight
                c = (255, 255, 100)  # Yellow
            else:
                c = (220, 220, 220)  # Grey/White

            text = self.font.render(line, True, c)
            self.screen.blit(text, (x_offset, y_offset))
            y_offset += 20

        pygame.display.flip()
