# ==========================================================================
# MÓDULO: npc.py
# PROPÓSITO: Veículos NPC e pedestres com pathfinding por waypoints
# DECISÃO DE DESIGN: NPCs seguem waypoints pré-definidos em vez de A*.
#   Em cruzamento simples, rotas fixas bastam e são O(1) por frame.
# ==========================================================================
from __future__ import annotations

import logging
import math
import random
from typing import Final

import pygame

from neurodrive.env.traffic_light import TrafficLight, TrafficLightState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
NPC_VEHICLE_WIDTH: Final[int] = 24
NPC_VEHICLE_HEIGHT: Final[int] = 40
NPC_VEHICLE_SPEED: Final[float] = 1.5
NPC_VEHICLE_COLOR: Final[tuple[int, int, int]] = (70, 130, 210)

PEDESTRIAN_WIDTH: Final[int] = 12
PEDESTRIAN_HEIGHT: Final[int] = 12
PEDESTRIAN_SPEED: Final[float] = 0.6
PEDESTRIAN_COLOR: Final[tuple[int, int, int]] = (230, 180, 80)

WAYPOINT_THRESHOLD: Final[float] = 5.0
STOP_DISTANCE: Final[float] = 30.0


class NPCVehicle:
    """Veículo NPC controlado por waypoint-following.

    Segue lista cíclica de waypoints, parando em semáforo vermelho.
    Usa modelo de ponto (não bicycle) — suficiente para obstáculo.

    Args:
        x: Posição X inicial (pixels).
        y: Posição Y inicial (pixels).
        waypoints: Pontos (x,y) que o NPC segue ciclicamente.
        speed: Velocidade em pixels/frame.
        color: Cor RGB do veículo.
    """

    def __init__(
        self, x: float, y: float,
        waypoints: list[tuple[float, float]],
        speed: float = NPC_VEHICLE_SPEED,
        color: tuple[int, int, int] = NPC_VEHICLE_COLOR,
    ) -> None:
        self.x = x
        self.y = y
        self.waypoints = waypoints
        self.speed = speed
        self.color = color
        self.heading: float = 0.0
        self._wp_index: int = 0
        self.active: bool = True
        self._stopped: bool = False
        if waypoints:
            self._update_heading()

    def _update_heading(self) -> None:
        """Atualiza heading para apontar para o waypoint atual."""
        if not self.waypoints:
            return
        tx, ty = self.waypoints[self._wp_index]
        dx, dy = tx - self.x, ty - self.y
        if abs(dx) > 0.01 or abs(dy) > 0.01:
            self.heading = math.atan2(dy, dx)

    def update(self, traffic_lights: list[TrafficLight] | None = None) -> None:
        """Avança 1 frame. Para em semáforo vermelho à frente.

        Args:
            traffic_lights: Semáforos para verificar parada.
        """
        if not self.active or not self.waypoints:
            return

        self._stopped = False
        if traffic_lights:
            for tl in traffic_lights:
                if tl.is_red or tl.is_yellow:
                    dist = math.hypot(tl.x - self.x, tl.y - self.y)
                    if dist < STOP_DISTANCE:
                        dx_tl = tl.x - self.x
                        dy_tl = tl.y - self.y
                        dot = dx_tl * math.cos(self.heading) + dy_tl * math.sin(self.heading)
                        if dot > 0:
                            self._stopped = True
                            return

        tx, ty = self.waypoints[self._wp_index]
        dx, dy = tx - self.x, ty - self.y
        dist = math.hypot(dx, dy)

        if dist < WAYPOINT_THRESHOLD:
            self._wp_index = (self._wp_index + 1) % len(self.waypoints)
            self._update_heading()
            return

        self.x += (dx / dist) * self.speed
        self.y += (dy / dist) * self.speed
        self._update_heading()

    def draw(self, surface: pygame.Surface) -> None:
        """Renderiza o NPC como retângulo rotacionado."""
        car_surf = pygame.Surface(
            (NPC_VEHICLE_WIDTH, NPC_VEHICLE_HEIGHT), pygame.SRCALPHA
        )
        pygame.draw.rect(car_surf, self.color,
                         (0, 0, NPC_VEHICLE_WIDTH, NPC_VEHICLE_HEIGHT),
                         border_radius=4)
        pygame.draw.rect(car_surf, (255, 255, 200),
                         (4, 0, NPC_VEHICLE_WIDTH - 8, 4))
        angle_deg = -math.degrees(self.heading) + 90
        rotated = pygame.transform.rotate(car_surf, angle_deg)
        rect = rotated.get_rect(center=(int(self.x), int(self.y)))
        surface.blit(rotated, rect)

    def get_rect(self) -> pygame.Rect:
        """AABB bounding box (para colisão e YOLO bbox)."""
        size = max(NPC_VEHICLE_WIDTH, NPC_VEHICLE_HEIGHT) // 2
        return pygame.Rect(int(self.x - size), int(self.y - size), size * 2, size * 2)


class Pedestrian:
    """Pedestre que caminha nas faixas do cruzamento.

    Args:
        x: Posição X inicial (pixels).
        y: Posição Y inicial (pixels).
        waypoints: Pontos de destino na faixa.
        speed: Velocidade em pixels/frame.
    """

    def __init__(
        self, x: float, y: float,
        waypoints: list[tuple[float, float]],
        speed: float = PEDESTRIAN_SPEED,
    ) -> None:
        self.x = x
        self.y = y
        self.waypoints = waypoints
        self.speed = speed
        self._wp_index: int = 0
        self.active: bool = True

    def update(self) -> None:
        """Move o pedestre em direção ao próximo waypoint."""
        if not self.active or not self.waypoints:
            return
        tx, ty = self.waypoints[self._wp_index]
        dx, dy = tx - self.x, ty - self.y
        dist = math.hypot(dx, dy)
        if dist < WAYPOINT_THRESHOLD:
            self._wp_index = (self._wp_index + 1) % len(self.waypoints)
            return
        self.x += (dx / dist) * self.speed
        self.y += (dy / dist) * self.speed

    def draw(self, surface: pygame.Surface) -> None:
        """Renderiza pedestre como círculo colorido."""
        pygame.draw.circle(surface, PEDESTRIAN_COLOR,
                           (int(self.x), int(self.y)), PEDESTRIAN_WIDTH // 2)
        pygame.draw.circle(surface, (180, 130, 50),
                           (int(self.x), int(self.y)), PEDESTRIAN_WIDTH // 2, width=1)

    def get_rect(self) -> pygame.Rect:
        """AABB bounding box do pedestre."""
        return pygame.Rect(
            int(self.x - PEDESTRIAN_WIDTH // 2),
            int(self.y - PEDESTRIAN_HEIGHT // 2),
            PEDESTRIAN_WIDTH, PEDESTRIAN_HEIGHT,
        )


def create_default_npc_vehicles(
    rng: random.Random | None = None,
) -> list[NPCVehicle]:
    """Cria 4 veículos NPC com rotas padrão no cruzamento.

    Args:
        rng: RNG para reprodutibilidade.

    Returns:
        Lista de NPCVehicle nas 4 vias.
    """
    colors = [(70, 130, 210), (210, 70, 70), (70, 180, 70), (200, 160, 50)]
    return [
        NPCVehicle(340, 50,  [(340, 280), (340, 590), (340, 50)],  color=colors[0]),
        NPCVehicle(300, 590, [(300, 360), (300, 50),  (300, 590)], color=colors[1]),
        NPCVehicle(590, 340, [(360, 340), (50, 340),  (590, 340)], color=colors[2]),
        NPCVehicle(50,  300, [(280, 300), (590, 300), (50, 300)],  color=colors[3]),
    ]


def create_default_pedestrians(
    rng: random.Random | None = None,
) -> list[Pedestrian]:
    """Cria 3 pedestres nas faixas do cruzamento."""
    return [
        Pedestrian(270, 285, [(370, 285), (270, 285)]),
        Pedestrian(370, 355, [(270, 355), (370, 355)]),
        Pedestrian(285, 370, [(285, 270), (285, 370)]),
    ]


# ── BLOCO DEMO ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    from neurodrive.env.traffic_light import TrafficLight as TL, TrafficLightController

    logging.basicConfig(level=logging.INFO)
    pygame.init()
    screen = pygame.display.set_mode((640, 640))
    pygame.display.set_caption("NeuroDrive — NPC Demo")
    clock = pygame.time.Clock()

    ns = [TL(320, 270), TL(320, 370)]
    ew = [TL(270, 320), TL(370, 320)]
    ctrl = TrafficLightController(ns, ew)
    vehicles = create_default_npc_vehicles()
    peds = create_default_pedestrians()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        ctrl.tick()
        for v in vehicles:
            v.update(ctrl.get_all_lights())
        for p in peds:
            p.update()
        screen.fill((80, 80, 80))
        pygame.draw.rect(screen, (50, 50, 50), (280, 0, 80, 640))
        pygame.draw.rect(screen, (50, 50, 50), (0, 280, 640, 80))
        ctrl.draw(screen)
        for v in vehicles:
            v.draw(screen)
        for p in peds:
            p.draw(screen)
        pygame.display.flip()
        clock.tick(30)
    pygame.quit()
    print("Demo NPC finalizado com sucesso.")
