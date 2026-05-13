# ==========================================================================
# MÓDULO: city_env.py
# PROPÓSITO: CityDriveEnv — Gymnasium environment para direção urbana 2D
# DECISÃO DE DESIGN: Dict observation space com detecções YOLO + ego_state
#   em vez de pixels brutos. Reduz dimensionalidade de 1.228.800 para ~65
#   floats, acelerando convergência do PPO em ~10x (Mnih et al., 2015).
#   Action space Discrete(5) simplifica exploração para contexto acadêmico.
# ==========================================================================
from __future__ import annotations

import logging
import math
import random
from typing import Any, Final

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces

from neurodrive.agent.reward import (
    GOAL_THRESHOLD_PX,
    compute_reward,
)
from neurodrive.env.npc import (
    NPCVehicle,
    Pedestrian,
    create_default_npc_vehicles,
    create_default_pedestrians,
)
from neurodrive.env.tilemap import (
    CityMap,
    MAP_HEIGHT,
    MAP_WIDTH,
    ROAD_EW_BOTTOM,
    ROAD_EW_TOP,
    ROAD_NS_LEFT,
    ROAD_NS_RIGHT,
)
from neurodrive.env.traffic_light import (
    TrafficLight,
    TrafficLightController,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes do Agente (Ego-Vehicle)
# ---------------------------------------------------------------------------
EGO_WIDTH: Final[int] = 20
EGO_HEIGHT: Final[int] = 36
EGO_COLOR: Final[tuple[int, int, int]] = (40, 220, 80)
EGO_MAX_SPEED: Final[float] = 4.0           # px/frame
EGO_ACCELERATION: Final[float] = 0.3        # px/frame²
EGO_DECELERATION: Final[float] = 0.5        # frenagem mais forte
EGO_TURN_RATE: Final[float] = 0.08          # rad/frame (~4.6°/frame)
EGO_FRICTION: Final[float] = 0.02           # desaceleração natural

# Número máximo de detecções no vetor de observação
MAX_DETECTIONS: Final[int] = 10

# Limite de steps por episódio (timeout)
MAX_STEPS: Final[int] = 2000

# Ações discretas
ACTION_ACCELERATE: Final[int] = 0
ACTION_BRAKE: Final[int] = 1
ACTION_TURN_LEFT: Final[int] = 2
ACTION_TURN_RIGHT: Final[int] = 3
ACTION_STOP: Final[int] = 4


class CityDriveEnv(gym.Env):
    """Gymnasium environment para direção autônoma em cidade 2D.

    O agente controla um veículo (ego-car) em um mapa urbano com
    cruzamento, semáforos, veículos NPC e pedestres. O objetivo é
    atingir um waypoint destino sem colidir e respeitando trânsito.

    Observation Space (Dict):
        'detections': Box(0, 1, shape=(MAX_DETECTIONS, 6))
            Cada linha: [class_id_norm, cx_norm, cy_norm, w_norm, h_norm, conf]
            Normalizado para [0, 1]. Linhas não usadas são zeros.
        'ego_state': Box(0, 1, shape=(5,))
            [pos_x_norm, pos_y_norm, velocity_norm, heading_norm, dist_to_goal_norm]

    Action Space: Discrete(5)
        0: ACELERAR, 1: FREAR, 2: VIRAR_ESQ, 3: VIRAR_DIR, 4: PARAR

    Render Modes: 'human' (pygame window), 'rgb_array' (numpy array)

    Note (decisão de design):
        Na Fase 1 as detecções são geradas via ground-truth (posição
        real dos objetos), sem passar pelo YOLO. Na Fase 3, o módulo
        vision/detector.py substituirá isso pelo pipeline YOLO real,
        sem alterar a interface da observation.
    """

    metadata: dict[str, Any] = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(
        self,
        render_mode: str | None = None,
        seed: int | None = None,
    ) -> None:
        """Inicializa o ambiente.

        Args:
            render_mode: 'human' para janela, 'rgb_array' para headless.
            seed: Seed para reprodutibilidade.
        """
        super().__init__()

        self.render_mode = render_mode

        # --- Observation Space ---
        self.observation_space = spaces.Dict({
            "detections": spaces.Box(
                low=0.0, high=1.0,
                shape=(MAX_DETECTIONS, 6),
                dtype=np.float32,
            ),
            "ego_state": spaces.Box(
                low=0.0, high=1.0,
                shape=(5,),
                dtype=np.float32,
            ),
        })

        # --- Action Space ---
        self.action_space = spaces.Discrete(5)

        # --- Componentes internos ---
        self._city_map = CityMap()
        self._rng = random.Random(seed)
        self._np_rng = np.random.RandomState(seed)

        # --- Estado (inicializado em reset) ---
        self._ego_x: float = 0.0
        self._ego_y: float = 0.0
        self._ego_speed: float = 0.0
        self._ego_heading: float = 0.0
        self._goal_x: float = 0.0
        self._goal_y: float = 0.0
        self._step_count: int = 0
        self._prev_action: int | None = None
        self._prev_dist_to_goal: float = 0.0

        self._npc_vehicles: list[NPCVehicle] = []
        self._pedestrians: list[Pedestrian] = []
        self._tl_controller: TrafficLightController | None = None

        # --- PyGame (lazy init) ---
        self._window: pygame.Surface | None = None
        self._clock: pygame.time.Clock | None = None
        self._pygame_initialized: bool = False

        logger.info("CityDriveEnv criado (render_mode=%s)", render_mode)

    def _init_pygame(self) -> None:
        """Inicializa PyGame se necessário (lazy init)."""
        if self._pygame_initialized:
            return
        pygame.init()
        if self.render_mode == "human":
            self._window = pygame.display.set_mode((MAP_WIDTH, MAP_HEIGHT))
            pygame.display.set_caption("NeuroDrive — CityDriveEnv")
        self._clock = pygame.time.Clock()
        self._pygame_initialized = True

    def _spawn_ego(self) -> None:
        """Posiciona o agente em um ponto aleatório válido na via.

        Note (decisão de design):
            Spawn nas pontas das vias (não no cruzamento) para evitar
            colisão imediata com NPCs no cruzamento.
        """
        spawn_points = [
            (300, 60, math.pi / 2),     # Norte, indo para sul
            (340, 580, -math.pi / 2),   # Sul, indo para norte
            (60, 340, 0.0),              # Oeste, indo para leste
            (580, 300, math.pi),         # Leste, indo para oeste
        ]
        x, y, heading = self._rng.choice(spawn_points)
        self._ego_x = float(x)
        self._ego_y = float(y)
        self._ego_heading = heading
        self._ego_speed = 0.0

    def _spawn_goal(self) -> None:
        """Define um waypoint destino em uma extremidade oposta ao spawn.

        Note (decisão de design):
            Goal é na extremidade oposta para forçar o agente a
            atravessar o cruzamento (onde estão os semáforos e NPCs).
        """
        goal_candidates = [
            (300, 60), (340, 580), (60, 340), (580, 300),
            (300, 580), (340, 60), (580, 340), (60, 300),
        ]
        # Filtra posições longe do ego
        valid = [
            (gx, gy) for gx, gy in goal_candidates
            if math.hypot(gx - self._ego_x, gy - self._ego_y) > 200
        ]
        if not valid:
            valid = goal_candidates
        gx, gy = self._rng.choice(valid)
        self._goal_x = float(gx)
        self._goal_y = float(gy)

    def _create_traffic_lights(self) -> None:
        """Cria semáforos sincronizados no cruzamento."""
        ns_lights = [
            TrafficLight(320, 265),  # Norte do cruzamento
            TrafficLight(320, 375),  # Sul do cruzamento
        ]
        ew_lights = [
            TrafficLight(265, 320),  # Oeste do cruzamento
            TrafficLight(375, 320),  # Leste do cruzamento
        ]
        self._tl_controller = TrafficLightController(ns_lights, ew_lights)

    def _dist_to_goal(self) -> float:
        """Distância euclidiana do ego ao goal."""
        return math.hypot(self._goal_x - self._ego_x, self._goal_y - self._ego_y)

    def _get_ego_rect(self) -> pygame.Rect:
        """Retorna AABB do ego-vehicle."""
        size = max(EGO_WIDTH, EGO_HEIGHT) // 2
        return pygame.Rect(
            int(self._ego_x - size), int(self._ego_y - size),
            size * 2, size * 2,
        )

    def _check_collision(self) -> bool:
        """Verifica colisão do ego com NPCs e pedestres (AABB)."""
        ego_rect = self._get_ego_rect()
        for npc in self._npc_vehicles:
            if npc.active and ego_rect.colliderect(npc.get_rect()):
                return True
        for ped in self._pedestrians:
            if ped.active and ego_rect.colliderect(ped.get_rect()):
                return True
        return False

    def _check_off_road(self) -> bool:
        """Verifica se o ego saiu da via."""
        return not self._city_map.is_road(self._ego_x, self._ego_y)

    def _check_red_light_violation(self) -> bool:
        """Verifica se o ego está avançando com sinal vermelho.

        Só conta como violação se:
            1. Está no cruzamento
            2. Está se movendo (speed > 0.5)
            3. O semáforo relevante está vermelho
        """
        if self._tl_controller is None:
            return False
        if self._ego_speed < 0.5:
            return False
        if not self._city_map.is_intersection(self._ego_x, self._ego_y):
            return False

        # Determina se está na via NS ou EW pelo heading
        heading_deg = abs(math.degrees(self._ego_heading) % 360)
        is_ns = (60 < heading_deg < 120) or (240 < heading_deg < 300)

        if is_ns:
            # Verifica semáforos NS
            for tl in self._tl_controller.ns_lights:
                if tl.is_red:
                    return True
        else:
            # Verifica semáforos EW
            for tl in self._tl_controller.ew_lights:
                if tl.is_red:
                    return True
        return False

    def _build_ground_truth_detections(self) -> np.ndarray:
        """Constrói vetor de detecções a partir de ground-truth.

        Na Fase 1, simula o output do YOLO usando posições reais.
        Na Fase 3, será substituído pelo pipeline YOLO real.

        Classes: 0=vehicle_npc, 1=traffic_light_red,
                 2=traffic_light_green, 3=pedestrian, 4=road_marking

        Returns:
            Array (MAX_DETECTIONS, 6) normalizado em [0, 1].
        """
        detections: list[list[float]] = []

        # NPCs
        for npc in self._npc_vehicles:
            if not npc.active:
                continue
            rect = npc.get_rect()
            detections.append([
                0.0 / 4.0,                          # class_id normalizado
                rect.centerx / MAP_WIDTH,
                rect.centery / MAP_HEIGHT,
                rect.width / MAP_WIDTH,
                rect.height / MAP_HEIGHT,
                1.0,                                  # confidence
            ])

        # Semáforos
        if self._tl_controller:
            for tl in self._tl_controller.get_all_lights():
                cls_id = 1.0 if tl.is_red else 2.0  # red=1, green=2
                if tl.is_yellow:
                    cls_id = 1.0  # yellow → treat as red for safety
                rect = tl.get_rect()
                detections.append([
                    cls_id / 4.0,
                    rect.centerx / MAP_WIDTH,
                    rect.centery / MAP_HEIGHT,
                    rect.width / MAP_WIDTH,
                    rect.height / MAP_HEIGHT,
                    1.0,
                ])

        # Pedestres
        for ped in self._pedestrians:
            if not ped.active:
                continue
            rect = ped.get_rect()
            detections.append([
                3.0 / 4.0,
                rect.centerx / MAP_WIDTH,
                rect.centery / MAP_HEIGHT,
                rect.width / MAP_WIDTH,
                rect.height / MAP_HEIGHT,
                1.0,
            ])

        # Pad/truncate para MAX_DETECTIONS
        result = np.zeros((MAX_DETECTIONS, 6), dtype=np.float32)
        n = min(len(detections), MAX_DETECTIONS)
        if n > 0:
            result[:n] = np.array(detections[:n], dtype=np.float32)

        return result

    def _get_ego_state(self) -> np.ndarray:
        """Constrói vetor de estado do ego normalizado.

        Returns:
            Array (5,) com [pos_x, pos_y, velocity, heading, dist_to_goal]
            todos normalizados para [0, 1].
        """
        diag = math.sqrt(MAP_WIDTH**2 + MAP_HEIGHT**2)
        return np.array([
            self._ego_x / MAP_WIDTH,
            self._ego_y / MAP_HEIGHT,
            self._ego_speed / EGO_MAX_SPEED,
            (self._ego_heading % (2 * math.pi)) / (2 * math.pi),
            min(self._dist_to_goal() / diag, 1.0),
        ], dtype=np.float32)

    def _get_obs(self) -> dict[str, np.ndarray]:
        """Constrói observação completa (Dict)."""
        return {
            "detections": self._build_ground_truth_detections(),
            "ego_state": self._get_ego_state(),
        }

    def _get_object_distances(self) -> tuple[list[float], list[float]]:
        """Calcula distâncias do ego a todos os objetos.

        Returns:
            (distances, weights) — listas pareadas.
        """
        distances: list[float] = []
        weights: list[float] = []
        for npc in self._npc_vehicles:
            if npc.active:
                d = math.hypot(npc.x - self._ego_x, npc.y - self._ego_y)
                distances.append(d)
                weights.append(1.0)
        for ped in self._pedestrians:
            if ped.active:
                d = math.hypot(ped.x - self._ego_x, ped.y - self._ego_y)
                distances.append(d)
                weights.append(1.5)  # pedestres têm peso maior
        return distances, weights

    # === Gymnasium API ===

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Reseta o ambiente para um novo episódio.

        Args:
            seed: Seed para reprodutibilidade.
            options: Opções extras (não usado).

        Returns:
            (observation, info) tuple.
        """
        super().reset(seed=seed)
        if seed is not None:
            self._rng = random.Random(seed)
            self._np_rng = np.random.RandomState(seed)

        self._step_count = 0
        self._prev_action = None

        # Spawn ego e goal
        self._spawn_ego()
        self._spawn_goal()
        self._prev_dist_to_goal = self._dist_to_goal()

        # Recria NPCs e semáforos
        self._npc_vehicles = create_default_npc_vehicles(self._rng)
        self._pedestrians = create_default_pedestrians(self._rng)
        self._create_traffic_lights()

        obs = self._get_obs()
        info: dict[str, Any] = {
            "goal_x": self._goal_x,
            "goal_y": self._goal_y,
            "ego_x": self._ego_x,
            "ego_y": self._ego_y,
        }

        logger.debug("Reset: ego=(%.0f,%.0f), goal=(%.0f,%.0f)",
                      self._ego_x, self._ego_y, self._goal_x, self._goal_y)

        if self.render_mode == "human":
            self.render()

        return obs, info

    def step(
        self, action: int,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """Executa uma ação no ambiente.

        Args:
            action: Ação discreta (0-4).

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        assert self.action_space.contains(action), f"Ação inválida: {action}"

        self._step_count += 1

        # --- Aplica ação ao ego-vehicle ---
        if action == ACTION_ACCELERATE:
            self._ego_speed = min(
                self._ego_speed + EGO_ACCELERATION, EGO_MAX_SPEED
            )
        elif action == ACTION_BRAKE:
            self._ego_speed = max(self._ego_speed - EGO_DECELERATION, 0.0)
        elif action == ACTION_TURN_LEFT:
            self._ego_heading -= EGO_TURN_RATE
        elif action == ACTION_TURN_RIGHT:
            self._ego_heading += EGO_TURN_RATE
        elif action == ACTION_STOP:
            self._ego_speed = max(self._ego_speed - EGO_DECELERATION * 2, 0.0)

        # Aplica fricção natural
        self._ego_speed = max(self._ego_speed - EGO_FRICTION, 0.0)

        # Atualiza posição
        self._ego_x += self._ego_speed * math.cos(self._ego_heading)
        self._ego_y += self._ego_speed * math.sin(self._ego_heading)

        # Clamp nas bordas do mapa
        self._ego_x = float(np.clip(self._ego_x, 5, MAP_WIDTH - 5))
        self._ego_y = float(np.clip(self._ego_y, 5, MAP_HEIGHT - 5))

        # --- Atualiza NPCs e semáforos ---
        if self._tl_controller:
            self._tl_controller.tick()
        all_lights = (
            self._tl_controller.get_all_lights()
            if self._tl_controller else []
        )
        for npc in self._npc_vehicles:
            npc.update(all_lights)
        for ped in self._pedestrians:
            ped.update()

        # --- Verifica condições terminais ---
        collision = self._check_collision()
        off_road = self._check_off_road()
        dist_to_goal = self._dist_to_goal()
        goal_reached = dist_to_goal < GOAL_THRESHOLD_PX
        red_light = self._check_red_light_violation()

        # --- Calcula recompensa ---
        obj_dists, obj_weights = self._get_object_distances()
        reward, reward_components = compute_reward(
            dist_to_goal_prev=self._prev_dist_to_goal,
            dist_to_goal_curr=dist_to_goal,
            distances_to_objects=obj_dists,
            object_weights=obj_weights,
            red_light_detected=red_light,
            action_prev=self._prev_action,
            action_curr=action,
            collision=collision,
            goal_reached=goal_reached,
            off_road=off_road,
        )

        # --- Atualiza estado ---
        self._prev_dist_to_goal = dist_to_goal
        self._prev_action = action

        # --- Terminação ---
        terminated = bool(collision or goal_reached or off_road)
        truncated = bool(self._step_count >= MAX_STEPS)

        # --- Info ---
        info: dict[str, Any] = {
            "reward_components": reward_components,
            "dist_to_goal": dist_to_goal,
            "collision": collision,
            "off_road": off_road,
            "goal_reached": goal_reached,
            "red_light_violation": red_light,
            "step": self._step_count,
        }

        # --- Render ---
        if self.render_mode == "human":
            self.render()

        return self._get_obs(), float(reward), terminated, truncated, info

    def render(self) -> np.ndarray | None:
        """Renderiza o frame atual.

        Returns:
            Array RGB (H, W, 3) se render_mode='rgb_array', None caso contrário.
        """
        self._init_pygame()

        # Cria surface de renderização
        canvas = pygame.Surface((MAP_WIDTH, MAP_HEIGHT))

        # 1. Mapa base
        self._city_map.draw(canvas)

        # 2. Goal (marcador dourado)
        pygame.draw.circle(
            canvas, (255, 215, 0),
            (int(self._goal_x), int(self._goal_y)), 12,
        )
        pygame.draw.circle(
            canvas, (200, 170, 0),
            (int(self._goal_x), int(self._goal_y)), 12, width=2,
        )

        # 3. NPCs
        for npc in self._npc_vehicles:
            npc.draw(canvas)
        for ped in self._pedestrians:
            ped.draw(canvas)

        # 4. Semáforos
        if self._tl_controller:
            self._tl_controller.draw(canvas)

        # 5. Ego-vehicle (retângulo verde rotacionado)
        ego_surf = pygame.Surface((EGO_WIDTH, EGO_HEIGHT), pygame.SRCALPHA)
        pygame.draw.rect(ego_surf, EGO_COLOR,
                         (0, 0, EGO_WIDTH, EGO_HEIGHT), border_radius=4)
        # Faróis
        pygame.draw.rect(ego_surf, (255, 255, 200),
                         (3, 0, EGO_WIDTH - 6, 3))
        angle_deg = -math.degrees(self._ego_heading) + 90
        rotated = pygame.transform.rotate(ego_surf, angle_deg)
        rect = rotated.get_rect(center=(int(self._ego_x), int(self._ego_y)))
        canvas.blit(rotated, rect)

        # 6. HUD (info text)
        try:
            font = pygame.font.SysFont("Arial", 12)
            hud_lines = [
                f"Step: {self._step_count}/{MAX_STEPS}",
                f"Speed: {self._ego_speed:.1f}",
                f"Dist: {self._dist_to_goal():.0f}",
            ]
            for i, line in enumerate(hud_lines):
                text = font.render(line, True, (255, 255, 255))
                canvas.blit(text, (5, 5 + i * 16))
        except Exception:
            pass

        if self.render_mode == "human" and self._window is not None:
            pygame.event.pump()
            self._window.blit(canvas, (0, 0))
            pygame.display.flip()
            if self._clock:
                self._clock.tick(self.metadata["render_fps"])
            return None

        elif self.render_mode == "rgb_array":
            return np.transpose(
                pygame.surfarray.array3d(canvas), axes=(1, 0, 2)
            )

        return None

    def close(self) -> None:
        """Libera recursos do PyGame."""
        if self._pygame_initialized:
            pygame.quit()
            self._pygame_initialized = False
            self._window = None
            self._clock = None
            logger.info("CityDriveEnv fechado.")


# ── BLOCO DEMO ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    env = CityDriveEnv(render_mode="human")
    obs, info = env.reset(seed=42)
    print(f"Obs keys: {obs.keys()}")
    print(f"Detections shape: {obs['detections'].shape}")
    print(f"Ego state shape: {obs['ego_state'].shape}")
    print(f"Goal: ({info['goal_x']:.0f}, {info['goal_y']:.0f})")

    # Roda 500 steps com agente aleatório
    total_reward = 0.0
    for step in range(500):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            reason = "GOAL!" if info.get("goal_reached") else (
                "COLISÃO" if info.get("collision") else (
                    "OFF-ROAD" if info.get("off_road") else "TIMEOUT"
                )
            )
            print(f"Episódio terminou: {reason} | "
                  f"Steps: {step+1} | Reward: {total_reward:.2f}")
            obs, info = env.reset()
            total_reward = 0.0

    env.close()
    print("Demo CityDriveEnv finalizado com sucesso.")
