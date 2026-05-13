# ==========================================================================
# MÓDULO: traffic_light.py
# PROPÓSITO: Máquina de Estados Finitos (FSM) para semáforos urbanos
# DECISÃO DE DESIGN: Ciclo baseado em contagem de frames (não wall-clock)
#   para garantir reprodutibilidade determinística no treinamento RL.
#   Alternativa descartada: time.time() — introduz não-determinismo entre
#   máquinas com clock speeds diferentes, quebrando seed-based replay.
# ==========================================================================
from __future__ import annotations

import enum
import logging
from typing import Final

import pygame

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes — durações em frames (a 30 FPS lógicos)
# ---------------------------------------------------------------------------
GREEN_DURATION: Final[int] = 150    # 5 segundos @ 30 FPS
YELLOW_DURATION: Final[int] = 45    # 1.5 segundos @ 30 FPS
RED_DURATION: Final[int] = 195      # = GREEN + YELLOW do oposto

# Dimensões visuais do semáforo (em pixels)
LIGHT_RADIUS: Final[int] = 6
POLE_WIDTH: Final[int] = 16
POLE_HEIGHT: Final[int] = 48

# Cores
COLOR_RED: Final[tuple[int, int, int]] = (220, 40, 40)
COLOR_YELLOW: Final[tuple[int, int, int]] = (240, 200, 30)
COLOR_GREEN: Final[tuple[int, int, int]] = (40, 200, 60)
COLOR_DARK: Final[tuple[int, int, int]] = (60, 60, 60)
COLOR_POLE: Final[tuple[int, int, int]] = (90, 90, 90)


class TrafficLightState(enum.Enum):
    """Estados possíveis de um semáforo.

    A transição segue a FSM clássica brasileira:
        GREEN → YELLOW → RED → GREEN → ...

    Note (decisão de design):
        Não incluímos estados como YELLOW_BLINKING (intermitente noturno)
        pois o agente não precisa lidar com essa complexidade no escopo
        do TCC. O modelo pode ser estendido futuramente.
    """

    RED = "red"
    YELLOW = "yellow"
    GREEN = "green"


class TrafficLight:
    """Semáforo individual com FSM baseada em frames.

    Cada instância mantém seu próprio contador de frames e transiciona
    automaticamente entre os estados quando o contador expira.

    Args:
        x: Posição X do centro do semáforo no mapa (pixels).
        y: Posição Y do centro do semáforo no mapa (pixels).
        initial_state: Estado inicial do semáforo.
        green_duration: Duração do estado verde em frames.
        yellow_duration: Duração do estado amarelo em frames.
        red_duration: Duração do estado vermelho em frames.

    Note (decisão de design):
        O semáforo armazena a posição em pixels (não em tiles) porque
        a renderização e a detecção YOLO operam em coordenadas de pixel.
        Converter de/para tiles adicionaria uma camada de abstração
        desnecessária sem benefício funcional.
    """

    def __init__(
        self,
        x: float,
        y: float,
        initial_state: TrafficLightState = TrafficLightState.RED,
        green_duration: int = GREEN_DURATION,
        yellow_duration: int = YELLOW_DURATION,
        red_duration: int = RED_DURATION,
    ) -> None:
        self.x = x
        self.y = y
        self.state = initial_state
        self._green_duration = green_duration
        self._yellow_duration = yellow_duration
        self._red_duration = red_duration
        self._frame_counter: int = 0

        logger.debug(
            "Semáforo criado em (%.0f, %.0f) com estado inicial %s",
            x, y, initial_state.value,
        )

    @property
    def is_red(self) -> bool:
        """Retorna True se o semáforo está vermelho."""
        return self.state == TrafficLightState.RED

    @property
    def is_green(self) -> bool:
        """Retorna True se o semáforo está verde."""
        return self.state == TrafficLightState.GREEN

    @property
    def is_yellow(self) -> bool:
        """Retorna True se o semáforo está amarelo."""
        return self.state == TrafficLightState.YELLOW

    def _current_duration(self) -> int:
        """Retorna a duração do estado atual em frames."""
        if self.state == TrafficLightState.GREEN:
            return self._green_duration
        elif self.state == TrafficLightState.YELLOW:
            return self._yellow_duration
        else:
            return self._red_duration

    def _next_state(self) -> TrafficLightState:
        """Retorna o próximo estado na FSM.

        Transição: GREEN → YELLOW → RED → GREEN.
        """
        if self.state == TrafficLightState.GREEN:
            return TrafficLightState.YELLOW
        elif self.state == TrafficLightState.YELLOW:
            return TrafficLightState.RED
        else:
            return TrafficLightState.GREEN

    def tick(self) -> None:
        """Avança o semáforo em 1 frame.

        Quando o contador atinge a duração do estado atual, transiciona
        para o próximo estado e reseta o contador.

        Note (decisão de design):
            Usamos contagem incremental (não decremental) por clareza.
            O custo computacional é idêntico — 1 comparação por frame.
        """
        self._frame_counter += 1
        if self._frame_counter >= self._current_duration():
            old_state = self.state
            self.state = self._next_state()
            self._frame_counter = 0
            logger.debug(
                "Semáforo (%.0f, %.0f): %s → %s",
                self.x, self.y, old_state.value, self.state.value,
            )

    def draw(self, surface: pygame.Surface) -> None:
        """Renderiza o semáforo no surface do PyGame.

        Desenha um poste cinza com 3 círculos (vermelho, amarelo, verde).
        O estado ativo brilha intensamente; os demais ficam escuros.

        Args:
            surface: Surface do PyGame onde desenhar.
        """
        # Poste (retângulo cinza)
        pole_rect = pygame.Rect(
            int(self.x - POLE_WIDTH // 2),
            int(self.y - POLE_HEIGHT // 2),
            POLE_WIDTH,
            POLE_HEIGHT,
        )
        pygame.draw.rect(surface, COLOR_POLE, pole_rect, border_radius=3)

        # Posições dos 3 círculos (de cima para baixo: R, Y, G)
        cx = int(self.x)
        spacing = POLE_HEIGHT // 4
        positions = [
            (cx, int(self.y - spacing)),       # Red (topo)
            (cx, int(self.y)),                  # Yellow (meio)
            (cx, int(self.y + spacing)),        # Green (base)
        ]

        # Cores: apagadas (escuro) ou acesas
        colors_off = [COLOR_DARK, COLOR_DARK, COLOR_DARK]
        if self.state == TrafficLightState.RED:
            colors_off[0] = COLOR_RED
        elif self.state == TrafficLightState.YELLOW:
            colors_off[1] = COLOR_YELLOW
        elif self.state == TrafficLightState.GREEN:
            colors_off[2] = COLOR_GREEN

        for (px, py), color in zip(positions, colors_off):
            pygame.draw.circle(surface, color, (px, py), LIGHT_RADIUS)

    def get_rect(self) -> pygame.Rect:
        """Retorna o retângulo de colisão do semáforo (para YOLO bbox)."""
        return pygame.Rect(
            int(self.x - POLE_WIDTH // 2),
            int(self.y - POLE_HEIGHT // 2),
            POLE_WIDTH,
            POLE_HEIGHT,
        )


class TrafficLightController:
    """Controlador de semáforos sincronizados em um cruzamento.

    Garante que semáforos em vias perpendiculares estejam sempre em
    estados opostos: quando N-S é verde, E-W é vermelho (e vice-versa).

    Args:
        ns_lights: Lista de semáforos da via Norte-Sul.
        ew_lights: Lista de semáforos da via Leste-Oeste.

    Note (decisão de design):
        Em vez de cada semáforo rodar sua FSM independentemente (o que
        exigiria sincronização complexa), o Controller é a única fonte
        de verdade. Ele inicializa N-S como GREEN e E-W como RED, e
        propaga tick() para todos de uma vez. Isso garante sincronia
        perfeita sem race conditions, mesmo com seeds aleatórias.
    """

    def __init__(
        self,
        ns_lights: list[TrafficLight],
        ew_lights: list[TrafficLight],
    ) -> None:
        self.ns_lights = ns_lights
        self.ew_lights = ew_lights

        # Inicializa N-S como verde, E-W como vermelho
        for light in self.ns_lights:
            light.state = TrafficLightState.GREEN
            light._frame_counter = 0
        for light in self.ew_lights:
            light.state = TrafficLightState.RED
            light._frame_counter = 0

        logger.info(
            "TrafficLightController: %d NS (green), %d EW (red)",
            len(ns_lights), len(ew_lights),
        )

    def tick(self) -> None:
        """Avança todos os semáforos em 1 frame.

        Como todos começam sincronizados e usam as mesmas durações,
        a sincronia se mantém automaticamente.
        """
        for light in self.ns_lights:
            light.tick()
        for light in self.ew_lights:
            light.tick()

    def get_all_lights(self) -> list[TrafficLight]:
        """Retorna todos os semáforos gerenciados."""
        return self.ns_lights + self.ew_lights

    def draw(self, surface: pygame.Surface) -> None:
        """Renderiza todos os semáforos no surface."""
        for light in self.get_all_lights():
            light.draw(surface)


# ── BLOCO DEMO ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    pygame.init()
    screen = pygame.display.set_mode((300, 300))
    pygame.display.set_caption("NeuroDrive — TrafficLight Demo")
    clock = pygame.time.Clock()

    # Cria 2 semáforos NS e 2 EW
    ns = [TrafficLight(100, 150), TrafficLight(200, 150)]
    ew = [TrafficLight(150, 100), TrafficLight(150, 200)]
    controller = TrafficLightController(ns, ew)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        controller.tick()

        screen.fill((40, 40, 40))
        controller.draw(screen)
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()
    print("Demo finalizado com sucesso.")
