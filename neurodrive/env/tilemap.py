# ==========================================================================
# MÓDULO: tilemap.py
# PROPÓSITO: Renderização do mapa urbano 2D (cruzamento com 4 quadrantes)
# DECISÃO DE DESIGN: Mapa 640x640 px desenhado programaticamente com
#   pygame.draw em vez de tilesets/TMX. Justificativa: (1) resolução
#   640x640 é nativa do YOLOv8, evitando resize; (2) sem dependência
#   externa de assets — reprodutibilidade total; (3) mapa estático não
#   justifica engine de tiles como Tiled/TMX.
# ==========================================================================
from __future__ import annotations

import logging
from typing import Final

import pygame

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes de Layout (em pixels)
# ---------------------------------------------------------------------------
MAP_WIDTH: Final[int] = 640
MAP_HEIGHT: Final[int] = 640

# Via: faixa central do cruzamento
ROAD_WIDTH: Final[int] = 80                 # largura de cada via
ROAD_CENTER_X: Final[int] = MAP_WIDTH // 2   # 320
ROAD_CENTER_Y: Final[int] = MAP_HEIGHT // 2  # 320

# Limites das vias (onde começa/termina o asfalto)
ROAD_NS_LEFT: Final[int] = ROAD_CENTER_X - ROAD_WIDTH // 2    # 280
ROAD_NS_RIGHT: Final[int] = ROAD_CENTER_X + ROAD_WIDTH // 2   # 360
ROAD_EW_TOP: Final[int] = ROAD_CENTER_Y - ROAD_WIDTH // 2     # 280
ROAD_EW_BOTTOM: Final[int] = ROAD_CENTER_Y + ROAD_WIDTH // 2  # 360

# Cores
COLOR_GRASS: Final[tuple[int, int, int]] = (90, 155, 70)
COLOR_ROAD: Final[tuple[int, int, int]] = (55, 55, 55)
COLOR_SIDEWALK: Final[tuple[int, int, int]] = (170, 165, 150)
COLOR_ROAD_LINE: Final[tuple[int, int, int]] = (200, 200, 200)
COLOR_CROSSWALK: Final[tuple[int, int, int]] = (220, 220, 220)
COLOR_BUILDING_LOJA: Final[tuple[int, int, int]] = (160, 140, 110)
COLOR_BUILDING_GAS: Final[tuple[int, int, int]] = (140, 140, 155)
COLOR_BUILDING_PRACA: Final[tuple[int, int, int]] = (70, 140, 60)
COLOR_BUILDING_PIZZA: Final[tuple[int, int, int]] = (155, 120, 100)
COLOR_AWNING_BLUE: Final[tuple[int, int, int]] = (60, 100, 180)
COLOR_AWNING_RED: Final[tuple[int, int, int]] = (180, 50, 50)
COLOR_FOUNTAIN: Final[tuple[int, int, int]] = (100, 170, 220)
COLOR_TREE_TRUNK: Final[tuple[int, int, int]] = (100, 70, 40)
COLOR_TREE_CROWN: Final[tuple[int, int, int]] = (50, 130, 50)
COLOR_GAS_PUMP: Final[tuple[int, int, int]] = (200, 60, 60)

# Largura da calçada
SIDEWALK_WIDTH: Final[int] = 20


class CityMap:
    """Mapa urbano 640x640 com cruzamento central e 4 quadrantes.

    Layout:
        NW: Loja (toldo azul listrado)
        NE: Posto de gasolina (bombas)
        SW: Praça pública (fonte + árvores)
        SE: Pizzaria (toldo vermelho listrado)

    As vias formam uma cruz "+" central com 80px de largura. Calçadas
    de 20px margeiam cada via. Os quadrantes preenchem o restante.

    Note (decisão de design):
        Tudo é desenhado com pygame.draw, sem sprites externos. Isso
        garante reprodutibilidade em qualquer máquina sem necessidade
        de distribuir assets gráficos. As formas geométricas simples
        são suficientes para gerar dados de treino YOLO (que detecta
        shapes, não texturas fotorrealistas).
    """

    def __init__(self) -> None:
        self._surface: pygame.Surface | None = None
        logger.info("CityMap inicializado (%dx%d)", MAP_WIDTH, MAP_HEIGHT)

    def _build_surface(self) -> pygame.Surface:
        """Constrói o surface estático do mapa (chamado uma vez).

        Returns:
            Surface com o mapa completo renderizado.
        """
        surf = pygame.Surface((MAP_WIDTH, MAP_HEIGHT))

        # 1. Fundo — grama
        surf.fill(COLOR_GRASS)

        # 2. Vias (asfalto)
        # Via Norte-Sul
        pygame.draw.rect(surf, COLOR_ROAD,
                         (ROAD_NS_LEFT, 0, ROAD_WIDTH, MAP_HEIGHT))
        # Via Leste-Oeste
        pygame.draw.rect(surf, COLOR_ROAD,
                         (0, ROAD_EW_TOP, MAP_WIDTH, ROAD_WIDTH))

        # 3. Calçadas (margens das vias)
        self._draw_sidewalks(surf)

        # 4. Linha central tracejada (divisória de faixas)
        self._draw_road_lines(surf)

        # 5. Faixas de pedestres
        self._draw_crosswalks(surf)

        # 6. Quadrantes (edifícios e elementos)
        self._draw_nw_shop(surf)
        self._draw_ne_gas_station(surf)
        self._draw_sw_park(surf)
        self._draw_se_pizzeria(surf)

        return surf

    def _draw_sidewalks(self, surf: pygame.Surface) -> None:
        """Desenha calçadas ao redor das vias."""
        sw = SIDEWALK_WIDTH
        # Calçada oeste da via NS
        pygame.draw.rect(surf, COLOR_SIDEWALK,
                         (ROAD_NS_LEFT - sw, 0, sw, MAP_HEIGHT))
        # Calçada leste da via NS
        pygame.draw.rect(surf, COLOR_SIDEWALK,
                         (ROAD_NS_RIGHT, 0, sw, MAP_HEIGHT))
        # Calçada norte da via EW
        pygame.draw.rect(surf, COLOR_SIDEWALK,
                         (0, ROAD_EW_TOP - sw, MAP_WIDTH, sw))
        # Calçada sul da via EW
        pygame.draw.rect(surf, COLOR_SIDEWALK,
                         (0, ROAD_EW_BOTTOM, MAP_WIDTH, sw))

    def _draw_road_lines(self, surf: pygame.Surface) -> None:
        """Desenha linhas centrais tracejadas nas vias."""
        dash_len = 20
        gap_len = 15
        center_x = ROAD_CENTER_X
        center_y = ROAD_CENTER_Y

        # Linha central da via NS (acima e abaixo do cruzamento)
        for y in range(0, ROAD_EW_TOP - 10, dash_len + gap_len):
            pygame.draw.line(surf, COLOR_ROAD_LINE,
                             (center_x, y), (center_x, min(y + dash_len, ROAD_EW_TOP - 10)), 2)
        for y in range(ROAD_EW_BOTTOM + 10, MAP_HEIGHT, dash_len + gap_len):
            pygame.draw.line(surf, COLOR_ROAD_LINE,
                             (center_x, y), (center_x, min(y + dash_len, MAP_HEIGHT)), 2)

        # Linha central da via EW (esquerda e direita do cruzamento)
        for x in range(0, ROAD_NS_LEFT - 10, dash_len + gap_len):
            pygame.draw.line(surf, COLOR_ROAD_LINE,
                             (x, center_y), (min(x + dash_len, ROAD_NS_LEFT - 10), center_y), 2)
        for x in range(ROAD_NS_RIGHT + 10, MAP_WIDTH, dash_len + gap_len):
            pygame.draw.line(surf, COLOR_ROAD_LINE,
                             (x, center_y), (min(x + dash_len, MAP_WIDTH), center_y), 2)

    def _draw_crosswalks(self, surf: pygame.Surface) -> None:
        """Desenha faixas de pedestres nos 4 lados do cruzamento."""
        stripe_w = 6
        stripe_gap = 4
        cw_depth = 15   # profundidade da faixa na via

        # Faixa Norte (topo do cruzamento)
        y_start = ROAD_EW_TOP - cw_depth
        for x in range(ROAD_NS_LEFT + 4, ROAD_NS_RIGHT - 4, stripe_w + stripe_gap):
            pygame.draw.rect(surf, COLOR_CROSSWALK,
                             (x, y_start, stripe_w, cw_depth))

        # Faixa Sul (base do cruzamento)
        y_start = ROAD_EW_BOTTOM
        for x in range(ROAD_NS_LEFT + 4, ROAD_NS_RIGHT - 4, stripe_w + stripe_gap):
            pygame.draw.rect(surf, COLOR_CROSSWALK,
                             (x, y_start, stripe_w, cw_depth))

        # Faixa Oeste (esquerda do cruzamento)
        x_start = ROAD_NS_LEFT - cw_depth
        for y in range(ROAD_EW_TOP + 4, ROAD_EW_BOTTOM - 4, stripe_w + stripe_gap):
            pygame.draw.rect(surf, COLOR_CROSSWALK,
                             (x_start, y, cw_depth, stripe_w))

        # Faixa Leste (direita do cruzamento)
        x_start = ROAD_NS_RIGHT
        for y in range(ROAD_EW_TOP + 4, ROAD_EW_BOTTOM - 4, stripe_w + stripe_gap):
            pygame.draw.rect(surf, COLOR_CROSSWALK,
                             (x_start, y, cw_depth, stripe_w))

    def _draw_awning(
        self, surf: pygame.Surface,
        x: int, y: int, w: int, h: int,
        color: tuple[int, int, int],
    ) -> None:
        """Desenha toldo listrado sobre um edifício."""
        stripe_w = 8
        for sx in range(x, x + w, stripe_w * 2):
            sw = min(stripe_w, x + w - sx)
            pygame.draw.rect(surf, color, (sx, y, sw, h))
        for sx in range(x + stripe_w, x + w, stripe_w * 2):
            sw = min(stripe_w, x + w - sx)
            pygame.draw.rect(surf, (255, 255, 255), (sx, y, sw, h))

    def _draw_nw_shop(self, surf: pygame.Surface) -> None:
        """Quadrante NW: Loja com toldo azul listrado."""
        qx, qy = 30, 30
        qw, qh = ROAD_NS_LEFT - SIDEWALK_WIDTH - 40, ROAD_EW_TOP - SIDEWALK_WIDTH - 40

        # Edifício
        pygame.draw.rect(surf, COLOR_BUILDING_LOJA, (qx, qy, qw, qh), border_radius=5)
        pygame.draw.rect(surf, (120, 100, 80), (qx, qy, qw, qh), width=2, border_radius=5)

        # Toldo azul
        self._draw_awning(surf, qx, qy + qh - 12, qw, 12, COLOR_AWNING_BLUE)

        # Texto "LOJA"
        try:
            font = pygame.font.SysFont("Arial", 14, bold=True)
            text = font.render("LOJA", True, (255, 255, 255))
            surf.blit(text, (qx + qw // 2 - text.get_width() // 2, qy + qh // 2 - 8))
        except Exception:
            pass

    def _draw_ne_gas_station(self, surf: pygame.Surface) -> None:
        """Quadrante NE: Posto de gasolina com bombas."""
        qx = ROAD_NS_RIGHT + SIDEWALK_WIDTH + 10
        qy = 30
        qw = MAP_WIDTH - qx - 30
        qh = ROAD_EW_TOP - SIDEWALK_WIDTH - 40

        # Cobertura do posto
        pygame.draw.rect(surf, COLOR_BUILDING_GAS, (qx, qy, qw, qh), border_radius=5)
        pygame.draw.rect(surf, (100, 100, 120), (qx, qy, qw, qh), width=2, border_radius=5)

        # Bombas de gasolina (retângulos vermelhos)
        pump_w, pump_h = 10, 20
        for i in range(3):
            px = qx + 20 + i * 30
            py = qy + qh // 2 - pump_h // 2
            pygame.draw.rect(surf, COLOR_GAS_PUMP, (px, py, pump_w, pump_h), border_radius=2)

        # Texto "GAS"
        try:
            font = pygame.font.SysFont("Arial", 14, bold=True)
            text = font.render("GAS", True, (255, 255, 255))
            surf.blit(text, (qx + qw // 2 - text.get_width() // 2, qy + 10))
        except Exception:
            pass

    def _draw_sw_park(self, surf: pygame.Surface) -> None:
        """Quadrante SW: Praça pública com fonte e árvores."""
        qx = 30
        qy = ROAD_EW_BOTTOM + SIDEWALK_WIDTH + 10
        qw = ROAD_NS_LEFT - SIDEWALK_WIDTH - 40
        qh = MAP_HEIGHT - qy - 30

        # Gramado da praça (mais claro)
        pygame.draw.rect(surf, (100, 170, 80), (qx, qy, qw, qh), border_radius=8)

        # Fonte central
        cx, cy = qx + qw // 2, qy + qh // 2
        pygame.draw.circle(surf, COLOR_FOUNTAIN, (cx, cy), 18)
        pygame.draw.circle(surf, (130, 200, 240), (cx, cy), 10)
        pygame.draw.circle(surf, (200, 230, 255), (cx, cy), 4)

        # Árvores
        tree_positions = [
            (qx + 25, qy + 25), (qx + qw - 25, qy + 25),
            (qx + 25, qy + qh - 25), (qx + qw - 25, qy + qh - 25),
        ]
        for tx, ty in tree_positions:
            pygame.draw.rect(surf, COLOR_TREE_TRUNK, (tx - 3, ty, 6, 12))
            pygame.draw.circle(surf, COLOR_TREE_CROWN, (tx, ty), 12)

    def _draw_se_pizzeria(self, surf: pygame.Surface) -> None:
        """Quadrante SE: Pizzaria com toldo vermelho listrado."""
        qx = ROAD_NS_RIGHT + SIDEWALK_WIDTH + 10
        qy = ROAD_EW_BOTTOM + SIDEWALK_WIDTH + 10
        qw = MAP_WIDTH - qx - 30
        qh = MAP_HEIGHT - qy - 30

        # Edifício
        pygame.draw.rect(surf, COLOR_BUILDING_PIZZA, (qx, qy, qw, qh), border_radius=5)
        pygame.draw.rect(surf, (120, 80, 60), (qx, qy, qw, qh), width=2, border_radius=5)

        # Toldo vermelho
        self._draw_awning(surf, qx, qy, qw, 12, COLOR_AWNING_RED)

        # Texto "PIZZA"
        try:
            font = pygame.font.SysFont("Arial", 14, bold=True)
            text = font.render("PIZZA", True, (255, 255, 255))
            surf.blit(text, (qx + qw // 2 - text.get_width() // 2, qy + qh // 2 - 8))
        except Exception:
            pass

    def get_surface(self) -> pygame.Surface:
        """Retorna o surface do mapa (lazy-build).

        O mapa é construído uma única vez e cacheado. Chamadas
        subsequentes retornam o mesmo surface sem recalcular.

        Returns:
            Surface 640x640 com o mapa completo.
        """
        if self._surface is None:
            self._surface = self._build_surface()
        return self._surface

    def draw(self, target: pygame.Surface) -> None:
        """Blita o mapa no surface alvo.

        Args:
            target: Surface destino (geralmente a tela).
        """
        target.blit(self.get_surface(), (0, 0))

    def is_road(self, x: float, y: float) -> bool:
        """Verifica se a posição (x,y) está sobre a via (asfalto).

        Args:
            x: Coordenada X em pixels.
            y: Coordenada Y em pixels.

        Returns:
            True se a posição está na via (NS ou EW).
        """
        on_ns = ROAD_NS_LEFT <= x <= ROAD_NS_RIGHT
        on_ew = ROAD_EW_TOP <= y <= ROAD_EW_BOTTOM
        return bool(on_ns or on_ew)

    def is_intersection(self, x: float, y: float) -> bool:
        """Verifica se a posição está no cruzamento central.

        Args:
            x: Coordenada X em pixels.
            y: Coordenada Y em pixels.

        Returns:
            True se está na interseção das duas vias.
        """
        return bool(
            ROAD_NS_LEFT <= x <= ROAD_NS_RIGHT
            and ROAD_EW_TOP <= y <= ROAD_EW_BOTTOM
        )


# ── BLOCO DEMO ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pygame.init()
    screen = pygame.display.set_mode((MAP_WIDTH, MAP_HEIGHT))
    pygame.display.set_caption("NeuroDrive — CityMap Demo")
    clock = pygame.time.Clock()

    city_map = CityMap()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        city_map.draw(screen)
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()
    print("Demo CityMap finalizado com sucesso.")
