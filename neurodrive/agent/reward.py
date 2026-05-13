# ==========================================================================
# MÓDULO: reward.py
# PROPÓSITO: Função de recompensa com potential-based reward shaping
# DECISÃO DE DESIGN: Reward shaping baseado em potencial (Ng et al., 1999)
#   garante que a política ótima do MDP original é preservada (invariância).
#   Alternativa descartada: recompensa esparsa (+100/-100) — convergência
#   extremamente lenta pois o agente não recebe sinal intermediário.
#   Cada componente é aditivo e independente para facilitar ablation study.
# ==========================================================================
from __future__ import annotations

import logging
import math
from typing import Final

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes de Recompensa — todos ajustáveis para experimentos
# ---------------------------------------------------------------------------

# Distância de segurança para objetos (2× largura média de NPC = 2×24 = 48,
# arredondamos para 80px para margem confortável)
SAFE_DISTANCE_PX: Final[float] = 80.0

# Peso da penalidade de segurança por objeto
SAFETY_WEIGHT: Final[float] = 0.5

# Penalidade por avançar sinal vermelho
RED_LIGHT_PENALTY: Final[float] = -10.0

# Penalidade por mudança brusca de ação
SMOOTHNESS_PENALTY: Final[float] = -0.1

# Recompensa terminal
GOAL_REWARD: Final[float] = 100.0
COLLISION_PENALTY: Final[float] = -100.0

# Distância para considerar que chegou ao destino (px)
GOAL_THRESHOLD_PX: Final[float] = 30.0


def potential(dist_to_goal: float) -> float:
    r"""Função de potencial para reward shaping.

    Definida como o negativo da distância ao goal normalizada:
        φ(s) = −dist(s, goal) / MAP_DIAGONAL

    A normalização por MAP_DIAGONAL (~905 px para 640×640) mantém
    φ ∈ [-1, 0], evitando que a escala da recompensa de progresso
    domine os demais componentes.

    Args:
        dist_to_goal: Distância euclidiana ao destino em pixels.

    Returns:
        Valor de potencial normalizado em [-1, 0].

    Note (decisão de design):
        Usamos normalização em vez de potencial bruto porque o PPO
        é sensível à escala de recompensa. Sem normalização, r_progress
        poderia atingir ~905 por step, dominando todos os outros sinais
        e efetivamente tornando a recompensa unidimensional.
    """
    map_diagonal = math.sqrt(640**2 + 640**2)  # ~905.1
    return -dist_to_goal / map_diagonal


def compute_reward(
    dist_to_goal_prev: float,
    dist_to_goal_curr: float,
    distances_to_objects: list[float],
    object_weights: list[float] | None,
    red_light_detected: bool,
    action_prev: int | None,
    action_curr: int,
    collision: bool,
    goal_reached: bool,
    off_road: bool,
) -> tuple[float, dict[str, float]]:
    """Calcula a recompensa total com 5 componentes independentes.

    Componentes:
        1. r_progress: φ(s') − φ(s) — progresso em direção ao goal
        2. r_safety: penalidade por proximidade a objetos
        3. r_traffic: penalidade por avançar sinal vermelho
        4. r_smooth: penalidade por mudança brusca de ação
        5. r_terminal: +100 (goal) ou -100 (colisão)

    Args:
        dist_to_goal_prev: Distância ao goal no step anterior (px).
        dist_to_goal_curr: Distância ao goal no step atual (px).
        distances_to_objects: Distâncias a cada objeto detectado (px).
        object_weights: Peso da penalidade por tipo de objeto.
            Se None, usa 1.0 para todos.
        red_light_detected: True se semáforo vermelho foi detectado
            e o agente está avançando (não parado).
        action_prev: Ação do step anterior (None no primeiro step).
        action_curr: Ação do step atual.
        collision: True se houve colisão (ego vs NPC ou ego vs calçada).
        goal_reached: True se o agente atingiu o destino.
        off_road: True se o agente saiu da via.

    Returns:
        Tupla (reward_total, componentes_dict) onde componentes_dict
        contém cada componente nomeado para logging/ablation.

    Note (decisão de design):
        Retornamos o dicionário de componentes junto com o total para
        permitir logging granular no TensorBoard. Isso é essencial
        para diagnosticar reward hacking — se um componente domina os
        demais, sabemos exatamente qual ajustar. Aprendemos isso no
        projeto anterior onde o agente fazia "donuts" para exploitar
        a recompensa de velocidade.
    """
    components: dict[str, float] = {}

    # ── 1. Progresso (potential-based) ──
    r_progress = potential(dist_to_goal_curr) - potential(dist_to_goal_prev)
    components["r_progress"] = r_progress

    # ── 2. Segurança (proximidade a objetos) ──
    r_safety = 0.0
    if distances_to_objects:
        if object_weights is None:
            object_weights = [1.0] * len(distances_to_objects)
        for dist, weight in zip(distances_to_objects, object_weights):
            violation = max(0.0, SAFE_DISTANCE_PX - dist)
            r_safety -= violation * SAFETY_WEIGHT * weight
    # Normaliza pelo número de objetos para não penalizar cenários densos
    if distances_to_objects:
        r_safety /= len(distances_to_objects)
    components["r_safety"] = r_safety

    # ── 3. Trânsito (semáforo vermelho) ──
    r_traffic = RED_LIGHT_PENALTY if red_light_detected else 0.0
    components["r_traffic"] = r_traffic

    # ── 4. Suavidade (mudança de ação) ──
    r_smooth = 0.0
    if action_prev is not None and action_prev != action_curr:
        r_smooth = SMOOTHNESS_PENALTY
    components["r_smooth"] = r_smooth

    # ── 5. Terminal ──
    r_terminal = 0.0
    if goal_reached:
        r_terminal = GOAL_REWARD
    elif collision:
        r_terminal = COLLISION_PENALTY
    elif off_road:
        r_terminal = COLLISION_PENALTY * 0.5  # menos severo que colisão
    components["r_terminal"] = r_terminal

    # ── Total ──
    r_total = r_progress + r_safety + r_traffic + r_smooth + r_terminal
    components["r_total"] = r_total

    return r_total, components


# ── BLOCO DEMO ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Cenário 1: Agente se aproximando do goal
    r, c = compute_reward(
        dist_to_goal_prev=200.0,
        dist_to_goal_curr=180.0,
        distances_to_objects=[100.0, 50.0],
        object_weights=None,
        red_light_detected=False,
        action_prev=0,
        action_curr=0,
        collision=False,
        goal_reached=False,
        off_road=False,
    )
    print(f"Cenário 1 (aproximando): r={r:.4f}, componentes={c}")

    # Cenário 2: Agente avançou sinal vermelho
    r, c = compute_reward(
        dist_to_goal_prev=200.0,
        dist_to_goal_curr=195.0,
        distances_to_objects=[],
        object_weights=None,
        red_light_detected=True,
        action_prev=0,
        action_curr=0,
        collision=False,
        goal_reached=False,
        off_road=False,
    )
    print(f"Cenário 2 (sinal vermelho): r={r:.4f}, componentes={c}")

    # Cenário 3: Agente atingiu o goal
    r, c = compute_reward(
        dist_to_goal_prev=35.0,
        dist_to_goal_curr=25.0,
        distances_to_objects=[],
        object_weights=None,
        red_light_detected=False,
        action_prev=0,
        action_curr=4,
        collision=False,
        goal_reached=True,
        off_road=False,
    )
    print(f"Cenário 3 (goal atingido): r={r:.4f}, componentes={c}")

    print("\nDemo reward.py finalizado com sucesso.")
