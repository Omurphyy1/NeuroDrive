# ==========================================================================
# MÓDULO: test_reward.py
# PROPÓSITO: Testes unitários para a função de recompensa
# ==========================================================================
from __future__ import annotations

import math

import pytest

from neurodrive.agent.reward import (
    COLLISION_PENALTY,
    GOAL_REWARD,
    RED_LIGHT_PENALTY,
    SAFE_DISTANCE_PX,
    SMOOTHNESS_PENALTY,
    compute_reward,
    potential,
)


class TestPotential:
    """Testes para a função de potencial φ(s) = −dist/diagonal."""

    def test_zero_distance_returns_zero(self) -> None:
        """φ(0) = 0 (no goal)."""
        assert potential(0.0) == 0.0

    def test_positive_distance_returns_negative(self) -> None:
        """φ(d) < 0 para d > 0."""
        assert potential(100.0) < 0.0

    def test_larger_distance_more_negative(self) -> None:
        """φ(d1) < φ(d2) quando d1 > d2 (monótona decrescente)."""
        assert potential(500.0) < potential(200.0)

    def test_normalized_range(self) -> None:
        """φ deve estar em [-1, 0] para qualquer distância no mapa."""
        diag = math.sqrt(640**2 + 640**2)
        val = potential(diag)
        assert -1.01 <= val <= 0.0  # tolerância flutuante


class TestComputeReward:
    """Testes para compute_reward com cenários isolados."""

    def test_progress_positive_when_approaching(self) -> None:
        """r_progress > 0 quando o agente se aproxima do goal."""
        r, c = compute_reward(
            dist_to_goal_prev=200.0,
            dist_to_goal_curr=180.0,
            distances_to_objects=[],
            object_weights=None,
            red_light_detected=False,
            action_prev=0,
            action_curr=0,
            collision=False,
            goal_reached=False,
            off_road=False,
        )
        assert c["r_progress"] > 0.0

    def test_progress_negative_when_retreating(self) -> None:
        """r_progress < 0 quando o agente se afasta do goal."""
        r, c = compute_reward(
            dist_to_goal_prev=180.0,
            dist_to_goal_curr=200.0,
            distances_to_objects=[],
            object_weights=None,
            red_light_detected=False,
            action_prev=0,
            action_curr=0,
            collision=False,
            goal_reached=False,
            off_road=False,
        )
        assert c["r_progress"] < 0.0

    def test_safety_penalty_when_too_close(self) -> None:
        """r_safety < 0 quando dist_to_object < SAFE_DISTANCE_PX."""
        r, c = compute_reward(
            dist_to_goal_prev=200.0,
            dist_to_goal_curr=200.0,
            distances_to_objects=[30.0],  # < 80px threshold
            object_weights=None,
            red_light_detected=False,
            action_prev=0,
            action_curr=0,
            collision=False,
            goal_reached=False,
            off_road=False,
        )
        assert c["r_safety"] < 0.0

    def test_no_safety_penalty_when_far(self) -> None:
        """r_safety = 0 quando todos os objetos estão longe."""
        r, c = compute_reward(
            dist_to_goal_prev=200.0,
            dist_to_goal_curr=200.0,
            distances_to_objects=[200.0, 300.0],
            object_weights=None,
            red_light_detected=False,
            action_prev=0,
            action_curr=0,
            collision=False,
            goal_reached=False,
            off_road=False,
        )
        assert c["r_safety"] == 0.0

    def test_red_light_penalty(self) -> None:
        """r_traffic = RED_LIGHT_PENALTY quando viola sinal vermelho."""
        r, c = compute_reward(
            dist_to_goal_prev=200.0,
            dist_to_goal_curr=200.0,
            distances_to_objects=[],
            object_weights=None,
            red_light_detected=True,
            action_prev=0,
            action_curr=0,
            collision=False,
            goal_reached=False,
            off_road=False,
        )
        assert c["r_traffic"] == RED_LIGHT_PENALTY

    def test_no_red_light_penalty(self) -> None:
        """r_traffic = 0 sem violação de sinal."""
        r, c = compute_reward(
            dist_to_goal_prev=200.0,
            dist_to_goal_curr=200.0,
            distances_to_objects=[],
            object_weights=None,
            red_light_detected=False,
            action_prev=0,
            action_curr=0,
            collision=False,
            goal_reached=False,
            off_road=False,
        )
        assert c["r_traffic"] == 0.0

    def test_smoothness_penalty_on_action_change(self) -> None:
        """r_smooth < 0 quando muda de ação consecutivamente."""
        r, c = compute_reward(
            dist_to_goal_prev=200.0,
            dist_to_goal_curr=200.0,
            distances_to_objects=[],
            object_weights=None,
            red_light_detected=False,
            action_prev=0,
            action_curr=1,  # mudou de ACELERAR para FREAR
            collision=False,
            goal_reached=False,
            off_road=False,
        )
        assert c["r_smooth"] == SMOOTHNESS_PENALTY

    def test_no_smoothness_penalty_same_action(self) -> None:
        """r_smooth = 0 quando mantém a mesma ação."""
        r, c = compute_reward(
            dist_to_goal_prev=200.0,
            dist_to_goal_curr=200.0,
            distances_to_objects=[],
            object_weights=None,
            red_light_detected=False,
            action_prev=0,
            action_curr=0,
            collision=False,
            goal_reached=False,
            off_road=False,
        )
        assert c["r_smooth"] == 0.0

    def test_goal_reached_terminal(self) -> None:
        """r_terminal = GOAL_REWARD ao atingir destino."""
        r, c = compute_reward(
            dist_to_goal_prev=35.0,
            dist_to_goal_curr=25.0,
            distances_to_objects=[],
            object_weights=None,
            red_light_detected=False,
            action_prev=0,
            action_curr=0,
            collision=False,
            goal_reached=True,
            off_road=False,
        )
        assert c["r_terminal"] == GOAL_REWARD

    def test_collision_terminal(self) -> None:
        """r_terminal = COLLISION_PENALTY em colisão."""
        r, c = compute_reward(
            dist_to_goal_prev=200.0,
            dist_to_goal_curr=200.0,
            distances_to_objects=[],
            object_weights=None,
            red_light_detected=False,
            action_prev=0,
            action_curr=0,
            collision=True,
            goal_reached=False,
            off_road=False,
        )
        assert c["r_terminal"] == COLLISION_PENALTY

    def test_total_is_sum_of_components(self) -> None:
        """r_total deve ser a soma exata dos componentes."""
        r, c = compute_reward(
            dist_to_goal_prev=200.0,
            dist_to_goal_curr=180.0,
            distances_to_objects=[50.0],
            object_weights=None,
            red_light_detected=True,
            action_prev=0,
            action_curr=1,
            collision=False,
            goal_reached=False,
            off_road=False,
        )
        expected = (
            c["r_progress"] + c["r_safety"] + c["r_traffic"]
            + c["r_smooth"] + c["r_terminal"]
        )
        assert abs(r - expected) < 1e-6

    def test_first_step_no_smooth_penalty(self) -> None:
        """Sem penalidade de suavidade no primeiro step (prev=None)."""
        r, c = compute_reward(
            dist_to_goal_prev=200.0,
            dist_to_goal_curr=180.0,
            distances_to_objects=[],
            object_weights=None,
            red_light_detected=False,
            action_prev=None,
            action_curr=0,
            collision=False,
            goal_reached=False,
            off_road=False,
        )
        assert c["r_smooth"] == 0.0
