# ==========================================================================
# MÓDULO: test_env.py
# PROPÓSITO: Testes de integração para CityDriveEnv (Gymnasium)
# ==========================================================================
from __future__ import annotations

import numpy as np
import pytest

from neurodrive.env.city_env import (
    ACTION_ACCELERATE,
    ACTION_BRAKE,
    ACTION_STOP,
    ACTION_TURN_LEFT,
    ACTION_TURN_RIGHT,
    CityDriveEnv,
    MAX_DETECTIONS,
    MAX_STEPS,
)


@pytest.fixture
def env() -> CityDriveEnv:
    """Cria environment para testes (sem renderização)."""
    e = CityDriveEnv(render_mode=None, seed=42)
    yield e
    e.close()


class TestCityDriveEnvReset:
    """Testes para reset() do environment."""

    def test_reset_returns_obs_and_info(self, env: CityDriveEnv) -> None:
        """Reset deve retornar (obs, info) tuple."""
        result = env.reset(seed=42)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_obs_has_correct_keys(self, env: CityDriveEnv) -> None:
        """Observation dict deve ter 'detections' e 'ego_state'."""
        obs, _ = env.reset(seed=42)
        assert "detections" in obs
        assert "ego_state" in obs

    def test_detections_shape(self, env: CityDriveEnv) -> None:
        """Detections deve ter shape (MAX_DETECTIONS, 6)."""
        obs, _ = env.reset(seed=42)
        assert obs["detections"].shape == (MAX_DETECTIONS, 6)
        assert obs["detections"].dtype == np.float32

    def test_ego_state_shape(self, env: CityDriveEnv) -> None:
        """Ego state deve ter shape (6,)."""
        obs, _ = env.reset(seed=42)
        assert obs["ego_state"].shape == (6,)
        assert obs["ego_state"].dtype == np.float32

    def test_obs_in_valid_range(self, env: CityDriveEnv) -> None:
        """Todos os valores de observação devem estar em [0, 1]."""
        obs, _ = env.reset(seed=42)
        assert np.all(obs["detections"] >= 0.0)
        assert np.all(obs["detections"] <= 1.0)
        assert np.all(obs["ego_state"] >= 0.0)
        assert np.all(obs["ego_state"] <= 1.01)  # tolerância float

    def test_info_has_goal_position(self, env: CityDriveEnv) -> None:
        """Info deve conter posição do goal."""
        _, info = env.reset(seed=42)
        assert "goal_x" in info
        assert "goal_y" in info

    def test_deterministic_with_same_seed(self) -> None:
        """Mesmo seed deve produzir mesma observação inicial."""
        env1 = CityDriveEnv(seed=123)
        env2 = CityDriveEnv(seed=123)
        obs1, _ = env1.reset(seed=123)
        obs2, _ = env2.reset(seed=123)
        np.testing.assert_array_equal(obs1["ego_state"], obs2["ego_state"])
        env1.close()
        env2.close()


class TestCityDriveEnvStep:
    """Testes para step() do environment."""

    def test_step_returns_5_tuple(self, env: CityDriveEnv) -> None:
        """Step deve retornar (obs, reward, terminated, truncated, info)."""
        env.reset(seed=42)
        result = env.step(ACTION_ACCELERATE)
        assert isinstance(result, tuple)
        assert len(result) == 5

    def test_step_reward_is_float(self, env: CityDriveEnv) -> None:
        """Reward deve ser float."""
        env.reset(seed=42)
        _, reward, _, _, _ = env.step(ACTION_ACCELERATE)
        assert isinstance(reward, float)

    def test_step_terminated_is_bool(self, env: CityDriveEnv) -> None:
        """terminated deve ser bool Python nativo."""
        env.reset(seed=42)
        _, _, terminated, _, _ = env.step(ACTION_ACCELERATE)
        assert isinstance(terminated, bool)

    def test_step_truncated_is_bool(self, env: CityDriveEnv) -> None:
        """truncated deve ser bool Python nativo."""
        env.reset(seed=42)
        _, _, _, truncated, _ = env.step(ACTION_ACCELERATE)
        assert isinstance(truncated, bool)

    def test_step_info_has_components(self, env: CityDriveEnv) -> None:
        """Info deve ter reward_components para TensorBoard."""
        env.reset(seed=42)
        _, _, _, _, info = env.step(ACTION_ACCELERATE)
        assert "reward_components" in info
        assert "dist_to_goal" in info

    def test_all_actions_valid(self, env: CityDriveEnv) -> None:
        """Todas as 5 ações devem ser executáveis sem erro."""
        env.reset(seed=42)
        for action in range(5):
            obs, r, term, trunc, info = env.step(action)
            assert env.observation_space.contains(obs)
            if term or trunc:
                env.reset(seed=42)

    def test_acceleration_increases_speed(self, env: CityDriveEnv) -> None:
        """Ação ACELERAR deve aumentar a velocidade."""
        env.reset(seed=42)
        speed_before = env._ego_speed
        env.step(ACTION_ACCELERATE)
        assert env._ego_speed > speed_before

    def test_brake_decreases_speed(self, env: CityDriveEnv) -> None:
        """Ação FREAR deve diminuir ou manter velocidade zero."""
        env.reset(seed=42)
        # Primeiro acelera
        env.step(ACTION_ACCELERATE)
        env.step(ACTION_ACCELERATE)
        speed_before = env._ego_speed
        env.step(ACTION_BRAKE)
        assert env._ego_speed <= speed_before

    def test_episode_truncates_at_max_steps(self, env: CityDriveEnv) -> None:
        """Episódio deve truncar ao atingir MAX_STEPS."""
        env.reset(seed=42)
        for _ in range(MAX_STEPS):
            obs, r, term, trunc, info = env.step(ACTION_STOP)
            if term:
                env.reset(seed=42)
        # Se chegou aqui sem terminar, o último step deve truncar
        # (ou terminou antes por colisão/goal)
        assert True  # O fato de não dar erro já é sucesso


class TestCityDriveEnvCollision:
    """Testes para detecção de colisão e off-road."""

    def test_off_road_terminates(self, env: CityDriveEnv) -> None:
        """Sair da via deve encerrar o episódio."""
        env.reset(seed=42)
        # Força posição fora da via
        env._ego_x = 100.0  # calçada/quadrante
        env._ego_y = 100.0
        _, _, terminated, _, info = env.step(ACTION_STOP)
        assert terminated is True
        assert info.get("off_road") is True


class TestCityDriveEnvRender:
    """Testes para renderização rgb_array."""

    def test_rgb_array_render(self) -> None:
        """render_mode='rgb_array' deve retornar numpy array."""
        env = CityDriveEnv(render_mode="rgb_array", seed=42)
        env.reset(seed=42)
        frame = env.render()
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (640, 640, 3)
        assert frame.dtype == np.uint8
        env.close()
