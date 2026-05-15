# ==========================================================================
# MÓDULO: evaluate.py
# PROPÓSITO: Avaliação e visualização do agente treinado
# DECISÃO DE DESIGN: Avaliação separada do treino para permitir (1) análise
#   post-hoc com diferentes seeds, (2) gravação de vídeo, (3) coleta de
#   métricas quantitativas sem overhead de treino.
# ==========================================================================
from __future__ import annotations

import argparse
import logging
import os
import time
from typing import Any, Final

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
DEFAULT_MODEL_PATH: Final[str] = "models/ppo_neurodrive_final.zip"
DEFAULT_EPISODES: Final[int] = 10
DEFAULT_SEED: Final[int] = 42


def evaluate(
    model_path: str = DEFAULT_MODEL_PATH,
    num_episodes: int = DEFAULT_EPISODES,
    render: bool = True,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Avalia o agente treinado e coleta métricas.

    Args:
        model_path: Caminho para o modelo .zip salvo.
        num_episodes: Número de episódios para avaliação.
        render: Se True, mostra visualização PyGame.
        seed: Seed para reprodutibilidade.

    Returns:
        Dict com métricas agregadas:
            - mean_reward: Recompensa média por episódio
            - std_reward: Desvio padrão
            - mean_steps: Steps médios por episódio
            - goal_rate: Taxa de chegada ao destino (%)
            - collision_rate: Taxa de colisão (%)
            - off_road_rate: Taxa de saída de via (%)
            - timeout_rate: Taxa de timeout (%)

    Note (decisão de design):
        Rodamos avaliação com env direto (sem VecEnv) para ter acesso
        ao info dict completo e poder coletar métricas granulares.
        Durante o treino usamos VecEnv para paralelismo, mas na
        avaliação priorizamos interpretabilidade sobre velocidade.
    """
    from stable_baselines3 import PPO

    from neurodrive.env.city_env import CityDriveEnv

    # Verifica se modelo existe
    if not os.path.exists(model_path):
        logger.error("Modelo não encontrado: %s", model_path)
        print(f"Erro: Modelo '{model_path}' não encontrado.")
        print("Certifique-se de treinar primeiro com: python -m neurodrive.agent.train")
        return {}

    # Carrega modelo
    logger.info("Carregando modelo: %s", model_path)
    model = PPO.load(model_path)

    # Cria environment
    render_mode = "human" if render else None
    env = CityDriveEnv(render_mode=render_mode, seed=seed)

    # Métricas
    episode_rewards: list[float] = []
    episode_steps: list[int] = []
    outcomes = {"goal": 0, "collision": 0, "off_road": 0, "timeout": 0}

    for ep in range(num_episodes):
        obs, info = env.reset(seed=seed + ep)
        total_reward = 0.0
        steps = 0
        done = False

        while not done:
            # Inferência do modelo
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))

            total_reward += reward
            steps += 1
            done = terminated or truncated

            if render:
                time.sleep(0.03)  # ~30 FPS

        # Registra resultado
        episode_rewards.append(total_reward)
        episode_steps.append(steps)

        if info.get("goal_reached"):
            outcomes["goal"] += 1
            outcome_str = "✅ GOAL"
        elif info.get("collision"):
            outcomes["collision"] += 1
            outcome_str = "💥 COLISÃO"
        elif info.get("off_road"):
            outcomes["off_road"] += 1
            outcome_str = "🚧 OFF-ROAD"
        else:
            outcomes["timeout"] += 1
            outcome_str = "⏰ TIMEOUT"

        logger.info(
            "Ep %d/%d: %s | Reward: %.1f | Steps: %d",
            ep + 1, num_episodes, outcome_str, total_reward, steps,
        )

    env.close()

    # Calcula métricas
    results = {
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "mean_steps": float(np.mean(episode_steps)),
        "goal_rate": outcomes["goal"] / num_episodes * 100,
        "collision_rate": outcomes["collision"] / num_episodes * 100,
        "off_road_rate": outcomes["off_road"] / num_episodes * 100,
        "timeout_rate": outcomes["timeout"] / num_episodes * 100,
        "episodes": num_episodes,
    }

    # Imprime relatório
    print("\n" + "=" * 60)
    print("    NeuroDrive — Relatório de Avaliação")
    print("=" * 60)
    print(f"  Modelo:           {model_path}")
    print(f"  Episódios:        {num_episodes}")
    print(f"  Reward médio:     {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"  Steps médios:     {results['mean_steps']:.0f}")
    print(f"  Taxa de sucesso:  {results['goal_rate']:.1f}%")
    print(f"  Taxa de colisão:  {results['collision_rate']:.1f}%")
    print(f"  Taxa off-road:    {results['off_road_rate']:.1f}%")
    print(f"  Taxa timeout:     {results['timeout_rate']:.1f}%")
    print("=" * 60)

    return results


def parse_args() -> argparse.Namespace:
    """Parse argumentos de linha de comando."""
    parser = argparse.ArgumentParser(
        description="NeuroDrive — Avaliação do Agente",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL_PATH,
        help="Caminho para o modelo .zip.",
    )
    parser.add_argument(
        "--episodes", type=int, default=DEFAULT_EPISODES,
        help="Número de episódios para avaliação.",
    )
    parser.add_argument(
        "--no-render", action="store_true",
        help="Desativa visualização PyGame.",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help="Seed para reprodutibilidade.",
    )
    return parser.parse_args()


# ── BLOCO DEMO ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = parse_args()
    evaluate(
        model_path=args.model,
        num_episodes=args.episodes,
        render=not args.no_render,
        seed=args.seed,
    )
