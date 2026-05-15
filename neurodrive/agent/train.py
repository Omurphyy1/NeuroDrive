# ==========================================================================
# MÓDULO: train.py
# PROPÓSITO: Script de treino PPO via Stable-Baselines3
# DECISÃO DE DESIGN: Usamos PPO on-policy em vez de SAC/TD3 off-policy
#   porque: (1) PPO é mais estável com action spaces discretos; (2) não
#   requer replay buffer (menor consumo de memória); (3) é o padrão
#   da indústria para controle com observações estruturadas (OpenAI Five,
#   Dota 2). SAC seria preferível para ações contínuas.
# ==========================================================================
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Final

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes de treino (defaults)
# ---------------------------------------------------------------------------
DEFAULT_TIMESTEPS: Final[int] = 1_000_000
DEFAULT_N_ENVS: Final[int] = 4
DEFAULT_LEARNING_RATE: Final[float] = 3e-4
DEFAULT_BATCH_SIZE: Final[int] = 64
DEFAULT_N_EPOCHS: Final[int] = 10
DEFAULT_GAMMA: Final[float] = 0.99
DEFAULT_GAE_LAMBDA: Final[float] = 0.95
DEFAULT_CLIP_RANGE: Final[float] = 0.2
DEFAULT_ENT_COEF: Final[float] = 0.01
DEFAULT_SEED: Final[int] = 42

# Diretórios
MODELS_DIR: Final[str] = "models"
LOGS_DIR: Final[str] = "tb_logs"


def make_env(seed: int = 42) -> Any:
    """Factory function para criar CityDriveEnv (para VecEnv).

    Args:
        seed: Seed do ambiente.

    Returns:
        Função callable que cria o environment.

    Note (decisão de design):
        Factory function em vez de instância direta porque o
        SubprocVecEnv precisa criar o env dentro de cada processo
        filho (serialização cross-process via pickle).
    """
    def _init() -> Any:
        from neurodrive.env.city_env import CityDriveEnv
        env = CityDriveEnv(render_mode=None, seed=seed)
        return env
    return _init


def train(args: argparse.Namespace) -> None:
    """Executa o treinamento PPO.

    Args:
        args: Argumentos de linha de comando.

    Note (decisão de design):
        Usamos DummyVecEnv em vez de SubprocVecEnv por padrão porque:
        1. SubprocVecEnv requer que o env seja picklable
        2. Em Windows, SubprocVecEnv exige if __name__ == '__main__' guard
        3. DummyVecEnv é suficiente para ambientes leves como o nosso
        Se performance for gargalo, pode-se trocar para SubprocVecEnv.
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import (
        CheckpointCallback,
        EvalCallback,
    )
    from stable_baselines3.common.vec_env import DummyVecEnv

    # Cria diretórios
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    logger.info("=== NeuroDrive PPO Training ===")
    logger.info("Timesteps: %d", args.timesteps)
    logger.info("N envs: %d", args.n_envs)
    logger.info("Seed: %d", args.seed)

    # --- Cria ambientes vetorizados ---
    env_fns = [make_env(seed=args.seed + i) for i in range(args.n_envs)]
    vec_env = DummyVecEnv(env_fns)

    # --- Ambiente de avaliação (separado) ---
    eval_env = DummyVecEnv([make_env(seed=args.seed + 1000)])

    # --- Callbacks ---
    checkpoint_cb = CheckpointCallback(
        save_freq=max(args.timesteps // 10, 10000),
        save_path=MODELS_DIR,
        name_prefix="ppo_neurodrive",
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=MODELS_DIR,
        log_path=LOGS_DIR,
        eval_freq=max(args.timesteps // 20, 5000),
        n_eval_episodes=5,
        deterministic=True,
    )

    # --- Modelo PPO ---
    model = PPO(
        "MultiInputPolicy",
        vec_env,
        learning_rate=args.lr,
        n_steps=2048,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        verbose=1,
        seed=args.seed,
        tensorboard_log=LOGS_DIR,
    )

    logger.info("Modelo PPO criado:")
    logger.info("  Policy: MultiInputPolicy (MLP para Dict obs)")
    logger.info("  LR: %f", args.lr)
    logger.info("  Batch size: %d", args.batch_size)
    logger.info("  Gamma: %f", args.gamma)
    logger.info("  Clip range: %f", args.clip_range)
    logger.info("  Entropy coef: %f", args.ent_coef)

    # --- Treino ---
    logger.info("Iniciando treino por %d timesteps...", args.timesteps)
    model.learn(
        total_timesteps=args.timesteps,
        callback=[checkpoint_cb, eval_cb],
        progress_bar=True,
    )

    # --- Salva modelo final ---
    final_path = os.path.join(MODELS_DIR, "ppo_neurodrive_final")
    model.save(final_path)
    logger.info("Modelo final salvo em: %s.zip", final_path)

    # --- Cleanup ---
    vec_env.close()
    eval_env.close()

    logger.info("=== Treino finalizado! ===")


def parse_args() -> argparse.Namespace:
    """Parse argumentos de linha de comando."""
    parser = argparse.ArgumentParser(
        description="NeuroDrive — Treino PPO",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--timesteps", type=int, default=DEFAULT_TIMESTEPS,
        help="Total de timesteps de treino.",
    )
    parser.add_argument(
        "--n_envs", type=int, default=DEFAULT_N_ENVS,
        help="Número de ambientes paralelos.",
    )
    parser.add_argument(
        "--lr", type=float, default=DEFAULT_LEARNING_RATE,
        help="Learning rate do PPO.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=DEFAULT_BATCH_SIZE,
        help="Batch size para SGD.",
    )
    parser.add_argument(
        "--n_epochs", type=int, default=DEFAULT_N_EPOCHS,
        help="Epochs por rollout update.",
    )
    parser.add_argument(
        "--gamma", type=float, default=DEFAULT_GAMMA,
        help="Fator de desconto.",
    )
    parser.add_argument(
        "--gae_lambda", type=float, default=DEFAULT_GAE_LAMBDA,
        help="Lambda para GAE.",
    )
    parser.add_argument(
        "--clip_range", type=float, default=DEFAULT_CLIP_RANGE,
        help="Clip range do PPO.",
    )
    parser.add_argument(
        "--ent_coef", type=float, default=DEFAULT_ENT_COEF,
        help="Coeficiente de entropia (exploração).",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help="Seed global para reprodutibilidade.",
    )
    return parser.parse_args()


# ── BLOCO DEMO ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = parse_args()
    train(args)
