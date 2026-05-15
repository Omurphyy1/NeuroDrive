# ==========================================================================
# MÓDULO: data_collector.py
# PROPÓSITO: Geração automática de dataset YOLO a partir do ambiente de jogo
# DECISÃO DE DESIGN: Anotações geradas programaticamente usando ground-truth
#   do simulador (posições reais dos objetos). Esta é a grande vantagem de
#   ambientes sintéticos — zero custo de anotação manual. Cada screenshot
#   gera automaticamente um arquivo .txt com bounding boxes no formato
#   YOLO (class_id cx cy w h), todos normalizados por 640×640.
#   Alternativa descartada: Anotação manual via LabelImg/Roboflow — seria
#   necessário anotar milhares de imagens manualmente, inviável para TCC.
# ==========================================================================
from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Final

import numpy as np
import pygame

from neurodrive.env.city_env import CityDriveEnv
from neurodrive.env.tilemap import MAP_HEIGHT, MAP_WIDTH

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
# Classes YOLO (devem ser consistentes com o treino)
YOLO_CLASSES: Final[dict[int, str]] = {
    0: "vehicle_npc",
    1: "traffic_light_red",
    2: "traffic_light_green",
    3: "pedestrian",
    4: "road_marking",
}

# Mínimo de pixels para considerar a bbox válida (evita anotações microscópicas)
MIN_BBOX_AREA_PX: Final[int] = 50

# Ações aleatórias para variação de cenário durante coleta
NUM_ACTIONS: Final[int] = 5


class DataCollector:
    """Coletor automático de dataset para fine-tune do YOLOv8.

    Roda o ambiente CityDriveEnv com ações aleatórias, captura screenshots
    em formato RGB e gera anotações YOLO automaticamente a partir das
    posições reais dos objetos (ground-truth do simulador).

    Args:
        output_dir: Diretório raiz para salvar o dataset.
        seed: Seed para reprodutibilidade.

    Note (decisão de design):
        O dataset é organizado no formato padrão Ultralytics:
            dataset/
            ├── images/
            │   ├── train/
            │   └── val/
            └── labels/
                ├── train/
                └── val/
        Isso permite uso direto com `yolo train data=dataset.yaml`
        sem conversão adicional.
    """

    def __init__(
        self,
        output_dir: str = "datasets/neurodrive_yolo",
        seed: int = 42,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.seed = seed
        self._rng = random.Random(seed)

        # Cria estrutura de diretórios
        for split in ("train", "val"):
            (self.output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (self.output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

        logger.info("DataCollector: output_dir=%s, seed=%d", output_dir, seed)

    def _extract_annotations(self, env: CityDriveEnv) -> list[tuple[int, float, float, float, float]]:
        """Extrai bounding boxes ground-truth do estado atual do environment.

        Itera sobre todos os objetos do ambiente (NPCs, semáforos, pedestres)
        e calcula suas bounding boxes normalizadas no formato YOLO.

        Args:
            env: Instância ativa do CityDriveEnv.

        Returns:
            Lista de tuplas (class_id, cx, cy, w, h) normalizadas [0, 1].

        Note (decisão de design):
            Usamos os rects do PyGame (get_rect()) que já estão em coordenadas
            de pixel, e normalizamos dividindo por MAP_WIDTH/MAP_HEIGHT.
            Isso garante consistência exata entre o que é renderizado e o que
            é anotado — impossível com anotação manual.
        """
        annotations: list[tuple[int, float, float, float, float]] = []

        # Veículos NPC (class_id = 0)
        for npc in env._npc_vehicles:
            if not npc.active:
                continue
            rect = npc.get_rect()
            if rect.width * rect.height < MIN_BBOX_AREA_PX:
                continue
            # Clamp dentro do mapa
            cx = max(0, min(rect.centerx, MAP_WIDTH)) / MAP_WIDTH
            cy = max(0, min(rect.centery, MAP_HEIGHT)) / MAP_HEIGHT
            w = min(rect.width, MAP_WIDTH) / MAP_WIDTH
            h = min(rect.height, MAP_HEIGHT) / MAP_HEIGHT
            annotations.append((0, cx, cy, w, h))

        # Semáforos (class_id = 1 ou 2)
        if env._tl_controller:
            for tl in env._tl_controller.get_all_lights():
                rect = tl.get_rect()
                if rect.width * rect.height < MIN_BBOX_AREA_PX:
                    continue
                # Vermelho ou amarelo → class 1, verde → class 2
                cls_id = 1 if (tl.is_red or tl.is_yellow) else 2
                cx = max(0, min(rect.centerx, MAP_WIDTH)) / MAP_WIDTH
                cy = max(0, min(rect.centery, MAP_HEIGHT)) / MAP_HEIGHT
                w = min(rect.width, MAP_WIDTH) / MAP_WIDTH
                h = min(rect.height, MAP_HEIGHT) / MAP_HEIGHT
                annotations.append((cls_id, cx, cy, w, h))

        # Pedestres (class_id = 3)
        for ped in env._pedestrians:
            if not ped.active:
                continue
            rect = ped.get_rect()
            if rect.width * rect.height < MIN_BBOX_AREA_PX:
                continue
            cx = max(0, min(rect.centerx, MAP_WIDTH)) / MAP_WIDTH
            cy = max(0, min(rect.centery, MAP_HEIGHT)) / MAP_HEIGHT
            w = min(rect.width, MAP_WIDTH) / MAP_WIDTH
            h = min(rect.height, MAP_HEIGHT) / MAP_HEIGHT
            annotations.append((3, cx, cy, w, h))

        return annotations

    def _save_annotation(
        self,
        filepath: Path,
        annotations: list[tuple[int, float, float, float, float]],
    ) -> None:
        """Salva anotações no formato YOLO .txt.

        Formato por linha: class_id cx cy w h
        Todos os valores normalizados para [0, 1].

        Args:
            filepath: Caminho do arquivo .txt.
            annotations: Lista de (class_id, cx, cy, w, h).
        """
        with open(filepath, "w") as f:
            for cls_id, cx, cy, w, h in annotations:
                f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

    def collect(
        self,
        num_images: int = 2000,
        val_split: float = 0.2,
        steps_between_captures: int = 5,
        max_episode_steps: int = 200,
    ) -> dict[str, int]:
        """Coleta dataset completo com screenshots e anotações.

        Executa o ambiente com ações aleatórias, capturando um frame
        a cada `steps_between_captures` steps. Quando o episódio
        termina (colisão, goal, timeout), reseta automaticamente.

        Args:
            num_images: Total de imagens a coletar.
            val_split: Fração para validação (0.2 = 20%).
            steps_between_captures: Frames entre capturas (para diversidade).
            max_episode_steps: Steps máximos por episódio durante coleta.

        Returns:
            Dict com contagem por split: {"train": N, "val": M}.

        Note (decisão de design):
            steps_between_captures=5 garante diversidade temporal:
            em vez de capturar frames consecutivos (que seriam quase
            idênticos), espaçamos para que os NPCs e semáforos mudem
            de posição/estado entre capturas. Isso é crucial para
            evitar overfitting do YOLO em poses estáticas.
        """
        env = CityDriveEnv(render_mode="rgb_array", seed=self.seed)

        num_val = int(num_images * val_split)
        num_train = num_images - num_val

        counts = {"train": 0, "val": 0}
        total_collected = 0
        image_idx = 0
        step_in_episode = 0

        obs, info = env.reset(seed=self.seed)
        logger.info("Iniciando coleta de %d imagens...", num_images)

        while total_collected < num_images:
            # Ação aleatória para variação de cenário
            action = self._rng.randint(0, NUM_ACTIONS - 1)
            obs, reward, terminated, truncated, info = env.step(action)
            step_in_episode += 1

            # Captura a cada N steps
            if step_in_episode % steps_between_captures == 0:
                # Renderiza frame como RGB array
                frame = env.render()
                if frame is None:
                    continue

                # Extrai anotações ground-truth
                annotations = self._extract_annotations(env)

                # Determina split (train vs val)
                split = "val" if total_collected >= num_train else "train"

                # Salva imagem (PNG para qualidade lossless)
                img_filename = f"frame_{image_idx:06d}.png"
                img_path = self.output_dir / "images" / split / img_filename
                # Converte numpy RGB para pygame surface e salva
                surf = pygame.surfarray.make_surface(
                    np.transpose(frame, (1, 0, 2))
                )
                pygame.image.save(surf, str(img_path))

                # Salva anotação YOLO
                label_filename = f"frame_{image_idx:06d}.txt"
                label_path = self.output_dir / "labels" / split / label_filename
                self._save_annotation(label_path, annotations)

                counts[split] += 1
                total_collected += 1
                image_idx += 1

                if total_collected % 100 == 0:
                    logger.info(
                        "Progresso: %d/%d imagens coletadas",
                        total_collected, num_images,
                    )

            # Reset se episódio terminou ou timeout
            if terminated or truncated or step_in_episode >= max_episode_steps:
                seed_val = self._rng.randint(0, 100000)
                obs, info = env.reset(seed=seed_val)
                step_in_episode = 0

        env.close()

        # Gera dataset.yaml para Ultralytics
        self._generate_dataset_yaml()

        logger.info(
            "Coleta finalizada: train=%d, val=%d",
            counts["train"], counts["val"],
        )
        return counts

    def _generate_dataset_yaml(self) -> None:
        """Gera arquivo dataset.yaml para treino Ultralytics YOLOv8.

        Note (decisão de design):
            O caminho usa path absoluto para evitar problemas com
            diretório de trabalho durante o treino YOLO. O Ultralytics
            resolve paths relativos ao CWD, não ao yaml, o que causa
            confusão em ambientes diferentes.
        """
        abs_path = self.output_dir.resolve()
        yaml_content = f"""# NeuroDrive YOLO Dataset
# Gerado automaticamente por DataCollector
# Formato: YOLOv8 (Ultralytics)

path: {abs_path}
train: images/train
val: images/val

# Classes (5 categorias do ambiente)
names:
  0: vehicle_npc
  1: traffic_light_red
  2: traffic_light_green
  3: pedestrian
  4: road_marking

# Notas:
# - Anotações geradas via ground-truth do simulador (zero anotação manual)
# - Resolução: 640x640 (nativa do YOLOv8, sem resize)
# - road_marking (class 4) será adicionada em versão futura
"""
        yaml_path = self.output_dir / "dataset.yaml"
        with open(yaml_path, "w") as f:
            f.write(yaml_content)
        logger.info("dataset.yaml gerado em %s", yaml_path)


# ── BLOCO DEMO ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Coleta pequena para demonstração (50 imagens)
    collector = DataCollector(output_dir="datasets/neurodrive_yolo", seed=42)
    counts = collector.collect(num_images=50, val_split=0.2)
    print(f"\nDataset gerado: {counts}")
    print(f"Diretório: datasets/neurodrive_yolo/")
    print("Para treinar YOLO: yolo train model=yolov8n.pt data=datasets/neurodrive_yolo/dataset.yaml")
