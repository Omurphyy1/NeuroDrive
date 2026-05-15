# ==========================================================================
# MÓDULO: state_encoder.py
# PROPÓSITO: Converte detecções YOLO em vetor de observação para o agente RL
# DECISÃO DE DESIGN: Encoda as N detecções mais confiantes em um tensor
#   fixo (MAX_DETECTIONS, 6) com padding de zeros. Normaliza class_id para
#   [0, 1] dividindo pelo número de classes. Isso garante que o observation
#   space seja sempre do mesmo shape, independente do número de objetos
#   detectados — requisito do Gymnasium.
#   Alternativa descartada: Grafo de detecções (GNN) — complexidade
#   desnecessária para 5 classes e máximo 10 objetos.
# ==========================================================================
from __future__ import annotations

import logging
from typing import Final

import numpy as np

from neurodrive.vision.detector import Detection

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
MAX_DETECTIONS: Final[int] = 10
NUM_CLASSES: Final[int] = 5     # 0-4
FEATURES_PER_DETECTION: Final[int] = 6   # [class_id_norm, cx, cy, w, h, conf]


class StateEncoder:
    """Converte lista de detecções YOLO em tensor fixo para o agente RL.

    O encoder recebe uma lista variável de Detection objects e produz
    um array numpy de shape fixo (MAX_DETECTIONS, 6) normalizado para
    [0, 1]. Detecções excedentes são truncadas (mantendo as mais
    confiantes); slots vazios são preenchidos com zeros.

    Args:
        max_detections: Número máximo de detecções no vetor.
        num_classes: Número total de classes (para normalização).

    Note (decisão de design):
        O padding com zeros (em vez de -1 ou NaN) é intencional:
        - Zeros são "neutros" para redes neurais (não ativam neurônios)
        - NaN causaria NaN propagation nos gradientes
        - -1 quebraria a normalização [0, 1]
        
        As detecções são ordenadas por confiança (desc) antes do truncamento.
        Isso garante que, se houver mais de MAX_DETECTIONS objetos, os
        mais relevantes (alta confiança) sejam preservados.
    """

    def __init__(
        self,
        max_detections: int = MAX_DETECTIONS,
        num_classes: int = NUM_CLASSES,
    ) -> None:
        self.max_detections = max_detections
        self.num_classes = num_classes

    def encode(self, detections: list[Detection]) -> np.ndarray:
        """Converte detecções em tensor fixo normalizado.

        Args:
            detections: Lista de Detection (já ordenada por confiança).

        Returns:
            Array float32 de shape (max_detections, 6).
            Cada linha: [class_id_norm, cx, cy, w, h, confidence].
            class_id_norm = class_id / (num_classes - 1) para [0, 1].

        Note (decisão de design):
            Normalizamos class_id em vez de usar one-hot encoding porque:
            1. One-hot expandiria cada detecção de 6 para 10 features
            2. Com apenas 5 classes ordinais, escalar normalizado funciona
            3. Mantém o observation space compacto (60 vs 100 floats)
        """
        result = np.zeros(
            (self.max_detections, FEATURES_PER_DETECTION),
            dtype=np.float32,
        )

        # Ordena por confiança (maior primeiro) se não estiver
        sorted_dets = sorted(
            detections, key=lambda d: d.confidence, reverse=True
        )

        n = min(len(sorted_dets), self.max_detections)
        for i in range(n):
            det = sorted_dets[i]
            # Normaliza class_id para [0, 1]
            cls_norm = det.class_id / max(self.num_classes - 1, 1)
            result[i] = [
                cls_norm,
                det.cx,
                det.cy,
                det.w,
                det.h,
                det.confidence,
            ]

        return result

    def encode_with_ego(
        self,
        detections: list[Detection],
        ego_x: float,
        ego_y: float,
        ego_speed: float,
        ego_heading: float,
        dist_to_goal: float,
        map_width: float = 640.0,
        map_height: float = 640.0,
        max_speed: float = 4.0,
    ) -> dict[str, np.ndarray]:
        """Constrói observação completa (Dict) para o agente.

        Combina as detecções YOLO encoded com o estado do ego-vehicle
        normalizado, produzindo o observation dict do Gymnasium.

        Args:
            detections: Lista de Detection do YOLO.
            ego_x: Posição X do agente (pixels).
            ego_y: Posição Y do agente (pixels).
            ego_speed: Velocidade do agente (px/frame).
            ego_heading: Heading do agente (radianos).
            dist_to_goal: Distância ao goal (pixels).
            map_width: Largura do mapa (pixels).
            map_height: Altura do mapa (pixels).
            max_speed: Velocidade máxima do agente.

        Returns:
            Dict com 'detections' (MAX_DETECTIONS, 6) e 'ego_state' (5,).
        """
        import math

        detection_tensor = self.encode(detections)

        diag = math.sqrt(map_width**2 + map_height**2)

        # Ângulo relativo do ego ao goal
        angle_to_goal = math.atan2(
            ego_y - ego_y,  # placeholder — caller should provide goal coords
            ego_x - ego_x,
        )
        # Para encode_with_ego, computamos angle_diff se goal coords fornecidos
        # Por padrão, usamos 0.5 (neutro) se não temos goal coords
        angle_diff_norm = 0.5

        ego_state = np.array([
            ego_x / map_width,
            ego_y / map_height,
            ego_speed / max_speed,
            (ego_heading % (2 * math.pi)) / (2 * math.pi),
            min(dist_to_goal / diag, 1.0),
            angle_diff_norm,
        ], dtype=np.float32)

        return {
            "detections": detection_tensor,
            "ego_state": ego_state,
        }


# ── BLOCO DEMO ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Cria detecções de exemplo
    dets = [
        Detection(0, 0.5, 0.3, 0.1, 0.15, 0.95),   # veículo NPC
        Detection(1, 0.45, 0.42, 0.03, 0.07, 0.88),  # semáforo vermelho
        Detection(3, 0.6, 0.5, 0.02, 0.03, 0.72),    # pedestre
    ]

    encoder = StateEncoder()

    # Encode apenas detecções
    tensor = encoder.encode(dets)
    print(f"Tensor shape: {tensor.shape}")
    print(f"Conteúdo (3 primeiras linhas):\n{tensor[:3]}")
    print(f"Zeros padding (linha 4):\n{tensor[3]}")

    # Encode completo com ego state
    obs = encoder.encode_with_ego(
        detections=dets,
        ego_x=300.0, ego_y=100.0,
        ego_speed=3.5, ego_heading=1.57,
        dist_to_goal=400.0,
    )
    print(f"\nObs keys: {obs.keys()}")
    print(f"Detections shape: {obs['detections'].shape}")
    print(f"Ego state: {obs['ego_state']}")
    print("\nDemo state_encoder.py finalizado com sucesso.")
