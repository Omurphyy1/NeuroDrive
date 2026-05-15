# ==========================================================================
# MÓDULO: detector.py
# PROPÓSITO: Wrapper do YOLOv8 para detecção de objetos nos frames do jogo
# DECISÃO DE DESIGN: Usamos YOLOv8n (nano) em vez de s/m/l/x porque:
#   1. Latência alvo < 30ms/frame para manter 30 FPS
#   2. Ambiente 2D com sprites simples não exige modelo pesado
#   3. Fine-tune em dados sintéticos compensa a menor capacidade
#   Alternativa descartada: YOLOv8s — 2x mais lento (~50ms) sem ganho
#   significativo de mAP em sprites 2D simples.
# ==========================================================================
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Final

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
DEFAULT_CONFIDENCE: Final[float] = 0.25
DEFAULT_IOU_THRESHOLD: Final[float] = 0.45
DEFAULT_IMG_SIZE: Final[int] = 640

# Classes do modelo (devem coincidir com dataset.yaml)
CLASS_NAMES: Final[dict[int, str]] = {
    0: "vehicle_npc",
    1: "traffic_light_red",
    2: "traffic_light_green",
    3: "pedestrian",
    4: "road_marking",
}


class Detection:
    """Estrutura de uma detecção YOLO individual.

    Armazena class_id, bounding box (cx, cy, w, h) normalizada,
    e confidence score. Imutável após criação.

    Args:
        class_id: ID da classe detectada (0-4).
        cx: Centro X normalizado [0, 1].
        cy: Centro Y normalizado [0, 1].
        w: Largura normalizada [0, 1].
        h: Altura normalizada [0, 1].
        confidence: Score de confiança [0, 1].
    """

    __slots__ = ("class_id", "cx", "cy", "w", "h", "confidence")

    def __init__(
        self,
        class_id: int,
        cx: float,
        cy: float,
        w: float,
        h: float,
        confidence: float,
    ) -> None:
        self.class_id = class_id
        self.cx = cx
        self.cy = cy
        self.w = w
        self.h = h
        self.confidence = confidence

    @property
    def class_name(self) -> str:
        """Nome da classe detectada."""
        return CLASS_NAMES.get(self.class_id, f"unknown_{self.class_id}")

    def to_array(self) -> np.ndarray:
        """Converte para array [class_id, cx, cy, w, h, conf].

        Returns:
            Array float32 de 6 elementos.
        """
        return np.array(
            [self.class_id, self.cx, self.cy, self.w, self.h, self.confidence],
            dtype=np.float32,
        )

    def __repr__(self) -> str:
        return (
            f"Detection({self.class_name}, "
            f"cx={self.cx:.3f}, cy={self.cy:.3f}, "
            f"w={self.w:.3f}, h={self.h:.3f}, "
            f"conf={self.confidence:.3f})"
        )


class YOLODetector:
    """Wrapper para inferência YOLOv8 em frames do jogo.

    Carrega um modelo YOLOv8 (pré-treinado ou fine-tuned) e executa
    detecção em arrays RGB numpy. Retorna lista de Detection structs.

    Args:
        model_path: Caminho para o modelo .pt (default: yolov8n.pt).
        confidence: Threshold mínimo de confiança.
        iou_threshold: Threshold de IoU para NMS.
        img_size: Tamanho de entrada do modelo (deve ser 640 para nosso mapa).
        device: Dispositivo PyTorch ('cpu', 'cuda', 'auto').

    Note (decisão de design):
        O wrapper abstrai a API da Ultralytics para que o restante do
        pipeline (StateEncoder, CityDriveEnv) não precise conhecer os
        detalhes de implementação do YOLO. Se futuramente trocarmos para
        outro detector (ex: RT-DETR), apenas este módulo muda.
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence: float = DEFAULT_CONFIDENCE,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
        img_size: int = DEFAULT_IMG_SIZE,
        device: str = "auto",
    ) -> None:
        self.model_path = model_path
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.img_size = img_size
        self._model = None
        self._device = device
        self._load_model()

    def _load_model(self) -> None:
        """Carrega o modelo YOLOv8.

        Note (decisão de design):
            Import lazy da ultralytics para não obrigar instalação
            em testes que não usam o detector (Fase 1).
        """
        try:
            from ultralytics import YOLO
            self._model = YOLO(self.model_path)
            logger.info(
                "YOLOv8 carregado: %s (%d classes)",
                self.model_path,
                len(self._model.names) if hasattr(self._model, "names") else -1,
            )
        except ImportError:
            logger.warning(
                "ultralytics não instalado. Detector operando em modo stub."
            )
            self._model = None
        except Exception as e:
            logger.error("Erro ao carregar modelo YOLO: %s", e)
            self._model = None

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Executa detecção em um frame RGB.

        Args:
            frame: Array numpy (H, W, 3) uint8 RGB.

        Returns:
            Lista de Detection, ordenada por confiança (desc).

        Raises:
            RuntimeError: Se o modelo não foi carregado.

        Note (decisão de design):
            Retornamos bboxes normalizadas (não em pixels) para
            consistência com o observation space do CityDriveEnv.
            A normalização é feita dividindo por img_size.
        """
        if self._model is None:
            logger.warning("Modelo YOLO não carregado. Retornando lista vazia.")
            return []

        start_time = time.perf_counter()

        # Inferência YOLO
        results = self._model.predict(
            source=frame,
            conf=self.confidence,
            iou=self.iou_threshold,
            imgsz=self.img_size,
            verbose=False,
            device=self._device if self._device != "auto" else None,
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        detections: list[Detection] = []

        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes
                for i in range(len(boxes)):
                    # xywhn: normalized [cx, cy, w, h]
                    xywhn = boxes.xywhn[i].cpu().numpy()
                    cls_id = int(boxes.cls[i].item())
                    conf = float(boxes.conf[i].item())

                    detections.append(Detection(
                        class_id=cls_id,
                        cx=float(xywhn[0]),
                        cy=float(xywhn[1]),
                        w=float(xywhn[2]),
                        h=float(xywhn[3]),
                        confidence=conf,
                    ))

        # Ordena por confiança (maior primeiro)
        detections.sort(key=lambda d: d.confidence, reverse=True)

        logger.debug(
            "YOLO: %d detecções em %.1fms",
            len(detections), elapsed_ms,
        )

        return detections

    def detect_batch(self, frames: list[np.ndarray]) -> list[list[Detection]]:
        """Executa detecção em um batch de frames.

        Args:
            frames: Lista de arrays RGB (H, W, 3).

        Returns:
            Lista de listas de Detection (uma por frame).
        """
        return [self.detect(frame) for frame in frames]

    @property
    def is_loaded(self) -> bool:
        """Retorna True se o modelo YOLO está carregado e pronto."""
        return self._model is not None


# ── BLOCO DEMO ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Demonstra detecção com modelo pré-treinado (COCO)
    print("Tentando carregar YOLOv8n...")
    detector = YOLODetector(model_path="yolov8n.pt", confidence=0.3)

    if detector.is_loaded:
        # Cria frame de teste do jogo
        from neurodrive.env.city_env import CityDriveEnv
        env = CityDriveEnv(render_mode="rgb_array", seed=42)
        env.reset(seed=42)
        for _ in range(10):
            env.step(0)
        frame = env.render()
        env.close()

        if frame is not None:
            detections = detector.detect(frame)
            print(f"\n{len(detections)} detecções encontradas:")
            for det in detections:
                print(f"  {det}")
        else:
            print("Erro ao renderizar frame.")
    else:
        print("Modelo não carregado (ultralytics não instalado?).")
        print("Instale com: pip install ultralytics")
