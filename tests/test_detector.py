# ==========================================================================
# MÓDULO: test_detector.py
# PROPÓSITO: Testes para detector YOLO + state encoder
# ==========================================================================
from __future__ import annotations

import numpy as np
import pytest

from neurodrive.vision.detector import CLASS_NAMES, Detection
from neurodrive.vision.state_encoder import (
    FEATURES_PER_DETECTION,
    MAX_DETECTIONS,
    StateEncoder,
)


# ── Testes para Detection ──────────────────────────────────────────────


class TestDetection:
    """Testes para a dataclass Detection."""

    def test_creation(self) -> None:
        """Detection deve ser criável com todos os campos."""
        det = Detection(0, 0.5, 0.3, 0.1, 0.15, 0.95)
        assert det.class_id == 0
        assert det.cx == 0.5
        assert det.confidence == 0.95

    def test_class_name(self) -> None:
        """class_name deve retornar nome legível."""
        det = Detection(0, 0.5, 0.3, 0.1, 0.15, 0.95)
        assert det.class_name == "vehicle_npc"

        det2 = Detection(1, 0.5, 0.3, 0.1, 0.15, 0.88)
        assert det2.class_name == "traffic_light_red"

        det3 = Detection(3, 0.5, 0.3, 0.1, 0.15, 0.72)
        assert det3.class_name == "pedestrian"

    def test_unknown_class(self) -> None:
        """Classe desconhecida deve retornar unknown_N."""
        det = Detection(99, 0.5, 0.3, 0.1, 0.15, 0.5)
        assert det.class_name == "unknown_99"

    def test_to_array(self) -> None:
        """to_array deve retornar float32 array de 6 elementos."""
        det = Detection(2, 0.4, 0.6, 0.05, 0.1, 0.9)
        arr = det.to_array()
        assert arr.shape == (6,)
        assert arr.dtype == np.float32
        assert arr[0] == 2.0   # class_id
        assert arr[5] == pytest.approx(0.9)  # confidence

    def test_repr(self) -> None:
        """repr deve ser legível."""
        det = Detection(0, 0.5, 0.3, 0.1, 0.15, 0.95)
        r = repr(det)
        assert "vehicle_npc" in r
        assert "0.950" in r


# ── Testes para StateEncoder ───────────────────────────────────────────


class TestStateEncoder:
    """Testes para o encoder de detecções → tensor fixo."""

    @pytest.fixture
    def encoder(self) -> StateEncoder:
        return StateEncoder()

    @pytest.fixture
    def sample_detections(self) -> list[Detection]:
        return [
            Detection(0, 0.5, 0.3, 0.1, 0.15, 0.95),
            Detection(1, 0.45, 0.42, 0.03, 0.07, 0.88),
            Detection(3, 0.6, 0.5, 0.02, 0.03, 0.72),
        ]

    def test_encode_shape(
        self, encoder: StateEncoder, sample_detections: list[Detection]
    ) -> None:
        """Output deve ter shape (MAX_DETECTIONS, 6)."""
        result = encoder.encode(sample_detections)
        assert result.shape == (MAX_DETECTIONS, FEATURES_PER_DETECTION)

    def test_encode_dtype(
        self, encoder: StateEncoder, sample_detections: list[Detection]
    ) -> None:
        """Output deve ser float32."""
        result = encoder.encode(sample_detections)
        assert result.dtype == np.float32

    def test_encode_values_in_range(
        self, encoder: StateEncoder, sample_detections: list[Detection]
    ) -> None:
        """Todos os valores devem estar em [0, 1]."""
        result = encoder.encode(sample_detections)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_encode_padding_zeros(
        self, encoder: StateEncoder, sample_detections: list[Detection]
    ) -> None:
        """Slots não usados devem ser zeros."""
        result = encoder.encode(sample_detections)
        # 3 detecções, slots 3-9 devem ser zeros
        assert np.all(result[3:] == 0.0)

    def test_encode_empty_list(self, encoder: StateEncoder) -> None:
        """Lista vazia deve retornar tensor de zeros."""
        result = encoder.encode([])
        assert result.shape == (MAX_DETECTIONS, FEATURES_PER_DETECTION)
        assert np.all(result == 0.0)

    def test_encode_truncation(self, encoder: StateEncoder) -> None:
        """Mais de MAX_DETECTIONS deve truncar mantendo os mais confiantes."""
        # Cria 15 detecções (mais que MAX_DETECTIONS=10)
        dets = [
            Detection(0, 0.5, 0.5, 0.1, 0.1, 0.5 + i * 0.03)
            for i in range(15)
        ]
        result = encoder.encode(dets)
        assert result.shape == (MAX_DETECTIONS, FEATURES_PER_DETECTION)
        # Verifica que nenhum slot está vazio (todos preenchidos)
        assert np.all(result[:, 5] > 0.0)  # confidence > 0

    def test_encode_sorted_by_confidence(
        self, encoder: StateEncoder
    ) -> None:
        """Detecções devem estar ordenadas por confiança desc."""
        dets = [
            Detection(0, 0.5, 0.5, 0.1, 0.1, 0.3),
            Detection(1, 0.5, 0.5, 0.1, 0.1, 0.9),
            Detection(2, 0.5, 0.5, 0.1, 0.1, 0.6),
        ]
        result = encoder.encode(dets)
        # Primeira linha deve ter a maior confiança
        assert result[0, 5] == pytest.approx(0.9)
        assert result[1, 5] == pytest.approx(0.6)
        assert result[2, 5] == pytest.approx(0.3)

    def test_encode_class_normalization(
        self, encoder: StateEncoder
    ) -> None:
        """class_id deve ser normalizado para [0, 1]."""
        dets = [
            Detection(0, 0.5, 0.5, 0.1, 0.1, 0.9),  # 0/4 = 0.0
            Detection(4, 0.5, 0.5, 0.1, 0.1, 0.8),  # 4/4 = 1.0
        ]
        result = encoder.encode(dets)
        assert result[0, 0] == pytest.approx(0.0)    # class 0 → 0.0
        assert result[1, 0] == pytest.approx(1.0)    # class 4 → 1.0

    def test_encode_with_ego(
        self, encoder: StateEncoder, sample_detections: list[Detection]
    ) -> None:
        """encode_with_ego deve retornar dict com detections e ego_state."""
        obs = encoder.encode_with_ego(
            detections=sample_detections,
            ego_x=300.0, ego_y=100.0,
            ego_speed=3.5, ego_heading=1.57,
            dist_to_goal=400.0,
        )
        assert "detections" in obs
        assert "ego_state" in obs
        assert obs["detections"].shape == (MAX_DETECTIONS, 6)
        assert obs["ego_state"].shape == (5,)
        assert obs["ego_state"].dtype == np.float32

    def test_ego_state_normalized(
        self, encoder: StateEncoder
    ) -> None:
        """Ego state deve estar normalizado em [0, 1]."""
        obs = encoder.encode_with_ego(
            detections=[],
            ego_x=320.0, ego_y=320.0,
            ego_speed=2.0, ego_heading=3.14,
            dist_to_goal=200.0,
        )
        assert np.all(obs["ego_state"] >= 0.0)
        assert np.all(obs["ego_state"] <= 1.01)


# ── Testes para CLASS_NAMES ───────────────────────────────────────────


class TestClassNames:
    """Testes para as constantes de classes."""

    def test_all_classes_defined(self) -> None:
        """Todas as 5 classes devem estar definidas."""
        assert len(CLASS_NAMES) == 5

    def test_class_ids_sequential(self) -> None:
        """IDs devem ser 0, 1, 2, 3, 4."""
        assert set(CLASS_NAMES.keys()) == {0, 1, 2, 3, 4}

    def test_expected_class_names(self) -> None:
        """Nomes devem corresponder à especificação."""
        assert CLASS_NAMES[0] == "vehicle_npc"
        assert CLASS_NAMES[1] == "traffic_light_red"
        assert CLASS_NAMES[2] == "traffic_light_green"
        assert CLASS_NAMES[3] == "pedestrian"
        assert CLASS_NAMES[4] == "road_marking"
