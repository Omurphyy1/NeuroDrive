# NeuroDrive — ADR Fase 2: Visão Computacional (YOLO)

> Decisões técnicas do pipeline de visão computacional.

---

## ADR-009: Dataset Sintético com Anotação Automática (Zero Manual Labeling)

**Data:** 2026-05-15  
**Status:** Aceita  

### Contexto
O YOLOv8 precisa de um dataset anotado (imagens + bounding boxes) para
fine-tuning. Anotação manual de milhares de imagens é inviável para TCC.

### Decisão
- `DataCollector` roda o jogo com ações aleatórias e captura screenshots
- Anotações YOLO geradas programaticamente via ground-truth do simulador
- Formato: YOLO txt (class_id cx cy w h), tudo normalizado [0, 1]
- Organização padrão Ultralytics (images/train, images/val, labels/train, labels/val)
- `dataset.yaml` gerado automaticamente

### Alternativas Descartadas
- **Anotação manual (LabelImg/Roboflow):** Milhares de imagens × 5 classes = centenas
  de horas de trabalho braçal
- **Semi-automatic (active learning):** Complexidade desnecessária quando temos
  ground-truth perfeito do simulador

### Justificativa Acadêmica
A geração automática de datasets a partir de ambientes sintéticos é uma técnica
estabelecida na literatura (Tremblay et al., 2018 "Training Deep Networks with
Synthetic Data"). A vantagem principal é o custo zero de anotação e a capacidade
de gerar datasets arbitrariamente grandes. A desvantagem (domain gap) é
inexistente aqui porque treino e inferência operam no mesmo domínio visual.

### Impacto
- 2.000+ imagens geráveis em ~5 minutos
- Zero erro de anotação (precision = 1.0 por construção)
- Dataset infinitamente expansível com variação de seeds

---

## ADR-010: YOLOv8n (Nano) para Detecção em Tempo Real

**Data:** 2026-05-15  
**Status:** Aceita  

### Contexto
O detector precisa rodar dentro do loop step() do ambiente RL,
com latência alvo < 30ms/frame para manter 30 FPS lógicos.

### Decisão
- YOLOv8n (3.2M parâmetros, ~6.5 GFLOPs)
- Fine-tune por 100 epochs em dataset sintético
- Input: 640×640 RGB (nativo do mapa, sem resize)
- Meta: mAP@0.5 ≥ 0.85 para vehicle_npc e traffic_light

### Alternativas Descartadas
- **YOLOv8s (11.2M params):** ~2x mais lento, sem ganho significativo
  em sprites 2D simples com poucos detalhes visuais
- **RT-DETR:** Transformer-based, mais lento que YOLO em CPU
- **YOLO pré-treinado no COCO sem fine-tune:** Classes diferentes
  (person, car reais vs sprites 2D), mAP seria <0.3

### Impacto
- Latência esperada: ~15-25ms/frame em CPU
- Trade-off: menor capacidade que modelos maiores, compensado por
  dados de treino do mesmo domínio visual (zero domain gap)

---

## ADR-011: StateEncoder — Tensor Fixo com Zero-Padding

**Data:** 2026-05-15  
**Status:** Aceita  

### Contexto
O agente PPO precisa de observações de shape fixo. YOLO retorna
número variável de detecções por frame.

### Decisão
- StateEncoder converte lista variável de Detection → tensor (10, 6)
- Detecções ordenadas por confiança (maior primeiro)
- Slots vazios preenchidos com zeros
- class_id normalizado escalarly: class_id / (num_classes - 1)

### Alternativas Descartadas
- **One-hot encoding para class_id:** Expandiria observation de 60 para
  100 features sem ganho para 5 classes ordinais
- **GNN (Graph Neural Network):** Overengineering para 10 detecções
- **Padding com -1:** Quebraria normalização [0, 1] e causaria
  ativações indesejadas em neurônios ReLU

### Impacto
- Shape fixo garante compatibilidade com Gymnasium spaces.Box
- Zeros são "neutros" para MLP (não ativam neurônios)
- O encoder é o ponto de integração entre Camadas 2 e 3

---

*Continuação de docs/adr/001-environment-design.md (ADR-001 a ADR-008).*
