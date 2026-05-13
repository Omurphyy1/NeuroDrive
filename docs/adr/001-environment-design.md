# NeuroDrive — Architecture Decision Records (ADR)
# Documento de Contexto e Decisões Técnicas do Projeto
# TCC: Engenharia de Controle e Automação

> Este documento registra formalmente todas as decisões de arquitetura e design
> do projeto NeuroDrive. Cada decisão inclui contexto, alternativas consideradas,
> justificativa e impacto. O objetivo é garantir que a banca examinadora e
> qualquer leitor consiga compreender o PORQUÊ de cada escolha técnica.

---

## ADR-001: Arquitetura em 3 Camadas (Ambiente → Visão → Agente)

**Data:** 2026-05-13  
**Status:** Aceita  

### Contexto
O projeto precisa de uma arquitetura que separe claramente a simulação
(jogo 2D), a percepção visual (detecção de objetos) e a tomada de decisão
(agente RL). Isso é necessário para:
1. Permitir ablation studies isolados (ex: YOLO vs ground-truth)
2. Facilitar manutenção e testes unitários por camada
3. Espelhar a arquitetura real do Tesla Autopilot (câmeras → rede neural → atuadores)

### Decisão
Adotamos 3 camadas desacopladas:
- **Camada 1 (Ambiente):** Jogo 2D em PyGame exposto como Gymnasium environment
- **Camada 2 (Visão):** YOLOv8n fine-tuned nos screenshots do jogo
- **Camada 3 (Agente):** PPO via Stable-Baselines3

### Alternativas Descartadas
- **Monolítico (tudo em um):** Dificulta testes, impossibilita ablation, acoplamento alto
- **Simulador externo (CARLA, AirSim):** Overkill para TCC, requer GPU potente,
  não permite controle total sobre os dados de treino

### Impacto
- Interface clara entre camadas via observation/action spaces do Gymnasium
- Cada camada pode ser desenvolvida e testada independentemente
- Trade-off: overhead de serialização entre camadas (~2ms, negligível)

---

## ADR-002: Mapa Urbano Estático 640×640 (Tilemap Programático)

**Data:** 2026-05-13  
**Status:** Aceita  

### Contexto
O agente precisa de um ambiente visual que represente uma cidade com
cruzamento, semáforos, veículos e pedestres. A resolução precisa ser
compatível com o input do YOLOv8.

### Decisão
- Mapa 640×640 pixels desenhado programaticamente com `pygame.draw`
- Cruzamento em "+" central com 4 quadrantes temáticos (Loja, Posto, Praça, Pizzaria)
- Vias de 80px de largura (comporta 1 faixa por sentido)

### Alternativas Descartadas
- **Tiled/TMX tilemap:** Adiciona dependência externa (Tiled editor + pytmx) sem
  benefício funcional para um mapa estático simples
- **Resolução maior (1280×1280):** Dobraria o custo de inferência YOLO sem ganho
  proporcional em detecção. 640×640 é o input nativo do YOLOv8
- **Mapa procedural:** Desnecessário — queremos consistência entre episódios para
  que as detecções YOLO sejam comparáveis. A variação vem dos NPCs e spawns

### Impacto
- Zero resize no pipeline YOLO (screenshot 640×640 → YOLO 640×640)
- Reprodutibilidade total sem assets externos
- O mapa é cacheado como Surface (renderizado 1 vez, blitado N vezes)

---

## ADR-003: FSM de Semáforos Baseada em Frames (não wall-clock)

**Data:** 2026-05-13  
**Status:** Aceita  

### Contexto
Semáforos precisam ciclar entre RED → GREEN → YELLOW → RED de forma
determinística para que o treinamento RL seja reprodutível com seeds fixas.

### Decisão
- `TrafficLightState` como Enum (RED, YELLOW, GREEN)
- Transições controladas por contagem de frames, não `time.time()`
- Semáforos perpendiculares sincronizados via `TrafficLightController`
  (N-S começa verde, E-W começa vermelho)

### Alternativas Descartadas
- **Timer wall-clock (`time.time()`):** Introduz não-determinismo entre
  máquinas com clock speeds diferentes. Impossibilita seed-based replay
- **YELLOW_BLINKING (intermitente):** Complexidade desnecessária para o
  escopo do TCC. Pode ser adicionado como extensão futura

### Impacto
- Determinismo total: mesma seed → mesma sequência de estados
- Durações configuráveis: GREEN=150 frames (5s@30FPS), YELLOW=45 (1.5s), RED=195 (6.5s)
- Controller garante sincronia perfeita sem race conditions

---

## ADR-004: Observation Space Dict vs Pixels Brutos

**Data:** 2026-05-13  
**Status:** Aceita  

### Contexto
O agente PPO precisa receber observações do ambiente. A escolha entre
pixels brutos (imagem RGB) e vetor estruturado impacta diretamente
a velocidade de convergência e o custo computacional.

### Decisão
- **Dict observation space** com dois campos:
  - `'detections'`: Box(shape=(10, 6)) — até 10 detecções [class_id, cx, cy, w, h, conf]
  - `'ego_state'`: Box(shape=(5,)) — [pos_x, pos_y, velocity, heading, dist_to_goal]
- Todos os valores normalizados para [0, 1]

### Alternativas Descartadas
- **Pixels brutos (640×640×3 = 1.228.800 dims):** Requer CNN profunda
  (ex: Nature DQN com 3 camadas conv), convergência ~10x mais lenta
  (Mnih et al., 2015), inviável sem GPU dedicada
- **Flat vector (concatenar tudo):** Perde a semântica entre detecções
  e estado do ego. Dict preserva a estrutura

### Justificativa Acadêmica
A comparação "YOLO+Dict vs pixels brutos" é exatamente o ablation study
da Fase 4 do projeto. Esperamos demonstrar que a abstração perceptual
(via YOLO) acelera a convergência em ordens de magnitude — analogamente
ao que o Autopilot real faz ao processar câmeras em redes de detecção
antes de alimentar o planejador de trajetória.

### Impacto
- Dimensionalidade reduzida: ~65 floats vs 1.228.800
- PPO com MLP (2 hidden layers de 64) é suficiente, sem CNN
- Trade-off: dependência da qualidade das detecções YOLO

---

## ADR-005: Action Space Discrete(5) vs Continuous Box(2)

**Data:** 2026-05-13  
**Status:** Aceita  

### Contexto
O projeto anterior (AutoDrive) usou action space contínuo Box(2)
para aceleração e esterçamento. Precisamos decidir se mantemos
ou mudamos para o novo projeto.

### Decisão
- **Discrete(5):** ACELERAR, FREAR, VIRAR_ESQ, VIRAR_DIR, PARAR
- Cada ação aplica um delta fixo à velocidade ou heading do ego

### Alternativas Descartadas
- **Continuous Box(2) [aceleração, esterçamento]:** Mais realista mas
  aumenta complexidade de exploração. Para PPO em navegação top-down,
  ações discretas permitem que o agente aprenda "vocabulário de manobras"
  mais rapidamente
- **MultiDiscrete:** Combinações (ex: acelerar + virar) adicionam
  complexidade sem benefício claro no escopo do TCC

### Justificativa Acadêmica
O Tesla Autopilot real usa ações contínuas (torque do motor, ângulo da
direção). Nós simplificamos para discreto por razões pedagógicas —
demonstrar os princípios do DRL sem os desafios adicionais de
distribuições contínuas (Gaussian policy, entropy bonus diferente).
Esta simplificação é explicitamente mencionada na seção de Limitações
do TCC.

### Impacto
- Exploração mais eficiente (5 ações vs espaço contínuo infinito)
- Política categórica (Softmax) é mais estável que Gaussian para PPO
- Trade-off: movimentos menos suaves que ações contínuas

---

## ADR-006: Reward Shaping Baseado em Potencial (Ng et al., 1999)

**Data:** 2026-05-13  
**Status:** Aceita  

### Contexto
O agente precisa de sinal de recompensa denso para convergir em tempo
razoável. Recompensa esparsa (+100/-100 apenas no final do episódio)
resulta em convergência extremamente lenta.

### Decisão
Recompensa composta por 5 componentes aditivos e independentes:
1. **r_progress** = φ(s') − φ(s) onde φ(s) = −dist(s, goal)/diagonal
2. **r_safety** = −Σ max(0, SAFE_DIST − dist_i) × weight_i
3. **r_traffic** = −10.0 se avança com semáforo vermelho
4. **r_smooth** = −0.1 por mudança de ação consecutiva
5. **r_terminal** = +100 (goal) | −100 (colisão) | −50 (off-road)

### Alternativas Descartadas
- **Recompensa esparsa:** Convergência ~100x mais lenta, o agente não
  recebe feedback intermediário sobre progresso
- **Recompensa não-baseada em potencial:** Pode introduzir ótimos locais
  espúrios e alterar a política ótima do MDP original (Ng et al., 1999)
- **Curiosity-driven (ICM):** Overkill para ambiente com goal fixo

### Justificativa Acadêmica
O paper "Policy Invariance Under Reward Transformations" (Ng, Harada,
Russell, 1999) prova formalmente que reward shaping baseado em potencial
preserva a política ótima. Isso é citável no TCC como garantia teórica.

A decomposição em componentes permite o ablation study da Fase 4:
treinar com/sem cada componente e medir o impacto na convergência.

### Impacto
- Cada componente retornado no `info` dict para logging granular
- TensorBoard mostra evolução individual de cada componente
- Previne reward hacking (como os "donuts" do projeto anterior)

---

## ADR-007: NPCs com Waypoint-Following vs Pathfinding A*

**Data:** 2026-05-13  
**Status:** Aceita  

### Contexto
Veículos NPC e pedestres precisam se mover de forma plausível no mapa
para gerar cenários de trânsito para o agente.

### Decisão
- NPCs seguem listas cíclicas de waypoints pré-definidos nas vias
- Veículos respeitam semáforos via verificação de proximidade + dot product
- Pedestres caminham nas faixas com velocidade 0.6 px/frame
- Modelo cinemático: ponto com velocidade + heading (sem bicycle model)

### Alternativas Descartadas
- **A* pathfinding:** Overengineering para 4 vias retas. O custo de
  manter um grafo de navegação não se justifica
- **Bicycle model para NPCs:** Os NPCs não precisam de física realista,
  apenas precisam existir como obstáculos detectáveis
- **Behavior trees:** Complexidade desnecessária para rotas fixas

### Impacto
- O(1) por frame por NPC (sem busca de caminho)
- Rotas determinísticas garantem reprodutibilidade
- 4 veículos + 3 pedestres por episódio (ajustável)

---

## ADR-008: Ground-Truth Detections na Fase 1 (sem YOLO)

**Data:** 2026-05-13  
**Status:** Aceita (temporária — será substituída na Fase 3)  

### Contexto
Na Fase 1 o YOLO ainda não foi treinado. Precisamos de um placeholder
para as detecções no observation space.

### Decisão
- `_build_ground_truth_detections()` gera o vetor de detecções usando
  as posições reais dos objetos (ground-truth do simulador)
- O formato é idêntico ao que o YOLO produzirá: [class_id, cx, cy, w, h, conf]
- Na Fase 3, este método será substituído pelo pipeline YOLO real

### Alternativas Descartadas
- **Pular detecções (usar só ego_state):** O observation space ficaria
  incompleto, e o agente não aprenderia a reagir a objetos
- **YOLO pré-treinado no COCO:** Classes diferentes (person, car em
  contexto real vs sprites 2D), mAP seria muito baixo

### Impacto
- A interface da observation não muda entre Fases 1 e 3
- Permite treinar o agente com detecções perfeitas (upper bound)
  e depois comparar com detecções YOLO ruidosas (ablation)

---

*Novas ADRs serão adicionadas conforme o projeto avança pelas Fases 2-5.*
