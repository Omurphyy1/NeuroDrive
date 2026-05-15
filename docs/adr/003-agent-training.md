# NeuroDrive — ADR Fase 3-4: Agente DRL (PPO)

> Decisões técnicas do pipeline de treinamento e avaliação.

---

## ADR-012: PPO com MultiInputPolicy para Dict Observation

**Data:** 2026-05-15  
**Status:** Aceita  

### Contexto
O observation space é um Dict com dois campos (detections + ego_state).
Precisamos de uma policy que processe ambos os campos.

### Decisão
- `MultiInputPolicy` do SB3: cria MLPs separadas para cada campo do Dict
  e concatena os embeddings antes da policy/value head
- Hyperparameters padrão do PPO (lr=3e-4, gamma=0.99, clip=0.2, ent_coef=0.01)
- DummyVecEnv (não SubprocVecEnv) para simplicidade no Windows

### Alternativas Descartadas
- **CnnPolicy com pixels brutos:** Requer CNN profunda, 10x mais lento
- **Custom Feature Extractor:** Desnecessário — MultiInputPolicy já separa
  os campos do Dict automaticamente
- **SubprocVecEnv:** Problemas de serialização no Windows, DummyVecEnv
  é suficiente para ambiente leve

### Impacto
- Treino funciona em CPU (sem GPU obrigatória)
- MultiInputPolicy trata detections (2D) e ego_state (1D) separadamente
- Entropy coefficient (0.01) encoraja exploração inicial

---

## ADR-013: Avaliação Separada do Treino

**Data:** 2026-05-15  
**Status:** Aceita  

### Contexto
Durante o treino usamos VecEnv para paralelismo. Mas para análise de
resultados precisamos de métricas granulares (goal rate, collision rate).

### Decisão
- evaluate.py usa env direto (não VecEnv) para acessar info dict completo
- Métricas coletadas: reward médio/std, steps médios, taxas de outcome
- Modo render opcional para visualização PyGame

### Alternativas Descartadas
- **Avaliação dentro do treino (via callback):** Limitada ao que o
  EvalCallback expõe (apenas mean reward), sem breakdown de outcomes

### Impacto
- Métricas quantitativas publicáveis no TCC
- Reprodutibilidade total via seed
- Comparação objetiva entre variantes (ablation study)

---

*Continuação de docs/adr/002-vision-pipeline.md (ADR-009 a ADR-011).*
