# NeuroDrive — Resultados do Treino PPO (Fase 4)

> Relatório do primeiro treino baseline (500k timesteps)

---

## Configuração do Experimento

| Parâmetro | Valor |
|-----------|-------|
| Algoritmo | PPO (Proximal Policy Optimization) |
| Policy | MultiInputPolicy |
| Total timesteps | 500.000 |
| N ambientes paralelos | 4 (DummyVecEnv) |
| Learning rate | 3e-4 |
| Batch size | 64 |
| N epochs | 10 |
| Gamma (γ) | 0.99 |
| GAE lambda | 0.95 |
| Clip range | 0.2 |
| Entropy coef | 0.01 |
| Seed | 42 |
| Device | CPU |
| Duração | 6min 44s (~1.247 FPS) |

## Métricas de Treino

| Métrica | Início | Final | Tendência |
|---------|--------|-------|-----------|
| explained_variance | 0.386 | 0.974 | ↗️ Convergindo |
| value_loss | 205 | 3.78 | ↘️ 54x redução |
| loss | 120 | 0.477 | ↘️ 251x redução |
| entropy | 1.60 | 0.632 | ↘️ Política mais determinística |
| approx_kl | 0.010 | 0.006 | ↘️ Estável |

## Resultados de Avaliação (20 episódios)

| Métrica | Valor |
|---------|-------|
| Reward médio | -203.85 ± 31.79 |
| Steps médios | 372 |
| Taxa de sucesso (goal) | 0.0% |
| Taxa de colisão | 75.0% |
| Taxa off-road | 25.0% |
| Taxa timeout | 0.0% |

## Análise

### O que está funcionando
- ✅ Pipeline completo roda sem erros (ambiente → treino → avaliação)
- ✅ Value function convergiu bem (explained_variance = 0.974)
- ✅ Entropia caiu naturalmente (exploração → exploração reduzida)
- ✅ KL divergence estável (~0.006) — sem colapso de policy

### O que precisa melhorar
- ❌ Taxa de sucesso 0% — o agente navega mas colide no cruzamento
- ❌ Predominância de colisões (75%) — provavelmente precisa de mais
  penalidade de proximidade ou mais timesteps de treino

### Próximos passos sugeridos
1. **Aumentar timesteps para 2M** — 500k pode ser insuficiente para
   aprender navegação completa com NPCs
2. **Ajustar reward shaping** — aumentar SAFETY_WEIGHT para penalizar
   mais a proximidade de objetos
3. **Fine-tune dos spawn points** — garantir que o agente não nasce
   em posições que levam inevitavelmente a colisão
4. **Ablation study** — comparar com/sem cada componente de recompensa

---

*Relatório gerado automaticamente em 2026-05-15*
