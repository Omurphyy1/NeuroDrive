# NeuroDrive

**Deep Reinforcement Learning para Direção Autônoma em Cidade 2D**

> TCC — Engenharia de Controle e Automação  
> Inspirado nos princípios fundamentais do Tesla Autopilot

---

## Resumo

NeuroDrive é um agente autônomo treinado com **Proximal Policy Optimization (PPO)**
para navegar em um ambiente de jogo 2D top-down de cidade. O agente "enxerga" o mundo
através de capturas de tela processadas pelo **YOLOv8** e toma decisões de direção
baseadas nas detecções visuais.

### Arquitetura em 3 Camadas

```
┌─────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  CAMADA 1       │     │  CAMADA 2        │     │  CAMADA 3        │
│  Ambiente PyGame│────▶│  Visão YOLOv8    │────▶│  Agente PPO      │
│  (Gymnasium)    │     │  (Detecção)      │     │  (SB3)           │
│                 │◀────│                  │◀────│                  │
└─────────────────┘     └──────────────────┘     └──────────────────┘
     screenshot              bboxes              ação discreta
```

## Formalização do MDP

| Componente | Definição |
|-----------|-----------|
| **Estado (S)** | Dict: detecções YOLO (10×6) + ego_state (5,) |
| **Ação (A)** | Discrete(5): ACELERAR, FREAR, VIRAR_ESQ, VIRAR_DIR, PARAR |
| **Transição (T)** | Cinemática simplificada + dinâmica de NPCs/semáforos |
| **Recompensa (R)** | Potential-based shaping: progresso + segurança + trânsito + suavidade + terminal |
| **γ (discount)** | 0.99 |

## Mapa Urbano

```
     N
     │
  ┌──┤──┐
  │NW│NE│
W─┤  ┼  ├─E
  │SW│SE│
  └──┤──┘
     │
     S

NW: Loja (toldo azul)    NE: Posto de gasolina
SW: Praça (fonte+árvores) SE: Pizzaria (toldo vermelho)
```

## Instalação

```bash
# 1. Clonar o repositório
git clone https://github.com/Omurphyy1/NeuroDrive.git
cd NeuroDrive

# 2. Criar ambiente virtual
python -m venv venv

# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 3. Instalar dependências
pip install -r requirements.txt
```

## Uso Rápido

```bash
# Demo visual do ambiente (agente aleatório)
python -m neurodrive.env.city_env

# Rodar testes
pytest tests/ -v

# Treinar agente (Fase 4)
python -m neurodrive.agent.train --timesteps 1000000
```

## Estrutura do Projeto

```
neurodrive/
├── env/
│   ├── city_env.py          # CityDriveEnv (Gymnasium)
│   ├── tilemap.py           # Mapa urbano 640×640
│   ├── npc.py               # Veículos NPC e pedestres
│   └── traffic_light.py     # FSM dos semáforos
├── vision/
│   ├── detector.py          # Wrapper YOLOv8
│   ├── state_encoder.py     # bboxes → vetor de observação
│   └── data_collector.py    # Geração automática de dataset
├── agent/
│   ├── train.py             # Script de treino PPO
│   ├── evaluate.py          # Avaliação e métricas
│   └── reward.py            # Função de recompensa
├── tests/                   # Suite de testes (pytest)
└── docs/adr/                # Architecture Decision Records
```

## Roadmap

| Fase | Semanas | Escopo | Status |
|------|---------|--------|--------|
| 1 | 1–3 | Ambiente de Jogo | ✅ Completo |
| 2 | 4–6 | Dataset + Fine-tune YOLO | ✅ Completo |
| 3 | 7–9 | Integração YOLO ↔ Gymnasium | ✅ Completo |
| 4 | 10–13 | Treino DRL + Experimentos | 🔄 Em progresso |
| 5 | 14–16 | Análise + Escrita TCC | ⬜ Pendente |

## Stack Tecnológico

| Tecnologia | Versão | Propósito |
|-----------|--------|-----------|
| Python | 3.11 | Linguagem base |
| PyGame-CE | 2.5.2 | Engine de renderização 2D |
| Gymnasium | 0.29.1 | Interface RL padronizada |
| YOLOv8 | 8.2.0 | Detecção de objetos |
| PyTorch | 2.2.2 | Backend de deep learning |
| Stable-Baselines3 | 2.3.2 | Algoritmo PPO |
| TensorBoard | 2.17.0 | Visualização de métricas |

## Referências

- Schulman et al. (2017). "Proximal Policy Optimization Algorithms"
- Ng, Harada, Russell (1999). "Policy Invariance Under Reward Transformations"
- Redmon & Farhadi (2018). "YOLOv3: An Incremental Improvement"
- Jocher et al. (2023). "Ultralytics YOLOv8"

## Licença

Este projeto é parte de um Trabalho de Conclusão de Curso (TCC) e está
disponível para fins acadêmicos e educacionais.
