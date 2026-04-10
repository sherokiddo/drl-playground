# DRL Playground

Проект предназначен для экспериментов с Deep Reinforcement Learning на двух уровнях:

- toy-задачи: `CartPole`, `FrozenLake`
- прикладная LTE-задача: downlink scheduler с custom Gymnasium environment

Сейчас основной фокус репозитория — LTE scheduler (планировщик):

- среда с `RAT0`, переменным числом UE и полосой
- traffic modes: `full_buffer`, `on_off`, `bursty`
- `true` и `reported` wideband CQI с периодичностью и aging
- custom masked DQN с shared per-UE encoder
- curriculum / domain randomization по `n_ue`, `bandwidth`, `wb_cqi_report_period_tti`

## Что где

### Основной LTE pipeline

- [envs/lte_scheduler_env.py](/drl_playground/envs/lte_scheduler_env.py)  
  Основная LTE-среда: RAT0-геометрия, буферы, CQI, reward, метрики, history.

- [envs/lte_padded_env.py](/drl_playground/envs/lte_padded_env.py)  
  Wrapper-адаптер над LTE-средой. Паддит observation и action mask до фиксированного `max_n_ue`, чтобы сеть и replay buffer работали с постоянной размерностью.

- [agents/lte_dqn_agent.py](/drl_playground/agents/lte_dqn_agent.py)  
  Custom masked DQN:
  - shared per-UE Q-network
  - action masking
  - target network
  - replay buffer

- [scripts/train_lte_dqn.py](/drl_playground/scripts/train_lte_dqn.py)  
  Основной train script для LTE DQN.

- [scripts/animate_scheduler.py](/drl_playground/scripts/animate_scheduler.py)  
  Live-визуализация scheduler-а:
  - resource grid
  - KPI-панель
  - rolling `SE / JFI(active) / JFI(all)`
  - per-UE throughput
  - allocation по последнему TTI

- [scripts/bench_inference.py](/drl_playground/scripts/bench_inference.py)  
  Benchmark инференса и `env.step`.

- [scripts/sanity_check.py](/drl_playground/scripts/sanity_check.py)  
  Быстрый smoke-test среды без обучения.

### Legacy / baseline

- [scripts/train_dqn_sb3.py](/drl_playground/scripts/train_dqn_sb3.py)  
  Старый baseline на `stable-baselines3.DQN`. Полезен как артефакт сравнения, но основной LTE-пайплайн уже не на нём.

- [utils/lte_transfer.py](/drl_playground/utils/lte_transfer.py)  
  Временный helper для transfer-экспериментов старой 3-UE модели на большее число UE. Для нового custom DQN не является основным путём.

### Toy RL

- [agents/dqn_agent.py](/drl_playground/agents/dqn_agent.py)
- [agents/ppo_agent.py](/drl_playground/agents/ppo_agent.py)
- [scripts/train_dqn_cartpole.py](/drl_playground/scripts/train_dqn_cartpole.py)
- [scripts/train_ppo_cartpole.py](/drl_playground/scripts/train_ppo_cartpole.py)
- [scripts/train_dqn_frozenlake.py](/drl_playground/scripts/train_dqn_frozenlake.py)

## Установка

### Windows + venv

```powershell
git clone https://github.com/sherokiddo/drl-playground drl_playground
cd drl_playground

python -m venv .venv
.\.venv\Scripts\activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` покрывает основной custom LTE DQN pipeline:

- `torch`
- `gymnasium`
- `numpy`
- `matplotlib`

Если нужен legacy SB3 baseline:

```powershell
pip install stable-baselines3
```

## LTE: кратко о постановке

Сейчас в среде моделируется:

- `RAT0`-распределение ресурсов по LTE bandwidth
- переменное число `UE`
- один шаг среды = назначение текущего `RBG` одному `UE`
- после всех `RBG` завершается `TTI`
- буферы, offered load и activity пользователей
- `true_wb_cqi` и `reported_wb_cqi`
- periodic CQI reporting и `CQI age`
- long-term throughput через EMA

### Reward и fairness

Reward сейчас сочетает:

- spectral efficiency term
- proportional-fair utility increment
- soft penalty, если `JFI(active) < target`

В проекте используются две fairness-метрики:

- `JFI(active)`  
  fairness только по активным/backlogged UE. Это основная метрика для scheduler-а и именно она идёт в reward.

- `JFI(all)`  
  fairness по всем UE, включая неактивных. Это диагностическая системная метрика.

## Быстрый старт

### 1. Проверить среду

```powershell
python scripts/sanity_check.py
```

Что это даёт:

- вывод shape observation
- проверка `action_mask`
- краткая сводка по `throughput / SE / JFI / reward`

### 2. Запустить обучение LTE DQN

```powershell
python scripts/train_lte_dqn.py
```

Артефакты:

- веса: [runs/lte_dqn/lte_dqn_shared_q.pt](/drl_playground/runs/lte_dqn/lte_dqn_shared_q.pt)
- train metrics: [runs/lte_dqn/train_metrics.csv](/drl_playground/runs/lte_dqn/train_metrics.csv)
- eval metrics: [runs/lte_dqn/eval_metrics.csv](/drl_playground/runs/lte_dqn/eval_metrics.csv)

### 3. Посмотреть scheduler вживую

```powershell
python scripts/animate_scheduler.py
```

### 4. Замерить производительность инференса

```powershell
python scripts/bench_inference.py
```

## Что делает `train_lte_dqn.py`

Скрипт обучает одну policy на смеси сценариев.

### Curriculum / scenario randomization

Во время обучения перемешиваются сценарии по:

- числу UE
- bandwidth через `n_rb_dl`
- периоду `wideband CQI report`

Сейчас в train/eval сценариях есть:

- `3 UE / 10 MHz / wb=5`
- `8 UE / 10 MHz / wb=5`
- `16 UE / 10 MHz / wb=5`
- `40 UE / 10 MHz / wb=5`
- `16 UE / 5 MHz / wb=5`
- `16 UE / 20 MHz / wb=5`
- `16 UE / 10 MHz / wb=1`
- `16 UE / 10 MHz / wb=10`

### Почему нужен `lte_padded_env`

`lte_scheduler_env.py` — это настоящая логика сети.  
`lte_padded_env.py` — это специальный ML-адаптер.

Он нужен, потому что:

- базовая LTE-среда живёт с реальным `n_ue`
- нейросеть и replay buffer хотят фиксированные размеры входа/выхода
- wrapper паддит observation и action mask до `max_n_ue`

Иными словами:

- `lte_scheduler_env` отвечает за simulation logic
- `lte_padded_env` отвечает за fixed-shape ML interface

## Что смотреть в результатах

Минимальный набор метрик:

- `mean_throughput_mbps`
- `mean_se_bps_hz`
- `mean_jfi_active`
- `mean_jfi_all`
- `mean_reward`
- `invalid_action_rate`

На что обращать внимание:

- высокий `SE` при плохом `JFI(active)` означает throughput-greedy policy
- большая разница между `JFI(active)` и `JFI(all)` часто говорит о том, что часть просадки fairness вызвана inactive UE, а не только scheduler-ом
- `invalid_action_rate` должен быть около нуля

## Визуализация

`animate_scheduler.py` показывает:

- resource grid
- live KPI
- rolling `SE / JFI(active) / JFI(all)`
- per-UE average throughput
- allocation по последнему TTI
- финальную summary в консоль

Это удобно для поиска:

- starvation
- domination одного UE
- деградации fairness при stale CQI
- неустойчивых периодов работы scheduler-а

## Benchmark

`bench_inference.py` сейчас меряет три вещи:

- `Only predict`
- `Only env.step`
- `Predict + step`

Сценарий бенча печатается в начале запуска:

- `n_ue`
- `n_rb_dl`
- `n_rbg`
- `traffic profile`
- `wb_cqi_report_period_tti`

Это полезно, потому что при сравнении latency важно не терять контекст сценария.

## Текущее состояние проекта

### Уже реализовано

- LTE env с `RAT0`
- activity-aware traffic
- stale wideband CQI
- `JFI(active)` и `JFI(all)`
- custom shared-UE masked DQN
- padded env для переменного числа UE
- curriculum по `n_ue / bandwidth / wb reporting period`
- live animation и benchmark

### Ещё не реализовано

- `subband_cqi`
- true/reported `sb_cqi`
- HARQ / BLER / retransmissions
- QCI / bearers
- полноценный PPO baseline для LTE
- перенос политики в realistic simulator / `srsRAN`

## Полезные скрипты

### LTE

```powershell
python scripts/sanity_check.py
python scripts/train_lte_dqn.py
python scripts/animate_scheduler.py
python scripts/bench_inference.py
```

### Toy RL

```powershell
python -m scripts.train_dqn_cartpole
python -m scripts.train_ppo_cartpole
python -m scripts.train_dqn_frozenlake
```

## Примечания

- Для LTE сейчас основным путём считается `custom DQN`, а не `SB3 DQN`.
- Старые SB3-скрипты лучше рассматривать как baseline-ветку и исторический артефакт.
- PPO для LTE ещё не интегрирован как полноценный pipeline, хотя инфраструктурно проект к этому уже близок.
