# DRL Playground (DQN + PPO)

Учебный проект для изучения Deep Reinforcement Learning на простых средах
(CartPole и toy-задачи), с возможностью расширения до прикладных задач в телекоммуникациях.

## Установка (Windows, venv, CPU)

```powershell
git clone <repo_url> drl_playground
cd drl_playground

python -m venv .venv
.\.venv\Scripts\activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Запуск скрипта (проверка)

```powershell
python -m scripts.train_dqn_cartpole
```

Дальше шаг за шагом планирую добавлять реализацию DQN и PPO.