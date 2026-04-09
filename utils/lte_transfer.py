import numpy as np

from envs.lte_scheduler_env import (
    CQI_BITS_PER_RE,
    MAX_AVG_TPUT,
    MAX_BUFFER_BYTES,
    RE_PER_RB_TTI,
)


def one_rbg_tput_bps(cqi: int, buffer_bytes: float, rb_count: int) -> float:
    bpre = CQI_BITS_PER_RE.get(int(cqi), 0.0)
    raw_bits = rb_count * RE_PER_RB_TTI * bpre
    return min(raw_bits, buffer_bytes * 8.0) * 1000.0


def select_topk_pf_candidates(
    env,
    k: int,
    min_avg_tput_bps: float = 1e5,
) -> tuple[np.ndarray, np.ndarray]:
    if k < 1:
        raise ValueError("k must be >= 1")
    if env.n_ue < k:
        raise ValueError(
            f"Cannot build a {k}-UE proxy from an environment with only {env.n_ue} UE."
        )

    rb_count = int(env.rbg_rb_sizes[env._current_rbg])
    inst_tput_bps = np.array(
        [
            one_rbg_tput_bps(env._reported_wb_cqi[i], env._buffer[i], rb_count)
            for i in range(env.n_ue)
        ],
        dtype=np.float64,
    )
    pf_score = inst_tput_bps / np.maximum(env._avg_tput, min_avg_tput_bps)
    action_mask = env._get_action_mask()

    # Tie-breaks slightly in favor of fresher reports, better channel and deeper buffer.
    sort_key = pf_score - 1e-6 * env._wb_cqi_age + 1e-6 * env._reported_wb_cqi + 1e-12 * env._buffer
    active_idx = np.flatnonzero(action_mask)
    inactive_idx = np.flatnonzero(~action_mask)

    active_sorted = active_idx[np.argsort(-sort_key[active_idx])]
    if len(active_sorted) >= k:
        candidates = active_sorted[:k]
    else:
        inactive_sorted = inactive_idx[np.argsort(-sort_key[inactive_idx])]
        candidates = np.concatenate([active_sorted, inactive_sorted[: k - len(active_sorted)]])

    return candidates.astype(np.int32), pf_score


def build_proxy_obs(env, ue_indices: np.ndarray) -> np.ndarray:
    ue_indices = np.asarray(ue_indices, dtype=np.int32)
    obs = np.zeros(
        len(ue_indices) * env.N_UE_FEATURES + env.N_CONTEXT_FEATURES,
        dtype=np.float32,
    )
    wb_cqi_age_norm = env._get_wb_cqi_age_norm()
    alloc_frac_per_ue = env._get_alloc_frac_per_ue()

    for slot, ue in enumerate(ue_indices):
        b = slot * env.N_UE_FEATURES
        obs[b + 0] = float(env._reported_wb_cqi[ue]) / 15.0
        obs[b + 1] = float(wb_cqi_age_norm[ue])
        obs[b + 2] = float(env._active_flag[ue])
        obs[b + 3] = float(env._buffer[ue]) / MAX_BUFFER_BYTES
        obs[b + 4] = float(env._avg_tput[ue]) / MAX_AVG_TPUT
        obs[b + 5] = float(alloc_frac_per_ue[ue])

    c = len(ue_indices) * env.N_UE_FEATURES
    obs[c + 0] = env._current_rbg / env.n_rbg
    obs[c + 1] = env._current_tti / max(env.episode_len, 1)
    obs[c + 2] = np.sum(env._rbg_alloc >= 0) / env.n_rbg

    return np.clip(obs, 0.0, 1.0)


def predict_transfer_action(
    model,
    env,
    model_n_ue: int,
    deterministic: bool = True,
    min_avg_tput_bps: float = 1e5,
) -> tuple[int, np.ndarray | None, np.ndarray, np.ndarray | None]:
    if env.n_ue == model_n_ue:
        obs = env._get_obs()
        action, _ = model.predict(obs, deterministic=deterministic)
        return int(action), None, obs, None

    if env.n_ue < model_n_ue:
        raise ValueError(
            f"Model expects {model_n_ue} UE but evaluation env has only {env.n_ue} UE."
        )

    candidates, pf_score = select_topk_pf_candidates(
        env,
        k=model_n_ue,
        min_avg_tput_bps=min_avg_tput_bps,
    )
    proxy_obs = build_proxy_obs(env, candidates)
    local_action, _ = model.predict(proxy_obs, deterministic=deterministic)
    local_action = int(local_action)

    if not 0 <= local_action < len(candidates):
        raise ValueError(
            f"Model produced local action {local_action}, expected 0..{len(candidates) - 1}."
        )

    return int(candidates[local_action]), candidates, proxy_obs, pf_score
