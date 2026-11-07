"""Reference window features for the starter kit (numpy / pandas only).

Every function takes a slice of a public drive table (`pandas.DataFrame` with the `vtt.schema.DRIVE_COLUMNS`
columns, normally the 600 rows of one 60 s window) and returns plain floats. Units: speed m/s, acceleration
m/s^2, steering angle deg, yaw rate rad/s, curvature 1/m, time s. The nominal sample period is 0.1 s.

These features are a reference implementation of the quantities the dataset was built around (acceleration /
deceleration episodes, lift-off deceleration, steering gain, car-following and the catch-up response after a lead
acceleration), not the official protocol itself: a model may use any function of the official inputs
(`vtt.config.OFFICIAL_INPUTS`). When the pipeline's own detectors (`vtt.features_core`) are importable they are
used for episodes and the response delay, so that the counts agree with the windows index; otherwise the local
implementation below (same rules, config.ACCEL_EP / config.LEAD_ACCEL_EVENT) applies. Lead-acceleration onsets are
always detected locally, with a short transition allowance between the quiet window and the hold (see
lead_accel_onsets), which the pipeline's stricter detector does not have.
Episode bounds are half-open sample indices [i0, i1).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

try:                                                     # the starter kit also runs outside the build repository
    from vtt import config as C
    ACCEL_EP = dict(C.ACCEL_EP)
    LEAD_ACCEL_EVENT = dict(C.LEAD_ACCEL_EVENT)
    THW_CF_S = C.THW_CF_S
    MOVING_V_MPS = C.MOVING_V_MPS
except ImportError:                                      # pragma: no cover - defaults mirror vtt.config
    ACCEL_EP = dict(min_s=3.0, min_dv_mps=2.0, min_mean_a=0.1, eps_mps=0.3, sg_win=5, sg_order=2)
    LEAD_ACCEL_EVENT = dict(slope_on=0.5, hold_s=1.0, quiet_s=2.0, quiet_abs=0.3, resp_a=0.3, resp_hold_s=0.5, horizon_s=8.0)
    THW_CF_S, MOVING_V_MPS = 6.0, 1.0

try:
    from vtt import features_core as CORE                # noqa: N812 - optional, written by the pipeline
except ImportError:                                      # pragma: no cover
    CORE = None

DT_S = 0.1

FEATURE_NAMES = [
    # speed
    "v_mean", "v_max", "v_std", "moving_frac", "standstill_frac", "a_std", "a_p95", "a_p05",
    # acceleration / deceleration episodes
    "n_accel_ep", "accel_dv_mean", "accel_mean_a", "accel_peak_a", "accel_t_peak_s", "accel_plateau_frac",
    "n_decel_ep", "decel_dv_mean", "decel_mean_a", "decel_peak_a", "decel_t_peak_s", "decel_plateau_frac",
    # lift-off (no pedal, no automation) deceleration
    "coast_decel_mean", "coast_frac",
    # steering
    "torque_abs_mean", "torque_abs_p95", "steer_pressed_frac", "steer_rate_abs_mean", "steer_angle_abs_mean", "k_steer_degm",
    # yaw / curvature
    "yaw_abs_mean", "yaw_abs_p95", "curv_abs_mean", "lat_acc_abs_p95", "yaw_valid_frac",
    # lead
    "lead_frac", "cf_frac", "thw_med", "thw_p10", "ttc_min", "lead_speed_mean", "lead_d_mean", "lead_vrel_std",
    # catch-up response to a lead acceleration
    "n_catchup_events", "catchup_responded_frac", "catchup_delay_mean_s", "catchup_gap_change_5s_mean_m", "catchup_accel_ratio_mean",
]


# ------------------------------------------------------------------------------------------------ smoothing
def sg_kernel(win: int, order: int, deriv: int = 0) -> np.ndarray:
    """Savitzky-Golay FIR kernel (centred least-squares polynomial fit) ready for `np.convolve`.

    deriv=0 smooths, deriv=1 returns the derivative per sample."""
    if win % 2 == 0 or win < order + 1:
        raise ValueError("win must be odd and > order")
    half = win // 2
    x = np.arange(-half, half + 1, dtype=float)
    pinv = np.linalg.pinv(np.vander(x, order + 1, increasing=True))
    return (pinv[deriv] * math.factorial(deriv))[::-1]


def _fill_nan(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, float).copy()
    bad = ~np.isfinite(y)
    if bad.all():
        return np.zeros_like(y)
    if bad.any():
        idx = np.arange(len(y))
        y[bad] = np.interp(idx[bad], idx[~bad], y[~bad])
    return y


def smooth(y, win: int = 5, order: int = 2, deriv: int = 0, dt: float = DT_S) -> np.ndarray:
    """SG smoothing of a 1-D signal (deriv=0) or its derivative per second (deriv=1). The first / last win//2
    samples use the polynomial fitted to the first / last window (so polynomials up to `order` are reproduced
    exactly everywhere, like scipy's mode='interp' and vtt.features_core.savgol_smooth)."""
    y = _fill_nan(y)
    n, half = len(y), win // 2
    if n < win:
        return y if deriv == 0 else (np.gradient(y, dt) if n > 1 else np.zeros_like(y))
    out = np.convolve(y, sg_kernel(win, order, deriv), mode="valid")
    out = np.concatenate([np.zeros(half), out, np.zeros(half)])
    x = np.arange(win, dtype=float)
    for sl, xs in ((slice(0, half), x[:half]), (slice(n - half, n), x[win - half:])):
        coef = np.polyfit(x, y[sl.start - (0 if sl.start == 0 else win - half):sl.start + (win if sl.start == 0 else half)], order)
        poly = np.polyder(np.poly1d(coef), deriv) if deriv else np.poly1d(coef)
        out[sl] = poly(xs)
    return out if deriv == 0 else out / dt ** deriv


# ------------------------------------------------------------------------------------------------ episodes
def monotone_runs(v_s: np.ndarray, sign: int, eps: float) -> list[tuple[int, int]]:
    """Maximal runs where sign*v_s never drops more than eps below its running maximum (inclusive bounds)."""
    s = sign * np.asarray(v_s, float)
    runs: list[tuple[int, int]] = []
    i, n = 0, len(s)
    while i < n - 1:
        j, run_max = i, s[i]
        while j + 1 < n and s[j + 1] >= run_max - eps:
            run_max = max(run_max, s[j + 1]); j += 1
        if j > i:
            runs.append((i, j))
        i = j + 1
    return runs


def accel_episodes(v, sign: int = 1, params: dict | None = None, dt: float = DT_S) -> list[tuple[int, int]]:
    """Acceleration (sign=+1) or deceleration (sign=-1) episodes as half-open (i0, i1) sample bounds.

    Rules (vtt.config.ACCEL_EP): the SG-smoothed speed is monotone within eps_mps, the run lasts >= min_s,
    |dv| >= min_dv_mps and the mean |acceleration| >= min_mean_a. Delegates to vtt.features_core when present."""
    p = dict(ACCEL_EP, **(params or {}))
    v = np.asarray(v, float)
    if CORE is not None:
        fn = CORE.accel_episodes if sign > 0 else CORE.decel_episodes
        return [(int(e[0]), int(e[1])) for e in fn(v, dt=dt, **p)]
    v_s = smooth(v, p["sg_win"], p["sg_order"])
    out = []
    for i0, i1 in monotone_runs(v_s, sign, p["eps_mps"]):
        dur = (i1 - i0) * dt
        dv = sign * (v_s[i1] - v_s[i0])
        if dur >= p["min_s"] and dv >= p["min_dv_mps"] and dv / dur >= p["min_mean_a"]:
            out.append((i0, i1 + 1))
    return out


def episode_stats(v, i0: int, i1: int, sign: int = 1, dt: float = DT_S) -> dict:
    """Shape of one episode [i0, i1): duration, |dv|, mean and peak |acceleration| of the smoothed speed, time from
    the start to the peak, plateau fraction (share of the episode with |acceleration| >= 75 % of its peak)."""
    a_s = sign * smooth(v, ACCEL_EP["sg_win"], ACCEL_EP["sg_order"], deriv=1, dt=dt)[i0:i1]
    v_s = smooth(v, ACCEL_EP["sg_win"], ACCEL_EP["sg_order"])
    peak = float(a_s.max()) if len(a_s) else np.nan
    return dict(dur_s=(i1 - i0) * dt, dv=float(sign * (v_s[i1 - 1] - v_s[i0])), mean_a=float(a_s.mean()) if len(a_s) else np.nan,
                peak_a=peak, t_peak_s=float(np.argmax(a_s)) * dt if len(a_s) else np.nan,
                plateau_frac=float(np.mean(a_s >= 0.75 * peak)) if peak > 0 else np.nan)


def episode_features(df: pd.DataFrame) -> dict:
    """Counts and mean episode shapes for acceleration and deceleration episodes of the window."""
    v = df["v_mps"].to_numpy(float)
    out: dict[str, float] = {}
    for sign, tag in ((1, "accel"), (-1, "decel")):
        eps = accel_episodes(v, sign)
        st = [episode_stats(v, i0, i1, sign) for i0, i1 in eps]
        out[f"n_{tag}_ep"] = len(eps)
        for key, name in (("dv", "dv_mean"), ("mean_a", "mean_a"), ("peak_a", "peak_a"), ("t_peak_s", "t_peak_s"), ("plateau_frac", "plateau_frac")):
            vals = [s[key] for s in st if np.isfinite(s[key])]
            out[f"{tag}_{name}"] = float(np.mean(vals)) if vals else np.nan
    return out


# ------------------------------------------------------------------------------------------------ other groups
def coast_features(df: pd.DataFrame, v_min: float = 3.0, a_max: float = -0.05, min_n: int = 5) -> dict:
    """Lift-off deceleration: mean acceleration while rolling (v > v_min) with no pedal, no automation and a < a_max."""
    m = ((df["v_mps"] > v_min) & ~df["gas_pressed"] & ~df["brake_pressed"] & ~df["das_any"] & ~df["standstill"]).to_numpy()
    a = df["a_mps2"].to_numpy(float)
    coast = m & (a < a_max)
    return dict(coast_decel_mean=float(a[coast].mean()) if coast.sum() >= min_n else np.nan,
                coast_frac=float(coast.mean()) if len(a) else np.nan)


def speed_features(df: pd.DataFrame) -> dict:
    v = df["v_mps"].to_numpy(float); a = df["a_mps2"].to_numpy(float)
    moving = v >= MOVING_V_MPS
    am = a[moving & np.isfinite(a)]
    return dict(v_mean=float(np.nanmean(v)), v_max=float(np.nanmax(v)), v_std=float(np.nanstd(v)),
                moving_frac=float(moving.mean()), standstill_frac=float(df["standstill"].mean()),
                a_std=float(am.std()) if len(am) else np.nan,
                a_p95=float(np.quantile(am, 0.95)) if len(am) else np.nan, a_p05=float(np.quantile(am, 0.05)) if len(am) else np.nan)


def steering_features(df: pd.DataFrame) -> dict:
    """Driver steering while moving and without automation steering (steer_torque_norm is per-drive normalised)."""
    m = ((df["v_mps"] >= MOVING_V_MPS) & ~df["op_lat_active"]).to_numpy()
    tq = np.abs(df["steer_torque_norm"].to_numpy(float))[m]
    tq = tq[np.isfinite(tq)]
    rate = np.abs(df["steer_rate_degps"].to_numpy(float))[m]
    ang = np.abs(df["steer_angle_deg"].to_numpy(float))[m]
    return dict(torque_abs_mean=float(tq.mean()) if len(tq) else np.nan, torque_abs_p95=float(np.quantile(tq, 0.95)) if len(tq) else np.nan,
                steer_pressed_frac=float(df["steer_pressed"].to_numpy()[m].mean()) if m.any() else np.nan,
                steer_rate_abs_mean=float(np.nanmean(rate)) if m.any() else np.nan,
                steer_angle_abs_mean=float(np.nanmean(ang)) if m.any() else np.nan, k_steer_degm=k_steer(df))


def k_steer(df: pd.DataFrame, v_min: float = 3.0, curv_min: float = 1e-3, min_n: int = 20) -> float:
    """Steering gain [deg*m]: least-squares slope through the origin of steer_angle_deg against curv_1pm over
    valid-yaw samples with v > v_min and |curvature| >= curv_min (a steering-ratio proxy; brand related)."""
    m = ((df["v_mps"] > v_min) & df["yaw_valid"]).to_numpy()
    c = df["curv_1pm"].to_numpy(float); ang = df["steer_angle_deg"].to_numpy(float)
    m &= np.isfinite(c) & np.isfinite(ang) & (np.abs(c) >= curv_min)
    if m.sum() < min_n:
        return np.nan
    den = float((c[m] * c[m]).sum())
    return float((ang[m] * c[m]).sum() / den) if den > 0 else np.nan


def yaw_features(df: pd.DataFrame) -> dict:
    m = (df["v_mps"] >= MOVING_V_MPS).to_numpy() & df["yaw_valid"].to_numpy()
    yaw = np.abs(df["yaw_rate_radps"].to_numpy(float))[m]; curv = np.abs(df["curv_1pm"].to_numpy(float))[m]
    latacc = np.abs(df["lat_acc_mps2"].to_numpy(float))[m]
    yaw, curv, latacc = yaw[np.isfinite(yaw)], curv[np.isfinite(curv)], latacc[np.isfinite(latacc)]
    return dict(yaw_abs_mean=float(yaw.mean()) if len(yaw) else np.nan, yaw_abs_p95=float(np.quantile(yaw, 0.95)) if len(yaw) else np.nan,
                curv_abs_mean=float(curv.mean()) if len(curv) else np.nan,
                lat_acc_abs_p95=float(np.quantile(latacc, 0.95)) if len(latacc) else np.nan,
                yaw_valid_frac=float(df["yaw_valid"].mean()))


def lead_features(df: pd.DataFrame) -> dict:
    lead = df["lead_valid"].to_numpy(bool)
    thw = df["thw_s"].to_numpy(float); ttc = df["ttc_s"].to_numpy(float)
    thw_v = thw[lead & np.isfinite(thw)]; ttc_v = ttc[lead & np.isfinite(ttc)]
    return dict(lead_frac=float(lead.mean()), cf_frac=float((thw_v <= THW_CF_S).mean()) if len(thw_v) else np.nan,
                thw_med=float(np.median(thw_v)) if len(thw_v) else np.nan, thw_p10=float(np.quantile(thw_v, 0.10)) if len(thw_v) else np.nan,
                ttc_min=float(ttc_v.min()) if len(ttc_v) else np.nan,
                lead_speed_mean=float(np.nanmean(df["lead_speed_mps"].to_numpy(float)[lead])) if lead.any() else np.nan,
                lead_d_mean=float(np.nanmean(df["lead_d_m"].to_numpy(float)[lead])) if lead.any() else np.nan,
                lead_vrel_std=float(np.nanstd(df["lead_v_rel_mps"].to_numpy(float)[lead])) if lead.sum() > 1 else np.nan)


# ------------------------------------------------------------------------------------------------ catch-up events
def _held(mask: np.ndarray, n: int) -> np.ndarray:
    """True at i when mask is True at i .. i+n-1 (a run of n starting at i)."""
    if n <= 1:
        return mask.copy()
    c = np.concatenate([[0], np.cumsum(mask.astype(int))])
    out = np.zeros(len(mask), bool)
    k = len(mask) - n + 1
    if k > 0:
        out[:k] = (c[n:n + k] - c[:k]) == n
    return out


def lead_accel_onsets(slope: np.ndarray, lead_valid: np.ndarray, p: dict, dt: float = DT_S, trans_s: float = 0.5) -> np.ndarray:
    """Onset sample indices of lead accelerations: the lead slope is >= slope_on for hold_s, preceded by quiet_s of
    |slope| < quiet_abs, with up to trans_s of unconstrained transition between the quiet window and the hold
    (a 1 s least-squares slope needs a few samples to climb from < quiet_abs to >= slope_on). All samples must
    carry a valid lead. trans_s = 0 is the stricter rule of vtt.features_core.lead_accel_events."""
    n_hold, n_quiet = (max(1, int(round(p[k] / dt))) for k in ("hold_s", "quiet_s"))
    n_trans = max(0, int(round(trans_s / dt)))
    s = np.asarray(slope, float); valid = np.asarray(lead_valid, bool)
    hi = _held(valid & (np.nan_to_num(s, nan=-np.inf) >= p["slope_on"]), n_hold)
    quiet = _held(valid & (np.nan_to_num(np.abs(s), nan=np.inf) < p["quiet_abs"]), n_quiet)
    idx = np.arange(n_quiet + n_trans, len(s))
    on = hi[idx] & ~np.concatenate([[False], hi[idx[:-1]]])          # first sample of each qualifying hold
    ok = np.zeros(len(idx), bool)
    for g in range(n_trans + 1):                                       # quiet window ending 0 .. n_trans samples before
        ok |= quiet[idx - n_quiet - g]
    return idx[on & ok]


def response_delay(a: np.ndarray, onset: int, p: dict, dt: float = DT_S) -> float:
    """Seconds from onset to the first sample where a >= resp_a holds for resp_hold_s within horizon_s (NaN if none)."""
    if CORE is not None:
        return float(CORE.ego_response_delay(a, int(onset), dt=dt, **p))
    n_resp, n_hor = (max(1, int(round(p[k] / dt))) for k in ("resp_hold_s", "horizon_s"))
    resp = _held(np.nan_to_num(np.asarray(a, float), nan=-np.inf) >= p["resp_a"], n_resp)
    r = np.flatnonzero(resp[onset:min(len(a), onset + n_hor + 1)])
    return float(r[0] * dt) if len(r) else np.nan


def catchup_events(df: pd.DataFrame, params: dict | None = None, human_only: bool = True, dt: float = DT_S) -> list[dict]:
    """Lead-acceleration events in car-following range and the ego response (vtt.config.LEAD_ACCEL_EVENT).

    An onset counts when the lead is in car-following range (thw <= THW_CF_S) and, with human_only, the
    automation is off at the onset. Per event: delay_s (response delay, NaN when none within horizon_s),
    gap_change_5s_m = lead_d(onset + 5 s) - lead_d(onset), accel_ratio = mean ego acceleration over the 2 s after
    the response divided by the mean lead slope over the hold period. Events closer than horizon_s to the previous
    one are skipped so that responses are not double counted."""
    p = dict(LEAD_ACCEL_EVENT, **(params or {}))
    n = len(df)
    lead = df["lead_valid"].to_numpy(bool); slope = df["lead_a_slope_mps2"].to_numpy(float)
    a = df["a_mps2"].to_numpy(float); d = df["lead_d_m"].to_numpy(float); thw = df["thw_s"].to_numpy(float)
    das = df["das_any"].to_numpy(bool)
    n_hold = max(1, int(round(p["hold_s"] / dt))); n_hor = int(round(p["horizon_s"] / dt)); n5 = int(round(5.0 / dt))
    events, last_end = [], -1
    for i in lead_accel_onsets(slope, lead, p, dt):
        i = int(i)
        if i <= last_end or not (np.isfinite(thw[i]) and thw[i] <= THW_CF_S) or (human_only and das[i]):
            continue
        delay = response_delay(a, i, p, dt)
        k = i + int(round(delay / dt)) if np.isfinite(delay) else -1
        gap = float(d[i + n5] - d[i]) if i + n5 < n and lead[i + n5] and lead[i] else np.nan
        lead_a = float(np.nanmean(slope[i:i + n_hold]))
        ratio = float(np.nanmean(a[k:k + 20]) / lead_a) if k >= 0 and lead_a > 0 else np.nan
        events.append(dict(onset_row=i, delay_s=delay, responded=bool(np.isfinite(delay)), gap_change_5s_m=gap,
                           accel_ratio=ratio, lead_a=lead_a, das_at_onset=bool(das[i])))
        last_end = i + n_hor
    return events


def catchup_features(df: pd.DataFrame) -> dict:
    ev = catchup_events(df)
    delays = [e["delay_s"] for e in ev if np.isfinite(e["delay_s"])]
    gaps = [e["gap_change_5s_m"] for e in ev if np.isfinite(e["gap_change_5s_m"])]
    ratios = [e["accel_ratio"] for e in ev if np.isfinite(e["accel_ratio"])]
    return dict(n_catchup_events=len(ev), catchup_responded_frac=float(np.mean([e["responded"] for e in ev])) if ev else np.nan,
                catchup_delay_mean_s=float(np.mean(delays)) if delays else np.nan,
                catchup_gap_change_5s_mean_m=float(np.mean(gaps)) if gaps else np.nan,
                catchup_accel_ratio_mean=float(np.mean(ratios)) if ratios else np.nan)


# ------------------------------------------------------------------------------------------------ assembly
def window_features(df: pd.DataFrame) -> dict:
    """All FEATURE_NAMES for one window slice (NaN where undefined)."""
    out: dict[str, float] = {}
    for fn in (speed_features, episode_features, coast_features, steering_features, yaw_features, lead_features, catchup_features):
        out.update(fn(df))
    return {k: out[k] for k in FEATURE_NAMES}


def features_table(windows: pd.DataFrame, load_drive, limit: int | None = None) -> pd.DataFrame:
    """Features for every row of a windows index; `load_drive(drive_id) -> DataFrame` reads one drive table.
    Drives are read once each (rows grouped by drive_id)."""
    rows = []
    sub = windows if limit is None else windows.head(limit)
    for drive_id, grp in sub.groupby("drive_id", sort=False):
        drive = load_drive(drive_id)
        for r in grp.itertuples(index=False):
            feats = window_features(drive.iloc[int(r.row0):int(r.row1)])
            feats["window_id"] = r.window_id
            rows.append(feats)
    return pd.DataFrame(rows, columns=["window_id", *FEATURE_NAMES])
