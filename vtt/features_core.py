"""Episode detectors shared by the window index (S6, vtt.windows) and the starter kit.

Pure numpy (no scipy), so the starter kit can import this module without the build environment:
  * Savitzky-Golay smoothing with a least-squares polynomial FIR kernel (savgol_kernel / savgol_smooth);
  * acceleration / deceleration episodes: the EV-paper definition (maximal runs over which the smoothed speed stays
    within eps of its running max / min, duration >= min_s, |dv| >= min_dv, mean |a| >= min_mean_a) rescaled from
    100 Hz to the 10 Hz public rate (config.ACCEL_EP: sg_win 5 samples ~ 0.5 s, order 2). A monotone run absorbs the
    steady cruise before and after a manoeuvre (within eps), which dilutes the mean-acceleration test of a short
    manoeuvre inside a long cruise; the optional `trim_win_s` switch (config.ACCEL_EP, 0 = literal rule) trims the
    run ends where the central slope of the smoothed speed is below min_mean_a before the tests (active_slope_mask);
  * lead-acceleration events (config.LEAD_ACCEL_EVENT) on the causal lead-speed slope column, inside lead episodes
    implied by lead_valid runs, and the ego response delay after such an onset.

Units: speeds in m/s, accelerations in m/s^2, times in s; `dt` is the sample period in s (0.1 at 10 Hz).
Indices are 0-based sample indices into the arrays passed in; episode bounds are [i0, i1) half-open.
"""
from __future__ import annotations

import math

import numpy as np

from vtt.config import ACCEL_EP, LEAD_ACCEL_EVENT, NOMINAL_DT_S

__all__ = ["savgol_kernel", "savgol_smooth", "monotone_runs", "active_slope_mask", "speed_episodes", "accel_episodes",
           "decel_episodes", "true_runs", "lead_accel_events", "ego_response_delay"]


# ----------------------------------------------------------------------------- Savitzky-Golay (least squares FIR)
def savgol_kernel(win: int, order: int, deriv: int = 0, dt: float = 1.0) -> np.ndarray:
    """Central Savitzky-Golay kernel of `win` taps (odd) fitting a polynomial of degree `order`.

    y[i] = sum_k c[k] * x[i - win//2 + k] is the value (deriv=0) or the deriv-th derivative (per unit of `dt`) of the
    least-squares polynomial through the window. deriv=0, win=5, order=2 gives (-3, 12, 17, 12, -3) / 35.
    """
    if win < 1 or win % 2 == 0:
        raise ValueError("win must be a positive odd integer")
    if order < 0 or order >= win:
        raise ValueError("order must satisfy 0 <= order < win")
    if deriv < 0 or deriv > order:
        raise ValueError("deriv must satisfy 0 <= deriv <= order")
    half = win // 2
    x = np.arange(-half, half + 1, dtype=float)
    vand = np.vander(x, order + 1, increasing=True)          # vand[k, p] = x_k ** p
    pinv = np.linalg.pinv(vand)                               # (order + 1, win): polynomial coefficients per tap
    return pinv[deriv] * math.factorial(deriv) / dt ** deriv


def savgol_smooth(x, win: int = ACCEL_EP["sg_win"], order: int = ACCEL_EP["sg_order"], deriv: int = 0,
                  dt: float = 1.0) -> np.ndarray:
    """Savitzky-Golay filtered copy of `x` (same length). Interior NaN are linearly interpolated and leading /
    trailing NaN held at the nearest finite value before filtering; the edges are padded with the edge value.
    The first / last win//2 samples are the polynomial fitted to the first / last window evaluated there (scipy's
    mode='interp'), so polynomials up to `order` are reproduced exactly everywhere.
    An all-NaN input returns NaN; an input shorter than `win` is returned unfiltered (deriv 0) or as np.gradient."""
    x = np.asarray(x, float)
    n = len(x)
    if n == 0:
        return x.copy()
    finite = np.isfinite(x)
    if not finite.any():
        return np.full(n, np.nan)
    xf = x
    if not finite.all():
        idx = np.flatnonzero(finite)
        xf = np.interp(np.arange(n), idx, x[idx])
    if n < win:
        return xf.copy() if deriv == 0 else np.gradient(xf, dt)
    c = savgol_kernel(win, order, deriv, dt)
    half = win // 2
    xp = np.pad(xf, half, mode="edge")
    y = np.correlate(xp, c, mode="valid")
    # edges: evaluate the least-squares polynomial of the first / last window at the outer taps (scipy mode='interp')
    xs = np.arange(-half, half + 1, dtype=float)
    pinv = np.linalg.pinv(np.vander(xs, order + 1, increasing=True))
    for head in (True, False):
        block = xf[:win] if head else xf[-win:]
        coef = pinv @ block                                 # polynomial coefficients in the window's local x
        pos = xs[:half] if head else xs[half + 1:]           # local x of the outer taps
        val = np.zeros(len(pos))
        for p_ in range(deriv, order + 1):
            val += coef[p_] * math.factorial(p_) / math.factorial(p_ - deriv) * pos ** (p_ - deriv)
        val /= dt ** deriv
        if head:
            y[:half] = val
        else:
            y[-half:] = val
    return y


# ----------------------------------------------------------------------------- monotone runs and speed episodes
def monotone_runs(v, eps: float, increasing: bool = True) -> np.ndarray:
    """Maximal runs [i0, i1] (inclusive, (n, 2) int array) over which `v` stays within `eps` of its running max
    (increasing=True) or running min (increasing=False); the EV-paper 'windowed-max monotone' rule. The next run
    starts right after the previous one ends; NaN samples belong to no run."""
    w = np.asarray(v, float) * (1.0 if increasing else -1.0)
    n = len(w)
    out: list[tuple[int, int]] = []
    i = 0
    while i < n:
        if not np.isfinite(w[i]):
            i += 1
            continue
        j = i
        run_max = w[i]
        while j + 1 < n and np.isfinite(w[j + 1]) and w[j + 1] >= run_max - eps:
            if w[j + 1] > run_max:
                run_max = w[j + 1]
            j += 1
        out.append((i, j))
        i = j + 1
    return np.array(out, dtype=np.int64).reshape(-1, 2)


def active_slope_mask(vs, dt: float, sign: int, min_a: float, win_s: float) -> np.ndarray:
    """True where the central-difference slope of the smoothed speed `vs` over `win_s` seconds (half a window on
    each side, clipped at the array ends) is >= min_a in the direction `sign` (+1 accelerating, -1 decelerating):
    the samples where the vehicle moves at the episode rate rather than cruising. Robust to speed noise because
    the slope is taken over the whole window instead of between neighbouring samples."""
    vs = np.asarray(vs, float)
    n = len(vs)
    if n == 0:
        return np.zeros(0, bool)
    k = max(1, int(round(win_s / dt / 2.0)))
    i = np.arange(n)
    hi = np.minimum(i + k, n - 1)
    lo = np.maximum(i - k, 0)
    span = np.maximum(hi - lo, 1) * dt
    with np.errstate(invalid="ignore"):
        return (sign * (vs[hi] - vs[lo]) / span) >= min_a


def speed_episodes(v, dt: float = NOMINAL_DT_S, sign: int = 1, min_s: float = ACCEL_EP["min_s"],
                   min_dv_mps: float = ACCEL_EP["min_dv_mps"], min_mean_a: float = ACCEL_EP["min_mean_a"],
                   eps_mps: float = ACCEL_EP["eps_mps"], sg_win: int = ACCEL_EP["sg_win"],
                   sg_order: int = ACCEL_EP["sg_order"], trim_win_s: float = ACCEL_EP.get("trim_win_s", 0.0),
                   **_ignored) -> list[tuple[int, int, float, float]]:
    """Acceleration (sign=+1) or deceleration (sign=-1) episodes of a speed trace `v` (m/s) sampled every `dt` s.

    The speed is Savitzky-Golay smoothed (sg_win, sg_order), split into monotone runs tolerating `eps_mps` dips
    (rises for decelerations), and a run is an episode when it has >= round(min_s / dt) samples, a signed speed
    change sign * dv >= min_dv_mps and a signed mean acceleration sign * dv / duration >= min_mean_a (m/s^2; the
    kinematic mean acceleration of the smoothed speed, which equals the mean of a_mps2 when that channel is the
    derivative of the speed and stays defined on vehicles with a degenerate aEgo).
    A monotone run includes the steady cruise around a manoeuvre (speed within eps), so with the literal rule
    (trim_win_s = 0) a short manoeuvre inside a long cruise is tested with the cruise in its duration. With
    trim_win_s > 0 the leading / trailing samples of each run whose central slope over trim_win_s seconds is below
    min_mean_a (active_slope_mask) are trimmed before the tests, so the count depends on the manoeuvre only
    (interior plateaus of a run are kept). Unknown keyword arguments are ignored so config.ACCEL_EP can be passed whole.
    Returns [(i0, i1_exclusive, dv, mean_a), ...] in time order.
    """
    v = np.asarray(v, float)
    if len(v) < 2:
        return []
    vs = savgol_smooth(v, sg_win, sg_order)
    min_n = max(2, int(round(min_s / dt)))
    active = active_slope_mask(vs, dt, sign, min_mean_a, trim_win_s) if trim_win_s and trim_win_s > 0 else None
    out: list[tuple[int, int, float, float]] = []
    for i0, i1 in monotone_runs(vs, eps_mps, increasing=(sign > 0)):
        if active is not None:
            act = active[i0:i1 + 1]
            if not act.any():
                continue
            i0 = int(i0 + np.argmax(act))
            i1 = int(i1 - np.argmax(act[::-1]))
        if i1 - i0 + 1 < min_n:
            continue
        dv = float(vs[i1] - vs[i0])
        mean_a = dv / ((i1 - i0) * dt)
        if sign * dv < min_dv_mps or sign * mean_a < min_mean_a:
            continue
        out.append((int(i0), int(i1) + 1, dv, float(mean_a)))
    return out


def accel_episodes(v, dt: float = NOMINAL_DT_S, **kw) -> list[tuple[int, int, float, float]]:
    """Acceleration episodes (see speed_episodes); kw override config.ACCEL_EP."""
    return speed_episodes(v, dt, 1, **kw)


def decel_episodes(v, dt: float = NOMINAL_DT_S, **kw) -> list[tuple[int, int, float, float]]:
    """Deceleration episodes (see speed_episodes); kw override config.ACCEL_EP."""
    return speed_episodes(v, dt, -1, **kw)


# ----------------------------------------------------------------------------- lead acceleration events
def true_runs(mask) -> list[tuple[int, int]]:
    """(i_start, i_end_exclusive) of the True runs of a boolean array."""
    m = np.asarray(mask, bool).astype(np.int8)
    d = np.diff(np.concatenate([[0], m, [0]]))
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


def lead_accel_events(lead_a_slope, lead_valid, dt: float = NOMINAL_DT_S, slope_on: float = LEAD_ACCEL_EVENT["slope_on"],
                      hold_s: float = LEAD_ACCEL_EVENT["hold_s"], quiet_s: float = LEAD_ACCEL_EVENT["quiet_s"],
                      quiet_abs: float = LEAD_ACCEL_EVENT["quiet_abs"], horizon_s: float | None = None,
                      **_ignored) -> np.ndarray:
    """Onset indices of lead-vehicle acceleration events.

    A lead episode is a run of lead_valid == True. Inside one, sample i is an onset when the lead acceleration
    (`lead_a_slope`, m/s^2, NaN = unknown) satisfies |a| < quiet_abs on every sample of the quiet_s before i and
    a >= slope_on on every sample of [i, i + hold_s), all inside the episode (NaN fails both tests). With
    `horizon_s`, the episode must also extend at least horizon_s past the onset (so an ego response can be
    measured). Two onsets are at least quiet_s apart by construction. Extra keyword arguments (the response
    constants of config.LEAD_ACCEL_EVENT) are accepted and ignored so the config dict can be passed whole.
    """
    s = np.asarray(lead_a_slope, float)
    valid = np.asarray(lead_valid, bool)
    if len(s) != len(valid):
        raise ValueError("lead_a_slope and lead_valid must have the same length")
    hold_n = max(1, int(round(hold_s / dt)))
    quiet_n = max(1, int(round(quiet_s / dt)))
    hor_n = int(round(horizon_s / dt)) if horizon_s else 0
    onsets: list[int] = []
    for r0, r1 in true_runs(valid):
        seg = s[r0:r1]
        m = len(seg)
        if m < quiet_n + hold_n:
            continue
        hi = np.nan_to_num(seg, nan=-np.inf) >= slope_on
        quiet = np.nan_to_num(np.abs(seg), nan=np.inf) < quiet_abs
        c_hi = np.concatenate([[0], np.cumsum(hi)])
        c_q = np.concatenate([[0], np.cumsum(quiet)])
        i = np.arange(quiet_n, m - hold_n + 1)
        ok = (c_hi[i + hold_n] - c_hi[i] == hold_n) & (c_q[i] - c_q[i - quiet_n] == quiet_n)
        if hor_n:
            ok &= i + hor_n <= m - 1
        onsets.extend((r0 + i[ok]).tolist())
    return np.array(onsets, dtype=np.int64)


def ego_response_delay(a, onset: int, dt: float = NOMINAL_DT_S, resp_a: float = LEAD_ACCEL_EVENT["resp_a"],
                       resp_hold_s: float = LEAD_ACCEL_EVENT["resp_hold_s"],
                       horizon_s: float = LEAD_ACCEL_EVENT["horizon_s"], **_ignored) -> float:
    """Seconds from `onset` to the first sample at which the ego acceleration `a` (m/s^2) stays >= resp_a for
    resp_hold_s, searching onsets in [onset, onset + horizon_s]; NaN when the ego does not respond in time
    (or the trace ends before the hold can be verified)."""
    a = np.asarray(a, float)
    n = len(a)
    hold_n = max(1, int(round(resp_hold_s / dt)))
    hor_n = int(round(horizon_s / dt))
    hi = np.nan_to_num(a, nan=-np.inf) >= resp_a
    c = np.concatenate([[0], np.cumsum(hi)])
    j_last = min(onset + hor_n, n - hold_n)
    for j in range(max(onset, 0), j_last + 1):
        if c[j + hold_n] - c[j] == hold_n:
            return (j - onset) * dt
    return float("nan")
