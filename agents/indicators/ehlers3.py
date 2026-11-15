"""John F. Ehlers' newer filters and oscillators (scout 15.11.2026, night 3) - ports of the published code in TASC
Traders' Tips (EasyLanguage / Pine / Python listings; EasyLanguage Cosine takes degrees, so cos(1.414*180/P) here is
cos(1.414*pi/P) in radians):

- ultimate_smoother / ehlers_uo: "The Ultimate Oscillator" (TASC 2025.04,
  https://www.traders.com/Documentation/FEEDbk_docs/2025/04/TradersTips.html): Signal = HP(close, BW*edge) -
  HP(close, edge), UO = Signal / RMS(Signal, 100). Defaults edge 20, bandwidth 2 (EasyLanguage / Pine). Not the
  Williams Ultimate Oscillator (that one is classic2.ultimate_oscillator).
- laguerre_filter: "Laguerre Filters" (TASC 2025.07,
  https://www.traders.com/Documentation/FEEDbk_docs/2025/07/TradersTips.html): L0 = UltimateSmoother(close, 40),
  4 Laguerre stages gamma 0.8, filter = (L0 + 4L1 + 6L2 + 4L3 + L4) / 16 (the EasyLanguage sum - the Pine / Python
  listings sum one stage more); oscillator gamma 0.5 / length 30 = (L0 - L1) / RMS100.
- continuation_index: "The Continuation Index" (TASC 2025.09,
  https://www.traders.com/Documentation/FEEDbk_docs/2025/09/TradersTips.html, Pine
  https://www.tradingview.com/script/5ZrOut79-TASC-2025-09-The-Continuation-Index/): US(close, 20) against an order-8
  Laguerre of US(close, 40), Ref = 2 (US - LG) / mean|US - LG| over 40 (TRAILING - the Python listing's centred
  np.convolve would look ahead), CI = tanh(Ref) in -1..1.
- one_euro: "The One Euro Filter" (TASC 2025.12, https://www.traders.com/Documentation/FEEDbk_docs/2025/12/TradersTips.html;
  Casiez, Roussel, Vogel, CHI 2012): adaptive EMA whose cutoff = 10 + 0.2 |smoothed change|; the oscillator = the
  same filter on HP(close, 54).
- pma_predict: "Removing Moving Average Lag" (TASC 2025.03,
  https://www.traders.com/Documentation/FEEDbk_docs/2025/03/TradersTips.html): projected MA = SMA + slope * L / 2 of the
  OLS line over L bars, Predict = PMA + 0.5 (slope - slope[2]) L; L 20 (EasyLanguage; Pine 30).
- reflex_trendflex: "Reflex: A New Zero-Lag Indicator" (TASC 2020.02; ProRealCode port
  https://www.prorealcode.com/prorealtime-indicators/reflex-and-trendflex-indicators-john-f-ehlers/): SuperSmoother
  (length / 2), Reflex = mean deviation from the line joining the filter now and length bars ago, Trendflex = mean
  deviation from the current filter; both normalised by their EMA(0.04) RMS.
- two_pole_predict: the simple 2-pole predictor of "Linear Predictive Filters" (TASC 2025.01,
  https://www.traders.com/Documentation/FEEDbk_docs/2025/01/TradersTips.html, Pine listing): LP = SS(HP(close, 15), 30),
  Predict = (LP - 1.8Q LP[1] + Q^2 LP[2]) / (1 - 1.8Q + Q^2), Q 0.35. (The Griffiths LMS predictor of the same article
  is not ported.)

All recursive filters run forward only - the value at bar i uses bars <= i (no repaint)."""
import numpy as np

from agents.indicators._core import S, cols


def _coef(p):
    a1 = np.exp(-1.414 * np.pi / p)
    b1 = 2 * a1 * np.cos(1.414 * np.pi / p)
    return b1, -a1 * a1


def ultimate_smoother(x, p=20):
    x = np.asarray(x, float)
    c2, c3 = _coef(p)
    c1 = (1 + c2 - c3) / 4
    y = x.copy()
    for i in range(4, len(x)):
        y[i] = ((1 - c1) * x[i] + (2 * c1 - c2) * x[i - 1] - (c1 + c3) * x[i - 2] + c2 * y[i - 1] + c3 * y[i - 2])
    return y


def _hp(x, p):
    x = np.asarray(x, float)
    c2, c3 = _coef(p)
    c1 = (1 + c2 - c3) / 4
    y = np.zeros(len(x))
    for i in range(4, len(x)):
        y[i] = c1 * (x[i] - 2 * x[i - 1] + x[i - 2]) + c2 * y[i - 1] + c3 * y[i - 2]
    return y


def _ss(x, p):
    x = np.asarray(x, float)
    c2, c3 = _coef(p)
    c1 = 1 - c2 - c3
    y = x.copy()
    for i in range(2, len(x)):
        y[i] = c1 * (x[i] + x[i - 1]) / 2 + c2 * y[i - 1] + c3 * y[i - 2]
    return y


def _rms(x, n=100):
    r = np.sqrt(S(np.asarray(x, float) ** 2).rolling(n, min_periods=10).mean().to_numpy())
    return r


def ehlers_uo(df, edge=20, bandwidth=2.0):
    """Ehlers' Ultimate Oscillator (2025): dict uo (about -3..3), roc (its change)."""
    c = cols(df)[3]
    sig = _hp(c, bandwidth * edge) - _hp(c, edge)
    r = _rms(sig, 100)
    with np.errstate(invalid="ignore", divide="ignore"):
        uo = np.where(r > 0, sig / r, 0.0)
    uo = np.nan_to_num(uo)
    return {"uo": uo, "roc": np.diff(uo, prepend=uo[0])}


def _lag_stage(x, g):
    """One Laguerre all-pass stage: y = -g x[1]... per Ehlers: Lk = -g*L(k-1)[1] + L(k-1)[1] + g*Lk[1]."""
    y = np.zeros(len(x))
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = -g * x[i - 1] + x[i - 1] + g * y[i - 1]
    return y


def laguerre_filter(df, length=40, gamma=0.8, osc_length=30, osc_gamma=0.5):
    """dict us (UltimateSmoother 40), filt (Laguerre filter), osc (Laguerre oscillator, RMS-normalised)."""
    c = cols(df)[3]
    l0 = ultimate_smoother(c, length)
    l1 = _lag_stage(l0, gamma)
    l2 = _lag_stage(l1, gamma)
    l3 = _lag_stage(l2, gamma)
    l4 = _lag_stage(l3, gamma)
    filt = (l0 + 4 * l1 + 6 * l2 + 4 * l3 + l4) / 16
    o0 = ultimate_smoother(c, osc_length)
    d = o0 - _lag_stage(o0, osc_gamma)
    r = _rms(d, 100)
    with np.errstate(invalid="ignore", divide="ignore"):
        osc = np.nan_to_num(np.where(r > 0, d / r, 0.0))
    return {"us": l0, "filt": filt, "osc": osc}


def continuation_index(df, length=40, gamma=0.8, order=8):
    """Continuation Index in -1..1 (tanh); > 0.5 strong up continuation, < -0.5 strong down."""
    c = cols(df)[3]
    us = ultimate_smoother(c, max(length // 2, 2))
    stage = ultimate_smoother(c, length)
    acc = stage.copy()
    for _ in range(2, order + 1):
        stage = _lag_stage(stage, gamma)
        acc = acc + stage
    lg = acc / order
    var = S(np.abs(us - lg)).rolling(length, min_periods=5).mean().to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        ref = np.where(var > 0, 2 * (us - lg) / var, 0.0)
    return np.tanh(np.nan_to_num(ref))


def _one_euro(x, period_min=10.0, beta=0.2):
    x = np.asarray(x, float)
    adx = 2 * np.pi / (4 * np.pi + 10)
    y = x.copy()
    dx = 0.0
    for i in range(1, len(x)):
        dx = adx * (x[i] - x[i - 1]) + (1 - adx) * dx
        cut = period_min + beta * abs(dx)
        a = 2 * np.pi / (4 * np.pi + cut)
        y[i] = a * x[i] + (1 - a) * y[i - 1]
    return y


def one_euro(df, period_min=10.0, beta=0.2, osc_hp=54):
    """dict filt (One Euro filter of the close), osc (the filter on HP(close, 54))."""
    c = cols(df)[3]
    return {"filt": _one_euro(c, period_min, beta), "osc": _one_euro(_hp(c, osc_hp), period_min, beta)}


def pma_predict(df, length=20):
    """dict pma (projected moving average), predict, slope (OLS slope per bar, positive = rising)."""
    c = cols(df)[3]
    n = len(c)
    L = int(length)
    k = np.arange(1, L + 1, dtype=float)              # k = 1 -> the current bar (Price[k-1])
    sx, sxx = k.sum(), (k * k).sum()
    den = L * sxx - sx * sx
    pma = np.full(n, np.nan)
    slope = np.full(n, np.nan)
    for i in range(L - 1, n):
        y = c[i - np.arange(L)]
        sy, sxy = y.sum(), (k * y).sum()
        sl = -(L * sxy - sx * sy) / den
        slope[i] = sl
        pma[i] = sy / L + sl * L / 2
    pred = pma + 0.5 * (slope - np.concatenate([[np.nan, np.nan], slope[:-2]])) * L
    return {"pma": pma, "predict": pred, "slope": slope}


def reflex_trendflex(df, length=20):
    """dict reflex, trendflex (both normalised, roughly -2..2)."""
    c = cols(df)[3]
    n = len(c)
    filt = _ss(c, max(length // 2, 2))
    refl = np.zeros(n)
    trnd = np.zeros(n)
    ms_r = ms_t = 0.0
    for i in range(length, n):
        slope = (filt[i - length] - filt[i]) / length
        j = np.arange(1, length + 1)
        sr = np.sum((filt[i] + j * slope) - filt[i - j]) / length
        st = np.sum(filt[i] - filt[i - j]) / length
        ms_r = 0.04 * sr * sr + 0.96 * ms_r
        ms_t = 0.04 * st * st + 0.96 * ms_t
        refl[i] = sr / np.sqrt(ms_r) if ms_r > 0 else 0.0
        trnd[i] = st / np.sqrt(ms_t) if ms_t > 0 else 0.0
    return {"reflex": refl, "trendflex": trnd}


def two_pole_predict(df, hp_len=15, lp_len=30, q=0.35):
    """dict lp (band-passed close), predict (2-pole forward estimate)."""
    c = cols(df)[3]
    lp = _ss(_hp(c, hp_len), lp_len)
    lp1 = np.concatenate([[0.0], lp[:-1]])
    lp2 = np.concatenate([[0.0, 0.0], lp[:-2]])
    pred = (lp - 1.8 * q * lp1 + q * q * lp2) / (1 - 1.8 * q + q * q)
    return {"lp": lp, "predict": pred}
