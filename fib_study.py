#!/usr/bin/env python3
"""
fib_study.py — Fibonacci düzeltme/uzantı seviyelerinin ampirik testi
=====================================================================

Zaman dilimleri: M (aylık), 2W, W (haftalık), 3D, D. Parametreler aylık ve haftalık için ayarlı.

Yaptıkları:
  1. Günlük veriyi indirir (yfinance) ve cache'ler, sonra M/2W/W/3D'ye resample eder.
  2. ATR tabanlı, repaint etmeyen (causal) ZigZag ile swing'leri bulur.
  3. Her itki -> düzeltme -> devam yapısı için ölçer:
       - düzeltme derinliği (0..1+), süre, hacim oranı
       - 1 / 1.272 / 1.618 / 2 / 2.618 uzantılarına ulaşma (stop = düzeltme dibi)
     İki modda:
       desc : düzeltme dibinden itibaren (tanımlayıcı, hindsight içerir)
       trd  : ZigZag dibi ONAYLADIĞI bardan itibaren, giriş = onay barı kapanışı (işlenebilir)
  4. Testler:
       - Kümelenme: düzeltme dipleri Fib seviyelerinde komşu bölgelere göre fazla mı?
         (+ rastgele seviyelere karşı yüzdelik / permütasyon)
       - Hazard: fiyat seviye X'e ulaştığında orada dönme olasılığı
       - Derinlik bandına göre sonuç tabloları + hacim filtresi
  5. --sweep: M ve W için ZigZag parametre taraması (sağlamlık kontrolü)

Kullanım:
  pip install -r requirements.txt
  streamlit run streamlit_app.py            # arayüz
  python fib_study.py                       # tickers.txt yoksa S&P 500 (Wikipedia)
  python fib_study.py --tickers tickers.txt --tf M,W
  python fib_study.py --sweep --tf M,W
  python fib_study.py --synthetic           # internetsiz test (rastgele yürüyüş)

Not: Wikipedia S&P 500 listesi BUGÜNKÜ üyelerdir -> survivorship bias.
"""
__version__ = 'v6.1 · adil kıyas (sadece hazır mumlar)'

import argparse
import zlib
import os
import sys

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------- ayarlar
FIB_LEVELS = [0.236, 0.382, 0.5, 0.618, 0.65, 0.786]
EXT_LEVELS = [1.0, 1.272, 1.618, 2.0, 2.618]
DEPTH_EDGES = [0, .236, .382, .5, .618, .65, .786, 1.0, 99]
DEPTH_LABELS = ['0-0.236', '0.236-0.382', '0.382-0.5', '0.5-0.618',
                '0.618-0.65 GP', '0.65-0.786', '0.786-1', '>1 bozuk']
TOL = 0.02  # kümelenme testi pencere yarı genişliği

# rev_mult: swing onayı için gereken ters hareket (ATR katı)
# min_impulse: itkinin sayılması için gereken min boy (ATR katı)
# horizon: sonuç takibi için max bar
TF_PARAMS = {
    'M':  dict(atr_len=12, rev_mult=2.0, min_impulse=4.0, horizon=36),
    '2W': dict(atr_len=14, rev_mult=2.5, min_impulse=5.0, horizon=52),
    'W':  dict(atr_len=14, rev_mult=2.5, min_impulse=5.0, horizon=78),
    '3D': dict(atr_len=14, rev_mult=3.0, min_impulse=6.0, horizon=120),
    'D':  dict(atr_len=14, rev_mult=3.0, min_impulse=6.0, horizon=250),
}
TF_ORDER = ['M', '2W', 'W', '3D', 'D']


# ----------------------------------------------------------------------------- veri
SP500_CSV = ('https://raw.githubusercontent.com/datasets/s-and-p-500-companies/'
             'main/data/constituents.csv')


def sp500_tickers():
    """S&P 500 listesi (bugünkü üyeler -> survivorship bias). lxml gerektirmez."""
    tbl = pd.read_csv(SP500_CSV)
    return [str(s).replace('.', '-').strip() for s in tbl['Symbol'].tolist()]


def load_universe(path):
    if path and os.path.exists(path):
        with open(path) as f:
            return [t.strip().upper() for t in f if t.strip() and not t.startswith('#')]
    print('tickers dosyası yok -> S&P 500 (bugünkü üyeler, survivorship bias!)')
    return sp500_tickers()


def _clean(df):
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(c in df.columns for c in cols):
        return None
    df = df[cols].dropna()
    df.index = pd.DatetimeIndex(df.index).tz_localize(None)
    return df if len(df) > 300 else None


def fetch_many(tickers, chunk=100, progress=None):
    """Toplu indirme (cache'siz). {ticker: df} döner."""
    import yfinance as yf
    out = {}
    for s in range(0, len(tickers), chunk):
        part = tickers[s:s + chunk]
        raw = yf.download(part, period='max', interval='1d', auto_adjust=True,
                          group_by='ticker', threads=True, progress=False)
        for tk in part:
            try:
                df = raw
                if isinstance(raw.columns, pd.MultiIndex):
                    if tk in raw.columns.get_level_values(0):
                        df = raw[tk]
                    elif tk in raw.columns.get_level_values(1):
                        df = raw.xs(tk, axis=1, level=1)
                df = _clean(df.copy())
                if df is not None:
                    out[tk] = df
            except (KeyError, ValueError):
                pass
        if progress:
            progress(min(s + chunk, len(tickers)), len(tickers))
    return out


def load_daily(ticker, cache_dir, refresh=False):
    path = os.path.join(cache_dir, f'{ticker}.csv')
    if os.path.exists(path) and not refresh:
        return pd.read_csv(path, index_col=0, parse_dates=True)
    import yfinance as yf
    df = yf.download(ticker, period='max', interval='1d', auto_adjust=True, progress=False)
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    df.to_csv(path)
    return df


def synthetic_daily(seed, years=30):
    rng = np.random.default_rng(seed)
    n = 252 * years
    vol = 0.02 * np.exp(np.cumsum(rng.normal(0, 0.02, n)) * 0.3)
    r = rng.normal(0.0003, 1, n) * vol
    c = 50 * np.exp(np.cumsum(r))
    o = c * np.exp(rng.normal(0, 0.3, n) * vol)
    h = np.maximum(o, c) * np.exp(np.abs(rng.normal(0, 0.5, n)) * vol)
    l = np.minimum(o, c) * np.exp(-np.abs(rng.normal(0, 0.5, n)) * vol)
    v = rng.lognormal(13, 0.4, n)
    idx = pd.bdate_range('1995-01-02', periods=n)
    return pd.DataFrame({'Open': o, 'High': h, 'Low': l, 'Close': c, 'Volume': v}, index=idx)


def resample(df, tf):
    agg = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}
    if tf == 'D':
        return df.copy()
    if tf == '3D':  # 3 işlem günü grupları
        g = np.arange(len(df)) // 3
        out = df.groupby(g).agg(agg)
        out.index = pd.DatetimeIndex(pd.Series(df.index).groupby(g).last().values)
        return out.iloc[:-1]
    rule = {'W': 'W-FRI', '2W': '2W-FRI', 'M': 'ME'}[tf]
    try:
        out = df.resample(rule).agg(agg)
    except ValueError:  # eski pandas
        out = df.resample('M' if tf == 'M' else rule).agg(agg)
    return out.dropna(subset=['Close']).iloc[:-1]  # son (yarım) barı at


# ----------------------------------------------------------------------------- göstergeler
def wilder_atr(h, l, c, n):
    pc = np.r_[np.nan, c[:-1]]
    tr = np.nanmax(np.vstack([h - l, np.abs(h - pc), np.abs(l - pc)]), axis=0)
    a = np.full(len(tr), np.nan)
    if len(tr) < n:
        return a
    a[n - 1] = tr[:n].mean()
    for i in range(n, len(tr)):
        a[i] = (a[i - 1] * (n - 1) + tr[i]) / n
    return a


def zigzag(h, l, a, rev_mult):
    """Causal ATR ZigZag. Dönüş: [(bar, fiyat, 'H'/'L', onay_bar), ...]"""
    n = len(h)
    start = int(np.argmax(~np.isnan(a))) if (~np.isnan(a)).any() else n
    piv, d = [], 0
    if start >= n - 1:
        return piv
    hi, hi_i, lo, lo_i = h[start], start, l[start], start
    for i in range(start + 1, n):
        thr = rev_mult * a[i]
        if d == 0:
            if h[i] > hi: hi, hi_i = h[i], i
            if l[i] < lo: lo, lo_i = l[i], i
            if hi - lo >= thr:
                if hi_i > lo_i:
                    piv.append((lo_i, lo, 'L', i)); d = 1
                else:
                    piv.append((hi_i, hi, 'H', i)); d = -1
                    k = hi_i + int(np.argmin(l[hi_i:i + 1])); lo, lo_i = l[k], k
            continue
        if d == 1:
            if h[i] > hi:
                hi, hi_i = h[i], i
            elif hi - l[i] >= thr:
                piv.append((hi_i, hi, 'H', i)); d = -1; lo, lo_i = l[i], i
        else:
            if l[i] < lo:
                lo, lo_i = l[i], i
            elif h[i] - lo >= thr:
                piv.append((lo_i, lo, 'L', i)); d = 1; hi, hi_i = h[i], i
    return piv


def track(h, l, start, horizon, n, sign, p0, imp, stop):
    """start'tan itibaren uzantı hedeflerini ve stop'u takip et."""
    end = min(n - 1, start + horizon - 1)
    res = {f'hit_{e}': False for e in EXT_LEVELS}
    res.update(stopped=False, bars_1272=np.nan, max_ext=np.nan)
    mx = -np.inf
    for j in range(start, end + 1):
        stop_hit = l[j] < stop if sign == 1 else h[j] > stop
        if stop_hit:  # aynı barda hedef+stop -> muhafazakâr: stop
            res['stopped'] = True
            break
        ext = sign * ((h[j] if sign == 1 else l[j]) - p0) / imp
        mx = max(mx, ext)
        for e in EXT_LEVELS:
            if not res[f'hit_{e}'] and ext >= e:
                res[f'hit_{e}'] = True
                if e == 1.272: res['bars_1272'] = j - start
    res['max_ext'] = mx if mx > -np.inf else np.nan
    res['complete'] = res['stopped'] or (end - start + 1 >= horizon)
    return res


# ----------------------------------------------------------------------------- Mum Gücü (Apex v666/v667 motoru)
MG_LOOKN = {'M': 120, '2W': 130, 'W': 260, '3D': 500, 'D': 500}
MG = dict(min_samp=15, rvol_len=12, beta_len=36, hi_vol=70, lo_vol=40, big_mv=70,
          sml_mv=40, rel_hi=65, rel_lo=35, wick=0.40)
BENCH = '^GSPC'  # 1927'den beri; SPY ile pratikte aynı


def _roll_pct(x, valid, look, min_samp):
    """Pine Dist.pct: önceki `look` mumdaki geçerli değerler içinde orta-sıra persentili."""
    from numpy.lib.stride_tricks import sliding_window_view
    n = len(x)
    out = np.full(n, np.nan)
    xv = np.where(valid & ~np.isnan(x), x, np.nan)
    W = sliding_window_view(np.concatenate([np.full(look, np.nan), xv]), look)[:n]
    cnt = np.sum(~np.isnan(W), 1)
    with np.errstate(invalid='ignore'):
        lt = np.sum(W < x[:, None], 1)
        le = np.sum(W <= x[:, None], 1)
    ok = (cnt >= min_samp) & ~np.isnan(x)
    out[ok] = (lt[ok] + le[ok]) / 2 / cnt[ok] * 100
    return out


def mum_gucu(df, bench_close, look, full=False):
    """Her mum için Pine v667 ile aynı cls / syn kodlarını döndürür."""
    m = MG
    o, h, l, c, v = (df[k].to_numpy(float) for k in ['Open', 'High', 'Low', 'Close', 'Volume'])
    pc = np.r_[np.nan, c[:-1]]
    mv = (c - pc) / pc * 100
    d = np.where(np.isnan(mv), 0, np.sign(mv)).astype(int)
    absmv = np.abs(mv)
    vavg = pd.Series(v).rolling(m['rvol_len']).mean().shift(1).to_numpy()
    rvol = np.where(vavg > 0, v / np.where(vavg > 0, vavg, 1), np.nan)
    rng = h - l
    hr = rng > 0
    safe = np.where(hr, rng, 1)
    clv = np.where(hr, (c - l) / safe, 0.5)
    upw = np.where(hr, (h - np.maximum(o, c)) / safe, 0.0)
    low_ = np.where(hr, (np.minimum(o, c) - l) / safe, 0.0)
    clstr = np.where(d == 1, clv, np.where(d == -1, 1 - clv, 0.5))

    b = np.asarray(bench_close, float)
    br = np.r_[np.nan, (b[1:] / b[:-1] - 1) * 100]
    s_mv, s_br = pd.Series(mv), pd.Series(br)
    L = m['beta_len']
    sdS = s_mv.rolling(L).std(ddof=0)
    sdB = s_br.rolling(L).std(ddof=0)
    corr = s_mv.rolling(L).corr(s_br)
    beta_now = (corr * sdS / sdB).where(sdB > 0)
    beta = beta_now.shift(1).fillna(1.0).to_numpy()
    alpha = mv - beta * br

    ms = m['min_samp']
    volP = _roll_pct(rvol, np.ones(len(c), bool), look, ms)
    upP = _roll_pct(absmv, d == 1, look, ms)
    dnP = _roll_pct(absmv, d == -1, look, ms)
    mvP = np.where(d == 1, upP, np.where(d == -1, dnP, np.nan))
    exP = _roll_pct(alpha, np.ones(len(c), bool), look, ms)

    ready = ~np.isnan(volP) & ~np.isnan(mvP)
    with np.errstate(invalid='ignore'):
        hv, lv = volP >= m['hi_vol'], volP <= m['lo_vol']
        bm, sm = mvP >= m['big_mv'], mvP <= m['sml_mv']
    W_ = m['wick']
    cls_up = np.select([hv & (upw >= W_), hv & sm & (low_ >= W_), hv & bm & (clstr >= .6),
                        hv & sm, bm & lv, hv | bm], [11, 10, 1, 11, 3, 2], 0)
    cls_dn = np.select([hv & (low_ >= W_), hv & sm & (upw >= W_), hv & bm & (clstr >= .6),
                        hv & sm, bm & lv, hv | bm], [10, 11, -1, 10, -3, -2], 0)
    cls = np.where(ready & (d == 1), cls_up, np.where(ready & (d == -1), cls_dn, 0))

    with np.errstate(invalid='ignore'):
        rS, rW = exP >= m['rel_hi'], exP <= m['rel_lo']
    syn = np.select(
        [cls == 1, cls == 2, cls == 3, cls == -1, cls == -2, cls == -3, cls == 10, cls == 11],
        [np.where(rS, 2, np.where(rW, 0, 1)), np.where(rS, 1, 0),
         np.where(rW, -4, np.where(rS, 1, 0)), np.where(rW, -2, np.where(rS, 0, -1)),
         np.where(rW, -1, 0), np.where(rS, 4, np.where(rW, -1, 0)),
         np.where(rS, 3, np.where(rW, 0, 1)), np.where(rW, -3, np.where(rS, 0, -1))], 0)
    syn = np.where(ready & ~np.isnan(exP), syn, 0)
    if full:
        return cls, syn, exP, dict(volP=volP, mvP=mvP, d=d, clv=clv, upw=upw, loww=low_, ready=ready)
    return cls, syn, exP


DIR = dict(wV=0.40, wI=0.35, fTh=0.25, strongTh=0.65)  # v666 varsayılanları (sektör yok)


def yon_motoru(exP, it):
    """v666 YÖN MOTORU, 'Kurallı' mod: +1 yukarı, 0 yatay, -1 aşağı (her mumun kapanışında)."""
    volP, mvP, d, clv, upw, loww, ready = (it[k] for k in ['volP', 'mvP', 'd', 'clv', 'upw', 'loww', 'ready'])
    effort = np.where(np.isnan(volP), 50, volP) / 100
    pMove = d * np.where(np.isnan(mvP), 0, mvP) / 100 * (0.4 + 0.6 * effort)
    pClose = (2 * clv - 1) * effort
    pWick = (loww - upw) * effort
    vF = np.clip(0.45 * pMove + 0.35 * pClose + 0.20 * pWick, -1, 1)
    iF = np.where(np.isnan(exP), 0, (exP - 50) / 50)
    wSum = DIR['wV'] + DIR['wI']
    netF = np.where(ready, DIR['wV'] / wSum * vF + DIR['wI'] / wSum * iF, 0.0)
    push = np.where(netF >= DIR['fTh'], 1, np.where(netF <= -DIR['fTh'], -1, 0))
    st = np.zeros(len(netF), np.int8)
    cur = 0
    for i in range(len(netF)):
        if ready[i]:
            p_ = push[i]
            if cur == 0:
                if p_ != 0:
                    cur = p_
            elif p_ == -cur:
                cur = p_ if abs(netF[i]) >= DIR['strongTh'] else 0
        st[i] = cur
    return st


def mg_features(cls, syn, exP, sign, i1, i2, c2):
    """Pine v667 setup bloğundaki Mum Gücü teyidi."""
    pc_, ps_ = cls[i1 + 1:i2 + 1], syn[i1 + 1:i2 + 1]
    lc, ls = cls[i2 + 1:c2 + 1], syn[i2 + 1:c2 + 1]
    if sign == 1:
        ab = np.any((pc_ == 10) | (ps_ == 3) | (ps_ == 4))
        di = np.any((ps_ == -2) | (ps_ == -3))
        st_ = np.any((ls == 2) | (lc == 1) | ((lc == 2) & (ls == 1)))
        bad = np.any((lc == 11) | (ls == -3) | (ls == -4))
    else:
        ab = np.any((pc_ == 11) | (ps_ == -3))
        di = np.any((ps_ == 2) | (ps_ == 3))
        st_ = np.any((ls == -2) | (lc == -1) | ((lc == -2) & (ls == -1)))
        bad = np.any((lc == 10) | (ls == 3) | (ls == 4))
    return dict(mg_abs_pull=bool(ab), mg_dis_pull=bool(di), mg_str_leg=bool(st_),
                mg_bad_leg=bool(bad), mg=int(ab) + int(st_) - int(bad) - int(di),
                mg_exP_conf=exP[c2], mg_syn_conf=int(syn[c2]))


TARGET_EXTS = [1.0, 1.272, 1.618, 2.0, 2.618]
CTL_DRAWS = 3  # olay başına rastgele kontrol girişi


def realized_scale(h, l, c, start, horizon, n, sign, entry, stop, t1, t2):
    """Yarısı t1'de kapanır, stop girişe çekilir, kalan yarı t2'de / girişte / vade sonunda."""
    risk = sign * (entry - stop)
    if risk <= 0:
        return np.nan
    end = min(n - 1, start + horizon - 1)
    half = None
    for j in range(start, end + 1):
        hi_, lo_ = h[j], l[j]
        if half is None:
            if (lo_ < stop) if sign == 1 else (hi_ > stop):
                return -1.0
            if (hi_ >= t1) if sign == 1 else (lo_ <= t1):
                half = 0.5 * sign * (t1 - entry) / risk
                if (hi_ >= t2) if sign == 1 else (lo_ <= t2):
                    return half + 0.5 * sign * (t2 - entry) / risk
        else:
            if (lo_ < entry) if sign == 1 else (hi_ > entry):
                return half
            if (hi_ >= t2) if sign == 1 else (lo_ <= t2):
                return half + 0.5 * sign * (t2 - entry) / risk
    if end - start + 1 < horizon:
        return np.nan
    rest = 0.5 * sign * (c[end] - entry) / risk
    return (half + rest) if half is not None else 2 * rest


def realized_R(h, l, c, start, horizon, n, sign, entry, stop, target):
    """Gerçekleşen R: 1.272 hedefi → +RR, stop → -1, ikisi de yoksa horizon sonunda kapanış."""
    risk = sign * (entry - stop)
    if risk <= 0:
        return np.nan
    end = min(n - 1, start + horizon - 1)
    for j in range(start, end + 1):
        if (l[j] < stop) if sign == 1 else (h[j] > stop):
            return -1.0
        if (h[j] >= target) if sign == 1 else (l[j] <= target):
            return sign * (target - entry) / risk
    if end - start + 1 < horizon:
        return np.nan  # veri bitti, sonuç belirsiz
    return sign * (c[end] - entry) / risk


# ----------------------------------------------------------------------------- olay çıkarımı
def extract_events(df, tf, ticker, p, bench_close=None):
    h, l, c, v = (df[k].to_numpy(float) for k in ['High', 'Low', 'Close', 'Volume'])
    n = len(h)
    a = wilder_atr(h, l, c, p['atr_len'])
    piv = zigzag(h, l, a, p['rev_mult'])
    mg = None
    if bench_close is not None:
        mg = mum_gucu(df, bench_close, MG_LOOKN.get(tf, 500))
    # rastgele giriş kontrolü için: tam vadesi olan, ATR'si geçerli mumlar
    crng = np.random.default_rng(zlib.crc32(f'{ticker}|{tf}'.encode()))
    lo_j = max(p['atr_len'] + 1, 0)
    hi_j = n - 1 - p['horizon']
    rows = []
    for k in range(len(piv) - 2):
        (i0, p0, t0, _), (i1, p1, _, _), (i2, p2, _, c2) = piv[k], piv[k + 1], piv[k + 2]
        sign = 1 if t0 == 'L' else -1
        imp = sign * (p1 - p0)
        if imp <= 0 or np.isnan(a[i1]) or imp < p['min_impulse'] * a[i1]:
            continue
        depth = sign * (p1 - p2) / imp
        vi, vp = v[i0 + 1:i1 + 1].mean(), v[i1 + 1:i2 + 1].mean() if i2 > i1 else np.nan
        row = dict(ticker=ticker, tf=tf, dir='up' if sign == 1 else 'down',
                   date_p0=df.index[i0], date_p1=df.index[i1], date_p2=df.index[i2],
                   date_conf=df.index[c2], depth=depth, impulse_atr=imp / a[i1],
                   impulse_bars=i1 - i0, pullback_bars=i2 - i1,
                   vol_ratio=vp / vi if vi > 0 else np.nan)
        # desc: dipten itibaren
        d = track(h, l, i2 + 1, p['horizon'], n, sign, p0, imp, p2)
        row.update({f'desc_{key}': val for key, val in d.items()})
        # trd: onay barından itibaren, giriş = onay kapanışı
        entry = c[c2]
        t = track(h, l, c2 + 1, p['horizon'], n, sign, p0, imp, p2)
        risk = sign * (entry - p2)
        tgt = p0 + sign * 1.272 * imp
        row['trd_rr_1272'] = sign * (tgt - entry) / risk if risk > 0 else np.nan
        row.update({f'trd_{key}': val for key, val in t.items()})
        row['trd_R'] = realized_R(h, l, c, c2 + 1, p['horizon'], n, sign, entry, p2, tgt)
        for e_ in TARGET_EXTS:
            row[f'trd_R_{e_}'] = realized_R(h, l, c, c2 + 1, p['horizon'], n, sign, entry, p2,
                                            p0 + sign * e_ * imp)
        row['trd_R_scale'] = realized_scale(h, l, c, c2 + 1, p['horizon'], n, sign, entry, p2,
                                            p0 + sign * 1.272 * imp, p0 + sign * 1.618 * imp)
        # --- Kontrol: aynı hisse, RASTGELE mum, aynı stop mesafesi (ATR) ve aynı R-katı hedefler ---
        if risk > 0 and not np.isnan(a[c2]) and hi_j > lo_j:
            risk_atr = risk / a[c2]
            rr = {e_: sign * (p0 + sign * e_ * imp - entry) / risk for e_ in TARGET_EXTS}
            ctl = {e_: [] for e_ in TARGET_EXTS}
            ctl['scale'] = []
            for _ in range(CTL_DRAWS):
                j = int(crng.integers(lo_j, hi_j + 1))
                if np.isnan(a[j]):
                    continue
                en, R1 = c[j], risk_atr * a[j]
                st = en - sign * R1
                for e_ in TARGET_EXTS:
                    ctl[e_].append(realized_R(h, l, c, j + 1, p['horizon'], n, sign, en, st,
                                              en + sign * rr[e_] * R1))
                ctl['scale'].append(realized_scale(h, l, c, j + 1, p['horizon'], n, sign, en, st,
                                                   en + sign * rr[1.272] * R1, en + sign * rr[1.618] * R1))
            for key, vals in ctl.items():
                vals = [x for x in vals if not np.isnan(x)]
                row[f'ctl_R_{key}'] = float(np.mean(vals)) if vals else np.nan
        if mg is not None:
            row.update(mg_features(*mg, sign, i1, i2, c2))
        rows.append(row)
    return rows


# ----------------------------------------------------------------------------- analizler
def clustering(depths, rng):
    d = depths[(depths > 0.05) & (depths < 1.2)]
    def ratio(L):
        obs = np.sum(np.abs(d - L) < TOL)
        sh = np.sum((np.abs(d - L) >= TOL) & (np.abs(d - L) < 3 * TOL)) / 2
        return obs / sh if sh > 0 else np.nan
    rand = np.array([ratio(x) for x in rng.uniform(0.15, 0.9, 1000)])
    rand = rand[~np.isnan(rand)]
    out = []
    for L in FIB_LEVELS:
        r = ratio(L)
        out.append(dict(level=L, n_window=int(np.sum(np.abs(d - L) < TOL)),
                        excess_ratio=round(r, 3),
                        pctile_vs_random=round(100 * np.mean(rand < r), 1) if len(rand) else np.nan))
    return pd.DataFrame(out)


def hazard(depths, step=0.02):
    d = depths[depths > 0]
    edges = np.arange(0.10, 1.02 + 1e-9, step)
    rows = []
    for lo in edges:
        reached = np.sum(d >= lo)
        stopped = np.sum((d >= lo) & (d < lo + step))
        rows.append(dict(level=round(lo, 2), reached=int(reached),
                         stop_prob=stopped / reached if reached else np.nan,
                         is_fib=any(abs(lo + step / 2 - f) <= step / 2 for f in FIB_LEVELS)))
    return pd.DataFrame(rows)


def outcome_table(ev, mode):
    e = ev[ev[f'{mode}_complete']].copy()
    e['band'] = pd.cut(e['depth'], DEPTH_EDGES, labels=DEPTH_LABELS, right=False)
    g = e.groupby('band', observed=False)
    t = pd.DataFrame({
        'n': g.size(),
        'P(1)': g[f'{mode}_hit_1.0'].mean(),
        'P(1.272)': g[f'{mode}_hit_1.272'].mean(),
        'P(1.618)': g[f'{mode}_hit_1.618'].mean(),
        'P(2.618)': g[f'{mode}_hit_2.618'].mean(),
        'med_max_ext': g[f'{mode}_max_ext'].median(),
    })
    if mode == 'trd':
        t['med_RR_1272'] = g['trd_rr_1272'].median()
        # kaba beklenen R: p*RR - (1-p)
        t['expR_1272'] = t['P(1.272)'] * t['med_RR_1272'] - (1 - t['P(1.272)'])
    return t.round(3)


def volume_table(ev):
    e = ev[ev['trd_complete'] & ev['vol_ratio'].notna()].copy()
    e['band'] = pd.cut(e['depth'], DEPTH_EDGES, labels=DEPTH_LABELS, right=False)
    med = e['vol_ratio'].median()
    e['vol'] = np.where(e['vol_ratio'] <= med, 'düşük_hacimli_düzeltme', 'yüksek_hacimli_düzeltme')
    return e.pivot_table(index='band', columns='vol', values='trd_hit_1.272',
                         aggfunc=['mean', 'size'], observed=False).round(3)


def plot_tf(ev, hz, tf, out_dir):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    d = ev['depth'][(ev['depth'] > 0) & (ev['depth'] < 1.2)]
    ax[0].hist(d, bins=np.arange(0, 1.21, 0.02), color='#4a7ab5')
    ax[1].plot(hz['level'] + 0.01, hz['stop_prob'], marker='o', ms=3)
    for a_ in ax:
        for f in FIB_LEVELS:
            a_.axvline(f, color='#d9822b', ls='--', lw=1)
    ax[0].set_title(f'{tf} — düzeltme derinliği dağılımı (n={len(d)})')
    ax[1].set_title(f'{tf} — seviyeye ulaşınca orada dönme olasılığı (hazard)')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f'{tf}_depth_hazard.png'), dpi=110)
    plt.close(fig)


# ----------------------------------------------------------------------------- ana akış
def collect_events(daily, tf, p, bench=None):
    rows = []
    bclose = bench['Close'].astype(float) if bench is not None else None
    for tk, df in daily.items():
        bars = resample(df, tf)
        if len(bars) < p['atr_len'] + 10:
            continue
        bc = None
        if bclose is not None:
            bc = bclose.reindex(bclose.index.union(bars.index)).ffill().reindex(bars.index).to_numpy()
        rows += extract_events(bars, tf, tk, p, bc)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- Mum Gücü testi
MG_FLAGS = [('mg_abs_pull', 'Düzeltmede absorpsiyon/silkeleme'),
            ('mg_str_leg', 'Dönüş bacağında güçlü mum'),
            ('mg_bad_leg', 'Dönüş bacağında dağıtım/fake'),
            ('mg_dis_pull', 'Düzeltmede ters A+ mum')]


def _grp_stats(e):
    return pd.Series(dict(n=len(e), P1=e['trd_hit_1.0'].mean(), P1272=e['trd_hit_1.272'].mean(),
                          P1618=e['trd_hit_1.618'].mean(), meanR=e['trd_R'].mean(),
                          medR=e['trd_R'].median()))


def _boot_diff(e, mask, col, reps, rng, strata=None):
    """mean(col | mask) - mean(col | ~mask), hisse bazında küme bootstrap %95 GA.
    strata verilirse: tabaka içi farkların ağırlıklı ortalaması (derinlik × süre etkisini ayıklar)."""
    uniq, inv = np.unique(e['ticker'].to_numpy(), return_inverse=True)
    x = e[col].to_numpy(float)
    ok = ~np.isnan(x)
    mk = mask.to_numpy()
    st_ = np.zeros(len(e), int) if strata is None else pd.factorize(strata)[0]
    U = len(uniq)
    parts = []
    for sidx in np.unique(st_):
        ins = st_ == sidx
        def sums(sel):
            sel = sel & ok & ins
            return (np.bincount(inv, weights=np.where(sel, x, 0), minlength=U),
                    np.bincount(inv, weights=sel.astype(float), minlength=U))
        parts.append(sums(mk) + sums(~mk))

    def combine(w):
        num = den = 0.0
        for s1, k1, s0, k0 in parts:
            K1, K0 = (w * k1).sum(), (w * k0).sum()
            if K1 > 0 and K0 > 0:
                wt = K1 * K0 / (K1 + K0)
                num += wt * ((w * s1).sum() / K1 - (w * s0).sum() / K0)
                den += wt
        return num / den if den > 0 else np.nan

    point = combine(np.ones(U))
    diffs = [combine(np.bincount(rng.integers(0, U, U), minlength=U)) for _ in range(reps)]
    diffs = [d for d in diffs if not np.isnan(d)]
    lo, hi = np.percentile(diffs, [2.5, 97.5]) if diffs else (np.nan, np.nan)
    return point, lo, hi


def _boot_mean(e, col, reps, rng, cluster='ticker'):
    """Küme bootstrap (varsayılan hisse bazlı) ile ortalama ve %95 GA."""
    x = e[col].to_numpy(float)
    ok = ~np.isnan(x)
    uniq, inv = np.unique(e[cluster].astype(str).to_numpy(), return_inverse=True)
    U = len(uniq)
    s_ = np.bincount(inv, weights=np.where(ok, x, 0), minlength=U)
    k_ = np.bincount(inv, weights=ok.astype(float), minlength=U)
    m = []
    for _ in range(reps):
        w = np.bincount(rng.integers(0, U, U), minlength=U)
        if (w * k_).sum() > 0:
            m.append((w * s_).sum() / (w * k_).sum())
    point = s_.sum() / k_.sum() if k_.sum() else np.nan
    lo, hi = np.percentile(m, [2.5, 97.5]) if m else (np.nan, np.nan)
    return point, lo, hi


TARGET_COLS = [(f'trd_R_{e}', f'Hedef {e}') for e in TARGET_EXTS] + [('trd_R_scale', 'Yarı 1.272 + yarı 1.618 (stop→giriş)')]


def target_analysis(ev, rng, reps=300):
    """Hangi hedef en yüksek ortalama R'yi veriyor? MG skoruna göre, yukarı setup'lar."""
    if 'trd_R_1.618' not in ev.columns:
        return None
    ev = _clipR(ev)
    e0 = ev[ev['trd_complete'] & (ev['depth'] < 1.0) & (ev['dir'] == 'up')].copy()
    cols = [c for c, _ in TARGET_COLS]
    e0 = e0.dropna(subset=cols)
    if len(e0) < 30:
        return None
    groups = [('Tümü', e0)]
    if 'mg' in e0.columns:
        groups += [('MG ≤-1', e0[e0.mg <= -1]), ('MG 0', e0[e0.mg == 0]),
                   ('MG 1', e0[e0.mg == 1]), ('MG ≥2 ★A+', e0[e0.mg >= 2])]
    mean_rows, diff_rows = [], []
    for gname, g in groups:
        if len(g) < 30:
            continue
        row = {'grup': gname, 'n': len(g)}
        for c, name in TARGET_COLS:
            row[name] = g[c].mean()
        mean_rows.append(row)
        # 1.272'ye göre eşleştirilmiş fark (aynı olaylar)
        for c, name in TARGET_COLS:
            if c == 'trd_R_1.272':
                continue
            g = g.assign(_d=g[c] - g['trd_R_1.272'])
            d, lo, hi = _boot_mean(g, '_d', reps, rng)
            diff_rows.append(dict(grup=gname, hedef=name, fark_R=d, GA95=f'[{lo:+.3f}, {hi:+.3f}]',
                                  anlamlı='EVET' if (lo > 0 or hi < 0) else 'hayır'))
    return dict(means=pd.DataFrame(mean_rows).set_index('grup').round(3),
                diffs=pd.DataFrame(diff_rows).round(3))


def _clipR(ev):
    """Veri hatalı (ATR≈0) tek olaylar R'yi binlerce yapabiliyor → R'yi [-1, 20] aralığına kırp."""
    cols = [c for c in ev.columns if c.startswith('trd_R') or c.startswith('ctl_R')]
    if cols:
        ev = ev.copy()
        ev[cols] = ev[cols].clip(-1, 20)
    return ev


def baseline_analysis(ev, rng, reps=300):
    """Setup, aynı hissede rastgele girişten (aynı stop/hedef R-katları) daha iyi mi?"""
    if 'ctl_R_1.272' not in ev.columns:
        return None
    ev = _clipR(ev)
    e0 = ev[ev['trd_complete'] & (ev['depth'] < 1.0) & (ev['dir'] == 'up')].copy()
    e0['dönem'] = np.where(pd.to_datetime(e0['date_conf']).dt.year < 2000, '<2000', '≥2000')
    pairs = [(f'trd_R_{e}', f'ctl_R_{e}', f'Hedef {e}') for e in TARGET_EXTS] + \
            [('trd_R_scale', 'ctl_R_scale', 'Kademeli 1.272+1.618')]
    groups = [('Tümü', e0), ('<2000', e0[e0['dönem'] == '<2000']), ('≥2000', e0[e0['dönem'] == '≥2000'])]
    if 'mg' in e0.columns:
        groups.append(('MG ≥2 ★A+', e0[e0.mg >= 2]))
    rows = []
    for gname, g in groups:
        for sc, cc, name in pairs:
            gg = g.dropna(subset=[sc, cc])
            if len(gg) < 30:
                continue
            gg = gg.assign(_d=gg[sc] - gg[cc])
            d, lo, hi = _boot_mean(gg, '_d', reps, rng)
            rows.append(dict(grup=gname, hedef=name, n=len(gg), setup_R=gg[sc].mean(), rastgele_R=gg[cc].mean(),
                             fark=d, GA95=f'[{lo:+.3f}, {hi:+.3f}]',
                             sonuç='SETUP İYİ' if lo > 0 else ('SETUP KÖTÜ' if hi < 0 else 'fark yok')))
    return pd.DataFrame(rows).round(3)


def mg_analysis(ev, rng, reps=300):
    """Mum Gücü teyidi Fib setup sonucunu iyileştiriyor mu?"""
    if 'mg' not in ev.columns:
        return None
    ev = _clipR(ev)
    e0 = ev[ev['trd_complete'] & (ev['depth'] < 1.0) & ev['trd_R'].notna()].copy()
    e0['mg_grp'] = pd.cut(e0['mg'], [-9, -1, 0, 1, 9], labels=['≤-1 ZAYIF', '0 NÖTR', '1 POZİTİF', '≥2 GÜÇLÜ'])
    e0['derinlik'] = np.where(e0['depth'] <= 0.618, 'sığ ≤0.618', 'derin >0.618')
    e0['dönem'] = np.where(pd.to_datetime(e0['date_conf']).dt.year < 2000, '<2000', '≥2000')
    e0['süre'] = pd.cut(e0['pullback_bars'], [-1, 3, 8, 10**6], labels=['≤3 mum', '4-8 mum', '>8 mum'])
    e0['tabaka'] = e0['derinlik'] + '|' + e0['süre'].astype(str)
    out = {}
    for d_ in ['up', 'down']:
        e = e0[e0.dir == d_]
        if len(e) < 30:
            continue
        r = {}
        r['by_score'] = e.groupby('mg_grp', observed=False).apply(_grp_stats).round(3)
        r['by_depth'] = e.groupby(['derinlik', 'mg_grp'], observed=False).apply(_grp_stats).round(3)
        r['by_era'] = e.groupby(['dönem', 'mg_grp'], observed=False).apply(_grp_stats).round(3)
        r['by_len'] = e.groupby(['süre', 'mg_grp'], observed=False).apply(_grp_stats).round(3)
        comp = []
        for col, name in MG_FLAGS + [('_ap', 'Skor ≥2 (★A+)')]:
            mask = (e['mg'] >= 2) if col == '_ap' else e[col].astype(bool)
            if mask.sum() < 10 or (~mask).sum() < 10:
                continue
            rawP = e.loc[mask, 'trd_hit_1.272'].mean() - e.loc[~mask, 'trd_hit_1.272'].mean()
            rawR = e.loc[mask, 'trd_R'].mean() - e.loc[~mask, 'trd_R'].mean()
            dP, pl, ph = _boot_diff(e, mask, 'trd_hit_1.272', reps, rng, e['tabaka'])
            dR, rl, rh = _boot_diff(e, mask, 'trd_R', reps, rng, e['tabaka'])
            comp.append(dict(özellik=name, n_var=int(mask.sum()), n_yok=int((~mask).sum()),
                             P1272_var=e.loc[mask, 'trd_hit_1.272'].mean(),
                             P1272_yok=e.loc[~mask, 'trd_hit_1.272'].mean(),
                             ham_fark_P=rawP, düz_fark_P=dP, GA95_P=f'[{pl:+.3f}, {ph:+.3f}]',
                             meanR_var=e.loc[mask, 'trd_R'].mean(), meanR_yok=e.loc[~mask, 'trd_R'].mean(),
                             ham_fark_R=rawR, düz_fark_R=dR, GA95_R=f'[{rl:+.2f}, {rh:+.2f}]',
                             anlamlı_P='EVET' if (pl > 0 or ph < 0) else 'hayır',
                             anlamlı_R='EVET' if (rl > 0 or rh < 0) else 'hayır'))
        r['components'] = pd.DataFrame(comp).round(3)
        out[d_] = r
    return out


def analyze(ev, rng):
    """Bir TF'nin olaylarından tüm tabloları üretir."""
    res = dict(
        n=len(ev), n_up=int(np.sum(ev.dir == 'up')), n_down=int(np.sum(ev.dir == 'down')),
        med_depth=float(ev.depth.median()),
        clustering=clustering(ev['depth'].to_numpy(), rng),
        hazard=hazard(ev['depth'].to_numpy()),
        trd=outcome_table(ev, 'trd'), desc=outcome_table(ev, 'desc'),
        volume=volume_table(ev), by_dir={})
    for d_ in ['up', 'down']:
        sub = ev[ev.dir == d_]
        if len(sub) > 30:
            res['by_dir'][d_] = outcome_table(sub, 'trd')
    res['mg'] = mg_analysis(ev, rng)
    res['targets'] = target_analysis(ev, rng)
    res['baseline'] = baseline_analysis(ev, rng)
    return res


# ----------------------------------------------------------------------------- Mum Gücü SİNYAL testi (Fib'den bağımsız)
SIG_HORIZONS = {'M': [1, 3, 6, 12], '2W': [1, 3, 6, 13], 'W': [1, 4, 13, 26], '3D': [1, 5, 20, 60], 'D': [1, 5, 20, 60]}
SIG_DEFS = [  # (ad, beklenen yön, koşul(sb))
    ('✅ Kurumsal alım A+ (syn 2)', 1, lambda f: f.syn == 2),
    ('🔵 Güçlü absorpsiyon A+ (syn 3)', 1, lambda f: f.syn == 3),
    ('🪤 Silkeleme olası (syn 4)', 1, lambda f: f.syn == 4),
    ('🟢 Güçlü yükseliş (cls 1, tümü)', 1, lambda f: f.cls == 1),
    ('🔵 Absorpsiyon (cls 10, tümü)', 1, lambda f: f.cls == 10),
    ('🟡 Hacimsiz yükseliş (cls 3)', -1, lambda f: f.cls == 3),
    ('❌ Fake yükseliş (syn -4)', -1, lambda f: f.syn == -4),
    ('🩸 Hisseye özel çıkış A+ (syn -2)', -1, lambda f: f.syn == -2),
    ('🟠 Güçlü dağıtım A+ (syn -3)', -1, lambda f: f.syn == -3),
    ('🔴 Güçlü düşüş (cls -1, tümü)', -1, lambda f: f.cls == -1),
    ('🟠 Dağıtım (cls 11, tümü)', -1, lambda f: f.cls == 11),
    ('Σ MG alım grubu (syn 2/3/4)', 1, lambda f: f.syn.isin([2, 3, 4])),
    ('Σ MG satış grubu (syn -2/-3/-4)', -1, lambda f: f.syn.isin([-2, -3, -4])),
    # --- YÖN MOTORU ---
    ('🧭 Yön → YUKARI döndü', 1, lambda f: (f.yon == 1) & (f.yon_once != 1)),
    ('🧭 Yön → AŞAĞI döndü', -1, lambda f: (f.yon == -1) & (f.yon_once != -1)),
    ('🧭 Yön → YATAY (yukarıdan)', -1, lambda f: (f.yon == 0) & (f.yon_once == 1)),
    ('🧭 Yön → YATAY (aşağıdan)', 1, lambda f: (f.yon == 0) & (f.yon_once == -1)),
    ('🧭 Yön YUKARI iken (her mum)', 1, lambda f: f.yon == 1),
    ('🧭 Yön AŞAĞI iken (her mum)', -1, lambda f: f.yon == -1),
    ('🧭 Yön YUKARI, 5+ mumdur', 1, lambda f: (f.yon == 1) & (f.yon_yas >= 5)),
    ('🧭 Yön AŞAĞI, 5+ mumdur', -1, lambda f: (f.yon == -1) & (f.yon_yas >= 5)),
]


PAIRS = [  # (A, B, etiket) → A − B
    ('🧭 Yön → YUKARI döndü', '🧭 Yön → AŞAĞI döndü', 'Yön yukarı döndü − aşağı döndü'),
    ('🧭 Yön YUKARI iken (her mum)', '🧭 Yön AŞAĞI iken (her mum)', 'Yön yukarıyken − aşağıyken'),
    ('🧭 Yön YUKARI, 5+ mumdur', '🧭 Yön AŞAĞI, 5+ mumdur', 'Yön 5+ mum yukarı − 5+ mum aşağı'),
    ('🟢 Güçlü yükseliş (cls 1, tümü)', '🔴 Güçlü düşüş (cls -1, tümü)', 'Güçlü yükseliş mumu − güçlü düşüş mumu'),
    ('🟢 Güçlü yükseliş (cls 1, tümü)', '🟡 Hacimsiz yükseliş (cls 3)', 'Hacimli yükseliş − hacimsiz yükseliş'),
    ('🔵 Absorpsiyon (cls 10, tümü)', '🟠 Dağıtım (cls 11, tümü)', 'Absorpsiyon − dağıtım'),
    ('Σ MG alım grubu (syn 2/3/4)', 'Σ MG satış grubu (syn -2/-3/-4)', 'MG alım grubu − satış grubu'),
]


def _pair_diff(sb, ma, mb, col, reps, rng):
    """mean(col|A) − mean(col|B), tarih bazlı küme bootstrap."""
    x = sb[col].to_numpy(float)
    ok = ~np.isnan(x)
    uniq, inv = np.unique(sb['date'].to_numpy(), return_inverse=True)
    U = len(uniq)
    def sums(m):
        m = m & ok
        return (np.bincount(inv, weights=np.where(m, x, 0), minlength=U),
                np.bincount(inv, weights=m.astype(float), minlength=U))
    sa, ka = sums(ma)
    s_b, kb = sums(mb)
    if ka.sum() < 30 or kb.sum() < 30:
        return np.nan, np.nan, np.nan
    pt = sa.sum() / ka.sum() - s_b.sum() / kb.sum()
    d = []
    for _ in range(reps):
        w = np.bincount(rng.integers(0, U, U), minlength=U)
        A, B = (w * ka).sum(), (w * kb).sum()
        if A > 0 and B > 0:
            d.append((w * sa).sum() / A - (w * s_b).sum() / B)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return pt, lo, hi


def signal_bars(daily, tf, bench):
    """Her hisse × her mum: cls, syn ve ileri getiriler (giriş = sinyal mumu kapanışı)."""
    H = SIG_HORIZONS.get(tf, [1, 5, 20])
    bclose = bench['Close'].astype(float)
    frames = []
    for tk, df in daily.items():
        bars = resample(df, tf)
        if len(bars) < 60:
            continue
        bc = bclose.reindex(bclose.index.union(bars.index)).ffill().reindex(bars.index).to_numpy()
        cls, syn, exP_, it = mum_gucu(bars, bc, MG_LOOKN.get(tf, 500), full=True)
        yon = yon_motoru(exP_, it)
        yon_once = np.r_[0, yon[:-1]].astype(np.int8)
        chg = np.r_[True, yon[1:] != yon[:-1]]
        yas = (np.arange(len(yon)) - np.maximum.accumulate(np.where(chg, np.arange(len(yon)), 0)) + 1)
        c = bars['Close'].to_numpy(float)
        d = {'ticker': tk, 'date': bars.index, 'cls': cls.astype(np.int8), 'syn': syn.astype(np.int8),
             'yon': yon, 'yon_once': yon_once, 'yon_yas': yas.astype(np.int16),
             'hazir': (it['ready'] & ~np.isnan(exP_))}
        for h in H:
            fwd = np.full(len(c), np.nan)
            fwd[:-h] = (c[h:] / c[:-h] - 1) * 100
            d[f'r{h}'] = fwd.astype(np.float32)
        frames.append(pd.DataFrame(d))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def signal_analysis(sb, tf, rng, reps=300):
    """Sinyal sonrası getiri − aynı tarihteki tüm hisselerin ortalaması (kesitsel fazla getiri).
    Piyasa yönü ve survivorship iki tarafta da aynı → fark = sinyalin bilgisi."""
    if sb is None or sb.empty:
        return None
    H = SIG_HORIZONS.get(tf, [1, 5, 20])
    sb = sb.copy()
    # ADİL KIYAS: sinyaller ancak Mum Gücü 'hazır' olunca (yeterli geçmiş) çıkabiliyor.
    # Hazır olmayan mumlar çoğunlukla hisselerin ilk yılları → bugünkü S&P 500 üyelerinde
    # (survivorship) bu dönemin getirisi şişik. Ortalama SADECE hazır mumlardan alınır.
    if 'hazir' in sb.columns:
        sb = sb[sb['hazir']].copy()
    for h in H:
        sb[f'x{h}'] = sb[f'r{h}'] - sb.groupby('date')[f'r{h}'].transform('mean')
    sb['dönem'] = np.where(pd.to_datetime(sb['date']).dt.year < 2000, '<2000', '≥2000')
    rows, era_rows = [], []
    hm = H[min(2, len(H) - 1)]  # dönem kırılımı için ana vade
    for name, exp, cond in SIG_DEFS:
        sel = sb[cond(sb).to_numpy()]
        if len(sel) < 30:
            continue
        r = dict(sinyal=name, beklenen='↑' if exp == 1 else '↓', n=len(sel))
        verdicts = []
        for h in H:
            e = sel.dropna(subset=[f'x{h}'])
            if len(e) < 30:
                continue
            m, lo, hi = _boot_mean(e, f'x{h}', reps, rng, cluster='date')
            r[f'+{h} fazla %'] = m
            r[f'+{h} GA95'] = f'[{lo:+.2f}, {hi:+.2f}]'
            r[f'+{h} isabet'] = float(((e[f'x{h}'] > 0) if exp == 1 else (e[f'x{h}'] < 0)).mean())
            sig = lo > 0 or hi < 0
            verdicts.append('✓' if sig and np.sign(m) == exp else ('✗ ters' if sig else '·'))
        r['anlamlı (vadeler)'] = ' '.join(verdicts)
        rows.append(r)
        for era, eg in sel.groupby('dönem'):
            e = eg.dropna(subset=[f'x{hm}'])
            if len(e) < 30:
                continue
            m, lo, hi = _boot_mean(e, f'x{hm}', reps, rng, cluster='date')
            era_rows.append(dict(sinyal=name, dönem=era, n=len(e), vade=f'+{hm}', fazla_getiri_pct=m,
                                 GA95=f'[{lo:+.2f}, {hi:+.2f}]'))
    base = {f'+{h} tüm mumlar ort. %': float(sb[f'r{h}'].mean()) for h in H}
    # ÇİFT KIYAS: iki zıt sinyalin farkı (ortak yanlılıklar birbirini götürür)
    pairs = []
    defs = {n: (e, c) for n, e, c in SIG_DEFS}
    for a_, b_, lab in PAIRS:
        if a_ not in defs or b_ not in defs:
            continue
        ma, mb = defs[a_][1](sb).to_numpy(), defs[b_][1](sb).to_numpy()
        r = dict(kıyas=lab)
        for h in H:
            m, lo, hi = _pair_diff(sb, ma, mb, f'x{h}', reps, rng)
            r[f'+{h} fark %'] = m
            r[f'+{h} GA95'] = f'[{lo:+.2f}, {hi:+.2f}]'
        pairs.append(r)
    return dict(table=pd.DataFrame(rows).round(3), era=pd.DataFrame(era_rows).round(3),
                pairs=pd.DataFrame(pairs).round(3),
                base=base, n_bars=len(sb), horizons=H)


def run(daily, tfs, params, out_dir, rng, write=True, bench=None):
    report = []
    all_ev = []
    for tf in tfs:
        ev = collect_events(daily, tf, params[tf], bench)
        if ev.empty:
            report.append(f'\n## {tf}: olay yok\n')
            continue
        all_ev.append(ev)
        r = analyze(ev, rng)
        hz = r['hazard']
        sec = [f'\n## {tf}  (parametreler: {params[tf]})',
               f'Olay sayısı: {r["n"]}  (up {r["n_up"]}, down {r["n_down"]})',
               f'Medyan derinlik: {r["med_depth"]:.3f}',
               '\n### Kümelenme (excess_ratio>1 ve pctile yüksek = Fib gerçekten "özel")',
               r['clustering'].to_string(index=False),
               '\n### Sonuç — işlenebilir (onay barından giriş, stop = düzeltme dibi)',
               r['trd'].to_string(),
               '\n### Sonuç — tanımlayıcı (dipten itibaren, hindsight)',
               r['desc'].to_string(),
               '\n### Hacim filtresi — P(1.272) işlenebilir',
               r['volume'].to_string()]
        for d_, t in r['by_dir'].items():
            sec += [f'\n### Yön: {d_} — işlenebilir', t.to_string()]
        if r.get('baseline') is not None:
            sec += ['\n### RASTGELE GİRİŞ KONTROLÜ — yukarı setup vs aynı hissede rastgele giriş',
                    r['baseline'].to_string(index=False)]
        if r.get('targets'):
            sec += ['\n### HEDEF TESTİ — yukarı setup, ortalama gerçekleşen R', r['targets']['means'].to_string(),
                    '1.272\'ye göre fark (eşleştirilmiş, hisse bazlı bootstrap):', r['targets']['diffs'].to_string(index=False)]
        if r.get('mg'):
            for d_, m in r['mg'].items():
                sec += [f'\n### MUM GÜCÜ TESTİ — {d_}', 'Skora göre:', m['by_score'].to_string(),
                        'Bileşenler (hisse bazlı bootstrap %95 GA):', m['components'].to_string(index=False),
                        'Derinliğe göre:', m['by_depth'].to_string(), 'Süreye göre:', m['by_len'].to_string(),
                        'Döneme göre:', m['by_era'].to_string()]
        report += sec
        if write:
            hz.to_csv(os.path.join(out_dir, f'{tf}_hazard.csv'), index=False)
            plot_tf(ev, hz, tf, out_dir)
        print('\n'.join(sec))
    if write and all_ev:
        pd.concat(all_ev).to_csv(os.path.join(out_dir, 'events.csv'), index=False)
        with open(os.path.join(out_dir, 'summary.md'), 'w') as f:
            f.write('# Fibonacci ampirik çalışma\n' + '\n'.join(report))
    return all_ev


def sweep(daily, tfs, rng, out_dir):
    rows = []
    for tf in tfs:
        for rm in [1.5, 2.0, 2.5, 3.0]:
            for mi in [3.0, 4.0, 5.0, 6.0]:
                p = {tf: dict(TF_PARAMS[tf], rev_mult=rm, min_impulse=mi)}
                evs = []
                for tk, df in daily.items():
                    bars = resample(df, tf)
                    if len(bars) > p[tf]['atr_len'] + 10:
                        evs += extract_events(bars, tf, tk, p[tf])
                ev = pd.DataFrame(evs)
                if len(ev) < 30:
                    continue
                cl = clustering(ev['depth'].to_numpy(), rng)
                e = ev[ev.trd_complete]
                gp = e[(e.depth >= .618) & (e.depth < .65)]
                rows.append(dict(tf=tf, rev_mult=rm, min_impulse=mi, n=len(ev),
                                 med_depth=round(ev.depth.median(), 3),
                                 fib_excess_mean=round(cl.excess_ratio.mean(), 3),
                                 fib_pctile_mean=round(cl.pctile_vs_random.mean(), 1),
                                 P1272_all=round(e['trd_hit_1.272'].mean(), 3),
                                 P1272_GP=round(gp['trd_hit_1.272'].mean(), 3) if len(gp) else np.nan,
                                 n_GP=len(gp)))
    t = pd.DataFrame(rows)
    print('\n### Parametre taraması (sağlamlık)\n' + t.to_string(index=False))
    t.to_csv(os.path.join(out_dir, 'sweep.csv'), index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tickers', default='tickers.txt')
    ap.add_argument('--tf', default='M,W,2W,3D,D')
    ap.add_argument('--cache', default='cache')
    ap.add_argument('--out', default='fib_out')
    ap.add_argument('--refresh', action='store_true')
    ap.add_argument('--sweep', action='store_true')
    ap.add_argument('--signals', action='store_true', help='Mum Gücü sinyal testi (Fib\'den bağımsız)')
    ap.add_argument('--synthetic', action='store_true')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    tfs = [t for t in TF_ORDER if t in args.tf.upper().split(',')]
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.cache, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    daily = {}
    bench = None
    if args.synthetic:
        for i in range(40):
            daily[f'SYN{i}'] = synthetic_daily(args.seed + i)
        bench = synthetic_daily(999)
    else:
        tickers = load_universe(args.tickers)
        for i, tk in enumerate(tickers, 1):
            try:
                df = load_daily(tk, args.cache, args.refresh)
                if df is not None and len(df) > 300:
                    daily[tk] = df
            except Exception as ex:
                print(f'{tk}: hata {ex}', file=sys.stderr)
            if i % 50 == 0:
                print(f'{i}/{len(tickers)} indirildi')
        try:
            bench = load_daily(BENCH, args.cache, args.refresh)
        except Exception as ex:
            print(f'Endeks ({BENCH}) alınamadı, Mum Gücü testi atlanacak: {ex}', file=sys.stderr)
    print(f'{len(daily)} hisse yüklendi, TF: {tfs}')

    if args.signals:
        if bench is None:
            sys.exit('Sinyal testi için endeks verisi gerekli.')
        for tf in tfs:
            r = signal_analysis(signal_bars(daily, tf, bench), tf, rng)
            if r:
                print(f'\n## {tf} — Mum Gücü sinyal testi ({r["n_bars"]} mum)')
                print(r['table'].to_string(index=False))
                print(r['era'].to_string(index=False))
                print(r['pairs'].to_string(index=False))
        return
    if args.sweep:
        sweep(daily, [t for t in tfs if t in ('M', 'W')] or tfs, rng, args.out)
    else:
        run(daily, tfs, TF_PARAMS, args.out, rng, bench=bench)


def _in_streamlit():
    try:
        from streamlit.runtime import exists
        return exists()
    except Exception:
        return False


if __name__ == '__main__':
    if _in_streamlit():  # Streamlit Cloud bu dosyayı ana dosya olarak çalıştırırsa arayüzü aç
        import streamlit_app
        streamlit_app.render()
    else:
        main()
