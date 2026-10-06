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
  pip install yfinance pandas numpy matplotlib lxml
  python fib_study.py                       # tickers.txt yoksa S&P 500 (Wikipedia)
  python fib_study.py --tickers tickers.txt --tf M,W
  python fib_study.py --sweep --tf M,W
  python fib_study.py --synthetic           # internetsiz test (rastgele yürüyüş)

Not: Wikipedia S&P 500 listesi BUGÜNKÜ üyelerdir -> survivorship bias.
"""
import argparse
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
def load_universe(path):
    if path and os.path.exists(path):
        with open(path) as f:
            return [t.strip().upper() for t in f if t.strip() and not t.startswith('#')]
    print('tickers dosyası yok -> S&P 500 (Wikipedia, bugünkü üyeler, survivorship bias!)')
    tbl = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    return [s.replace('.', '-') for s in tbl['Symbol'].tolist()]


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


# ----------------------------------------------------------------------------- olay çıkarımı
def extract_events(df, tf, ticker, p):
    h, l, c, v = (df[k].to_numpy(float) for k in ['High', 'Low', 'Close', 'Volume'])
    n = len(h)
    a = wilder_atr(h, l, c, p['atr_len'])
    piv = zigzag(h, l, a, p['rev_mult'])
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
def run(daily, tfs, params, out_dir, rng, write=True):
    report = []
    all_ev = []
    for tf in tfs:
        rows = []
        for tk, df in daily.items():
            bars = resample(df, tf)
            if len(bars) < params[tf]['atr_len'] + 10:
                continue
            rows += extract_events(bars, tf, tk, params[tf])
        ev = pd.DataFrame(rows)
        if ev.empty:
            report.append(f'\n## {tf}: olay yok\n')
            continue
        all_ev.append(ev)
        hz = hazard(ev['depth'].to_numpy())
        cl = clustering(ev['depth'].to_numpy(), rng)
        sec = [f'\n## {tf}  (parametreler: {params[tf]})',
               f'Olay sayısı: {len(ev)}  (up {np.sum(ev.dir == "up")}, down {np.sum(ev.dir == "down")})',
               f'Medyan derinlik: {ev.depth.median():.3f}',
               '\n### Kümelenme (excess_ratio>1 ve pctile yüksek = Fib gerçekten "özel")',
               cl.to_string(index=False),
               '\n### Sonuç — işlenebilir (onay barından giriş, stop = düzeltme dibi)',
               outcome_table(ev, 'trd').to_string(),
               '\n### Sonuç — tanımlayıcı (dipten itibaren, hindsight)',
               outcome_table(ev, 'desc').to_string(),
               '\n### Hacim filtresi — P(1.272) işlenebilir',
               volume_table(ev).to_string()]
        for d_ in ['up', 'down']:
            sub = ev[ev.dir == d_]
            if len(sub) > 30:
                sec += [f'\n### Yön: {d_} — işlenebilir', outcome_table(sub, 'trd').to_string()]
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
    ap.add_argument('--synthetic', action='store_true')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    tfs = [t for t in TF_ORDER if t in args.tf.upper().split(',')]
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.cache, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    daily = {}
    if args.synthetic:
        for i in range(40):
            daily[f'SYN{i}'] = synthetic_daily(args.seed + i)
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
    print(f'{len(daily)} hisse yüklendi, TF: {tfs}')

    if args.sweep:
        sweep(daily, [t for t in tfs if t in ('M', 'W')] or tfs, rng, args.out)
    else:
        run(daily, tfs, TF_PARAMS, args.out, rng)


if __name__ == '__main__':
    main()
