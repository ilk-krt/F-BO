"""Fibonacci ampirik çalışma — Streamlit arayüzü (fib_study.py'yi kullanır)."""
import io

import numpy as np
import pandas as pd
import streamlit as st

import fib_study as fs

PCT_COLS = ['P(1)', 'P(1.272)', 'P(1.618)', 'P(2.618)']


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def get_sp500():
    return fs.sp500_tickers()


@st.cache_data(ttl=24 * 3600, show_spinner=False, max_entries=3)
def get_data(tickers: tuple):
    bar = st.progress(0.0, text='Veri indiriliyor…')
    data = fs.fetch_many(list(tickers),
                         progress=lambda d, n: bar.progress(d / n, text=f'Veri indiriliyor… {d}/{n}'))
    bar.empty()
    return {k: v.astype('float32') for k, v in data.items()}


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def get_bench():
    for sym in [fs.BENCH, 'SPY']:  # ^GSPC olmazsa SPY
        try:
            df = fs.fetch_many([sym]).get(sym)
            if df is not None:
                return df
        except Exception:
            pass
    return None


def fig_depth_hazard(ev, hz, tf):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, figsize=(9, 6.5))
    d = ev['depth'][(ev['depth'] > 0) & (ev['depth'] < 1.2)]
    ax[0].hist(d, bins=np.arange(0, 1.21, 0.02), color='#4a7ab5')
    ax[1].plot(hz['level'] + 0.01, hz['stop_prob'], marker='o', ms=3, color='#4a7ab5')
    for a in ax:
        for f in fs.FIB_LEVELS:
            a.axvline(f, color='#d9822b', ls='--', lw=1)
        a.grid(alpha=0.2)
    ax[0].set_title(f'{tf} — düzeltme derinliği dağılımı (n={len(d)})')
    ax[1].set_title('Seviyeye ulaşınca orada dönme olasılığı (hazard)')
    fig.tight_layout()
    return fig


def fmt(t):
    sty = {c: '{:.1%}' for c in PCT_COLS if c in t.columns}
    for c in ['med_max_ext', 'med_RR_1272', 'expR_1272']:
        if c in t.columns:
            sty[c] = '{:.2f}'
    return t.style.format(sty, na_rep='–')


def render():
    st.set_page_config(page_title='Fibonacci Çalışması', layout='wide')
    st.title('Fibonacci Ampirik Çalışma')
    st.caption('Düzeltme dipleri Fib seviyelerinde gerçekten birikiyor mu, '
               've hangi derinlikten sonra trend devam ediyor?')

    with st.sidebar:
        st.header('Ayarlar')
        uni = st.radio('Hisse evreni', ['S&P 500', 'Kendi listem'])
        if uni == 'Kendi listem':
            txt = st.text_area('Semboller (virgül veya satır)', 'AAPL, MSFT, NVDA, AMZN, META')
            tickers = [t.strip().upper() for t in txt.replace(',', '\n').split() if t.strip()]
        else:
            try:
                tickers = get_sp500()
            except Exception as ex:
                st.error(f'S&P 500 listesi alınamadı: {ex}')
                tickers = []
            n_max = st.slider('Kaç hisse (hız için)', 20, max(len(tickers), 20),
                              min(len(tickers), 500), step=10)
            tickers = tickers[:n_max]
        tfs = st.multiselect('Zaman dilimleri', fs.TF_ORDER, default=['M', 'W'])

        params = {}
        with st.expander('ZigZag parametreleri'):
            for tf in fs.TF_ORDER:
                p = dict(fs.TF_PARAMS[tf])
                if tf in tfs:
                    c1, c2 = st.columns(2)
                    p['rev_mult'] = c1.number_input(f'{tf} onay (ATR)', 0.5, 10.0,
                                                    p['rev_mult'], 0.5, key=f'rm_{tf}')
                    p['min_impulse'] = c2.number_input(f'{tf} min itki (ATR)', 1.0, 20.0,
                                                       p['min_impulse'], 0.5, key=f'mi_{tf}')
                params[tf] = p
        go = st.button('Çalıştır', type='primary', width='stretch',
                       disabled=not (tickers and tfs))

    if go:
        data = get_data(tuple(tickers))
        if not data:
            st.error('Hiç veri indirilemedi (Yahoo erişimi / sembol hatası).')
            return
        bench = get_bench()
        if bench is None:
            st.warning('Endeks verisi alınamadı; Mum Gücü testi atlanacak.')
        rng = np.random.default_rng(42)
        results = {}
        for tf in [t for t in fs.TF_ORDER if t in tfs]:
            with st.spinner(f'{tf} analiz ediliyor…'):
                ev = fs.collect_events(data, tf, params[tf], bench)
                if not ev.empty:
                    results[tf] = (ev, fs.analyze(ev, rng), params[tf])
        st.session_state['res'] = results
        st.session_state['n_loaded'] = len(data)

    results = st.session_state.get('res')
    if not results:
        st.info('Soldan ayarları seçip **Çalıştır**\'a bas. '
                'S&P 500 için ilk indirme birkaç dakika sürebilir; sonra 24 saat cache\'lenir.')
        return

    st.success(f"{st.session_state.get('n_loaded', 0)} hisse yüklendi.")
    tabs = st.tabs(list(results.keys()))
    for tab, (tf, (ev, r, p)) in zip(tabs, results.items()):
        with tab:
            c = st.columns(4)
            c[0].metric('Olay', r['n'])
            c[1].metric('Yukarı / Aşağı', f"{r['n_up']} / {r['n_down']}")
            c[2].metric('Medyan derinlik', f"{r['med_depth']:.3f}")
            c[3].metric('Parametre', f"{p['rev_mult']} / {p['min_impulse']} ATR")

            st.subheader('Kümelenme')
            st.caption('excess_ratio > 1 ve pctile_vs_random yüksek → dipler o seviyede komşu '
                       'bölgelere göre fazla birikiyor. Tek TF\'de çıkan sonuç gürültü olabilir.')
            st.dataframe(r['clustering'], hide_index=True, width='stretch')
            st.pyplot(fig_depth_hazard(ev, r['hazard'], tf))

            st.subheader('Sonuç — işlenebilir')
            st.caption('Giriş: dip ZigZag ile onaylandıktan sonra, onay barı kapanışı. '
                       'Stop: düzeltme dibi. Gerçek trade\'e en yakın tablo.')
            st.dataframe(fmt(r['trd']), width='stretch')

            for d_, t in r['by_dir'].items():
                with st.expander(f"Yön: {'yukarı' if d_ == 'up' else 'aşağı'}"):
                    st.dataframe(fmt(t), width='stretch')

            with st.expander('Hacim filtresi — P(1.272)'):
                st.dataframe(r['volume'], width='stretch')
            with st.expander('Sonuç — tanımlayıcı (dipten itibaren, hindsight → iyimser)'):
                st.dataframe(fmt(r['desc']), width='stretch')

            mg = r.get('mg')
            if mg:
                st.subheader('🕯️ Mum Gücü testi')
                st.caption('Soru: Fib setup\'ında Mum Gücü teyidi sonucu iyileştiriyor mu? '
                           '"düz_fark" = derinlik ve düzeltme süresi etkisi ayıklanmış fark. '
                           'GA95 hisse bazlı bootstrap. Sıfırı içermiyorsa → anlamlı. '
                           'Çok sayıda test var: tek bir "EVET" şans eseri olabilir; '
                           'M ve W\'de ve iki dönemde de tutarlı olmalı.')
                for d_, m in mg.items():
                    st.markdown(f"**{'Yukarı' if d_ == 'up' else 'Aşağı'} setup'lar**")
                    st.dataframe(m['components'], hide_index=True, width='stretch')
                    st.dataframe(m['by_score'].style.format({'P1': '{:.1%}', 'P1272': '{:.1%}', 'P1618': '{:.1%}',
                                                             'meanR': '{:.2f}', 'medR': '{:.2f}', 'n': '{:.0f}'}),
                                 width='stretch')
                    with st.expander('Derinlik / süre / dönem kırılımı'):
                        st.dataframe(m['by_depth'], width='stretch')
                        st.dataframe(m['by_len'], width='stretch')
                        st.dataframe(m['by_era'], width='stretch')

            buf = io.StringIO()
            ev.to_csv(buf, index=False)
            st.download_button(f'{tf} olaylarını indir (CSV)', buf.getvalue(),
                               file_name=f'fib_events_{tf}.csv', mime='text/csv')


if __name__ == '__main__':
    render()
