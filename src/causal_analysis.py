"""
causal_analysis.py: why do Kyoto's cherries bloom earlier?

Causal discovery and effect estimation on 146 years of Kyoto weather (JMA, 1881-2026)
and the full-bloom record of Prunus jamasakura in Arashiyama (Aono, 812-2026).

The script is the science only. It writes tables to results/ and prints a summary;
it draws nothing.

    python src/causal_analysis.py            # full run, KCI tests included (about a minute)
    python src/causal_analysis.py --fast     # skips the kernel (KCI) tests, fewer bootstraps (~20 s)

Pipeline
    1. Data           clean.build(): JMA quality flags, bloom-year re-indexing (Oct-Dec of Y-1 -> Y)
    2. Description    long-record eras, Theil-Sen trends, Pettitt change points
    3. Validation     our 1929-2025 monthly warming rates against JMA's published Kyoto values
    4. Trend removal  year-to-year differences, so slow shared drivers cannot pose as causes
    5. Assumptions    Gaussianity of every variable (LiNGAM/ANM need non-Gaussianity)
    6. Discovery      blind: DirectLiNGAM, BottomUpParceLiNGAM, RCD, ANM, PC, CD-NOD
                      time-ordered: DirectLiNGAM (+bootstrap), PC (Fisher-z, KCI), FCI, CD-NOD
    7. Effects        first-differenced regressions, Newey-West (HAC) errors, robustness specs
    8. Attribution    counterfactual decomposition of the 1881-1910 -> 1997-2026 shift,
                      moving-block bootstrap; urban vs regional split from JMA rural benchmark
    9. Forecast test  train 1881-1996, forecast the warmer 1997-2026: linear vs tree ensembles
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import clean  # noqa: E402

RESULTS = ROOT / "results"
SEED = 42

# JMA Heat-Island Monitoring Report, Kyoto vs the mean of 15 low-urbanisation stations,
# monthly mean temperature trend 1929-2025, °C per 100 years (Jan..Dec). Cited constants.
JMA_KYOTO = [2.7, 3.0, 3.6, 3.2, 3.0, 2.6, 2.2, 2.7, 3.3, 3.4, 2.9, 2.0]
JMA_RURAL15 = [1.8, 2.2, 2.5, 2.1, 2.0, 1.9, 1.4, 1.5, 2.1, 2.1, 1.7, 1.0]

# Variables for causal discovery (n = 133 complete years, 1890-2026)
VARS = {"gmst_ONDJFM": "GlobalT", "temp_OND": "T_autumn", "temp_m1": "T_Jan", "temp_m2": "T_Feb",
        "temp_m3": "T_Mar", "precip_m3": "P_Mar", "sun_m3": "Sun_Mar", "doy": "Bloom"}
# Time order: a bloom in April cannot change March weather; nothing local causes global temperature
TIER = {"GlobalT": 0, "T_autumn": 1, "T_Jan": 2, "T_Feb": 3, "T_Mar": 4, "P_Mar": 4, "Sun_Mar": 4, "Bloom": 5}

# Monthly temperatures of the dormancy year used for effect sizes (p = previous calendar year)
MON = ["temp_m10p", "temp_m11p", "temp_m12p", "temp_m1", "temp_m2", "temp_m3"]
PRETTY = {"temp_m10p": "Oct (prev)", "temp_m11p": "Nov (prev)", "temp_m12p": "Dec (prev)",
          "temp_m1": "Jan", "temp_m2": "Feb", "temp_m3": "Mar", "year": "year (trend)"}


# ----------------------------------------------------------------------------- helpers
def silent(fn, *a, **k):
    """Run a causal-learn call without its progress bars and verbose prints."""
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return fn(*a, **k)


def lowess_detrend(s: pd.Series) -> pd.Series:
    """Anomalies from a ~31-year LOWESS trend."""
    return s - sm.nonparametric.lowess(s.values, s.index.values, frac=31 / len(s), return_sorted=False)


def hac_ols(df: pd.DataFrame, xs: list[str], y: str = "doy"):
    """OLS with Newey-West (HAC, 2 lags) standard errors."""
    return sm.OLS(df[y], sm.add_constant(df[xs])).fit(cov_type="HAC", cov_kwds={"maxlags": 2})


def theil(y: pd.Series) -> tuple[float, float, float]:
    """Theil-Sen slope per decade with its 95% interval."""
    y = y.dropna()
    r = stats.theilslopes(y.values, y.index.values)
    return r.slope * 10, r.low_slope * 10, r.high_slope * 10


def pettitt(s: pd.Series) -> dict:
    """Pettitt change-point test: last year of the 'before' segment, p-value, segment means."""
    s = s.dropna(); x = s.values; n = len(x)
    U = np.array([np.sign(x[:k + 1, None] - x[None, k + 1:]).sum() for k in range(n - 1)])
    k = np.abs(U).argmax(); K = np.abs(U).max()
    return dict(last_year_before=int(s.index[k]), p=2 * np.exp(-6 * K**2 / (n**3 + n**2)),
                mean_before=x[:k + 1].mean(), mean_after=x[k + 1:].mean())


def pcorr(df: pd.DataFrame, a: str, b: str, cond: list[str]) -> float:
    """Partial correlation of a and b given cond."""
    Zc = np.c_[np.ones(len(df)), df[cond].values] if cond else np.ones((len(df), 1))
    ra = df[a] - Zc @ np.linalg.lstsq(Zc, df[a], rcond=None)[0]
    rb = df[b] - Zc @ np.linalg.lstsq(Zc, df[b], rcond=None)[0]
    return float(np.corrcoef(ra, rb)[0, 1])


def edge(G, a: str, b: str) -> str:
    """The CPDAG/PAG mark between a and b as text, e.g. 'T_Mar --> Bloom' or 'T_Mar o-o P_Mar'."""
    L = {"TAIL": "-", "ARROW": "<", "CIRCLE": "o"}; R = {"TAIL": "-", "ARROW": ">", "CIRCLE": "o"}
    for e in G.get_graph_edges():
        n1, n2 = e.get_node1().get_name(), e.get_node2().get_name()
        if {n1, n2} == {a, b}:
            return f"{n1} {L[e.get_endpoint1().name]}-{R[e.get_endpoint2().name]} {n2}"
    return "no edge"


def all_edges(G) -> list[str]:
    return [edge(G, e.get_node1().get_name(), e.get_node2().get_name()) for e in G.get_graph_edges()]


def tiers_bk(nodes, tier: dict):
    from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
    bk = BackgroundKnowledge()
    for nd in nodes:
        bk.add_node_to_tier(nd, tier[nd.get_name()])
    return bk


def save(df: pd.DataFrame, name: str, index: bool = False) -> None:
    df.round(4).to_csv(RESULTS / name, index=index)


# ----------------------------------------------------------------------------- 1-3 data, description, validation
def describe(p: pd.DataFrame, m: pd.DataFrame, w: pd.DataFrame) -> dict:
    b = p.doy.dropna()
    eras = [(812, 1399), (1400, 1699), (1700, 1880), (1881, 1950), (1951, 1990), (1991, 2026)]
    era_tab = pd.DataFrame([(f"{a}-{z}", len(b.loc[a:z]), b.loc[a:z].mean(), b.loc[a:z].std(), b.loc[a:z].min()) for a, z in eras],
                           columns=["era", "n_years", "mean_doy", "sd", "earliest_doy"])
    save(era_tab, "01_eras.csv")

    trends = pd.DataFrame([(f"bloom_doy_since_{a}", *theil(m.doy.loc[a:])) for a in (1881, 1950, 1980)],
                          columns=["series", "days_per_decade", "ci95_low", "ci95_high"])
    save(trends, "01_trends_theil_sen.csv")
    cps = pd.DataFrame({c: pettitt(m[c]) for c in ["doy", "temp_FM", "temp_m3", "temp_OND", "temp_annual"]}).T
    save(cps, "01_change_points_pettitt.csv", index=True)

    # Validation: our OLS 1929-2025 monthly trends must reproduce JMA's published Kyoto values
    mw = pd.DataFrame({
        "month": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
        "ours_ols_1929_2025": [np.polyfit(w.loc[1929:2025].index, w.loc[1929:2025, f"temp_{k:02d}"], 1)[0] * 100 for k in range(1, 13)],
        "jma_kyoto_1929_2025": JMA_KYOTO, "jma_rural15_1929_2025": JMA_RURAL15})
    mw["urban_excess"] = mw.jma_kyoto_1929_2025 - mw.jma_rural15_1929_2025
    mw["urban_share"] = mw.urban_excess / mw.jma_kyoto_1929_2025
    save(mw, "02_jma_validation_and_urban_benchmark.csv")
    max_err = float((mw.ours_ols_1929_2025.round(1) - mw.jma_kyoto_1929_2025).abs().max())
    return {"earliest_on_record": b.sort_values().head(3).astype(int).to_dict(),
            "bloom_trend_since_1950_days_per_decade": trends.iloc[1].days_per_decade,
            "bloom_change_point_last_year_before": cps.loc["doy", "last_year_before"],
            "jma_validation_max_abs_error_C_per_century": max_err}


# ----------------------------------------------------------------------------- 4 trend removal
def trend_free_screen(m: pd.DataFrame) -> pd.DataFrame:
    """Every candidate variable in levels and in year-to-year differences.

    This is a pre-screen, not causal evidence. Differencing removes everything that drifts
    slowly (global warming, urban growth, observer change), so a variable whose link to bloom
    vanishes once differenced was riding the shared trend, not driving the bloom.
    """
    cand = [c for c in m.columns if c.startswith(("temp_", "precip_", "sun_", "wind_", "snow_"))] + ["gmst_ONDJFM"]
    rows = []
    for c in cand:
        d = m[["doy", c]].dropna()
        if len(d) < 40 or d[c].std() == 0:
            continue
        dd = d.diff().dropna()
        rows.append((c, len(d), d.doy.corr(d[c]), dd.doy.corr(dd[c])))
    screen = pd.DataFrame(rows, columns=["variable", "n", "r_levels", "r_year_to_year"]).dropna()
    screen = screen.reindex(screen.r_year_to_year.abs().sort_values(ascending=False).index)
    save(screen, "03_trend_free_screen.csv")
    return screen


# ----------------------------------------------------------------------------- 5-6 causal discovery
def discovery_data(m: pd.DataFrame):
    D = m[list(VARS)].rename(columns=VARS).dropna()
    names = list(D.columns)
    ix = {n: i for i, n in enumerate(names)}
    Z = ((D - D.mean()) / D.std()).values
    A = D.apply(lowess_detrend)
    ZA = ((A - A.mean()) / A.std()).values
    gauss = pd.DataFrame({"shapiro_p": [stats.shapiro(D[c]).pvalue for c in names],
                          "skew": [stats.skew(D[c]) for c in names],
                          "excess_kurtosis": [stats.kurtosis(D[c]) for c in names],
                          "shapiro_p_detrended": [stats.shapiro(A[c]).pvalue for c in names]}, index=names)
    save(gauss, "04_gaussianity.csv", index=True)
    return D, names, ix, Z, ZA


def blind_tournament(D, names, ix, Z, ZA, fast: bool) -> pd.DataFrame:
    """Can algorithms find 'March warmth -> bloom' without being told time order?"""
    import lingam
    from causallearn.search.ConstraintBased.CDNOD import cdnod
    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.search.FCMBased.ANM.ANM import ANM

    tour = []
    for label, data in [("DirectLiNGAM (blind, raw)", Z), ("DirectLiNGAM (blind, detrended)", ZA)]:
        mdl = lingam.DirectLiNGAM(random_state=SEED).fit(data)
        Bm = mdl.adjacency_matrix_
        fwd, back = Bm[ix["Bloom"], ix["T_Mar"]], Bm[ix["T_Mar"], ix["Bloom"]]
        bs = lingam.DirectLiNGAM(random_state=SEED).bootstrap(data, n_sampling=60 if fast else 200).get_probabilities(min_causal_effect=0.05)
        tour.append((label, "T_Mar -> Bloom" if abs(fwd) > abs(back) else "Bloom -> T_Mar",
                     f"P(fwd)={bs[ix['Bloom'], ix['T_Mar']]:.2f}, P(back)={bs[ix['T_Mar'], ix['Bloom']]:.2f}",
                     " > ".join(names[i] for i in mdl.causal_order_)))

    bu = lingam.BottomUpParceLiNGAM(alpha=0.1).fit(Z)
    flat = []
    for g in bu.causal_order_:
        flat += [g] if isinstance(g, list) else [[g]]
    pos = {names[k]: i for i, grp in enumerate(flat) for k in grp}
    tour.append(("BottomUpParceLiNGAM (latent-aware)", "T_Mar -> Bloom" if pos["T_Mar"] < pos["Bloom"] else "Bloom -> T_Mar",
                 "order groups: " + str([[names[k] for k in g] for g in flat]), ""))

    rcd = lingam.RCD(max_explanatory_num=2, cor_alpha=0.01, ind_alpha=0.01, shapiro_alpha=0.01).fit(Z)
    v = rcd.adjacency_matrix_[ix["Bloom"], ix["T_Mar"]]
    tour.append(("RCD (latent-aware LiNGAM)", "abstains: flags latent confounder" if np.isnan(v) else f"coef {v:+.2f}",
                 f"{int(np.isnan(rcd.adjacency_matrix_).sum() / 2)} pairs flagged as confounded", ""))

    anm = ANM(); rng = np.random.default_rng(SEED)
    xa, ya = Z[:, [ix["T_Mar"]]], Z[:, [ix["Bloom"]]]
    pf, pb = anm.cause_or_effect(xa, ya)
    draws = [rng.integers(0, len(xa), len(xa)) for _ in range(10 if fast else 30)]
    wins = np.mean([np.greater(*anm.cause_or_effect(xa[i], ya[i])) for i in draws])
    tour.append(("ANM (GP + HSIC)", "T_Mar -> Bloom" if pf > pb else "Bloom -> T_Mar",
                 f"p_fwd={pf:.2f}, p_back={pb:.2f}, bootstrap P(fwd wins)={wins:.2f}", ""))

    cg = pc(Z, 0.05, "fisherz", node_names=names, show_progress=False)
    tour.append(("PC (Fisher-z, no priors)", edge(cg.G, "T_Mar", "Bloom"), "skeleton found, direction undetermined", ""))

    if not fast:
        C = np.asarray(D.index, dtype=float).reshape(-1, 1)
        cgn = silent(cdnod, Z, C, 0.05, "kci", show_progress=False)
        xname = {f"X{i + 1}": n for i, n in enumerate(names + ["C_time"])}
        e_ = edge(cgn.G, "X5", "X8")
        tour.append(("CD-NOD (KCI, no priors)", " ".join(xname.get(t, t) for t in e_.split()), "", ""))

    out = pd.DataFrame(tour, columns=["method", "verdict_T_Mar_Bloom", "support", "causal_order"])
    save(out, "05_blind_tournament.csv")
    return out, cg


def time_ordered_discovery(D, names, ix, Z, cg_blind, fast: bool):
    """DirectLiNGAM with prior knowledge (+bootstrap), PC (Fisher-z, KCI) and FCI with tiers."""
    import lingam
    from causallearn.search.ConstraintBased.FCI import fci
    from causallearn.search.ConstraintBased.PC import pc

    pk = -np.ones((len(names),) * 2)  # lingam convention: pk[i, j] = 0 means "j does not cause i"
    for a in names:
        for b in names:
            if a != b and (TIER[b] > TIER[a] or a == "GlobalT"):
                pk[ix[a], ix[b]] = 0
    lp = lingam.DirectLiNGAM(random_state=SEED, prior_knowledge=pk).fit(Z)
    lp_bs = lingam.DirectLiNGAM(random_state=SEED, prior_knowledge=pk).bootstrap(Z, n_sampling=100 if fast else 300)
    P_edge = lp_bs.get_probabilities(min_causal_effect=0.05)
    TE = pd.DataFrame(lp_bs.get_total_causal_effects(min_causal_effect=0.01))
    TE["from"] = TE["from"].map(lambda i: names[i]); TE["to"] = TE["to"].map(lambda i: names[i])

    nodes = cg_blind.G.get_nodes()
    cg_fz = silent(pc, Z, 0.05, "fisherz", node_names=names, background_knowledge=tiers_bk(nodes, TIER), show_progress=False)
    cg_kci = None if fast else silent(pc, Z, 0.05, "kci", node_names=names, background_knowledge=tiers_bk(nodes, TIER), show_progress=False)
    G0, _ = silent(fci, Z, "fisherz", 0.05, node_names=names, show_progress=False, verbose=False)
    G_fci, _ = silent(fci, Z, "fisherz", 0.05, node_names=names, background_knowledge=tiers_bk(G0.get_nodes(), TIER),
                      show_progress=False, verbose=False)

    rows = []
    for c in [n for n in names if n != "Bloom"]:
        te = TE.query("`from` == @c and to == 'Bloom'")
        rows.append((c, lp.adjacency_matrix_[ix["Bloom"], ix[c]], P_edge[ix["Bloom"], ix[c]],
                     te.effect.mean() if len(te) else 0.0, edge(cg_fz.G, c, "Bloom"),
                     "skipped (--fast)" if cg_kci is None else edge(cg_kci.G, c, "Bloom"), edge(G_fci, c, "Bloom")))
    parents = pd.DataFrame(rows, columns=["candidate_cause", "lingam_direct_sd", "lingam_bootstrap_p_edge", "lingam_total_effect_sd",
                                          "pc_fisherz", "pc_kci", "fci"])
    save(parents, "06_parents_of_bloom_time_ordered.csv")
    graphs = pd.DataFrame({"method": ["PC Fisher-z (tiers)"] * len(all_edges(cg_fz.G)) + ["FCI (tiers)"] * len(all_edges(G_fci)),
                           "edge": all_edges(cg_fz.G) + all_edges(G_fci)})
    save(graphs, "06_full_graphs_time_ordered.csv")
    return lp, P_edge, cg_fz, G_fci, parents


def cdnod_mechanism_change(D, names, Z, fast: bool) -> pd.DataFrame:
    """CD-NOD with the time index C as a surrogate for anything that drifts.
    An edge C_time -> X means X's causal mechanism changes over time."""
    from causallearn.search.ConstraintBased.CDNOD import cdnod
    from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge

    C = np.asarray(D.index, dtype=float).reshape(-1, 1)
    tierC = {k: v + 1 for k, v in TIER.items()}; tierC["C_time"] = 0  # causal-learn tiers must be >= 0
    xn = {f"X{i + 1}": n for i, n in enumerate(names + ["C_time"])}
    rows = []
    for test in (("fisherz",) if fast else ("fisherz", "kci")):
        g0 = silent(cdnod, Z, C, 0.05, test, show_progress=False)
        bk = BackgroundKnowledge()
        for nd in g0.G.get_nodes():
            bk.add_node_to_tier(nd, tierC[xn[nd.get_name()]])
        g = silent(cdnod, Z, C, 0.05, test, background_knowledge=bk, show_progress=False)
        for e in all_edges(g.G):
            rows.append((test, " ".join(xn.get(t, t) for t in e.split())))
    out = pd.DataFrame(rows, columns=["test", "edge"])
    out["touches_time"] = out.edge.str.contains("C_time")
    save(out, "07_cdnod_edges.csv")
    return out


def faithfulness_check(D, names, ix, Z, cg_blind, lp) -> dict:
    """Bloom tracks March so tightly that conditioning on Bloom masks March's own causes."""
    from causallearn.search.FCMBased.ANM.ANM import ANM
    pf, pb = ANM().cause_or_effect(Z[:, [ix["GlobalT"]]], Z[:, [ix["T_Mar"]]])
    return {"pc_sepset_GlobalT_T_Mar": [[names[k] for k in s] for s in (cg_blind.sepset[ix["GlobalT"], ix["T_Mar"]] or [])],
            "corr_GlobalT_T_Mar": pcorr(D, "GlobalT", "T_Mar", []),
            "corr_GlobalT_T_Mar_given_Bloom": pcorr(D, "GlobalT", "T_Mar", ["Bloom"]),
            "corr_GlobalT_Bloom_given_T_Feb_T_Mar": pcorr(D, "GlobalT", "Bloom", ["T_Feb", "T_Mar"]),
            "anm_GlobalT_vs_T_Mar": "GlobalT -> T_Mar" if pf > pb else "T_Mar -> GlobalT",
            "lingam_time_ordered_GlobalT_to_T_Mar_sd": float(lp.adjacency_matrix_[ix["T_Mar"], ix["GlobalT"]])}


# ----------------------------------------------------------------------------- 7 effect sizes
def effect_sizes(m: pd.DataFrame):
    """Days per °C on year-to-year differences (HAC errors) plus robustness specifications."""
    Dm = m[["doy"] + MON].dropna().copy()
    Dm["year"] = Dm.index.astype(float)
    specs = {"Year-to-year differences": (Dm.diff().dropna(), MON),
             "LOWESS anomalies": (Dm[["doy"] + MON].apply(lowess_detrend), MON),
             "Differences 1881-1953": (Dm.loc[:1953].diff().dropna(), MON),
             "Differences 1954-2026": (Dm.loc[1954:].diff().dropna(), MON),
             "Levels": (Dm, MON), "Levels + year trend": (Dm, MON + ["year"])}
    rows = []
    for k, (df, xs) in specs.items():
        f = hac_ols(df, xs)
        for x in xs:
            rows.append((k, PRETTY[x], f.params[x], f.bse[x], f.pvalues[x], f.rsquared, int(f.nobs)))
    tab = pd.DataFrame(rows, columns=["specification", "term", "days_per_C", "se_hac", "p", "r2", "n"])
    save(tab, "08_effects_days_per_C.csv")
    beta = hac_ols(Dm.diff().dropna(), MON).params[MON]

    # Do snow, wind, sunshine or rain add anything once temperature is known? (1953+, all variables exist)
    q = m.loc[1953:]; extra_rows = []
    for extra in ["snow_DJF", "wind_FM", "sun_FM", "precip_FM"]:
        cols = ["temp_m10p", "temp_m11p", "temp_m2", "temp_m3", extra]
        f = hac_ols(q[["doy"] + cols].dropna().diff().dropna(), cols)
        extra_rows.append((extra, f.params[extra], f.pvalues[extra]))
    save(pd.DataFrame(extra_rows, columns=["added_variable", "coef_per_unit", "p"]), "08_other_weather_given_temperature.csv")

    # Stability of the Feb-Mar sensitivity in rolling 40-year windows
    r2 = m[["doy", "temp_FM"]].dropna(); roll = []
    for s0 in range(1881, 1988, 3):
        s = r2.loc[s0:s0 + 39].diff().dropna()
        f = sm.OLS(s.doy, sm.add_constant(s.temp_FM)).fit()
        roll.append((s0 + 19.5, f.params.temp_FM, *f.conf_int().loc["temp_FM"], r2.loc[s0:s0 + 39].temp_FM.mean()))
    save(pd.DataFrame(roll, columns=["center_year", "days_per_C", "ci95_low", "ci95_high", "mean_T_FM"]), "08_rolling_sensitivity.csv")
    return Dm, beta, r2


# ----------------------------------------------------------------------------- 8 attribution
def decompose(p, Dm, r2, e1, e2, B=2000):
    """Counterfactual decomposition of the shift in mean bloom between two eras.
    Sensitivities come from year-to-year differences, resampled with a 5-year moving-block bootstrap."""
    a, z = p.loc[e1[0]:e1[1]], p.loc[e2[0]:e2[1]]
    dT = z[MON].mean() - a[MON].mean(); obs = z.doy.mean() - a.doy.mean()
    dd = Dm.diff().dropna(); rng = np.random.default_rng(SEED); idx = dd.index.values; out = []
    for _ in range(B):
        st = rng.integers(0, len(idx) - 5, len(idx) // 5)
        s = dd.loc[np.concatenate([idx[i:i + 5] for i in st])]
        bb = sm.OLS(s.doy, sm.add_constant(s[MON])).fit().params[MON]; c = bb * dT
        out.append([c.temp_m3, c.temp_m2, c[["temp_m10p", "temp_m11p", "temp_m12p"]].sum(), c.temp_m1, c.sum()])
    out = np.array(out)
    lab = ["March warming", "February warming", "Autumn brake (Oct-Dec)", "January", "All measured temperature"]
    res = pd.DataFrame({"days": np.median(out, 0), "ci95_low": np.percentile(out, 2.5, 0), "ci95_high": np.percentile(out, 97.5, 0)}, index=lab)
    res.loc["Observed shift"] = [obs, np.nan, np.nan]
    allT = res.loc["All measured temperature"]
    res.loc["Unexplained residual"] = [obs - allT.days, obs - allT.ci95_high, obs - allT.ci95_low]
    dfm = r2.diff().dropna()
    fm_sens = sm.OLS(dfm.doy, sm.add_constant(dfm.temp_FM)).fit().params.temp_FM
    res.loc["Spring-only model (Feb-Mar mean)"] = [fm_sens * (z.temp_FM.mean() - a.temp_FM.mean()), np.nan, np.nan]
    return res, dT


def attribution(p, m, Dm, beta, r2, fast: bool) -> dict:
    B = 400 if fast else 2000
    att, dT = decompose(p, Dm, r2, (1881, 1910), (1997, 2026), B=B)
    save(att.reset_index(names="component"), "09_attribution_1881-1910_to_1997-2026.csv")
    save(dT.rename(PRETTY).rename_axis("month").rename("warming_C").reset_index(), "09_warming_between_eras.csv")

    # Robustness: baselines after the December 1913 station relocation
    rob = []
    for base in [(1914, 1943), (1914, 1953)]:
        r_, _ = decompose(p, Dm, r2, base, (1997, 2026), B=300)
        rob.append((f"{base[0]}-{base[1]}", *r_.loc[["Observed shift", "All measured temperature",
                                                       "Spring-only model (Feb-Mar mean)", "Unexplained residual"], "days"]))
    save(pd.DataFrame(rob, columns=["baseline", "observed", "all_temperature", "spring_only", "residual"]), "09_attribution_post_relocation_baselines.csv")

    # Station-move step (net of global temperature)
    qq = m[["temp_OND", "temp_FM", "gmst_ONDJFM"]].dropna().assign(post1913=lambda d: (d.index >= 1914).astype(float))
    steps = {c: sm.OLS(qq[c], sm.add_constant(qq[["gmst_ONDJFM", "post1913"]])).fit() for c in ["temp_OND", "temp_FM"]}
    save(pd.DataFrame([(c, f.params.post1913, f.pvalues.post1913) for c, f in steps.items()],
                      columns=["series", "step_C_after_1913_move", "p"]), "09_station_move_step.csv")

    # Urban vs regional split, 1929-2025: JMA's published trends (x 0.96 centuries) times our sensitivities
    mmap = {"temp_m10p": 9, "temp_m11p": 10, "temp_m12p": 11, "temp_m1": 0, "temp_m2": 1, "temp_m3": 2}
    split = pd.DataFrame([(PRETTY[k], beta[k] * (JMA_KYOTO[i] - JMA_RURAL15[i]) * 0.96, beta[k] * JMA_RURAL15[i] * 0.96)
                          for k, i in mmap.items()], columns=["month", "urban_excess_days", "regional_global_days"]).set_index("month")
    split.loc["Total"] = split.sum()
    save(split, "10_urban_vs_regional_split_1929-2025.csv", index=True)
    spring = split.loc[["Feb", "Mar"]].sum()

    # Counterfactual bloom series
    base = p.loc[1881:1910, MON].mean()
    cf_full = m.doy - ((m[MON] - base) * beta).sum(axis=1, min_count=6)
    dfm = r2.diff().dropna()
    fm_beta = sm.OLS(dfm.doy, sm.add_constant(dfm.temp_FM)).fit().params.temp_FM
    cf_spring = m.doy - (m.temp_FM - p.loc[1881:1910, "temp_FM"].mean()) * fm_beta
    save(pd.DataFrame({"year": m.index, "observed_doy": m.doy.values, "cf_no_spring_warming": cf_spring.values,
                       "cf_no_measured_warming": cf_full.values}), "10_counterfactual_bloom.csv")

    dT_FM = p.loc[1997:2026, "temp_FM"].mean() - p.loc[1881:1910, "temp_FM"].mean()
    dG = p.loc[1997:2026, "gmst_ONDJFM"].mean() - p.loc[1881:1910, "gmst_ONDJFM"].mean()
    return {"observed_shift_days": att.loc["Observed shift", "days"],
            "march_days": att.loc["March warming", "days"], "february_days": att.loc["February warming", "days"],
            "autumn_brake_days": att.loc["Autumn brake (Oct-Dec)", "days"],
            "all_temperature_days": att.loc["All measured temperature", "days"],
            "residual_days": att.loc["Unexplained residual", "days"],
            "spring_only_days": att.loc["Spring-only model (Feb-Mar mean)", "days"],
            "kyoto_FM_warming_C": dT_FM, "global_ONDJFM_warming_C": dG,
            "urban_share_of_spring_accelerator_1929_2025": spring.urban_excess_days / spring.sum(),
            "autumn_step_after_1913_move_C": steps["temp_OND"].params.post1913}


# ----------------------------------------------------------------------------- 9 forecast test
def forecast_test(m: pd.DataFrame) -> pd.DataFrame:
    """Train 1881-1996, forecast 1997-2026, a period warmer than anything in training."""
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import LinearRegression, RidgeCV
    feats = ["temp_m10p", "temp_m11p", "temp_m12p", "temp_m1", "temp_m2", "temp_m3", "temp_m4", "precip_m3", "precip_FM", "precip_OND"]
    Dml = m[["doy"] + feats].dropna(); tr, te = Dml.loc[:1996], Dml.loc[1997:]
    models = {"Training-period mean (no weather)": None,
              "Linear, March temp": (LinearRegression(), ["temp_m3"]),
              "Linear, Feb + Mar temp": (LinearRegression(), ["temp_m2", "temp_m3"]),
              "Linear, Oct-Mar temps (incl. brake)": (LinearRegression(), MON),
              "Ridge, all 10 features": (RidgeCV(alphas=np.logspace(-2, 3, 30)), feats),
              "Random forest, all 10": (RandomForestRegressor(500, min_samples_leaf=3, random_state=SEED), feats),
              "Gradient boosting, all 10": (GradientBoostingRegressor(n_estimators=300, max_depth=2, learning_rate=0.03,
                                                                      subsample=0.8, random_state=SEED), feats)}
    res = []
    for k, v in models.items():
        pr = np.full(len(te), tr.doy.mean()) if v is None else v[0].fit(tr[v[1]], tr.doy).predict(te[v[1]])
        e = te.doy.values - pr
        res.append((k, np.sqrt(np.mean(e**2)), np.mean(np.abs(e)), e.mean()))
    ml = pd.DataFrame(res, columns=["model", "rmse_days", "mae_days", "bias_obs_minus_pred"])
    ml.attrs["n_test_marches_warmer_than_training"] = int((te.temp_m3 > tr.temp_m3.max()).sum())
    ml.attrs["n_test"] = len(te)
    save(ml, "11_forecast_test_train1881-1996_test1997-2026.csv")
    return ml


# ----------------------------------------------------------------------------- graph export
def export_graph(names, ix, lp, P_edge, cg_fz, G_fci, beta, n_years, years) -> None:
    Bp = pd.DataFrame(lp.adjacency_matrix_, index=names, columns=names)  # Bp.loc[effect, cause]
    node_meta = {"GlobalT": ("Global temperature (Oct-Mar)", 0, "driver"), "Urban": ("Urban heat island", 0, "latent"),
                 "T_autumn": ("Oct-Dec temperature (prev. year)", 1, "climate"), "T_Jan": ("January temperature", 2, "climate"),
                 "T_Feb": ("February temperature", 3, "climate"), "T_Mar": ("March temperature", 4, "climate"),
                 "P_Mar": ("March precipitation", 4, "climate"), "Sun_Mar": ("March sunshine", 4, "climate"),
                 "Bloom": ("Full-bloom date", 5, "outcome"), "Residual": ("Unexplained trend", 5, "latent")}
    fd_key = {"T_Mar": "temp_m3", "T_Feb": "temp_m2", "T_autumn": "temp_m10p", "T_Jan": "temp_m1"}
    edges = []
    for eff in names:
        for cau in names:
            wgt = Bp.loc[eff, cau]
            if abs(wgt) > 0.05:
                e = {"source": cau, "target": eff, "kind": "direct", "lingam_coef_sd": round(float(wgt), 4),
                     "lingam_bootstrap_p": round(float(P_edge[ix[eff], ix[cau]]), 4),
                     "pc_fisherz": edge(cg_fz.G, cau, eff), "fci": edge(G_fci, cau, eff)}
                if eff == "Bloom" and cau in fd_key:
                    e["days_per_C_trend_free"] = round(float(beta[fd_key[cau]]), 4)
                edges.append(e)
    edges += [{"source": "Urban", "target": t, "kind": "latent",
               "evidence": "JMA 1929-2025: Kyoto Tmin +3.9 vs rural +2.1 °C/100yr; Tmax equal"} for t in ("T_autumn", "T_Feb", "T_Mar")]
    edges += [{"source": "Residual", "target": "Bloom", "kind": "latent",
               "evidence": "about 5 days of the shift not explained once the autumn brake is included"}]
    graph = {"nodes": [{"id": k, "label": v[0], "tier": v[1], "group": v[2]} for k, v in node_meta.items()],
             "edges": edges, "n_years": n_years, "years": years}
    (RESULTS / "causal_graph.json").write_text(json.dumps(graph, indent=2, ensure_ascii=False))


# ----------------------------------------------------------------------------- main
def main(fast: bool = False) -> dict:
    RESULTS.mkdir(exist_ok=True)
    T = clean.build(save=True)
    p = T["panel"].set_index("year")
    m = p.loc[1881:2026]
    w = T["climate_monthly_wide"].set_index("year")

    print("1-3  data, description, JMA validation")
    summary = {"mode": "fast" if fast else "full"} | describe(p, m, w)
    print("4    trend-free screen")
    trend_free_screen(m)
    print("5    causal discovery: assumptions and blind tournament")
    D, names, ix, Z, ZA = discovery_data(m)
    tour, cg_blind = blind_tournament(D, names, ix, Z, ZA, fast)
    print("6    causal discovery with time order")
    lp, P_edge, cg_fz, G_fci, parents = time_ordered_discovery(D, names, ix, Z, cg_blind, fast)
    cd = cdnod_mechanism_change(D, names, Z, fast)
    faith = faithfulness_check(D, names, ix, Z, cg_blind, lp)
    print("7    effect sizes")
    Dm, beta, r2 = effect_sizes(m)
    print("8    attribution")
    att = attribution(p, m, Dm, beta, r2, fast)
    print("9    forecast test")
    ml = forecast_test(m)
    export_graph(names, ix, lp, P_edge, cg_fz, G_fci, beta, len(D), [int(D.index.min()), int(D.index.max())])

    par = parents.set_index("candidate_cause")
    summary |= {
        "n_years_discovery": len(D), "discovery_years": [int(D.index.min()), int(D.index.max())],
        "blind_methods_correct_direction": int(tour.verdict_T_Mar_Bloom.str.contains("T_Mar -> Bloom").sum()),
        "blind_methods_total": len(tour),
        "time_ordered_T_Mar_to_Bloom": {"lingam_bootstrap_p": par.loc["T_Mar", "lingam_bootstrap_p_edge"],
                                        "pc_fisherz": par.loc["T_Mar", "pc_fisherz"], "fci": par.loc["T_Mar", "fci"]},
        "time_ordered_GlobalT_to_Bloom_direct": {"pc_fisherz": par.loc["GlobalT", "pc_fisherz"], "fci": par.loc["GlobalT", "fci"]},
        "cdnod_time_to_bloom_edge": bool(cd.edge.str.contains("C_time").any() and cd[cd.touches_time].edge.str.contains("Bloom").any()),
        "faithfulness": faith,
        "days_per_C_trend_free": {PRETTY[k]: float(v) for k, v in beta.items()},
        "attribution": att,
        "forecast_rmse_days": dict(zip(ml.model, ml.rmse_days)),
        "forecast_test_marches_warmer_than_training": f"{ml.attrs['n_test_marches_warmer_than_training']} of {ml.attrs['n_test']}",
    }

    def clean_json(o):
        if isinstance(o, dict): return {str(k): clean_json(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)): return [clean_json(v) for v in o]
        if isinstance(o, (np.floating, float)): return None if np.isnan(o) else round(float(o), 3)
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, np.bool_): return bool(o)
        return o
    summary = clean_json(summary)
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    a = summary["attribution"]; b = summary["days_per_C_trend_free"]
    print(f"""
Findings ({summary['mode']} run, results/ holds every table)
  Blind discovery found March -> bloom in {summary['blind_methods_correct_direction']} of {summary['blind_methods_total']} methods;
  with time order: LiNGAM bootstrap P = {summary['time_ordered_T_Mar_to_Bloom']['lingam_bootstrap_p']}, PC '{summary['time_ordered_T_Mar_to_Bloom']['pc_fisherz']}', FCI '{summary['time_ordered_T_Mar_to_Bloom']['fci']}'
  Global temperature -> bloom, direct edge: PC '{summary['time_ordered_GlobalT_to_Bloom_direct']['pc_fisherz']}', FCI '{summary['time_ordered_GlobalT_to_Bloom_direct']['fci']}'
  Trend-free effects: March {b['Mar']:+.2f} d/°C, February {b['Feb']:+.2f} d/°C, previous October {b['Oct (prev)']:+.2f} d/°C
  Shift 1881-1910 -> 1997-2026: observed {a['observed_shift_days']:+.1f} d = March {a['march_days']:+.1f} + Feb {a['february_days']:+.1f}
      + autumn brake {a['autumn_brake_days']:+.1f} + January and residual; unexplained {a['residual_days']:+.1f} d
  Kyoto Feb-Mar warmed {a['kyoto_FM_warming_C']:+.2f} °C vs global {a['global_ONDJFM_warming_C']:+.2f} °C;
      urban share of the spring accelerator since 1929: {a['urban_share_of_spring_accelerator_1929_2025']:.0%}
  Forecasting 1997-2026: linear Feb+Mar RMSE {summary['forecast_rmse_days']['Linear, Feb + Mar temp']:.2f} d,
      random forest {summary['forecast_rmse_days']['Random forest, all 10']:.2f} d, gradient boosting {summary['forecast_rmse_days']['Gradient boosting, all 10']:.2f} d
""")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fast", action="store_true", help="skip kernel (KCI) tests and use fewer bootstrap draws")
    main(fast=ap.parse_args().fast)
