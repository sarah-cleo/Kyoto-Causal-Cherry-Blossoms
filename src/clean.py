"""
clean.py — turn the raw JMA workbook + Aono bloom record into tidy, analysis-ready tables.

JMA quality codes found in the workbook (Kyoto, block 47759):
    plain number   reliable value
    "12.3 )"       quasi-reliable (a few missing days)      -> keep, flag = "quasi"
    "12.3 ]"       insufficient data (>20% missing)          -> drop, flag = "insufficient"
    "×"            missing                                   -> NaN,  flag = "missing"
    "--"           phenomenon absent (e.g. no snowfall)      -> 0,    flag = "none_observed"
    "　"       blank (before the record starts / future) -> NaN,  flag = "blank"
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
SHEETS = {  # sheet name in workbook -> (variable id, unit, how months combine into seasons)
    "air_temp": ("temp", "°C", "mean"),
    "precipitation ": ("precip", "mm", "sum"),
    "sunshine_duration": ("sun", "h", "sum"),
    "wind": ("wind", "m/s", "mean"),
    "snowfall_depth": ("snow", "cm", "sum"),
}
NUM = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*([)\]])?\s*$")


def parse_cell(v) -> tuple[float, str]:
    """Return (value, flag) for one JMA cell."""
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v), "ok"
    if v is None:
        return np.nan, "blank"
    s = str(v).strip().replace("　", "")
    if s == "":
        return np.nan, "blank"
    if s == "--":
        return 0.0, "none_observed"
    if s in ("×", "x", "X"):
        return np.nan, "missing"
    m = NUM.match(s)
    if m:
        val, mark = float(m.group(1)), m.group(2)
        if mark == ")":
            return val, "quasi"
        if mark == "]":
            return np.nan, "insufficient"
        return val, "ok"
    return np.nan, f"unparsed:{s}"


def load_jma(path: Path = RAW / "Kyoto_Climate_Variables.xlsx") -> pd.DataFrame:
    """Long table: year, month (1-12), var, unit, value, flag."""
    rows = []
    for sheet, (var, unit, _) in SHEETS.items():
        df = pd.read_excel(path, sheet_name=sheet, dtype=object)
        for _, r in df.iterrows():
            year = int(r["Year"])
            for mi, mname in enumerate(MONTHS, start=1):
                val, flag = parse_cell(r[mname])
                rows.append((year, mi, var, unit, val, flag))
    long = pd.DataFrame(rows, columns=["year", "month", "var", "unit", "value", "flag"])
    bad = long[long.flag.str.startswith("unparsed")]
    assert bad.empty, f"Unparsed JMA cells:\n{bad}"
    return long


def load_bloom(path: Path = RAW / "kyoto_bloom_aono.csv") -> pd.DataFrame:
    b = pd.read_csv(path)
    b = b.rename(columns={"bloom_doy": "doy"})[["year", "bloom_date", "doy"]]
    b["bloom_date"] = pd.to_datetime(b["bloom_date"], errors="coerce", format="mixed")
    return b.sort_values("year").reset_index(drop=True)


def load_global(path: Path = RAW / "global_temp_annual.csv", source: str = "GCAG") -> pd.DataFrame:
    g = pd.read_csv(path)
    g = g[g.Source == source].rename(columns={"Year": "year", "Mean": "gmst"})
    g["year"] = g["year"].astype(int)
    return g[["year", "gmst"]].sort_values("year").reset_index(drop=True)


def load_global_season(path: Path = RAW / "global_temp_monthly.csv", source: str = "GCAG") -> pd.DataFrame:
    """Global mean surface temperature anomaly averaged over the dormancy season Oct(Y-1)–Mar(Y)."""
    g = pd.read_csv(path)
    g = g[g.Source == source].copy()
    g["date"] = pd.to_datetime(g["Year"], format="%Y-%m")
    g["bloom_year"] = g.date.dt.year + (g.date.dt.month >= 10).astype(int)
    g = g[g.date.dt.month.isin([10, 11, 12, 1, 2, 3])]
    s = g.groupby("bloom_year")["Mean"].agg(["mean", "size"])
    s = s[s["size"] == 6]["mean"].rename("gmst_ONDJFM")
    s.index.name = "year"
    return s.reset_index()


def wide_monthly(long: pd.DataFrame) -> pd.DataFrame:
    """One row per calendar year, columns like temp_03 (March temperature)."""
    w = long.pivot_table(index="year", columns=["var", "month"], values="value", aggfunc="first")
    w.columns = [f"{v}_{m:02d}" for v, m in w.columns]
    return w.sort_index()


def bloom_year_features(w: pd.DataFrame) -> pd.DataFrame:
    """
    Re-index climate onto the *bloom year* Y. Dormancy starts the previous autumn, so
    Oct–Dec of Y-1 belong to bloom year Y.  Naming: temp_m10p = October of the previous year.
    Seasonal aggregates are only computed when every month in the window is present.
    """
    prev = w.shift(1, fill_value=np.nan)  # row Y now holds year Y-1 values
    prev.index = w.index
    out = pd.DataFrame(index=w.index)
    for var in ["temp", "precip", "sun", "wind", "snow"]:
        for m in (10, 11, 12):
            c = f"{var}_{m:02d}"
            if c in prev:
                out[f"{var}_m{m}p"] = prev[c]
        for m in range(1, 5):
            c = f"{var}_{m:02d}"
            if c in w:
                out[f"{var}_m{m}"] = w[c]
    agg = {"temp": "mean", "precip": "sum", "sun": "sum", "wind": "mean", "snow": "sum"}

    def season(var, cols):
        cols = [c for c in cols if c in out]
        block = out[cols]
        f = block.mean(axis=1) if agg[var] == "mean" else block.sum(axis=1, min_count=len(cols))
        return f.where(block.notna().all(axis=1))

    for var in agg:
        out[f"{var}_OND"] = season(var, [f"{var}_m10p", f"{var}_m11p", f"{var}_m12p"])   # chilling / dormancy
        out[f"{var}_DJF"] = season(var, [f"{var}_m12p", f"{var}_m1", f"{var}_m2"])       # winter
        out[f"{var}_FM"] = season(var, [f"{var}_m2", f"{var}_m3"])                       # forcing window
    out["temp_annual"] = w[[f"temp_{m:02d}" for m in range(1, 13)]].mean(axis=1).where(
        w[[f"temp_{m:02d}" for m in range(1, 13)]].notna().all(axis=1))
    return out


def build(save: bool = True) -> dict[str, pd.DataFrame]:
    long = load_jma()
    w = wide_monthly(long)
    feats = bloom_year_features(w)
    bloom = load_bloom()
    gmst = load_global()
    gseason = load_global_season()
    panel = (bloom.set_index("year")[["doy", "bloom_date"]]
             .join(feats, how="outer")
             .join(gmst.set_index("year"), how="left")
             .join(gseason.set_index("year"), how="left"))
    panel.index.name = "year"
    panel = panel.loc[panel.index >= 812]
    panel["station_era"] = np.where(panel.index <= 1913, "Imperial Palace grounds (1880–1913)",
                                    "Nishinokyo, Nakagyo-ku (1914–)")
    tables = {"climate_long": long, "climate_monthly_wide": w.reset_index(),
              "bloom": bloom, "gmst": gmst, "panel": panel.reset_index()}
    if save:
        OUT.mkdir(parents=True, exist_ok=True)
        for k, v in tables.items():
            v.to_csv(OUT / f"{k}.csv", index=False)
    return tables


if __name__ == "__main__":
    t = build()
    long = t["climate_long"]
    print(long.groupby(["var", "flag"]).size().unstack(fill_value=0))
    p = t["panel"]
    m = p[(p.year >= 1881)]
    print("bloom+FM temp overlap years:", m.dropna(subset=["doy", "temp_FM"]).shape[0])
