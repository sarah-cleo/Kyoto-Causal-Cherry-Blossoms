# Data

Everything in `raw/` is redistributed unchanged with its source and terms below. `processed/` is rebuilt from `raw/` by `src/clean.py`, which `src/causal_analysis.py` calls first.

## Sources and terms

| File | What | Years | Source | Terms |
|---|---|---|---|---|
| `raw/kyoto_bloom_aono.csv` | Full-flowering date of *Prunus jamasakura*, Arashiyama, Kyoto (day of year) | 812–2026, sparse before 1880 | Compiled by Yasuyuki Aono (Aono & Kazui 2008; Aono & Saito 2010), via the [GMU peak-bloom-prediction repository](https://github.com/GMU-CherryBlossomCompetition/peak-bloom-prediction) (`data/kyoto.csv`) | © Yasuyuki Aono. Data obtained from http://atmenv.envi.osakafu-u.ac.jp/aono/kyophenotemp4/. Shared under the GMU repository's CC BY-NC-SA 4.0 terms. |
| `raw/Kyoto_Climate_Variables.xlsx` | Monthly air temperature, precipitation, sunshine duration, wind speed and snowfall, Kyoto Local Meteorological Observatory (block 47759) | 1880–2026 (sunshine 1889+, wind 1940+, snow 1953+) | Source: Japan Meteorological Agency website ([Kyoto monthly values](https://www.data.jma.go.jp/stats/etrn/view/monthly_s3_en.php?block_no=47759&view=12)), arranged into one workbook | [JMA terms of use](https://www.jma.go.jp/jma/en/copyright.html): Public Data License 1.0, compatible with CC BY 4.0. Values are unchanged; quality flags are interpreted in `clean.py`. |
| `raw/global_temp_monthly.csv`, `raw/global_temp_annual.csv` | Global land and ocean temperature anomaly (NOAA GCAG and NASA GISTEMP; the analysis uses GCAG) | 1850–2026 | [datasets/global-temp](https://datahub.io/core/global-temp) | ODC-PDDL-1.0 (public domain) |

Cited constants, not files: Kyoto and 15-station rural warming trends for 1929–2025 come from JMA's *Heat Island Monitoring Report*. They are hard-coded in `src/causal_analysis.py` (`JMA_KYOTO`, `JMA_RURAL15`), and the script checks its own Kyoto trends against them. All twelve months match at JMA's published precision of 0.1 °C per century.

## Cleaning (`src/clean.py`)

**JMA quality codes**

| Code | Meaning | Treatment |
|---|---|---|
| plain number | reliable | kept |
| `)` | quasi-reliable (a few missing days) | kept, flagged |
| `]` | insufficient data (more than 20% missing) | dropped |
| `×` | missing | dropped |
| `--` | phenomenon absent, such as a month with no snow | set to 0 |

**Bloom year.** Dormancy starts the previous autumn, so October to December of year *Y−1* are attached to bloom year *Y*. Those columns end in `p`; for example, `temp_m10p` is the previous October.

**Seasons.** `OND` (Oct–Dec), `DJF` (Dec–Feb) and `FM` (Feb–Mar) are computed only when every month in the window is present.

**Station history.** The observatory moved in December 1913. `station_era` marks this, and the analysis tests its effect.

## Main table: `processed/panel.csv`

One row per year from 812 to 2026.

| Column | Meaning |
|---|---|
| `doy` | Full-bloom day of year |
| `temp_m1` … `temp_m4` | Mean temperature for January to April, °C |
| `temp_m10p`, `temp_m11p`, `temp_m12p` | Mean temperature for the previous October, November and December, °C |
| `temp_FM`, `temp_OND`, `temp_DJF` | Seasonal means, °C |
| `precip_*`, `sun_*`, `wind_*`, `snow_*` | The same layout for precipitation (mm), sunshine (h), wind (m/s) and snowfall (cm) |
| `gmst_ONDJFM` | Global temperature anomaly averaged over October to March (GCAG, °C) |
| `station_era` | Observatory location |
