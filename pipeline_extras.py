"""
Pipeline Extras — Transfer Comps + Best/Worst Fit Narrative
"""

import pandas as pd
import numpy as np
import io


# ── Transfer Comps from Projections File ──────────────────────────────────────

def parse_transfer_comps(file_bytes, player_name, max_comps=3):
    xl = pd.ExcelFile(io.BytesIO(file_bytes))
    report_sheet = None
    for s in xl.sheet_names:
        if player_name.lower() in s.lower():
            report_sheet = s
            break

    if "Transfer Comps" in xl.sheet_names:
        df_comps = pd.read_excel(xl, sheet_name="Transfer Comps")
    else:
        df_comps = None

    params = {}
    if "Parameters" in xl.sheet_names:
        df_params = pd.read_excel(xl, sheet_name="Parameters")
        for _, row in df_params.iterrows():
            field = str(row.iloc[0]).strip() if pd.notna(row.iloc[0]) else ""
            value = row.iloc[1] if len(row) > 1 else ""
            if field:
                params[field] = value

    origin_league = params.get("Origin League", "")
    target_league = params.get("Target League", "")
    adjustment = params.get("League Adjustment", "")

    # Auto-detect league names from Transfer Comps column prefixes
    if df_comps is not None and not df_comps.empty and (not origin_league or not target_league):
        cols = [str(c) for c in df_comps.columns]
        for c in cols:
            if " Team" in c and c != "Player":
                prefix = c.replace(" Team", "").strip()
                if not origin_league:
                    origin_league = prefix
                elif prefix != origin_league and not target_league:
                    target_league = prefix
    # Fallback from Parameters title row
    if not origin_league or not target_league:
        for key in params:
            if "→" in key:
                parts = key.split("→")
                if len(parts) == 2:
                    if not origin_league:
                        origin_league = parts[0].strip().split()[-1] if parts[0].strip() else "Origin"
                    if not target_league:
                        target_league = parts[1].strip().split()[0] if parts[1].strip() else "Target"
    if not origin_league:
        origin_league = "Origin League"
    if not target_league:
        target_league = "Target League"

    if report_sheet:
        result = _parse_comps_from_report_sheet(
            xl, report_sheet, player_name,
            origin_league, target_league, adjustment, max_comps)
        if result:
            return result

    # Also try parsing projections from the report sheet
    if report_sheet:
        result = _parse_projections_from_report_sheet(
            xl, report_sheet, player_name,
            origin_league, target_league, params)
        if result:
            # Append Transfer Comps below if available
            if df_comps is not None and not df_comps.empty:
                tc = _parse_comps_from_transfer_sheet(
                    xl, df_comps, player_name, origin_league, target_league,
                    adjustment, max_comps)
                if tc:
                    result = result + "\n\n" + tc
            return result

    if df_comps is not None and not df_comps.empty:
        return _parse_comps_from_transfer_sheet(
            xl, df_comps, player_name, origin_league, target_league,
            adjustment, max_comps)

    return ""


def _parse_comps_from_report_sheet(xl, sheet_name, player_name,
                                    origin_league, target_league,
                                    adjustment, max_comps):
    df = pd.read_excel(xl, sheet_name=sheet_name, header=None)
    lines = []
    lines.append("## League Transition Comparison")
    lines.append("")
    lines.append("**{} → {}**".format(origin_league, target_league))
    if adjustment:
        lines.append("**League Adjustment Factor:** {}".format(adjustment))
    lines.append("")

    comp_start = None
    for i, row in df.iterrows():
        val = str(row.iloc[0]) if pd.notna(row.iloc[0]) else ""
        if "TOP TRANSFER COMPS" in val.upper():
            comp_start = i
            break
    if comp_start is None:
        for i, row in df.iterrows():
            val = str(row.iloc[0]) if pd.notna(row.iloc[0]) else ""
            if "Comp #1" in val:
                comp_start = i - 2
                break
    if comp_start is None:
        return ""

    comp_num = 0
    i = comp_start
    while i < len(df) and comp_num < max_comps:
        val = str(df.iloc[i, 0]) if pd.notna(df.iloc[i, 0]) else ""
        if val.startswith("Comp #"):
            comp_num += 1
            lines.append("### {}".format(val))
            i += 1
            if i < len(df):
                transfer_info = str(df.iloc[i, 0]) if pd.notna(df.iloc[i, 0]) else ""
                gp_info = ""
                for col_idx in range(1, min(7, len(df.columns))):
                    cell = df.iloc[i, col_idx] if pd.notna(df.iloc[i, col_idx]) else ""
                    if "GP:" in str(cell) or "MPG:" in str(cell):
                        gp_info = str(cell)
                        break
                lines.append("**{}**".format(transfer_info))
                if gp_info:
                    lines.append("*{}*".format(gp_info))
                lines.append("")
            i += 1
            if i < len(df):
                lines.append("| Stat | {} P40 | {} %ile | {} P40 | {} %ile | Change |".format(
                    origin_league, origin_league, target_league, target_league))
                lines.append("|---|---|---|---|---|---|")
                i += 1
                while i < len(df):
                    stat = str(df.iloc[i, 0]) if pd.notna(df.iloc[i, 0]) else ""
                    if not stat or stat == "nan" or "Tags" in stat:
                        break
                    vals = []
                    for col_idx in range(1, min(7, len(df.columns))):
                        cell = df.iloc[i, col_idx]
                        vals.append(str(cell) if pd.notna(cell) else "—")
                    while len(vals) < 5:
                        vals.append("—")
                    lines.append("| {} | {} | {} | {} | {} | {} |".format(
                        stat, vals[0], vals[1], vals[2], vals[3], vals[4]))
                    i += 1
                lines.append("")
            while i < len(df):
                val = str(df.iloc[i, 0]) if pd.notna(df.iloc[i, 0]) else ""
                if "Tags" in val and pd.notna(df.iloc[i, 1]):
                    lines.append("**{}** {}".format(val, str(df.iloc[i, 1])))
                    i += 1
                elif val and val != "nan" and "Comp #" not in val:
                    i += 1
                else:
                    break
            lines.append("")
            lines.append("---")
            lines.append("")
        else:
            i += 1
    return "\n".join(lines)


def _parse_projections_from_report_sheet(xl, sheet_name, player_name,
                                          origin_league, target_league, params):
    """Parse projections + percentiles from a player report sheet (non-comp format)."""
    df = pd.read_excel(xl, sheet_name=sheet_name, header=None)

    # Extract header context (rows 0-8)
    header_info = {}
    for i in range(min(9, len(df))):
        key = str(df.iloc[i, 0]).rstrip(":") if pd.notna(df.iloc[i, 0]) else ""
        val = str(df.iloc[i, 1]) if df.shape[1] > 1 and pd.notna(df.iloc[i, 1]) else ""
        if key and val:
            header_info[key] = val

    lines = []
    lines.append("## Data Projections ({} → {})".format(origin_league, target_league))
    lines.append("")

    # Projector context line
    transfer_pairs = params.get("Transfer Pairs Used", "")
    mape = params.get("Optimization Error (MAPE)", "")
    if transfer_pairs:
        proj_line = "**Projector:** {} → {}".format(origin_league, target_league)
        proj_line += " | Calibrated on {} direct transfers".format(transfer_pairs)
        if mape:
            if isinstance(mape, float):
                proj_line += " | MAPE: {:.2f}%".format(mape * 100)
            else:
                proj_line += " | MAPE: {}".format(mape)
        lines.append(proj_line)

    # Parameters line
    adjustment = params.get("League Adjustment", "")
    fg_mult = params.get("FG% Multiplier", "")
    three_mult = params.get("3PT% Multiplier", params.get("3P% Multiplier", ""))
    ft_mult = params.get("FT% Multiplier", "")
    param_parts = []
    if adjustment:
        param_parts.append("League Adjustment: {}".format(adjustment))
    if fg_mult:
        param_parts.append("FG Multiplier: {}".format(fg_mult))
    if three_mult:
        param_parts.append("3PT Multiplier: {}".format(three_mult))
    if ft_mult:
        param_parts.append("FT Multiplier: {}".format(ft_mult))
    if param_parts:
        lines.append("**Parameters:** {}".format(" | ".join(param_parts)))
    lines.append("")

    # Player context block
    ctx = []
    for label, key in [("Current Team", "EUR Team"), ("Season", "EUR Season"),
                        ("GP", "EUR Games Played"), ("MPG", "EUR Minutes/Game"),
                        ("Projected Destination", "Projected NCAA Destination")]:
        val = header_info.get(key, "")
        if val:
            ctx.append("**{}:** {}".format(label, val))
    if ctx:
        lines.append(" | ".join(ctx))
        lines.append("")

    # Find key sections
    actual_start = projected_start = pctile_start = summary_start = None
    for i in range(len(df)):
        val = str(df.iloc[i, 0]) if pd.notna(df.iloc[i, 0]) else ""
        vu = val.upper()
        if "ACTUAL STATS" in vu:
            actual_start = i
        elif "PROJECTED STATS" in vu and "MINUTE" in vu:
            projected_start = i
        elif "PERCENTILE" in vu and "RADAR" not in vu:
            pctile_start = i
        elif "SCOUTING SUMMARY" in vu:
            summary_start = i

    # Actual stats
    if actual_start is not None:
        hdr = actual_start + 1
        dat = actual_start + 2
        if dat < len(df):
            title = str(df.iloc[actual_start, 0]) if pd.notna(df.iloc[actual_start, 0]) else ""
            lines.append("### {}".format(title))
            lines.append("")
            headers = [str(df.iloc[hdr, c]) for c in range(df.shape[1]) if pd.notna(df.iloc[hdr, c])]
            vals = [str(df.iloc[dat, c]) for c in range(len(headers)) if pd.notna(df.iloc[dat, c])]
            if headers and vals:
                lines.append("| " + " | ".join(headers) + " |")
                lines.append("|" + "|".join(["---"] * len(headers)) + "|")
                lines.append("| " + " | ".join(vals) + " |")
                lines.append("")

    # Projected stats by minutes
    if projected_start is not None:
        title = str(df.iloc[projected_start, 0]) if pd.notna(df.iloc[projected_start, 0]) else ""
        lines.append("### {}".format(title))
        lines.append("")
        hdr = projected_start + 1
        if hdr < len(df):
            headers = [str(df.iloc[hdr, c]) for c in range(df.shape[1]) if pd.notna(df.iloc[hdr, c])]
            lines.append("| " + " | ".join(headers) + " |")
            lines.append("|" + "|".join(["---"] * len(headers)) + "|")
            r = hdr + 1
            while r < len(df):
                first = str(df.iloc[r, 0]) if pd.notna(df.iloc[r, 0]) else ""
                if not first or first == "nan":
                    break
                vals = []
                for c in range(len(headers)):
                    cell = df.iloc[r, c]
                    vals.append(str(cell) if pd.notna(cell) else "—")
                lines.append("| " + " | ".join(vals) + " |")
                r += 1
            lines.append("")

    # Percentile rankings
    if pctile_start is not None:
        title = str(df.iloc[pctile_start, 0]) if pd.notna(df.iloc[pctile_start, 0]) else ""
        lines.append("### {} Percentiles".format(target_league))
        lines.append("")
        hdr = pctile_start + 1
        if hdr < len(df):
            headers = [str(df.iloc[hdr, c]) for c in range(df.shape[1]) if pd.notna(df.iloc[hdr, c])]
            lines.append("| " + " | ".join(headers) + " |")
            lines.append("|" + "|".join(["---"] * len(headers)) + "|")
            r = hdr + 1
            while r < len(df):
                first = str(df.iloc[r, 0]) if pd.notna(df.iloc[r, 0]) else ""
                if not first or first == "nan":
                    break
                vals = []
                for c in range(len(headers)):
                    cell = df.iloc[r, c]
                    vals.append(str(cell) if pd.notna(cell) else "—")
                lines.append("| " + " | ".join(vals) + " |")
                r += 1
            lines.append("")

    # Scouting summary
    if summary_start is not None:
        lines.append("### Scouting Summary")
        lines.append("")
        r = summary_start + 1
        while r < len(df):
            val = str(df.iloc[r, 0]) if pd.notna(df.iloc[r, 0]) else ""
            if not val or val == "nan":
                break
            lines.append("- {}".format(val))
            r += 1
        lines.append("")

    lines.append("---")
    result = "\n".join(lines)
    if actual_start is not None or projected_start is not None or pctile_start is not None:
        return result
    return ""


def _load_projection_pool(xl, target_mpg=22):
    """Load the projection sheet closest to target_mpg and compute percentile ranks."""
    sheets = {10: "10min", 15: "15min", 22: "22min", 30: "30min"}
    # Pick closest minute sheet
    best = min(sheets.keys(), key=lambda x: abs(x - target_mpg))
    sheet_name = sheets.get(best)
    if sheet_name not in xl.sheet_names:
        return None, None
    df = pd.read_excel(xl, sheet_name=sheet_name, header=None)
    headers = [str(df.iloc[0, c]) for c in range(df.shape[1])]
    df.columns = headers
    df = df.iloc[1:].reset_index(drop=True)
    # Convert numeric
    for col in ["Proj PPG", "Proj RPG", "Proj APG", "Proj SPG", "Proj TOV", "Proj BPG",
                 "EUR GP", "EUR MPG"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # Convert pct columns
    for col in ["FG%", "3P%", "FT%", "TS%", "eFG%"]:
        if col in df.columns:
            df[col + "_num"] = df[col].apply(
                lambda x: float(str(x).replace("%", "")) if pd.notna(x) and str(x) != "nan" else np.nan)
    return df, best


def _compute_pctile(pool_series, value, invert=False):
    """Compute percentile rank of value within pool."""
    valid = pool_series.dropna()
    if len(valid) == 0:
        return None
    if invert:
        rank = (valid > value).sum()
    else:
        rank = (valid < value).sum()
    return int(round(100 * rank / len(valid)))


def _ordinal(n):
    if n is None:
        return "—"
    s = str(n)
    if s.endswith("11") or s.endswith("12") or s.endswith("13"):
        return s + "th"
    last = s[-1]
    if last == "1":
        return s + "st"
    elif last == "2":
        return s + "nd"
    elif last == "3":
        return s + "rd"
    return s + "th"


def _parse_comps_from_transfer_sheet(xl, df_comps, player_name, origin_league,
                                      target_league, adjustment, max_comps):
    cols = [str(c) for c in df_comps.columns]

    # Detect column prefixes
    o_pre, t_pre = "", ""
    for c in cols:
        if " Team" in c and c != "Player":
            prefix = c.replace(" Team", "").strip()
            if not o_pre:
                o_pre = prefix
            elif prefix != o_pre and not t_pre:
                t_pre = prefix
    if not o_pre:
        o_pre = "Origin"
    if not t_pre:
        t_pre = "Target"

    def _col(prefix, suffix):
        exact = "{} {}".format(prefix, suffix)
        if exact in cols:
            return exact
        for c in cols:
            if c.lower() == exact.lower():
                return c
        return None

    # Detect format
    has_per40 = any("P40" in c or "/40" in c for c in cols)
    has_pctile = any("%ile" in c for c in cols)

    lines = []
    lines.append("## Verified Transfer Comparisons")
    lines.append("")
    lines.append("**{} → {}**".format(origin_league, target_league))
    if adjustment:
        lines.append("")
        lines.append("**League Adjustment Factor:** {}".format(adjustment))
    lines.append("")
    lines.append("*{} players found who played in both leagues.*".format(len(df_comps)))
    lines.append("")

    if has_per40 or has_pctile:
        # Detailed per-40 + percentile format (Demarcus-style)
        for idx, row in df_comps.head(max_comps).iterrows():
            name = row.get("Player", "Unknown")
            lines.append("### {}".format(name))
            o_team = row.get(_col(o_pre, "Team") or "Origin Team", "")
            o_season = row.get(_col(o_pre, "Season") or "Origin Season", "")
            t_team = row.get(_col(t_pre, "Team") or "Target Team", "")
            t_season = row.get(_col(t_pre, "Season") or "Target Season", "")
            lines.append("**{} ({}) → {} ({})**".format(o_team, o_season, t_team, t_season))
            o_gp = row.get(_col(o_pre, "GP") or "Origin GP", "?")
            t_gp = row.get(_col(t_pre, "GP") or "Target GP", "?")
            o_mpg = row.get(_col(o_pre, "MPG") or "Origin MPG", "?")
            t_mpg = row.get(_col(t_pre, "MPG") or "Target MPG", "?")
            lines.append("*GP: {}/{} | MPG: {}/{}*".format(o_gp, t_gp, o_mpg, t_mpg))
            lines.append("")
            stat_pairs = [
                ("Points", "PTS"), ("Rebounds", "REB"), ("Assists", "AST"),
                ("Steals", "STL"), ("Blocks", "BLK"), ("Turnovers", "TOV"),
                ("FG%", "FG%"), ("3P%", "3P%"), ("FT%", "FT%"),
            ]
            lines.append("| Stat | {} P40 | {} %ile | {} P40 | {} %ile |".format(
                origin_league, origin_league, target_league, target_league))
            lines.append("|---|---|---|---|---|")
            for stat_name, stat_key in stat_pairs:
                op = row.get("Orig {}/40".format(stat_key), row.get("Orig {}".format(stat_key), "—"))
                oc = row.get("Orig {} %ile".format(stat_key), "—")
                tp = row.get("Tgt {}/40".format(stat_key), row.get("Tgt {}".format(stat_key), "—"))
                tc = row.get("Tgt {} %ile".format(stat_key), "—")
                if pd.isna(op): op = "—"
                if pd.isna(oc): oc = "—"
                if pd.isna(tp): tp = "—"
                if pd.isna(tc): tc = "—"
                lines.append("| {} | {} | {} | {} | {} |".format(stat_name, op, oc, tp, tc))
            lines.append("")
            lines.append("---")
            lines.append("")
    else:
        # Summary format — individual comp profiles with percentile comparison
        # Load projection pool for percentile computation
        pool, pool_mpg = _load_projection_pool(xl, target_mpg=22)
        subject_row = None
        if pool is not None:
            matches = pool[pool["Player"].str.contains(player_name.split()[-1], case=False, na=False)]
            # Narrow to exact match if multiple
            for _, mr in matches.iterrows():
                if player_name.lower() in str(mr.get("Player", "")).lower():
                    subject_row = mr
                    break
            if subject_row is None and not matches.empty:
                subject_row = matches.iloc[0]

        # Compute subject percentiles
        subject_pctiles = {}
        if pool is not None and subject_row is not None:
            stat_map = [
                ("Points", "Proj PPG", False), ("Rebounds", "Proj RPG", False),
                ("Assists", "Proj APG", False), ("Steals", "Proj SPG", False),
                ("Blocks", "Proj BPG", False), ("Turnovers", "Proj TOV", True),
                ("FG%", "FG%_num", False), ("3P%", "3P%_num", False),
                ("FT%", "FT%_num", False), ("TS%", "TS%_num", False),
                ("eFG%", "eFG%_num", False),
            ]
            for label, col, inv in stat_map:
                if col in pool.columns and pd.notna(subject_row.get(col)):
                    subject_pctiles[label] = _compute_pctile(pool[col], float(subject_row[col]), inv)

        # Show each top comp as individual profile
        for comp_idx, (_, row) in enumerate(df_comps.head(max_comps).iterrows()):
            name = row.get("Player", "Unknown")
            if pd.isna(name):
                name = "Unknown"
            o_team = row.get(_col(o_pre, "Team") or "", "")
            o_season = row.get(_col(o_pre, "Season") or "", "")
            t_team = row.get(_col(t_pre, "Team") or "", "")
            t_season = row.get(_col(t_pre, "Season") or "", "")
            o_gp = row.get(_col(o_pre, "GP") or "", "—")
            t_gp = row.get(_col(t_pre, "GP") or "", "—")
            o_mpg = row.get(_col(o_pre, "MPG") or "", "—")
            t_mpg = row.get(_col(t_pre, "MPG") or "", "—")
            o_ppg = row.get(_col(o_pre, "PPG") or "", "—")
            t_ppg = row.get(_col(t_pre, "PPG") or "", "—")
            ppg_chg = row.get("PPG Change", "—")
            mpg_chg = row.get("MPG Change", "—")

            lines.append("### Comp #{}: {}".format(comp_idx + 1, name))
            lines.append("")
            lines.append("**{} ({}) → {} ({})**".format(o_team, o_season, t_team, t_season))
            lines.append("*GP: {}/{} | MPG: {}/{}*".format(o_gp, t_gp, o_mpg, t_mpg))
            lines.append("")

            # Stats comparison table
            lines.append("| Stat | {} ({}) | {} ({}) | Translation |".format(
                name, o_pre, name, t_pre))
            lines.append("|---|---|---|---|")
            lines.append("| **PPG** | {} | {} | {} |".format(o_ppg, t_ppg, _fmt_change(ppg_chg)))
            lines.append("| **MPG** | {} | {} | {} |".format(o_mpg, t_mpg, _fmt_change(mpg_chg)))
            lines.append("| **GP** | {} | {} | — |".format(o_gp, t_gp))
            lines.append("")

            # Percentile profile comparison: subject vs comp
            comp_pctiles = {}
            if pool is not None:
                # First try to find comp in projection pool (current EUR players)
                comp_row = None
                comp_matches = pool[pool["Player"].str.contains(
                    name.split()[-1] if name != "Unknown" else "XXXXXXX",
                    case=False, na=False)]
                for _, cmr in comp_matches.iterrows():
                    if name.lower() in str(cmr.get("Player", "")).lower():
                        comp_row = cmr
                        break
                if comp_row is not None:
                    stat_map = [
                        ("Points", "Proj PPG", False), ("Rebounds", "Proj RPG", False),
                        ("Assists", "Proj APG", False), ("Steals", "Proj SPG", False),
                        ("Blocks", "Proj BPG", False), ("Turnovers", "Proj TOV", True),
                        ("FG%", "FG%_num", False), ("3P%", "3P%_num", False),
                        ("FT%", "FT%_num", False),
                    ]
                    for label, col, inv in stat_map:
                        if col in pool.columns and pd.notna(comp_row.get(col)):
                            comp_pctiles[label] = _compute_pctile(pool[col], float(comp_row[col]), inv)
                else:
                    # Comp not in pool — use actual NCAA stats from Transfer Comps
                    # Rank their actual stats against projected stats in pool
                    ncaa_ppg = row.get(_col(t_pre, "PPG") or "", None)
                    if ncaa_ppg is not None and not (isinstance(ncaa_ppg, float) and pd.isna(ncaa_ppg)):
                        try:
                            comp_pctiles["Points"] = _compute_pctile(
                                pool["Proj PPG"], float(ncaa_ppg), False)
                        except (ValueError, TypeError):
                            pass

            # Show percentile radar comparison — always two columns
            if subject_pctiles:
                lines.append("**Projected {} Percentile Profile — {} vs {}**".format(
                    target_league, player_name, name))
                lines.append("")
                lines.append("| Dimension | {} | {} |".format(player_name, name))
                lines.append("|---|---|---|")
                for dim in ["Points", "Rebounds", "Assists", "Steals", "Blocks",
                            "FG%", "3P%", "FT%"]:
                    sp = _ordinal(subject_pctiles.get(dim))
                    cp = _ordinal(comp_pctiles.get(dim)) if comp_pctiles.get(dim) is not None else "—"
                    lines.append("| {} | {} | {} |".format(dim, sp, cp))
                lines.append("")

            lines.append("---")
            lines.append("")

        # Full comparison table at the end
        lines.append("### All Verified Transfers")
        lines.append("")
        lines.append("| Player | {} Team | {} GP | {} MPG | {} PPG | {} Team | {} GP | {} MPG | {} PPG | PPG Chg | MPG Chg |".format(
            o_pre, o_pre, o_pre, o_pre, t_pre, t_pre, t_pre, t_pre))
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|")

        def _clean(val):
            if val is None or (isinstance(val, float) and pd.isna(val)):
                return "—"
            return val

        for _, row in df_comps.iterrows():
            lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                _clean(row.get("Player", "Unknown")),
                _clean(row.get(_col(o_pre, "Team") or "", "")),
                _clean(row.get(_col(o_pre, "GP") or "", "—")),
                _clean(row.get(_col(o_pre, "MPG") or "", "—")),
                _clean(row.get(_col(o_pre, "PPG") or "", "—")),
                _clean(row.get(_col(t_pre, "Team") or "", "")),
                _clean(row.get(_col(t_pre, "GP") or "", "—")),
                _clean(row.get(_col(t_pre, "MPG") or "", "—")),
                _clean(row.get(_col(t_pre, "PPG") or "", "—")),
                _clean(row.get("PPG Change", "—")),
                _clean(row.get("MPG Change", "—"))))
        lines.append("")
        lines.append("---")
        lines.append("")
    return "\n".join(lines)


def _fmt_change(val):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "—"
    try:
        v = float(val)
        if v > 0:
            return "+{}".format(val)
        return str(val)
    except (ValueError, TypeError):
        return str(val)


# ── Best/Worst Fit Narrative ──────────────────────────────────────────────────

def _extract_partner(duo_str, player_name):
    parts = duo_str.split(" + ")
    if len(parts) != 2:
        return duo_str
    a, b = parts[0].strip(), parts[1].strip()
    pn = player_name.lower().strip()
    if pn in a.lower():
        return b
    if pn in b.lower():
        return a
    pn_last = pn.split()[-1] if pn.split() else pn
    a_last = a.lower().split()[-1] if a.split() else ""
    b_last = b.lower().split()[-1] if b.split() else ""
    if pn_last == a_last:
        return b
    if pn_last == b_last:
        return a
    return b


def _prepare_duo_data(duo_df, player_name):
    import re
    if "Duo" in duo_df.columns:
        pattern = re.escape(player_name)
        player_duos = duo_df[duo_df["Duo"].str.contains(pattern, case=False, na=False)].copy()
        if player_duos.empty:
            last_name = player_name.strip().split()[-1]
            pattern = re.escape(last_name)
            player_duos = duo_df[duo_df["Duo"].str.contains(pattern, case=False, na=False)].copy()
        if not player_duos.empty:
            duo_df = player_duos
    if "Duo" in duo_df.columns:
        duo_df = duo_df.copy()
        duo_df["_partner"] = duo_df["Duo"].apply(lambda d: _extract_partner(d, player_name))
        duo_df = duo_df.drop_duplicates(subset=["_partner"], keep="first")
    sort_col = "Net Rating Diff"
    if sort_col not in duo_df.columns:
        for c in duo_df.columns:
            if "diff" in c.lower() and "net" in c.lower():
                sort_col = c
                break
        else:
            sort_col = duo_df.columns[-1]
    return duo_df.sort_values(sort_col, ascending=False), sort_col


def _build_duo_summary(sorted_duos, sort_col, player_name, n_best=3, n_worst=3):
    rows = []
    best = sorted_duos.head(n_best)
    worst = sorted_duos.tail(n_worst).iloc[::-1]
    for label, subset in [("BEST", best), ("WORST", worst)]:
        for _, row in subset.iterrows():
            partner = _extract_partner(row.get("Duo", ""), player_name)
            info = {
                "partner": partner, "category": label,
                "net_rtg_diff": row.get(sort_col, np.nan),
                "net_rtg_on": row.get("Net Rating (ON)", np.nan),
                "net_rtg_off": row.get("Net Rating (OFF)", np.nan),
                "off_rtg_on": row.get("Off Pts/Poss (ON)", np.nan),
                "off_rtg_off": row.get("Off Pts/Poss (OFF)", np.nan),
                "def_rtg_on": row.get("Def Pts/Poss (ON)", np.nan),
                "def_rtg_off": row.get("Def Pts/Poss (OFF)", np.nan),
                "efg_on": row.get("Off eFG% (ON)", np.nan),
                "efg_off": row.get("Off eFG% (OFF)", np.nan),
                "tov_on": row.get("Off TOV% (ON)", np.nan),
                "tov_off": row.get("Off TOV% (OFF)", np.nan),
                "minutes": row.get("Minutes together", np.nan),
            }
            cleaned = {}
            for k, v in info.items():
                if isinstance(v, float) and pd.notna(v):
                    cleaned[k] = round(v, 2)
                elif isinstance(v, float):
                    cleaned[k] = "N/A"
                else:
                    cleaned[k] = v
            rows.append(cleaned)
    return rows


def generate_fit_narrative(duo_df, p2a_profile, p2b_results, player_name, api_key=None):
    if duo_df is None or duo_df.empty:
        return ""
    sorted_duos, sort_col = _prepare_duo_data(duo_df, player_name)
    duo_summaries = _build_duo_summary(sorted_duos, sort_col, player_name)

    subject_context = ""
    if p2a_profile:
        tags = p2a_profile.get("tags", [])
        cluster = p2a_profile.get("major_cluster", "")
        position = p2a_profile.get("position", "")
        tag_str = ", ".join(tags[:6]) if isinstance(tags, list) else str(tags)
        subject_context = "Subject: {}, Position: {}, Archetype: {}, Tags: {}".format(
            player_name, position, cluster, tag_str)

    narratives = None
    if api_key:
        try:
            narratives = _generate_ai_narratives(
                api_key, player_name, subject_context, duo_summaries)
        except Exception:
            narratives = None

    lines = []
    lines.append("## Lineup Analytics")
    lines.append("")
    lines.append("### Current Teammate Compatibility Analysis")
    lines.append("")

    best_rows = [d for d in duo_summaries if d["category"] == "BEST"]
    lines.append("**Best Fits:**")
    lines.append("")
    lines.append("| Player | Net Rtg Diff | Why |")
    lines.append("|---|---|---|")
    for i, d in enumerate(best_rows):
        diff = d["net_rtg_diff"]
        if isinstance(diff, (int, float)) and diff > 0:
            diff_str = "+{}".format(diff)
        else:
            diff_str = str(diff)
        why = _get_narrative(narratives, "best", i, d)
        lines.append("| {} | {} | {} |".format(d["partner"], diff_str, why))
    lines.append("")

    worst_rows = [d for d in duo_summaries if d["category"] == "WORST"]
    lines.append("**Worst Fits:**")
    lines.append("")
    lines.append("| Player | Net Rtg Diff | Why |")
    lines.append("|---|---|---|")
    for i, d in enumerate(worst_rows):
        diff_str = str(d["net_rtg_diff"])
        why = _get_narrative(narratives, "worst", i, d)
        lines.append("| {} | {} | {} |".format(d["partner"], diff_str, why))
    lines.append("")

    if p2b_results:
        future_narratives = None
        if api_key:
            try:
                future_narratives = _generate_future_fit_narratives(
                    api_key, player_name, subject_context, p2b_results, p2a_profile)
            except Exception:
                future_narratives = None

        lines.append("### Future Teammate Compatibility Analysis")
        lines.append("")
        lines.append("| Player | Role | Compatibility | Why it could work |")
        lines.append("|---|---|---|---|")
        for idx, r in enumerate(p2b_results[:5]):
            r_name = r.get("name", "Unknown")
            r_cluster = r.get("major_cluster", "")
            r_score = r.get("similarity_score", 0)
            if r_score >= 0.6: compat = "Excellent"
            elif r_score >= 0.45: compat = "High"
            elif r_score >= 0.35: compat = "Good"
            elif r_score >= 0.25: compat = "Moderate"
            else: compat = "Low"
            if future_narratives and "future_{}".format(idx) in future_narratives:
                why = future_narratives["future_{}".format(idx)]
            else:
                why = "Complementary skill set as {}".format(r_cluster) if r_cluster else "Complementary skill set"
            lines.append("| {} | {} | {} | {} |".format(r_name, r_cluster, compat, why))
        lines.append("")

    lines.append("---")
    return "\n".join(lines)


def _get_narrative(narratives, category, index, fallback_data):
    if narratives:
        key = "{}_{}".format(category, index)
        if key in narratives:
            return narratives[key]
    verb = "improves" if category == "best" else "drops"
    diff = fallback_data.get("net_rtg_diff", "N/A")
    return "Net rating {} by {} when paired".format(verb, diff)


def _generate_ai_narratives(api_key, player_name, subject_context, duo_summaries):
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    duo_lines = []
    for d in duo_summaries:
        line = "- {cat} FIT: {p} | Net Rtg Diff: {nrd} | ORtg ON: {oro} / OFF: {orf} | DRtg ON: {dro} / OFF: {drf} | eFG ON: {eo} / OFF: {ef} | TOV ON: {to} / OFF: {tf} | Min: {m}".format(
            cat=d["category"], p=d["partner"], nrd=d["net_rtg_diff"],
            oro=d["off_rtg_on"], orf=d["off_rtg_off"],
            dro=d["def_rtg_on"], drf=d["def_rtg_off"],
            eo=d["efg_on"], ef=d["efg_off"],
            to=d["tov_on"], tf=d["tov_off"],
            m=d["minutes"])
        duo_lines.append(line)
    data_block = "\n".join(duo_lines)
    prompt = "You are a professional basketball scout writing teammate compatibility analysis.\n\n"
    prompt += subject_context + "\n\n"
    prompt += "Here are the duo stats for " + player_name + "'s best and worst teammate pairings:\n\n"
    prompt += data_block + "\n\n"
    prompt += "For each pairing, write a short Why explanation (1-2 sentences max, ~15-25 words) that:\n"
    prompt += "- Uses basketball terminology (spacing, pick-and-roll chemistry, defensive switching, rim protection, ball movement, pace, etc.)\n"
    prompt += "- Explains WHY the pairing works or doesn't from an on-court dynamics perspective\n"
    prompt += "- For best fits: explain the basketball reason the duo is effective\n"
    prompt += "- For worst fits: explain the basketball reason the duo struggles\n\n"
    prompt += "CRITICAL: Do NOT restate numbers. Translate data into basketball insights.\n\n"
    prompt += "Return EXACTLY in this format (one line per duo):\n\n"
    prompt += "BEST_0: [why]\nBEST_1: [why]\nBEST_2: [why]\nWORST_0: [why]\nWORST_1: [why]\nWORST_2: [why]"
    response = client.messages.create(
        model="claude-sonnet-4-20250514", max_tokens=800,
        messages=[{"role": "user", "content": prompt}])
    raw = response.content[0].text
    narratives = {}
    prefixes = ["BEST_0:", "BEST_1:", "BEST_2:", "WORST_0:", "WORST_1:", "WORST_2:"]
    for line in raw.strip().split("\n"):
        line = line.strip()
        if not line: continue
        for prefix in prefixes:
            if line.startswith(prefix):
                narratives[prefix.replace(":", "").lower()] = line[len(prefix):].strip()
                break
    return narratives if narratives else None


def _generate_future_fit_narratives(api_key, player_name, subject_context, p2b_results, p2a_profile):
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    player_lines = []
    for idx, r in enumerate(p2b_results[:5]):
        tags = r.get("tags", [])
        tag_str = ", ".join(tags[:4]) if isinstance(tags, list) else str(tags)
        line = "- FUTURE_{}: {} | Role: {} | Position: {} | Tags: {}".format(
            idx, r.get("name", "Unknown"), r.get("major_cluster", ""),
            r.get("position", ""), tag_str)
        player_lines.append(line)
    subject_tags = ""
    if p2a_profile and isinstance(p2a_profile.get("tags"), list):
        subject_tags = ", ".join(p2a_profile["tags"][:6])
    data_block = "\n".join(player_lines)
    prompt = "You are a professional basketball scout analyzing future teammate compatibility.\n\n"
    prompt += subject_context + "\n"
    prompt += "Subject tags: " + subject_tags + "\n\n"
    prompt += "These are players on the TARGET TEAM that " + player_name + " would play alongside:\n\n"
    prompt += data_block + "\n\n"
    prompt += "For each player, write a short explanation (1-2 sentences, ~15-25 words) of why they would fit well together on court.\n"
    prompt += "Use basketball terminology: spacing, ball movement, defensive versatility, transition play, pick-and-roll, etc.\n"
    prompt += "Focus on how their skills COMPLEMENT each other, not how similar they are.\n\n"
    prompt += "Return EXACTLY in this format:\n\n"
    prompt += "FUTURE_0: [why]\nFUTURE_1: [why]\nFUTURE_2: [why]\nFUTURE_3: [why]\nFUTURE_4: [why]"
    response = client.messages.create(
        model="claude-sonnet-4-20250514", max_tokens=600,
        messages=[{"role": "user", "content": prompt}])
    raw = response.content[0].text
    narratives = {}
    for line in raw.strip().split("\n"):
        line = line.strip()
        if not line: continue
        for i in range(5):
            prefix = "FUTURE_{}:".format(i)
            if line.startswith(prefix):
                narratives["future_{}".format(i)] = line[len(prefix):].strip()
                break
    return narratives if narratives else None

