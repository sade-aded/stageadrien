"""
Pipeline 1 -- On/Off Four Factors + Duo Synergy Analysis
Ported from playerlynk_recon_oncourt.py and one_player_duo_onoff.py
"""

import pandas as pd
import numpy as np
import re
from collections import defaultdict
from itertools import combinations

# ── Helpers ──────────────────────────────────────────────────────────────────

SUFFIXES = {"Jr.", "Jr", "Sr.", "Sr", "II", "III", "IV", "V"}


def minutes_to_float(min_str):
    if pd.isna(min_str):
        return np.nan
    s = str(min_str).strip()
    if ":" in s:
        parts = s.split(":")
        try:
            if len(parts) == 2:
                mm, ss = map(int, parts)
                return round(mm + ss / 60.0, 2)
            elif len(parts) == 3:
                hh, mm, ss = map(int, parts)
                return round(hh * 60.0 + mm + ss / 60.0, 2)
        except:
            return np.nan
    s = s.replace(",", ".")
    try:
        return float(s)
    except:
        return np.nan


def extract_names(lineup_str):
    if lineup_str is None or (isinstance(lineup_str, float) and np.isnan(lineup_str)):
        return []
    s = str(lineup_str).strip()
    if s in {"-", "", "nan"}:
        return []
    for suf in SUFFIXES:
        s = s.replace(f", {suf}", f" {suf}")
    parts = [p.strip() for p in s.split(",") if p.strip()]
    names = []
    for p in parts:
        m = re.match(r"^\d+\s+(.*)$", p)
        if m:
            names.append(m.group(1).strip())
        else:
            p2 = re.sub(r"^\d+\s*", "", p).strip()
            p2 = re.sub(r"\s+", " ", p2)
            if p2 in SUFFIXES and names:
                names[-1] = f"{names[-1]} {p2}".strip()
            elif p2:
                names.append(p2)
    return names


def clean_df(df):
    df.columns = df.columns.str.strip()
    cols_to_clean = [c for c in df.columns if c not in ["Lineup", "Unnamed: 1", "Minutes"]]
    for col in cols_to_clean:
        df[col] = df[col].astype(str).str.replace('%', '', regex=False).str.replace(',', '.', regex=False)
        df[col] = pd.to_numeric(df[col], errors='coerce')
    if "Minutes" in df.columns:
        df["Minutes"] = df["Minutes"].apply(minutes_to_float)
    return df


# ── Individual On/Off ────────────────────────────────────────────────────────

def _safe(val):
    return val if pd.notna(val) else 0


def _accumulate_stats(stats_dict, player, row):
    """Accumulate raw stats for a player from a stint row."""
    fields = [
        "Minutes", "Poss_team", "Poss_opp", "Points_team", "Points_opp",
        "ORB_team", "ORB_opp", "TOV_team", "TOV_opp", "STL_team", "STL_opp",
        "FTA_team", "FTA_opp", "FGA_team", "FGA_opp", "FGM_team", "FGM_opp",
        "3PM_team", "3PM_opp", "2PA_team", "2PA_opp", "2PM_team", "2PM_opp",
        "3PA_team", "3PA_opp",
    ]
    for f in fields:
        key = f + "_total" if f != "Minutes" else "Minutes_total"
        if pd.notna(row.get(f)):
            stats_dict[player][key] += row[f]


def _compute_four_factors(df_stats):
    """Compute per-100 possession metrics on accumulated stats."""
    df = df_stats.copy()
    df["OFFENSE: Pts/Poss"] = df["Points_team_total"] / df["Poss_team_total"] * 100
    df["DEFENSE: Pts/Poss"] = df["Points_opp_total"] / df["Poss_opp_total"] * 100

    df["OFFENSE: eFG%"] = (df["FGM_team_total"] + df["3PM_team_total"] * 0.5) / df["FGA_team_total"] * 100
    df["DEFENSE: eFG%"] = (df["FGM_opp_total"] + df["3PM_opp_total"] * 0.5) / df["FGA_opp_total"] * 100

    df["OFFENSE: TOV%"] = df["TOV_team_total"] / df["Poss_team_total"] * 100
    df["DEFENSE: TOV%"] = df["TOV_opp_total"] / df["Poss_opp_total"] * 100

    df["OFFENSE: ORB%"] = df["ORB_team_total"] / (df["FGA_team_total"] - df["FGM_team_total"]) * 100
    df["DEFENSE: ORB%"] = df["ORB_opp_total"] / (df["FGA_opp_total"] - df["FGM_opp_total"]) * 100

    df["OFFENSE: FT Rate"] = df["FTA_team_total"] / df["FGA_team_total"] * 100
    df["DEFENSE: FT Rate"] = df["FTA_opp_total"] / df["FGA_opp_total"] * 100

    df["OFFENSE: 3P%"] = df["3PM_team_total"] / df["3PA_team_total"] * 100
    df["OFFENSE: 2P%"] = df["2PM_team_total"] / df["2PA_team_total"] * 100
    df["DEFENSE: 3P%"] = df["3PM_opp_total"] / df["3PA_opp_total"] * 100
    df["DEFENSE: 2P%"] = df["2PM_opp_total"] / df["2PA_opp_total"] * 100

    df["OFFENSE: 3PA FREQ%"] = df["3PA_team_total"] / df["FGA_team_total"] * 100
    df["OFFENSE: 2PA FREQ%"] = df["2PA_team_total"] / df["FGA_team_total"] * 100
    df["DEFENSE: 3PA FREQ%"] = df["3PA_opp_total"] / df["FGA_opp_total"] * 100
    df["DEFENSE: 2PA FREQ%"] = df["2PA_opp_total"] / df["FGA_opp_total"] * 100

    return df.round(2)


def _make_default_stats():
    return {
        "Poss_team_total": 0, "Poss_opp_total": 0, "Minutes_total": 0,
        "Points_team_total": 0, "Points_opp_total": 0,
        "ORB_team_total": 0, "ORB_opp_total": 0,
        "TOV_team_total": 0, "TOV_opp_total": 0,
        "STL_team_total": 0, "STL_opp_total": 0,
        "FTA_team_total": 0, "FTA_opp_total": 0,
        "FGA_team_total": 0, "FGA_opp_total": 0,
        "FGM_team_total": 0, "FGM_opp_total": 0,
        "3PM_team_total": 0, "3PM_opp_total": 0,
        "2PA_team_total": 0, "2PA_opp_total": 0,
        "2PM_team_total": 0, "2PM_opp_total": 0,
        "3PA_team_total": 0, "3PA_opp_total": 0,
    }


def parse_lineups(df):
    """Merge every two rows (team + OPP) into stint records."""
    merged = []
    for i in range(0, len(df) - 1, 2):
        team = df.iloc[i]
        opp = df.iloc[i + 1]
        players = extract_names(team.get("Lineup"))
        if not players:
            continue
        row = {
            "Players": players,
            "Minutes": team.get("Minutes"),
            "Poss_team": team.get("Possessions"), "Poss_opp": opp.get("Possessions"),
            "ORB_team": team.get("Offensive rebounds"), "ORB_opp": opp.get("Offensive rebounds"),
            "TOV_team": team.get("Turnovers"), "TOV_opp": opp.get("Turnovers"),
            "STL_team": team.get("Steals"), "STL_opp": opp.get("Steals"),
            "Points_team": team.get("Points"), "Points_opp": opp.get("Points"),
            "FTA_team": team.get("Free throws attempted"), "FTA_opp": opp.get("Free throws attempted"),
            "FGA_team": team.get("Field goals attempted"), "FGA_opp": opp.get("Field goals attempted"),
            "FGM_team": team.get("Field goals made"), "FGM_opp": opp.get("Field goals made"),
            "3PM_team": team.get("3-pt field goals made"), "3PM_opp": opp.get("3-pt field goals made"),
            "2PA_team": team.get("2-pt field goals attempted"), "2PA_opp": opp.get("2-pt field goals attempted"),
            "2PM_team": team.get("2-pt field goals made"), "2PM_opp": opp.get("2-pt field goals made"),
            "3PA_team": team.get("3-pt field goals attempted"), "3PA_opp": opp.get("3-pt field goals attempted"),
        }
        merged.append(row)
    return pd.DataFrame(merged)


def compute_individual_onoff(df_lineups, min_minutes=100):
    """Compute individual on/off four factors for all players."""
    # ON stats
    player_on = defaultdict(_make_default_stats)
    for _, row in df_lineups.iterrows():
        for player in row["Players"]:
            _accumulate_stats(player_on, player, row)

    df_on = pd.DataFrame.from_dict(dict(player_on), orient='index')
    df_on.index.name = "Player"
    df_on = _compute_four_factors(df_on)

    # OFF stats
    all_players = df_on.index.tolist()
    player_off = defaultdict(_make_default_stats)
    for _, row in df_lineups.iterrows():
        lineup_set = set(row["Players"])
        for player in all_players:
            if player not in lineup_set:
                _accumulate_stats(player_off, player, row)

    df_off = pd.DataFrame.from_dict(dict(player_off), orient='index')
    df_off.index.name = "Player"
    df_off = _compute_four_factors(df_off)

    # Join ON/OFF
    df_onoff = df_on.join(df_off, lsuffix='_ON', rsuffix='_OFF', how='inner')

    # Compute deltas
    metric_cols = [
        "OFFENSE: Pts/Poss", "DEFENSE: Pts/Poss",
        "OFFENSE: eFG%", "DEFENSE: eFG%",
        "OFFENSE: TOV%", "DEFENSE: TOV%",
        "OFFENSE: ORB%", "DEFENSE: ORB%",
        "OFFENSE: FT Rate", "DEFENSE: FT Rate",
        "OFFENSE: 3P%", "DEFENSE: 3P%",
        "OFFENSE: 2P%", "DEFENSE: 2P%",
        "OFFENSE: 3PA FREQ%", "DEFENSE: 3PA FREQ%",
        "OFFENSE: 2PA FREQ%", "DEFENSE: 2PA FREQ%",
    ]
    for col in metric_cols:
        df_onoff[col] = df_onoff[f"{col}_ON"] - df_onoff[f"{col}_OFF"]

    df_onoff["Diff"] = df_onoff["OFFENSE: Pts/Poss"] - df_onoff["DEFENSE: Pts/Poss"]
    df_onoff["OFF POSS ADD"] = df_onoff["OFFENSE: ORB%"] - df_onoff["OFFENSE: TOV%"]
    df_onoff["DEF POSS ADD"] = df_onoff["DEFENSE: TOV%"] - df_onoff["DEFENSE: ORB%"]
    df_onoff["TOT POSS ADD"] = df_onoff["OFF POSS ADD"] + df_onoff["DEF POSS ADD"]

    df_onoff = df_onoff.round(2)

    # Filter by minutes
    df_onoff = df_onoff[df_onoff["Minutes_total_ON"] >= min_minutes]

    return df_onoff.reset_index()


# ── Duo Synergy ──────────────────────────────────────────────────────────────

def _detect_team_tag(df):
    """Auto-detect the team tag from the Unnamed: 1 column."""
    if "Unnamed: 1" not in df.columns:
        return None
    tags = df["Unnamed: 1"].dropna().astype(str).str.strip().str.upper()
    tags = tags[tags != ""]
    # The team tag is whichever value isn't "OPP"
    non_opp = tags[tags != "OPP"].unique()
    if len(non_opp) == 1:
        return non_opp[0]
    elif len(non_opp) > 1:
        # Multiple tags — pick most frequent
        from collections import Counter
        c = Counter(tags[tags != "OPP"])
        return c.most_common(1)[0][0]
    return None


def _compute_stint_rates(row):
    """Compute per-stint rate stats from raw parsed lineup data."""
    pts_t = row.get("Points_team", np.nan)
    pts_o = row.get("Points_opp", np.nan)
    poss_t = row.get("Poss_team", np.nan)
    poss_o = row.get("Poss_opp", np.nan)
    fga_t = row.get("FGA_team", np.nan)
    fgm_t = row.get("FGM_team", np.nan)
    fg3m_t = row.get("3PM_team", np.nan)
    fta_t = row.get("FTA_team", np.nan)
    orb_t = row.get("ORB_team", np.nan)
    tov_t = row.get("TOV_team", np.nan)
    fga_o = row.get("FGA_opp", np.nan)
    fgm_o = row.get("FGM_opp", np.nan)
    fg3m_o = row.get("3PM_opp", np.nan)
    fta_o = row.get("FTA_opp", np.nan)
    orb_o = row.get("ORB_opp", np.nan)
    tov_o = row.get("TOV_opp", np.nan)

    # Defensive rebounds needed for ORB% = ORB / (ORB + opp DRB)
    # We approximate DRB from FGA-FGM-ORB when not available
    drb_o = row.get("DRB_opp", np.nan)
    drb_t = row.get("DRB_team", np.nan)

    def safe_div(a, b):
        if pd.notna(a) and pd.notna(b) and b > 0:
            return a / b
        return np.nan

    off_ppp = safe_div(pts_t, poss_t) * 100 if pd.notna(safe_div(pts_t, poss_t)) else np.nan
    off_efg = safe_div(fgm_t + 0.5 * fg3m_t, fga_t) * 100 if (pd.notna(fgm_t) and pd.notna(fg3m_t) and pd.notna(safe_div(fgm_t + 0.5 * fg3m_t, fga_t))) else np.nan
    off_tov_rate = safe_div(tov_t, poss_t) * 100 if pd.notna(safe_div(tov_t, poss_t)) else np.nan
    off_ftr = safe_div(fta_t, fga_t) * 100 if pd.notna(safe_div(fta_t, fga_t)) else np.nan

    def_ppp = safe_div(pts_o, poss_o) * 100 if pd.notna(safe_div(pts_o, poss_o)) else np.nan
    def_efg = safe_div(fgm_o + 0.5 * fg3m_o, fga_o) * 100 if (pd.notna(fgm_o) and pd.notna(fg3m_o) and pd.notna(safe_div(fgm_o + 0.5 * fg3m_o, fga_o))) else np.nan
    def_tov_rate = safe_div(tov_o, poss_o) * 100 if pd.notna(safe_div(tov_o, poss_o)) else np.nan
    def_ftr = safe_div(fta_o, fga_o) * 100 if pd.notna(safe_div(fta_o, fga_o)) else np.nan

    # ORB%: team ORB / (team ORB + opp DRB) for offense
    # We don't have DRB directly from parse_lineups; approximate from missed shots
    if pd.notna(fga_o) and pd.notna(fgm_o) and pd.notna(orb_o):
        opp_missed = fga_o - fgm_o
        raw = safe_div(orb_t, orb_t + (opp_missed - orb_o)) if pd.notna(orb_t) else np.nan
        off_orb = raw * 100 if pd.notna(raw) else np.nan
    else:
        off_orb = np.nan

    if pd.notna(fga_t) and pd.notna(fgm_t) and pd.notna(orb_t):
        team_missed = fga_t - fgm_t
        raw = safe_div(orb_o, orb_o + (team_missed - orb_t)) if pd.notna(orb_o) else np.nan
        def_orb = raw * 100 if pd.notna(raw) else np.nan
    else:
        def_orb = np.nan

    return {
        "Off Pts/Poss": off_ppp, "Off eFG%": off_efg, "Off TOV%": off_tov_rate,
        "Off ORB%": off_orb, "Off FT Rate": off_ftr,
        "Def Pts/Poss": def_ppp, "Def eFG%": def_efg, "Def TOV%": def_tov_rate,
        "Def ORB%": def_orb, "Def FT Rate": def_ftr,
    }


def build_stints(df, team_tag):
    """Build stint records from a single sheet, handling team/OPP pairs.
    Auto-detects team tag from data if the provided tag doesn't match."""
    if df.empty or "Lineup" not in df.columns:
        return pd.DataFrame()

    has_tag_col = "Unnamed: 1" in df.columns

    if has_tag_col:
        # Auto-detect the actual tag in the file
        detected_tag = _detect_team_tag(df)
        actual_tag = detected_tag if detected_tag else team_tag.upper()

        data = df[df["Unnamed: 1"].astype(str).str.upper().isin([actual_tag, "OPP"])].reset_index(drop=True)
        rows = []
        i = 0
        while i < len(data) - 1:
            team_row = data.iloc[i]
            opp_row = data.iloc[i + 1]
            tag_up = str(team_row["Unnamed: 1"]).upper()
            nxt_up = str(opp_row["Unnamed: 1"]).upper()

            if (tag_up == actual_tag and nxt_up == "OPP") or (tag_up == "OPP" and nxt_up == actual_tag):
                if tag_up == "OPP":
                    team_row, opp_row = opp_row, team_row

                players = extract_names(team_row["Lineup"])
                fga = team_row.get("Field goals attempted")
                fgm = team_row.get("Field goals made")
                fg3a = team_row.get("3-pt field goals attempted")
                fg3m = team_row.get("3-pt field goals made")
                fta = team_row.get("Free throws attempted")
                orb = team_row.get("Offensive rebounds")
                drb = team_row.get("Defensive rebounds")
                tov = team_row.get("Turnovers")
                pts = team_row.get("Points")
                poss = team_row.get("Possessions")
                mins = team_row.get("Minutes")

                opp_fga = opp_row.get("Field goals attempted")
                opp_fgm = opp_row.get("Field goals made")
                opp_fg3a = opp_row.get("3-pt field goals attempted")
                opp_fg3m = opp_row.get("3-pt field goals made")
                opp_fta = opp_row.get("Free throws attempted")
                opp_orb = opp_row.get("Offensive rebounds")
                opp_drb = opp_row.get("Defensive rebounds")
                opp_tov = opp_row.get("Turnovers")
                opp_pts = opp_row.get("Points")
                opp_poss = opp_row.get("Possessions")

                off_ppp = (pts / poss) * 100 if (pd.notna(pts) and pd.notna(poss) and poss > 0) else np.nan
                off_efg = ((fgm + 0.5 * fg3m) / fga) * 100 if (pd.notna(fgm) and pd.notna(fg3m) and pd.notna(fga) and fga > 0) else np.nan
                off_tov = (tov / poss) * 100 if (pd.notna(tov) and pd.notna(poss) and poss > 0) else np.nan
                off_orb = (orb / (orb + opp_drb)) * 100 if (pd.notna(orb) and pd.notna(opp_drb) and (orb + opp_drb) > 0) else np.nan
                off_ftr = (fta / fga) * 100 if (pd.notna(fta) and pd.notna(fga) and fga > 0) else np.nan

                def_ppp = (opp_pts / opp_poss) * 100 if (pd.notna(opp_pts) and pd.notna(opp_poss) and opp_poss > 0) else np.nan
                def_efg = ((opp_fgm + 0.5 * opp_fg3m) / opp_fga) * 100 if (pd.notna(opp_fgm) and pd.notna(opp_fg3m) and pd.notna(opp_fga) and opp_fga > 0) else np.nan
                def_tov = (opp_tov / opp_poss) * 100 if (pd.notna(opp_tov) and pd.notna(opp_poss) and opp_poss > 0) else np.nan
                def_orb = (opp_orb / (opp_orb + drb)) * 100 if (pd.notna(opp_orb) and pd.notna(drb) and (opp_orb + drb) > 0) else np.nan
                def_ftr = (opp_fta / opp_fga) * 100 if (pd.notna(opp_fta) and pd.notna(opp_fga) and opp_fga > 0) else np.nan

                rows.append({
                    "Players": players, "Minutes": mins, "Possessions": poss,
                    "Off Pts/Poss": off_ppp, "Off eFG%": off_efg, "Off TOV%": off_tov,
                    "Off ORB%": off_orb, "Off FT Rate": off_ftr,
                    "Def Pts/Poss": def_ppp, "Def eFG%": def_efg, "Def TOV%": def_tov,
                    "Def ORB%": def_orb, "Def FT Rate": def_ftr,
                })
                i += 2
            else:
                i += 1

        stints = pd.DataFrame(rows)
        if not stints.empty:
            return stints[stints["Players"].map(len) > 0].reset_index(drop=True)

    # Fallback: no tag column or tag-based parsing failed.
    # Use simple pair merging (same as parse_lineups) and compute rates.
    return _build_stints_from_pairs(df)


def _build_stints_from_pairs(df):
    """Fallback: build duo stints by pairing every two rows (team + OPP)."""
    if df.empty or "Lineup" not in df.columns:
        return pd.DataFrame()

    rows = []
    for i in range(0, len(df) - 1, 2):
        team = df.iloc[i]
        opp = df.iloc[i + 1]
        players = extract_names(team.get("Lineup"))
        if not players:
            continue

        pts_t = team.get("Points")
        poss_t = team.get("Possessions")
        pts_o = opp.get("Points")
        poss_o = opp.get("Possessions")
        fga_t = team.get("Field goals attempted")
        fgm_t = team.get("Field goals made")
        fg3m_t = team.get("3-pt field goals made")
        fta_t = team.get("Free throws attempted")
        orb_t = team.get("Offensive rebounds")
        tov_t = team.get("Turnovers")
        fga_o = opp.get("Field goals attempted")
        fgm_o = opp.get("Field goals made")
        fg3m_o = opp.get("3-pt field goals made")
        fta_o = opp.get("Free throws attempted")
        orb_o = opp.get("Offensive rebounds")
        drb_o = opp.get("Defensive rebounds")
        tov_o = opp.get("Turnovers")
        drb_t = team.get("Defensive rebounds")

        def safe_div(a, b):
            if pd.notna(a) and pd.notna(b) and b > 0:
                return a / b
            return np.nan

        off_ppp = safe_div(pts_t, poss_t) * 100 if pd.notna(safe_div(pts_t, poss_t)) else np.nan
        off_efg = safe_div(fgm_t + 0.5 * fg3m_t, fga_t) * 100 if (pd.notna(fgm_t) and pd.notna(fg3m_t) and pd.notna(safe_div(fgm_t + 0.5 * fg3m_t, fga_t))) else np.nan
        off_tov = safe_div(tov_t, poss_t) * 100 if pd.notna(safe_div(tov_t, poss_t)) else np.nan
        off_ftr = safe_div(fta_t, fga_t) * 100 if pd.notna(safe_div(fta_t, fga_t)) else np.nan
        off_orb_rate = safe_div(orb_t, orb_t + drb_o) * 100 if (pd.notna(orb_t) and pd.notna(drb_o) and pd.notna(safe_div(orb_t, orb_t + drb_o))) else np.nan

        def_ppp = safe_div(pts_o, poss_o) * 100 if pd.notna(safe_div(pts_o, poss_o)) else np.nan
        def_efg = safe_div(fgm_o + 0.5 * fg3m_o, fga_o) * 100 if (pd.notna(fgm_o) and pd.notna(fg3m_o) and pd.notna(safe_div(fgm_o + 0.5 * fg3m_o, fga_o))) else np.nan
        def_tov = safe_div(tov_o, poss_o) * 100 if pd.notna(safe_div(tov_o, poss_o)) else np.nan
        def_ftr = safe_div(fta_o, fga_o) * 100 if pd.notna(safe_div(fta_o, fga_o)) else np.nan
        def_orb_rate = safe_div(orb_o, orb_o + drb_t) * 100 if (pd.notna(orb_o) and pd.notna(drb_t) and pd.notna(safe_div(orb_o, orb_o + drb_t))) else np.nan

        rows.append({
            "Players": players, "Minutes": team.get("Minutes"), "Possessions": poss_t,
            "Off Pts/Poss": off_ppp, "Off eFG%": off_efg, "Off TOV%": off_tov,
            "Off ORB%": off_orb_rate, "Off FT Rate": off_ftr,
            "Def Pts/Poss": def_ppp, "Def eFG%": def_efg, "Def TOV%": def_tov,
            "Def ORB%": def_orb_rate, "Def FT Rate": def_ftr,
        })

    stints = pd.DataFrame(rows)
    if stints.empty:
        return stints
    return stints[stints["Players"].map(len) > 0].reset_index(drop=True)


def wavg(series, weights):
    s = pd.to_numeric(series, errors="coerce")
    w = pd.to_numeric(weights, errors="coerce")
    m = s.notna() & w.notna()
    if m.sum() == 0:
        return np.nan
    sw = w[m].sum()
    if sw == 0:
        return np.nan
    return (s[m] * w[m]).sum() / sw


def _fuzzy_match_player(player_name, all_players):
    """
    Smart player name matching that handles:
    - Exact match: "Q. Ellis" → "Q. Ellis"
    - Substring: "Ellis" → "Q. Ellis"
    - Full-to-initial: "Quinn Ellis" → "Q. Ellis"
    - Initial-to-full: "Q. Ellis" → "Quinn Ellis"
    - Last name only: "Ellis" → "Q. Ellis"
    - Reversed order: "Ellis, Quinn" → "Quinn Ellis" / "Q. Ellis"

    Returns (matched_name, was_fuzzy) or (None, False).
    """
    if not player_name or not all_players:
        return None, False

    name = player_name.strip()
    name_lower = name.lower()

    # 1) Exact match
    for p in all_players:
        if p.lower() == name_lower:
            return p, False

    # 2) Substring match (case-insensitive)
    matches = [p for p in all_players if name_lower in p.lower() or p.lower() in name_lower]
    if len(matches) == 1:
        return matches[0], True
    elif len(matches) > 1:
        return min(matches, key=len), True

    # 3) Parse input name into parts
    # Handle "Last, First" format
    if "," in name:
        parts = [x.strip() for x in name.split(",", 1)]
        name = f"{parts[1]} {parts[0]}"
        name_lower = name.lower()

    name_parts = name.split()
    if not name_parts:
        return None, False

    input_last = name_parts[-1].lower()
    input_firsts = [p.lower() for p in name_parts[:-1]]

    for p in all_players:
        p_parts = p.split()
        if not p_parts:
            continue
        p_last = p_parts[-1].lower()
        p_firsts = [x.lower() for x in p_parts[:-1]]

        # Last name must match
        if p_last != input_last:
            continue

        # If only last name given, match
        if not input_firsts:
            return p, True

        # Try matching first names / initials
        if _initials_match(input_firsts, p_firsts):
            return p, True

    # 4) difflib fallback
    from difflib import get_close_matches
    close = get_close_matches(name_lower, [p.lower() for p in all_players], n=1, cutoff=0.6)
    if close:
        for p in all_players:
            if p.lower() == close[0]:
                return p, True

    return None, False


def _initials_match(firsts_a, firsts_b):
    """Check if two first-name lists match, allowing initials.
    "Quinn" matches "Q.", "Q" matches "Quinn", etc."""
    if not firsts_a or not firsts_b:
        return True  # one side has no first name → match on last name alone

    for a, b in zip(firsts_a, firsts_b):
        a_clean = a.rstrip(".")
        b_clean = b.rstrip(".")

        if a_clean == b_clean:
            continue
        # One is an initial of the other
        if len(a_clean) == 1 and b_clean.startswith(a_clean):
            continue
        if len(b_clean) == 1 and a_clean.startswith(b_clean):
            continue
        return False
    return True


def _resolve_player_name(stints, player_name):
    """Find the actual player name in stints that matches the input."""
    all_players = sorted({p for lst in stints["Players"] for p in lst})
    matched, _ = _fuzzy_match_player(player_name, all_players)
    return matched


def compute_duo_onoff(stints, min_together=100.0, target_player=None):
    """Compute duo on/off synergy. If target_player is set, compute
    player-centric OFF = 'target on court, partner off court'."""
    if stints.empty:
        return pd.DataFrame()

    # Resolve target player name to exact match in data
    resolved_target = None
    if target_player:
        resolved_target = _resolve_player_name(stints, target_player)

    players = sorted({p for lst in stints["Players"] for p in lst})
    out = []

    for a, b in combinations(players, 2):
        on_mask = stints["Players"].apply(lambda lst: a in lst and b in lst)
        on_df = stints[on_mask]
        if on_df.empty:
            continue

        minutes_together = pd.to_numeric(on_df["Minutes"], errors="coerce").sum()
        if pd.isna(minutes_together) or minutes_together < min_together:
            continue

        # Determine OFF stints
        if resolved_target and (resolved_target == a or resolved_target == b):
            # Player-centric: OFF = target player on, partner off
            partner = b if resolved_target == a else a
            off_mask = stints["Players"].apply(
                lambda lst: resolved_target in lst and partner not in lst
            )
            off_df = stints[off_mask]
        else:
            # Generic: OFF = not both on
            off_df = stints[~on_mask]

        if off_df.empty:
            continue

        if "Possessions" in on_df.columns and on_df["Possessions"].notna().any():
            w_on, w_off = on_df["Possessions"], off_df["Possessions"]
        else:
            w_on, w_off = on_df["Minutes"], off_df["Minutes"]

        row = {"Duo": f"{a} + {b}", "Minutes together": round(float(minutes_together), 1)}
        for col in ["Off Pts/Poss", "Off eFG%", "Off TOV%", "Off ORB%", "Off FT Rate",
                     "Def Pts/Poss", "Def eFG%", "Def TOV%", "Def ORB%", "Def FT Rate"]:
            on_val = wavg(on_df[col], w_on)
            off_val = wavg(off_df[col], w_off)
            row[f"{col} (ON)"] = round(on_val, 2) if pd.notna(on_val) else np.nan
            row[f"{col} (OFF)"] = round(off_val, 2) if pd.notna(off_val) else np.nan
            row[f"{col} Diff"] = round(on_val - off_val, 2) if (pd.notna(on_val) and pd.notna(off_val)) else np.nan

        off_on = row["Off Pts/Poss (ON)"]
        def_on = row["Def Pts/Poss (ON)"]
        off_off = row["Off Pts/Poss (OFF)"]
        def_off = row["Def Pts/Poss (OFF)"]
        net_on = (off_on - def_on) if pd.notna(off_on) and pd.notna(def_on) else np.nan
        net_off = (off_off - def_off) if pd.notna(off_off) and pd.notna(def_off) else np.nan

        row["Net Rating (ON)"] = round(net_on, 2) if pd.notna(net_on) else np.nan
        row["Net Rating (OFF)"] = round(net_off, 2) if pd.notna(net_off) else np.nan
        row["Net Rating Diff"] = round(net_on - net_off, 2) if (pd.notna(net_on) and pd.notna(net_off)) else np.nan
        out.append(row)

    duo = pd.DataFrame(out)
    if duo.empty:
        return duo
    return duo.sort_values("Net Rating Diff", ascending=False).reset_index(drop=True)


# ── Main Pipeline Entry Point ────────────────────────────────────────────────

def run_onoff_pipeline(file_bytes, player_name, team_tag, min_minutes=100, min_duo_minutes=100):
    """
    Run the full On/Off + Duo pipeline.
    Returns (onoff_df, duo_df, player_onoff_md, player_duo_md, warnings)
    """
    import io
    warnings = []

    # Read all sheets
    xls = pd.ExcelFile(io.BytesIO(file_bytes))
    all_lineups = []
    all_stints = []

    for sheet in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet)
        df = clean_df(df)

        # For individual on/off: simple pair merging
        lineups = parse_lineups(df)
        if not lineups.empty:
            all_lineups.append(lineups)

        # For duo: use build_stints (auto-detects tag, falls back to pair merging)
        st = build_stints(df, team_tag)
        if not st.empty:
            all_stints.append(st)
        elif not lineups.empty:
            # Extra fallback: convert parse_lineups output to stint format for duos
            stint_rows = []
            for _, row in lineups.iterrows():
                rates = _compute_stint_rates(row)
                rates["Players"] = row["Players"]
                rates["Minutes"] = row["Minutes"]
                rates["Possessions"] = row.get("Poss_team", np.nan)
                stint_rows.append(rates)
            if stint_rows:
                fallback_stints = pd.DataFrame(stint_rows)
                fallback_stints = fallback_stints[fallback_stints["Players"].map(len) > 0]
                if not fallback_stints.empty:
                    all_stints.append(fallback_stints)

    if not all_lineups:
        return None, None, "", ["No valid lineup data found in the file."]

    df_lineups = pd.concat(all_lineups, ignore_index=True)
    df_onoff = compute_individual_onoff(df_lineups, min_minutes=min_minutes)

    if df_onoff.empty:
        warnings.append(f"No players found with >= {min_minutes} minutes.")

    # Find the target player (fuzzy match)
    all_players = df_onoff["Player"].tolist()
    resolved_name, was_fuzzy = _fuzzy_match_player(player_name, all_players)

    if resolved_name:
        if was_fuzzy:
            warnings.append(f"Matched '{player_name}' → '{resolved_name}'.")
        player_name = resolved_name
    else:
        warnings.append(f"Player '{player_name}' not found. Available players: {', '.join(all_players[:15])}")

    # Duo analysis
    duo_df = pd.DataFrame()
    if all_stints:
        stints = pd.concat(all_stints, ignore_index=True)
        duo_df = compute_duo_onoff(stints, min_together=min_duo_minutes, target_player=player_name)
        if duo_df.empty:
            warnings.append(f"No duos found with >= {min_duo_minutes} minutes together.")

    # Generate markdown
    md = generate_onoff_markdown(df_onoff, duo_df, player_name, team_tag, warnings)

    return df_onoff, duo_df, md, warnings


def _fmt(val, force_sign=False):
    """Format a numeric value with optional +/- sign."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "--"
    if isinstance(val, (int, float)):
        if force_sign:
            return f"+{val:.2f}" if val > 0 else f"{val:.2f}"
        return f"{val:.2f}"
    return str(val)


def _fmt_int(val):
    """Format as integer."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "--"
    return str(int(round(val)))


def generate_onoff_markdown(df_onoff, duo_df, player_name, team_tag, warnings):
    """Generate the markdown report matching the exact PlayerLynk format."""
    from datetime import datetime
    lines = []

    # ── Header ──
    lines.append(f"# Scouting Report: {player_name}")
    lines.append(f"**Team:** {team_tag}")
    lines.append(f"**Data:** Lineup stints (season aggregated)")
    lines.append(f"**Date:** {datetime.now().strftime('%d %b %Y')}")
    lines.append("")

    # Find the target player row
    player_row = df_onoff[df_onoff["Player"].str.contains(player_name, case=False, na=False)]

    if player_row.empty:
        if warnings:
            for w in warnings:
                lines.append(f"> **Warning:** {w}")
        return "\n".join(lines)

    pr = player_row.iloc[0]

    # ── Sample Size ──
    lines.append("## Sample Size")
    on_min = pr.get("Minutes_total_ON", 0)
    off_min = pr.get("Minutes_total_OFF", 0)
    on_poss = pr.get("Poss_team_total_ON", 0)
    off_poss = pr.get("Poss_team_total_OFF", 0)
    lines.append(f"- **ON-court minutes:** {_fmt(on_min)}")
    lines.append(f"- **OFF-court minutes:** {_fmt(off_min)}")
    lines.append(f"- **ON-court possessions (team):** {_fmt_int(on_poss)}")
    lines.append(f"- **OFF-court possessions (team):** {_fmt_int(off_poss)}")
    lines.append("")

    if warnings:
        for w in warnings:
            lines.append(f"> **Warning:** {w}")
        lines.append("")

    # ── On/Off Impact Table (single combined table) ──
    lines.append("## On/Off Impact (per 100 possessions)")
    lines.append("")
    lines.append("| Metric | ON | OFF | DIFF |")
    lines.append("|--------|---:|----:|-----:|")

    # Net Rating first (bolded)
    off_on = pr.get("OFFENSE: Pts/Poss_ON", np.nan)
    def_on = pr.get("DEFENSE: Pts/Poss_ON", np.nan)
    off_off = pr.get("OFFENSE: Pts/Poss_OFF", np.nan)
    def_off = pr.get("DEFENSE: Pts/Poss_OFF", np.nan)
    net_on = off_on - def_on if pd.notna(off_on) and pd.notna(def_on) else np.nan
    net_off = off_off - def_off if pd.notna(off_off) and pd.notna(def_off) else np.nan
    net_diff = pr.get("Diff", np.nan)
    lines.append(f"| **Net Rating** | {_fmt(net_on, True)} | {_fmt(net_off, True)} | **{_fmt(net_diff, True)}** |")

    # All metrics in order matching the example
    metric_order = [
        ("OFFENSE: Pts/Poss", "OFFENSE: Pts/Poss"),
        ("DEFENSE: Pts/Poss", "DEFENSE: Pts/Poss"),
        ("OFFENSE: eFG%", "OFFENSE: eFG%"),
        ("DEFENSE: eFG%", "DEFENSE: eFG%"),
        ("OFFENSE: TOV%", "OFFENSE: TOV%"),
        ("DEFENSE: TOV%", "DEFENSE: TOV%"),
        ("OFFENSE: ORB%", "OFFENSE: ORB%"),
        ("DEFENSE: ORB%", "DEFENSE: ORB%"),
        ("OFFENSE: FT Rate", "OFFENSE: FT Rate"),
        ("DEFENSE: FT Rate", "DEFENSE: FT Rate"),
        ("OFFENSE: 2P%", "OFFENSE: 2P%"),
        ("OFFENSE: 3P%", "OFFENSE: 3P%"),
        ("DEFENSE: 2P%", "DEFENSE: 2P%"),
        ("DEFENSE: 3P%", "DEFENSE: 3P%"),
        ("OFFENSE: 3PA FREQ%", "OFFENSE: 3PA FREQ%"),
        ("OFFENSE: 2PA FREQ%", "OFFENSE: 2PA FREQ%"),
        ("DEFENSE: 3PA FREQ%", "DEFENSE: 3PA FREQ%"),
        ("DEFENSE: 2PA FREQ%", "DEFENSE: 2PA FREQ%"),
    ]

    for label, key in metric_order:
        on_val = pr.get(f"{key}_ON", np.nan)
        off_val = pr.get(f"{key}_OFF", np.nan)
        diff_val = pr.get(key, np.nan)
        lines.append(f"| {label} | {_fmt(on_val)} | {_fmt(off_val)} | {_fmt(diff_val, True)} |")

    lines.append("")

    # ── Offensive Impact Summary (narrative bullets) ──
    lines.append("## Offensive Impact Summary")
    lines.append("")

    off_rtg_on = pr.get("OFFENSE: Pts/Poss_ON", 0)
    off_rtg_off = pr.get("OFFENSE: Pts/Poss_OFF", 0)
    off_rtg_diff = pr.get("OFFENSE: Pts/Poss", 0)
    direction = "worse" if off_rtg_diff < 0 else "better"
    lines.append(f"- **Offensive Rating:** Team scores **{abs(off_rtg_diff):.1f} pts/100 poss {direction}** with {player_name} on court ({off_rtg_on:.1f} ON vs {off_rtg_off:.1f} OFF).")

    efg_diff = pr.get("OFFENSE: eFG%", 0)
    direction = "lower" if efg_diff < 0 else "higher"
    lines.append(f"- **eFG%:** {abs(efg_diff):.1f}pp {direction} with him on court --{'reduced' if efg_diff < 0 else 'improved'} shooting efficiency.")

    tov_diff = pr.get("OFFENSE: TOV%", 0)
    direction = "fewer" if tov_diff < 0 else "more"
    quality = "a positive for ball security" if tov_diff < 0 else "a concern for ball security"
    lines.append(f"- **TOV%:** {abs(tov_diff):.1f}pp {direction} turnovers per possession --{quality}.")

    orb_diff = pr.get("OFFENSE: ORB%", 0)
    direction = "lower" if orb_diff < 0 else "higher"
    lines.append(f"- **ORB%:** {abs(orb_diff):.1f}pp {direction} offensive rebounding rate.")

    ftr_diff = pr.get("OFFENSE: FT Rate", 0)
    direction = "lower" if ftr_diff < 0 else "higher"
    quality = "fewer" if ftr_diff < 0 else "more"
    lines.append(f"- **FT Rate:** {abs(ftr_diff):.1f}pp {direction} free throw rate --{quality} trips to the line.")

    # ON-court shot profile
    o3pa_freq = pr.get("OFFENSE: 3PA FREQ%_ON", 0)
    o3p_pct = pr.get("OFFENSE: 3P%_ON", 0)
    o2pa_freq = pr.get("OFFENSE: 2PA FREQ%_ON", 0)
    o2p_pct = pr.get("OFFENSE: 2P%_ON", 0)
    lines.append(f"- **ON-court shot profile:** {o3pa_freq:.1f}% 3PA frequency at {o3p_pct:.1f}% 3P%, {o2pa_freq:.1f}% 2PA frequency at {o2p_pct:.1f}% 2P%.")

    lines.append("")

    # ── Defensive Impact Summary (narrative bullets) ──
    lines.append("## Defensive Impact Summary")
    lines.append("")

    def_rtg_on = pr.get("DEFENSE: Pts/Poss_ON", 0)
    def_rtg_off = pr.get("DEFENSE: Pts/Poss_OFF", 0)
    def_rtg_diff = pr.get("DEFENSE: Pts/Poss", 0)
    direction = "more" if def_rtg_diff > 0 else "fewer"
    signal = "a negative signal" if def_rtg_diff > 0 else "a positive signal"
    lines.append(f"- **Defensive Rating:** Opponents score **{abs(def_rtg_diff):.1f} pts/100 poss {direction}** with {player_name} on court ({def_rtg_on:.1f} ON vs {def_rtg_off:.1f} OFF) --{signal}.")

    defg_diff = pr.get("DEFENSE: eFG%", 0)
    direction = "higher" if defg_diff > 0 else "lower"
    quality = "worse" if defg_diff > 0 else "better"
    lines.append(f"- **Opp eFG%:** {abs(defg_diff):.1f}pp {direction} --{quality} shot contestation.")

    dtov_diff = pr.get("DEFENSE: TOV%", 0)
    direction = "fewer" if dtov_diff < 0 else "more"
    quality = "a negative for disruption" if dtov_diff < 0 else "a positive for disruption"
    lines.append(f"- **Forced TOV%:** {abs(dtov_diff):.1f}pp {direction} turnovers forced --{quality}.")

    dorb_diff = pr.get("DEFENSE: ORB%", 0)
    direction = "more" if dorb_diff > 0 else "fewer"
    quality = "a concern for defensive rebounding" if dorb_diff > 0 else "a positive for defensive rebounding"
    lines.append(f"- **Opp ORB%:** {abs(dorb_diff):.1f}pp {direction} opponent offensive rebounds --{quality}.")

    dftr_diff = pr.get("DEFENSE: FT Rate", 0)
    direction = "fewer" if dftr_diff < 0 else "more"
    quality = "a positive for foul discipline" if dftr_diff < 0 else "a concern for foul discipline"
    lines.append(f"- **Opp FT Rate:** {abs(dftr_diff):.1f}pp {direction} opponent free throws --{quality}.")

    # ON-court opp shot profile
    d3pa_freq = pr.get("DEFENSE: 3PA FREQ%_ON", 0)
    d3p_pct = pr.get("DEFENSE: 3P%_ON", 0)
    d2pa_freq = pr.get("DEFENSE: 2PA FREQ%_ON", 0)
    d2p_pct = pr.get("DEFENSE: 2P%_ON", 0)
    lines.append(f"- **ON-court opp shot profile:** {d3pa_freq:.1f}% 3PA frequency at {d3p_pct:.1f}% 3P%, {d2pa_freq:.1f}% 2PA frequency at {d2p_pct:.1f}% 2P%.")

    lines.append("")

    # ── Two-Man Lineup Splits (player-centric, sorted by Net Rtg Diff) ──
    if not duo_df.empty:
        pattern = re.escape(player_name)
        player_duos = duo_df[duo_df["Duo"].str.contains(pattern, case=False, na=False)].copy()

        if not player_duos.empty:
            lines.append("## Two-Man Lineup Splits")
            min_min = int(duo_df['Minutes together'].min()) if len(duo_df) else 100
            lines.append(f"*{player_name} paired with each teammate -- minimum {min_min} minutes together.*")
            lines.append(f"*Sorted by Net Rating differential (ON together vs. {player_name} on without partner).*")
            lines.append("")
            lines.append("| Partner | Min Together | Net Rtg (ON) | Net Rtg (OFF) | Net Rtg Diff | Off Rtg (ON) | Def Rtg (ON) | Off eFG% (ON) | Off TOV% (ON) |")
            lines.append("|---------|------------:|-------------:|--------------:|-------------:|-------------:|-------------:|--------------:|--------------:|")

            # Sort by Net Rating Diff descending
            player_duos = player_duos.sort_values("Net Rating Diff", ascending=False)

            for _, r in player_duos.iterrows():
                # Extract partner name from duo string
                duo_str = r["Duo"]
                parts = duo_str.split(" + ")
                partner = parts[1] if player_name.lower() in parts[0].lower() else parts[0]

                min_tog = _fmt_int(r.get("Minutes together"))
                net_on = _fmt(r.get("Net Rating (ON)"), True)
                net_off = _fmt(r.get("Net Rating (OFF)"), True)
                net_diff = r.get("Net Rating Diff", np.nan)
                net_diff_str = f"**{_fmt(net_diff, True)}**"
                off_rtg = _fmt(r.get("Off Pts/Poss (ON)"))
                def_rtg = _fmt(r.get("Def Pts/Poss (ON)"))

                # eFG% and TOV% ON --format as whole numbers like in the example
                efg_on = r.get("Off eFG% (ON)", np.nan)
                efg_str = f"{efg_on:.1f}" if pd.notna(efg_on) else "--"
                tov_on = r.get("Off TOV% (ON)", np.nan)
                tov_str = f"{tov_on:.1f}" if pd.notna(tov_on) else "--"

                lines.append(f"| {partner} | {min_tog} | {net_on} | {net_off} | {net_diff_str} | {off_rtg} | {def_rtg} | {efg_str} | {tov_str} |")

            lines.append("")

            # ── Duo Key Findings ──
            lines.append("### Duo Key Findings")
            lines.append("")

            best = player_duos.iloc[0]
            best_duo = best["Duo"]
            best_parts = best_duo.split(" + ")
            best_partner = best_parts[1] if player_name.lower() in best_parts[0].lower() else best_parts[0]
            best_min = _fmt_int(best.get("Minutes together"))
            best_diff = _fmt(best.get("Net Rating Diff"), True)
            best_net_on = _fmt(best.get("Net Rating (ON)"), True)
            best_net_off = _fmt(best.get("Net Rating (OFF)"), True)
            lines.append(f"- **Best pairing:** {best_partner} ({best_min} min together, Net Rtg Diff: {best_diff}). Net rating of {best_net_on} when paired vs {best_net_off} when {player_name} plays without them.")

            worst = player_duos.iloc[-1]
            worst_duo = worst["Duo"]
            worst_parts = worst_duo.split(" + ")
            worst_partner = worst_parts[1] if player_name.lower() in worst_parts[0].lower() else worst_parts[0]
            worst_min = _fmt_int(worst.get("Minutes together"))
            worst_diff = _fmt(worst.get("Net Rating Diff"), True)
            worst_net_on = _fmt(worst.get("Net Rating (ON)"), True)
            lines.append(f"- **Worst pairing:** {worst_partner} ({worst_min} min together, Net Rtg Diff: {worst_diff}). Net rating drops to {worst_net_on} when paired.")

            lines.append("")

    # ── Footer ──
    lines.append("---")
    lines.append(f"*Report generated from lineup stint data. All rate stats are per 100 possessions. Four Factors computed from aggregated totals (not possession-weighted averages of stint-level rates). Duo metrics use possession-weighted averaging.*")

    return "\n".join(lines)
