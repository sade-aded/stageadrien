"""
Pipeline 2 — Percentiles + Tags + Clusters + Similarity
Mode A: Single player profile
Mode B: Find similar players on a target team
"""

import pandas as pd
import numpy as np
import re
from difflib import get_close_matches
from scipy.spatial.distance import euclidean


# ── Fuzzy Team Matching ─────────────────────────────────────────────────────

def _normalize_team(name):
    """Normalize a team name for comparison: lowercase, strip punctuation, common abbreviations."""
    s = str(name).lower().strip()
    # Remove common suffixes/noise
    for remove in ["university", "college", "state", " u "]:
        s = s.replace(remove, " ")
    # Normalize abbreviations
    s = s.replace("st.", "st").replace("saint", "st")
    # Strip all punctuation except spaces
    s = re.sub(r"[^\w\s]", "", s)
    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


def fuzzy_match_team(target_team, available_teams, cutoff=0.5):
    """
    Find the best matching team name from available teams.
    Returns (matched_name, warning_msg_or_None).

    Strategy:
    1. Exact case-insensitive match (full string)
    2. Normalized exact match
    3. Word-boundary-aware substring match (prefer shortest / most precise)
    4. difflib fuzzy match on normalized names
    """
    target_lower = target_team.lower().strip()
    target_norm = _normalize_team(target_team)

    # Build normalized lookup
    norm_map = {}  # normalized -> original
    for t in available_teams:
        n = _normalize_team(t)
        norm_map[n] = t

    # 1) Exact full-string match (case-insensitive)
    for t in available_teams:
        if target_lower == t.lower().strip():
            return t, None

    # 2) Exact normalized match
    if target_norm in norm_map:
        return norm_map[target_norm], None

    # 3) Word-boundary-aware substring match
    #    "Kentucky" should match "Kentucky Wildcats" but NOT "Eastern Kentucky"
    #    Strategy: collect all substring matches, then pick the best one
    #    Best = shortest name (fewest extra words) + target appears at start
    substring_matches = []
    target_words = target_norm.split()

    for n, orig in norm_map.items():
        n_words = n.split()

        # Check if all target words appear in the team name
        if all(tw in n_words for tw in target_words):
            # Score: how many extra words does the team name have?
            extra_words = len(n_words) - len(target_words)
            # Bonus: does the team name START with the target words?
            starts_with = n.startswith(target_norm)
            score = (0 if starts_with else 1, extra_words, len(n))
            substring_matches.append((score, orig))
        elif target_norm in n or n in target_norm:
            # Fallback: raw substring (less reliable)
            extra = abs(len(n) - len(target_norm))
            substring_matches.append(((2, extra, len(n)), orig))

    if substring_matches:
        # Sort by score tuple — lowest is best
        substring_matches.sort(key=lambda x: x[0])
        best_match = substring_matches[0][1]
        if best_match.lower().strip() != target_lower:
            return best_match, f"Matched '{target_team}' to '{best_match}'."
        return best_match, None

    # 4) Fuzzy match via difflib — use a HIGH cutoff to avoid garbage matches
    #    Short names like "France" can falsely match "Ukraine" at low cutoffs.
    #    Use 0.75 minimum (or the caller's cutoff if higher).
    fuzzy_cutoff = max(cutoff, 0.75)
    close = get_close_matches(target_norm, list(norm_map.keys()), n=1, cutoff=fuzzy_cutoff)
    if close:
        matched_orig = norm_map[close[0]]
        return matched_orig, f"Fuzzy matched '{target_team}' to '{matched_orig}'."

    return None, f"Team '{target_team}' not found. Available: {', '.join(available_teams[:15])}"

# ── Formatting Helpers ──────────────────────────────────────────────────────

def _ordinal(n):
    """Convert integer to ordinal string: 1→'1st', 2→'2nd', 93→'93rd', 44→'44th'."""
    n = int(n)
    if 11 <= (n % 100) <= 13:
        return f"{n}th"
    suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _pct_display(val):
    """Format a 0-1 percentile value as ordinal string, or '--' if missing."""
    if val is None:
        return "--"
    if isinstance(val, str):
        return "--"
    try:
        if pd.isna(val):
            return "--"
    except (TypeError, ValueError):
        return "--"
    return _ordinal(round(float(val) * 100))


# ── Data Cleaning ────────────────────────────────────────────────────────────

def minutes_to_float(min_str):
    if pd.isna(min_str):
        return np.nan
    s = str(min_str).strip()
    if s in {"-", "", "nan", "None"}:
        return np.nan
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


def clean_league_data(df):
    """Clean a league box score DataFrame."""
    df = df.copy()
    df.columns = df.columns.str.strip()

    # Clean all columns
    for col in df.columns:
        if col in ["Player", "Team", "Name"]:
            continue
        # Convert to string, strip %, replace dashes with NaN
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].str.replace('%', '', regex=False)
        df[col] = df[col].str.replace(',', '.', regex=False)
        df[col] = df[col].replace(['-', 'nan', '', 'None', 'N/A'], np.nan)
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Convert Minutes
    if "Minutes" in df.columns:
        # Re-read original minutes before numeric conversion
        df["Minutes"] = df["Minutes"]  # already processed above but may have lost format
    # We need to handle minutes specially — re-load from original if MM:SS
    return df


def load_and_clean(file_bytes, sheet_name="Box score"):
    """Load an Excel file and clean it."""
    import io
    xls = pd.ExcelFile(io.BytesIO(file_bytes))

    # Try to find the right sheet
    if sheet_name in xls.sheet_names:
        df_raw = pd.read_excel(xls, sheet_name=sheet_name)
    elif len(xls.sheet_names) == 1:
        df_raw = pd.read_excel(xls, sheet_name=xls.sheet_names[0])
    else:
        # Try common names
        for name in ["Box score", "Box Score", "Stats", "Players", xls.sheet_names[0]]:
            if name in xls.sheet_names:
                df_raw = pd.read_excel(xls, sheet_name=name)
                break
        else:
            df_raw = pd.read_excel(xls, sheet_name=xls.sheet_names[0])

    # Save raw minutes before cleaning
    df_raw_cols = df_raw.columns.str.strip().tolist()
    raw_minutes = None
    if "Minutes" in df_raw_cols:
        idx = df_raw_cols.index("Minutes")
        raw_minutes = df_raw.iloc[:, idx].copy()

    df = clean_league_data(df_raw)

    # Re-apply minutes conversion from raw
    if raw_minutes is not None:
        df["Minutes"] = raw_minutes.apply(minutes_to_float)

    return df


# ── Per-40 Normalization ─────────────────────────────────────────────────────

# Stats that should be normalized to per-40 minutes (counting stats)
COUNTING_STATS = [
    "Points", "Assists", "Rebounds", "Offensive rebounds", "Defensive rebounds",
    "Steals", "Blocks", "Turnovers",
    "Field goals made", "Field goals attempted",
    "3-pt field goals made", "3-pt field goals attempted",
    "2-pt field goals made", "2-pt field goals attempted",
    "Free throws made", "Free throws attempted",
    "Catch and shoot made", "Catch and shoot shots made",
    "Cuts made", "Catch and drive made", "Drives with shot", "Drives made",
    "Screens off attempted", "Screens off made",
    "Transitions made", "Transitions attempted", "Transition attacks",
    "Isolations made", "Isolations attempted",
    "Secondary Assist", "Points off assists",
    "Post up made", "Post up attempted",
    "Hand off made", "Hand off attempted",
    "Pick-n-roll made", "Pick-n-roll attempted",
    "Pick-n-pop made", "Pick-n-pop attempted",
]

# Defensive usage columns (also counting-ish, normalize per 40)
DEF_USAGE_PATTERN = r"^(Opp |Def\. usage )"


def compute_per40(df):
    """Add per-40 minute columns for counting stats."""
    df = df.copy()
    if "Minutes" not in df.columns or df["Minutes"].isna().all():
        return df

    for col in df.columns:
        if col in COUNTING_STATS or re.match(DEF_USAGE_PATTERN, col):
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                per40_col = f"{col}_per40"
                df[per40_col] = np.where(
                    df["Minutes"] > 0,
                    df[col] / df["Minutes"] * 40,
                    np.nan
                )
    return df


# ── Per-36 and Safe-Column Helpers ──────────────────────────────────────────

def per36(df, col):
    """Convert per-game stat to per-36-minutes rate."""
    if col not in df.columns:
        return pd.Series(0.0, index=df.index)
    mpg = df.get("Minutes", pd.Series(0.0, index=df.index))
    out = df[col] * 36.0 / mpg
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def _safe_col(df, col, default=0.0):
    """Return column if it exists, otherwise Series of default."""
    if col in df.columns:
        return df[col].fillna(default)
    return pd.Series(default, index=df.index)


def _safe_pct(df, col_name, default=0.0):
    """Get a percentile column from df, trying multiple name patterns."""
    for suffix in [f"{col_name}_pct", f"{col_name}_per40_pct"]:
        if suffix in df.columns:
            return df[suffix].fillna(default)
    return pd.Series(default, index=df.index)


# ── Percentile Computation ───────────────────────────────────────────────────

def compute_percentiles(df, min_games=5, min_mpg=0):
    """
    Compute percentile ranks (0-1) for all numeric columns.
    Adds _pct suffix columns.
    """
    df = df.copy()

    # Filter qualifying players
    if "Games played" in df.columns:
        df = df[df["Games played"] >= min_games].copy()
    if min_mpg > 0 and "avg_minutes" in df.columns:
        df = df[df["avg_minutes"] >= min_mpg].copy()
    elif min_mpg > 0 and "Minutes" in df.columns and "Games played" in df.columns:
        df["avg_minutes"] = df["Minutes"] / df["Games played"]
        df = df[df["avg_minutes"] >= min_mpg].copy()

    # Compute percentiles for all numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude = ["Games played", "Minutes", "avg_minutes"]

    pct_data = {}
    for col in numeric_cols:
        if col in exclude:
            continue
        pct_data[f"{col}_pct"] = df[col].rank(pct=True, na_option='keep')

    df = pd.concat([df, pd.DataFrame(pct_data, index=df.index)], axis=1)

    return df


# ── Position Inference (v2) ──────────────────────────────────────────────────

def derive_position(df):
    """Derive position from stat profile using composite big/guard scoring.
    Works on a full DataFrame, returns a Series of position strings.

    v2.1: Reweighted for modern stretch-4/5 archetypes.
    - Reduced post-up weight (not all bigs post up)
    - Added offensive rebounds to big_score
    - Reduced 3PA weight on guard side (many modern 4s shoot threes)
    - Added hard override: elite DREB + BLK forces at least Forward
    """
    def _rank01(s):
        return s.rank(pct=True).fillna(0.5)

    dreb36 = per36(df, "Defensive rebounds")
    blk36 = per36(df, "Blocks")
    orb36 = per36(df, "Offensive rebounds")
    post36 = per36(df, "Posts up attempted")
    ast36 = per36(df, "Assists")
    tpa36 = per36(df, "3-pt field goals attempted")
    drive36 = per36(df, "Drives with shot")

    r_dreb = _rank01(dreb36)
    r_blk = _rank01(blk36)
    r_orb = _rank01(orb36)
    r_post = _rank01(post36)
    r_ast = _rank01(ast36)
    r_tpa = _rank01(tpa36)
    r_drive = _rank01(drive36)

    big_score = (
        r_dreb * 0.30 +
        r_blk * 0.25 +
        r_orb * 0.15 +
        r_post * 0.10 +
        (1 - r_tpa) * 0.10 +
        (1 - r_ast) * 0.10
    )

    guard_score = (
        r_ast * 0.40 +
        r_drive * 0.25 +
        r_tpa * 0.10 +
        (1 - r_dreb) * 0.10 +
        (1 - r_blk) * 0.10 +
        (1 - r_orb) * 0.05
    )

    size_axis = big_score - guard_score

    pos = pd.Series("Wing", index=df.index)
    pos[size_axis >= 0.25] = "Center"
    pos[(size_axis >= 0.08) & (size_axis < 0.25)] = "Forward"
    pos[(size_axis >= -0.10) & (size_axis < 0.08)] = "Wing"
    pos[size_axis < -0.10] = "Guard"

    # Hard override: elite rebounders + shot-blockers are at least Forward
    # (catches stretch-4s who shoot threes but rebound/block like bigs)
    clear_big = (r_dreb >= 0.80) & (r_blk >= 0.65)
    pos[clear_big & (pos == "Guard")] = "Forward"
    pos[clear_big & (pos == "Wing")] = "Forward"

    return pos.fillna("Wing")


def infer_position(row):
    """Legacy single-row position inference (fallback only)."""
    reb_pct = row.get("Rebounds_pct", row.get("Defensive rebounds_pct", 0.5))
    blk_pct = row.get("Blocks_pct", 0.5)
    ast_pct = row.get("Assists_pct", 0.5)
    if pd.isna(reb_pct): reb_pct = 0.5
    if pd.isna(blk_pct): blk_pct = 0.5
    if pd.isna(ast_pct): ast_pct = 0.5
    if reb_pct > 0.7 and blk_pct > 0.5: return "Center"
    if ast_pct > 0.65 and blk_pct < 0.4: return "Guard"
    if reb_pct > 0.7 and blk_pct > 0.3: return "Forward"
    if ast_pct > 0.5 and reb_pct < 0.5: return "Guard"
    if reb_pct > 0.65 and ast_pct < 0.5: return "Forward"
    return "Wing"


# ── Tag Assignment ───────────────────────────────────────────────────────────

def _get_pct(row, col_name):
    """Get a percentile value, trying multiple column name patterns."""
    # Try exact match first
    val = row.get(f"{col_name}_pct")
    if val is not None and not pd.isna(val):
        return val
    # Try per40 version
    val = row.get(f"{col_name}_per40_pct")
    if val is not None and not pd.isna(val):
        return val
    return 0.0


# ── v2 Tag System — Archetype Gates + Tiers ─────────────────────────────────
# Two-stage: 1) archetype gate (raw thresholds) 2) tier within qualified subset

TIER_LABELS = ["Below avg", "Average", "Above avg", "Elite"]


def _pct_col(df, col, default=0.0):
    """Get a percentage column normalized to 0-1 scale.
    Handles both 0.469 and 46.9 formats transparently."""
    if col not in df.columns:
        return pd.Series(default, index=df.index)
    vals = df[col].fillna(default)
    # If max value > 1.5, these are whole-number percentages (e.g., 46.9 not 0.469)
    if vals.abs().max() > 1.5:
        vals = vals / 100.0
    return vals


def tier_from_pct(pct):
    """4 tiers by percentile within qualified subset.
       Bottom 25 / 25-50 / 50-85 / 85+."""
    if pd.isna(pct):
        return None
    if pct >= 0.85:
        return "Elite"
    if pct >= 0.50:
        return "Above avg"
    if pct >= 0.25:
        return "Average"
    return "Below avg"


def _add_tag(tags_dict, player_idx, archetype, tier):
    """Attach an archetype tag with its tier to a player."""
    if player_idx not in tags_dict:
        tags_dict[player_idx] = []
    tags_dict[player_idx].append(f"{archetype} ({tier})")


# ── Archetype functions ───────────────────────────────────────────────────────
# Each returns (qualifies: bool Series, score: numeric Series)
# qualifies = raw basketball threshold gate
# score = metric for ranking within qualified subset

def _arch_three_pt_shooter(df):
    """Shoots threes with real volume and acceptable accuracy."""
    tpa36 = per36(df, "3-pt field goals attempted")
    fg3_pct = _pct_col(df, "3-pt field goals, %")
    if fg3_pct.sum() == 0:
        fg3_pct = _pct_col(df, "3-pt field goals %")
    qualifies = (tpa36 >= 4.0) & (fg3_pct >= 0.33)
    tpm36 = per36(df, "3-pt field goals made")
    score = tpm36 * fg3_pct
    return qualifies, score


def _arch_spot_up_shooter(df):
    """Catch-and-shoot is a real part of his diet, and it goes in."""
    cas36 = per36(df, "Catch and shoot attempted")
    if cas36.sum() == 0:
        cas36 = per36(df, "Catch and shoot shots made")
    cs_pct = _pct_col(df, "Catch and shoot shots made, %")
    if cs_pct.sum() == 0:
        cs_pct = _pct_col(df, "Catch and shoot shots made %")
    qualifies = (cas36 >= 2.0) & (cs_pct >= 0.35)
    score = per36(df, "Catch and shoot made") * cs_pct
    return qualifies, score


def _arch_movement_shooter(df):
    """Shoots off screens — running off curls/staggers."""
    so36 = per36(df, "Screens off attempted")
    so_pct = _pct_col(df, "Screens off, %")
    if so_pct.sum() == 0:
        so_pct = _pct_col(df, "Screens off %")
    qualifies = (so36 >= 1.0) & (so_pct >= 0.33)
    score = per36(df, "Screens off made") * so_pct
    return qualifies, score


def _arch_transition_threat(df):
    """Gets out and finishes in transition."""
    tr36 = per36(df, "Transitions attempted")
    tr_pct = _pct_col(df, "Transition attacks, %")
    if tr_pct.sum() == 0:
        tr_pct = _pct_col(df, "Transition attacks %")
    qualifies = (tr36 >= 1.5) & (tr_pct >= 0.45)
    score = per36(df, "Transitions made") * tr_pct
    return qualifies, score


def _arch_interior_finisher(df):
    """Finishes at the rim — 2pt volume + efficiency."""
    twopa36 = per36(df, "2-pt field goals attempted")
    fg2_pct = _pct_col(df, "2-pt field goals, %")
    if fg2_pct.sum() == 0:
        fg2_pct = _pct_col(df, "2-pt field goals %")
    qualifies = (twopa36 >= 3.5) & (fg2_pct >= 0.50)
    score = per36(df, "2-pt field goals made") * fg2_pct
    return qualifies, score


def _arch_offensive_rebounder(df):
    """Crashes the offensive glass."""
    orb36 = per36(df, "Offensive rebounds")
    qualifies = orb36 >= 2.0
    score = orb36
    return qualifies, score


def _arch_playmaker(df):
    """Real distribution role with usable A/TO."""
    ast36 = per36(df, "Assists")
    ast_tov = _safe_col(df, "Assists to turnovers", 0)
    if ast_tov.sum() == 0:
        ast_tov = _safe_col(df, "Assist to turnover", 0)
    qualifies = (ast36 >= 4.0) & (ast_tov >= 1.5)
    score = ast36 * ast_tov
    return qualifies, score


def _arch_secondary_creator(df):
    """Helps the offense flow without being the primary engine."""
    ast36 = per36(df, "Assists")
    ast_tov = _safe_col(df, "Assists to turnovers", 0)
    if ast_tov.sum() == 0:
        ast_tov = _safe_col(df, "Assist to turnover", 0)
    qualifies = (ast36 >= 2.5) & (ast36 < 4.0) & (ast_tov >= 1.2)
    score = ast36 * ast_tov
    return qualifies, score


def _arch_pnr_ball_handler(df):
    """Initiates ball-screen offense as a handler."""
    pnr36 = per36(df, "PnR Handlers attempted")
    qualifies = pnr36 >= 2.0
    pnr_pct = _pct_col(df, "PR Handler, %")
    score = per36(df, "PnR Handlers made") * pnr_pct
    return qualifies, score


def _arch_pnr_roll_man(df):
    """Rolls and finishes off ball screens."""
    pnr36 = per36(df, "PnR Rollers attempted")
    qualifies = pnr36 >= 1.0
    pnr_pct = _pct_col(df, "PR Roller, %")
    score = per36(df, "PnR Rollers made") * pnr_pct
    return qualifies, score


def _arch_pick_n_pop(df):
    """Bigs who shoot off the pop."""
    pnp36 = per36(df, "PnP attempted")
    if pnp36.sum() == 0:
        pnp36 = per36(df, "Pick-n-pop attempted")
    qualifies = pnp36 >= 0.7
    pnp_pct = _pct_col(df, "Pick-n-pops, %")
    if pnp_pct.sum() == 0:
        pnp_pct = _pct_col(df, "Pick-n-pop %")
    pnp_made = per36(df, "PnP made")
    if pnp_made.sum() == 0:
        pnp_made = per36(df, "Pick-n-pop made")
    score = pnp_made * pnp_pct
    return qualifies, score


def _arch_post_scorer(df):
    """Real post-up game. Restricted to Forward/Center."""
    post36 = per36(df, "Posts up attempted")
    if post36.sum() == 0:
        post36 = per36(df, "Post up attempted")
    post_pct = _pct_col(df, "Post up, %")
    if post_pct.sum() == 0:
        post_pct = _pct_col(df, "Post up %")
    is_big = df["DerivedPosition"].isin(["Forward", "Center"]) if "DerivedPosition" in df.columns else pd.Series(True, index=df.index)
    qualifies = (post36 >= 1.5) & (post_pct >= 0.40) & is_big
    post_made = per36(df, "Posts up made")
    if post_made.sum() == 0:
        post_made = per36(df, "Post up made")
    score = post_made * post_pct
    return qualifies, score


def _arch_isolation_scorer(df):
    """Creates and scores from isolation."""
    iso36 = per36(df, "Isolations attempted")
    iso_pct = _pct_col(df, "Isolation, %")
    if iso_pct.sum() == 0:
        iso_pct = _pct_col(df, "Isolation %")
    qualifies = (iso36 >= 1.5) & (iso_pct >= 0.40)
    score = per36(df, "Isolations made") * iso_pct
    return qualifies, score


def _arch_cutter(df):
    """Off-ball mover who scores on cuts."""
    cut36 = per36(df, "Cuts attempted")
    if cut36.sum() == 0:
        cut36 = per36(df, "Cuts made")
    cut_pct = _pct_col(df, "Cuts, %")
    if cut_pct.sum() == 0:
        cut_pct = _pct_col(df, "Cuts %")
    qualifies = (cut36 >= 1.0) & (cut_pct >= 0.50)
    score = per36(df, "Cuts made") * cut_pct
    return qualifies, score


def _arch_driver(df):
    """Attacks the rim off the dribble."""
    dr36 = per36(df, "Drives with shot")
    drv_pct = _pct_col(df, "Drives, %")
    if drv_pct.sum() == 0:
        drv_pct = _pct_col(df, "Drives %")
    qualifies = (dr36 >= 3.0) & (drv_pct >= 0.40)
    score = dr36 * drv_pct
    return qualifies, score


def _arch_scorer(df):
    """Volume scorer — total points per 36 with real usage."""
    pts36 = per36(df, "Points")
    usage = _pct_col(df, "Usage Percentage")
    ts = _pct_col(df, "True shooting percentage", 0.5)
    qualifies = (pts36 >= 14.0) & (usage >= 0.18)
    score = pts36 * ts
    return qualifies, score


def _arch_efficient_scorer(df):
    """Scores efficiently — TS% with real volume."""
    pts36 = per36(df, "Points")
    ts = _pct_col(df, "True shooting percentage")
    qualifies = (pts36 >= 10.0) & (ts >= 0.58)
    score = ts
    return qualifies, score


def _arch_ball_security(df):
    """Takes care of the ball at meaningful usage."""
    tov36 = per36(df, "Turnovers")
    usage = _pct_col(df, "Usage Percentage")
    qualifies = (usage >= 0.15) & (tov36 <= 1.8) & (tov36 > 0)
    score = -tov36  # lower turnovers = better
    return qualifies, score


# ----------- Defensive archetypes -----------

def _arch_steals_disruptor(df):
    """Generates turnovers via steals."""
    stl36 = per36(df, "Steals")
    qualifies = stl36 >= 1.3
    score = stl36
    return qualifies, score


def _arch_rim_protector(df):
    """Blocks shots. Restricted to Forward/Center."""
    blk36 = per36(df, "Blocks")
    is_big = df["DerivedPosition"].isin(["Forward", "Center"]) if "DerivedPosition" in df.columns else pd.Series(True, index=df.index)
    qualifies = (blk36 >= 1.0) & is_big
    score = blk36
    return qualifies, score


def _arch_defensive_playmaker(df):
    """Generates events on both fronts — steals + blocks."""
    stl36 = per36(df, "Steals")
    blk36 = per36(df, "Blocks")
    qualifies = (stl36 >= 0.9) & (blk36 >= 0.6)
    score = stl36 + blk36
    return qualifies, score


def _arch_iso_defender(df):
    """Defends isolations well."""
    opp_att = _safe_col(df, "Opp Isolations shots", 0)
    opp_pct = _pct_col(df, "Opponent Isolation shots made, %", 1.0)
    qualifies = (opp_att >= 0.3) & (opp_pct <= 0.55)
    score = -opp_pct
    return qualifies, score


def _arch_post_defender(df):
    """Defends the post."""
    opp_att = _safe_col(df, "Opp Post up shots", 0)
    opp_pct = _pct_col(df, "Opponent Post up shots made, %", 1.0)
    qualifies = (opp_att >= 0.3) & (opp_pct <= 0.60)
    score = -opp_pct
    return qualifies, score


def _arch_perimeter_defender(df):
    """Defends catch-and-shoot well."""
    opp_att = _safe_col(df, "Opp catch and shoot shots", 0)
    if opp_att.sum() == 0:
        opp_att = _safe_col(df, "Opp Catch and shoot shots", 0)
    opp_pct = _pct_col(df, "Opp Catch and shoot shots made, %", 1.0)
    qualifies = (opp_att >= 0.6) & (opp_pct <= 0.50)
    score = -opp_pct
    return qualifies, score


def _arch_drive_defender(df):
    """Holds up against drivers."""
    opp_att = _safe_col(df, "Opp Drives shots", 0)
    opp_pct = _pct_col(df, "Opponent Drives shots made, %", 1.0)
    qualifies = (opp_att >= 0.6) & (opp_pct <= 0.60)
    score = -opp_pct
    return qualifies, score

# ── Archetype + Role Tag registries ────────────────────────────────────────

ARCHETYPES = [
    # (display_name, archetype_function)
    # Shooting
    ("3-Point Shooter",      _arch_three_pt_shooter),
    ("Spot-Up Shooter",      _arch_spot_up_shooter),
    ("Movement Shooter",     _arch_movement_shooter),
    # Scoring
    ("Scorer",               _arch_scorer),
    ("Efficient Scorer",     _arch_efficient_scorer),
    ("Interior Finisher",    _arch_interior_finisher),
    ("Post Scorer",          _arch_post_scorer),
    ("Isolation Scorer",     _arch_isolation_scorer),
    ("Driver",               _arch_driver),
    ("Transition Threat",    _arch_transition_threat),
    ("Cutter",               _arch_cutter),
    # Playmaking
    ("Primary Playmaker",    _arch_playmaker),
    ("Secondary Creator",    _arch_secondary_creator),
    ("PnR Ball Handler",     _arch_pnr_ball_handler),
    ("Ball Security",        _arch_ball_security),
    # Bigs / off-ball offense
    ("PnR Roll Man",         _arch_pnr_roll_man),
    ("Pick-n-Pop Big",       _arch_pick_n_pop),
    ("Offensive Rebounder",  _arch_offensive_rebounder),
    # Defense
    ("Rim Protector",        _arch_rim_protector),
    ("Steals Disruptor",     _arch_steals_disruptor),
    ("Defensive Playmaker",  _arch_defensive_playmaker),
    ("Iso Defender",         _arch_iso_defender),
    ("Post Defender",        _arch_post_defender),
    ("Perimeter Defender",   _arch_perimeter_defender),
    ("Drive Defender",       _arch_drive_defender),
]

ROLE_TAGS = [
    ("Starter",    lambda df: _safe_col(df, "MinPerGame", _safe_col(df, "Minutes", 0)) >= 25.0),
    ("Rotation",   lambda df: (_safe_col(df, "MinPerGame", _safe_col(df, "Minutes", 0)) >= 15.0) & (_safe_col(df, "MinPerGame", _safe_col(df, "Minutes", 0)) < 25.0)),
    ("Bench",      lambda df: (_safe_col(df, "MinPerGame", _safe_col(df, "Minutes", 0)) >= 8.0) & (_safe_col(df, "MinPerGame", _safe_col(df, "Minutes", 0)) < 15.0)),
    ("High Usage", lambda df: _pct_col(df, "Usage Percentage") >= 0.25),
    ("Low Usage",  lambda df: _pct_col(df, "Usage Percentage") <= 0.15),
]


def apply_tags(df):
    """Apply v2 two-stage archetype tags — exact port of tag_v2.py logic.

    Stage 1: archetype gate — does the player DO this thing? (raw thresholds)
    Stage 2: tier — how good is he relative to peers who also qualify?
             (percentile within the qualified subset → 4 tiers)

    Returns dict {index -> list[str]} of flat tag lists like
    ["Primary Playmaker (Elite)", "Steals Disruptor (Above avg)", "Starter"]

    Sample-size floor: >= 5 games AND >= 8 min/game.
    """
    # Sample-size floor
    gp = _safe_col(df, "Games played", default=0)
    mpg = _safe_col(df, "Minutes", default=0)
    eligible = (gp >= 5) & (mpg >= 8)
    tags = {idx: [] for idx in df.index if eligible.loc[idx]}

    for name, fn in ARCHETYPES:
        try:
            qualifies, score = fn(df)
        except Exception:
            continue
        qualifies = qualifies.fillna(False) & eligible
        # Percentile within qualified subset only
        qual_scores = score[qualifies]
        if len(qual_scores) < 4:
            # Not enough qualifiers to tier — give everyone "Average"
            for idx in qual_scores.index:
                _add_tag(tags, idx, name, "Average")
            continue
        pct = qual_scores.rank(pct=True)
        for idx, p in pct.items():
            tier = tier_from_pct(p)
            if tier is not None:
                _add_tag(tags, idx, name, tier)

    # Role tags (no tiering — binary)
    for role_name, role_fn in ROLE_TAGS:
        try:
            mask = role_fn(df) & eligible
            for idx in df.index[mask.fillna(False)]:
                if idx in tags:
                    tags[idx].append(role_name)
        except Exception:
            continue

    return tags


# ── Cluster Assignment ───────────────────────────────────────────────────────

CLUSTER_PRIORITY = [
    "Defensive Playmaker",
    "Paint Protector",
    "Passing Ball-Handler",
    "Offensive Ball-Handler",
    "Shooting Ball-Handler",
    "Defensive Ball-Handler",
    "Driver",
    "Wing 3&D",
    "Big 3&D",
    "Versatile Big Man",
    "Off-Ball Playmaker",
]

CLUSTER_POSITION_GATES = {
    "Paint Protector": ["Forward", "Center"],
    "Big 3&D": ["Forward", "Center"],
    "Versatile Big Man": ["Forward", "Center"],
    "Wing 3&D": ["Guard", "Wing", "Forward"],
    "Driver": ["Guard", "Wing"],
    "Offensive Ball-Handler": ["Guard", "Wing"],
    "Defensive Ball-Handler": ["Guard", "Wing"],
    "Passing Ball-Handler": ["Guard", "Wing"],
    "Shooting Ball-Handler": ["Guard", "Wing"],
}


def assign_clusters(row):
    """Assign Major and Minor clusters based on percentile values."""
    position = row.get("Position", "Wing")
    points = _get_pct(row, "Points")
    assists = _get_pct(row, "Assists")
    rebounds = _get_pct(row, "Rebounds")
    def_reb = _get_pct(row, "Defensive rebounds")
    steals = _get_pct(row, "Steals")
    blocks = _get_pct(row, "Blocks")
    fg3_made_p40 = _get_pct(row, "3-pt field goals made_per40")
    cs_made_p40 = _get_pct(row, "Catch and shoot made_per40")
    cuts_made_p40 = _get_pct(row, "Cuts made_per40")
    cd_made_p40 = _get_pct(row, "Catch and drive made_per40")
    drives_made_p40 = _get_pct(row, "Drives made_per40") or _get_pct(row, "Drives with shot_per40")

    matches = []

    def check_gate(cluster_name):
        if cluster_name in CLUSTER_POSITION_GATES:
            return position in CLUSTER_POSITION_GATES[cluster_name]
        return True

    # Check each cluster rule
    if check_gate("Offensive Ball-Handler") and points > 0.65 and assists > 0.65:
        matches.append("Offensive Ball-Handler")
    if check_gate("Defensive Ball-Handler") and steals > 0.65 and assists > 0.65:
        matches.append("Defensive Ball-Handler")
    if check_gate("Passing Ball-Handler") and assists > 0.70 and points < 0.60:
        matches.append("Passing Ball-Handler")
    if check_gate("Shooting Ball-Handler") and points > 0.65 and fg3_made_p40 > 0.65:
        matches.append("Shooting Ball-Handler")
    if check_gate("Wing 3&D") and fg3_made_p40 > 0.65 and steals > 0.50 and assists < 0.50:
        matches.append("Wing 3&D")
    if check_gate("Big 3&D") and fg3_made_p40 > 0.50 and rebounds > 0.59:
        matches.append("Big 3&D")
    if check_gate("Paint Protector") and blocks > 0.70 and def_reb > 0.50:
        matches.append("Paint Protector")
    if cuts_made_p40 > 0.65 and cs_made_p40 > 0.65:
        matches.append("Off-Ball Playmaker")
    if check_gate("Driver") and cd_made_p40 > 0.75 and drives_made_p40 > 0.75:
        matches.append("Driver")
    if check_gate("Versatile Big Man") and assists > 0.49 and rebounds > 0.59:
        matches.append("Versatile Big Man")
    if blocks > 0.65 and steals > 0.65:
        matches.append("Defensive Playmaker")

    # Sort by priority
    matches_sorted = sorted(matches, key=lambda x: CLUSTER_PRIORITY.index(x) if x in CLUSTER_PRIORITY else 99)

    if len(matches_sorted) >= 2:
        return matches_sorted[0], matches_sorted[1]
    elif len(matches_sorted) == 1:
        # Minor from role tags (v2 flat list)
        tags_list = row.get("tags", [])
        if isinstance(tags_list, list):
            role_tags = [t for t in tags_list if t in ("Starter", "Rotation", "Bench", "High Usage", "Low Usage")]
            minor = role_tags[0] if role_tags else "Role Player"
        elif isinstance(tags_list, dict):
            pt_tags = tags_list.get("Playing Time", [])
            minor = pt_tags[0] if pt_tags else "Role Player"
        else:
            minor = "Role Player"
        return matches_sorted[0], minor
    else:
        # Fallback
        best_stat = max(
            [("Scorer", points), ("Interior Role Player", rebounds + blocks),
             ("Distributor", assists), ("Defensive Role Player", steals + blocks)],
            key=lambda x: x[1] if not pd.isna(x[1]) else 0
        )
        tags_list = row.get("tags", [])
        if isinstance(tags_list, list):
            role_tags = [t for t in tags_list if t in ("Starter", "Rotation", "Bench", "High Usage", "Low Usage")]
            minor = role_tags[0] if role_tags else "Role Player"
        elif isinstance(tags_list, dict):
            pt_tags = tags_list.get("Playing Time", [])
            minor = pt_tags[0] if pt_tags else "Role Player"
        else:
            minor = "Role Player"
        return f"{best_stat[0]} (fallback)", minor


# ── Similarity Scoring ───────────────────────────────────────────────────────

CLUSTER_PERCENTILE_KEYS = [
    "Points_pct", "Assists_pct", "Rebounds_pct", "Defensive rebounds_pct",
    "Steals_pct", "Blocks_pct", "3-pt field goals made_per40_pct",
    "Catch and shoot made_per40_pct", "Cuts made_per40_pct",
    "Catch and drive made_per40_pct", "Drives with shot_per40_pct",
]


_POS_ORDER = {"Guard": 0, "Wing": 1, "Forward": 2, "Center": 3}


def _position_similarity(pos_a, pos_b):
    """Score position similarity: same=1.0, adjacent=0.6, 2 apart=0.25, 3 apart=0.0."""
    a = _POS_ORDER.get(pos_a, 1)
    b = _POS_ORDER.get(pos_b, 1)
    gap = abs(a - b)
    return {0: 1.0, 1: 0.6, 2: 0.25, 3: 0.0}.get(gap, 0.0)


def compute_similarity(subject, candidate):
    """
    Score similarity between subject player profile and candidate.
    Returns score 0-1 (higher = more similar).

    Components:
    - Cluster match (25%)
    - Tag overlap (35%)
    - Stat distance (25%)
    - Position match (15%)
    """
    # 1. Cluster match (25%)
    cluster_score = 0
    if subject.get("major_cluster") == candidate.get("major_cluster"):
        cluster_score = 1.0
    elif subject.get("major_cluster") == candidate.get("minor_cluster"):
        cluster_score = 0.5
    elif subject.get("minor_cluster") == candidate.get("major_cluster"):
        cluster_score = 0.5

    # 2. Tag overlap (35%) — handles both v2 flat list and legacy dict
    subj_tags_raw = subject.get("tags", [])
    cand_tags_raw = candidate.get("tags", [])
    if isinstance(subj_tags_raw, list):
        # Strip tier suffixes: "Playmaker (Elite)" -> "Playmaker"
        subj_tags = set(re.sub(r'\s*\(.*?\)$', '', t) for t in subj_tags_raw)
    elif isinstance(subj_tags_raw, dict):
        subj_tags = set()
        for cat_tags in subj_tags_raw.values():
            subj_tags.update(cat_tags)
    else:
        subj_tags = set()

    if isinstance(cand_tags_raw, list):
        cand_tags = set(re.sub(r'\s*\(.*?\)$', '', t) for t in cand_tags_raw)
    elif isinstance(cand_tags_raw, dict):
        cand_tags = set()
        for cat_tags in cand_tags_raw.values():
            cand_tags.update(cat_tags)
    else:
        cand_tags = set()

    if subj_tags or cand_tags:
        overlap = len(subj_tags & cand_tags)
        total = len(subj_tags | cand_tags)
        tag_score = overlap / total if total > 0 else 0
    else:
        tag_score = 0

    # 3. Euclidean distance on cluster percentiles (25%)
    subj_vec = []
    cand_vec = []
    for key in CLUSTER_PERCENTILE_KEYS:
        sv = subject.get("percentiles", {}).get(key, 0.5)
        cv = candidate.get("percentiles", {}).get(key, 0.5)
        if pd.isna(sv):
            sv = 0.5
        if pd.isna(cv):
            cv = 0.5
        subj_vec.append(sv)
        cand_vec.append(cv)

    dist = euclidean(subj_vec, cand_vec)
    max_dist = np.sqrt(len(CLUSTER_PERCENTILE_KEYS))  # max possible distance
    distance_score = 1 - (dist / max_dist)

    # 4. Position match (15%)
    pos_score = _position_similarity(
        subject.get("position", "Wing"),
        candidate.get("position", "Wing")
    )

    total_score = (cluster_score * 0.25
                   + tag_score * 0.35
                   + distance_score * 0.25
                   + pos_score * 0.15)
    return round(total_score, 3)


# ── Pipeline Entry Points ────────────────────────────────────────────────────

def _find_player(df, player_name, team_name=None):
    """Find a player row by name, optionally filtered by team."""
    # Try exact match first
    name_col = "Player" if "Player" in df.columns else "Name" if "Name" in df.columns else None
    team_col = "Team" if "Team" in df.columns else None

    if name_col is None:
        return None, "No 'Player' or 'Name' column found in data."

    # Resolve team name with fuzzy matching
    resolved_team = None
    team_warn = None
    if team_name and team_col:
        available_teams = df[team_col].dropna().astype(str).unique().tolist()
        resolved_team, team_warn = fuzzy_match_team(team_name, available_teams)
        if not resolved_team:
            resolved_team = team_name  # fall back to original

    mask = df[name_col].astype(str).str.strip().str.lower() == player_name.strip().lower()
    if resolved_team and team_col:
        team_mask = df[team_col].astype(str).str.strip().str.lower() == resolved_team.strip().lower()
        combined = mask & team_mask
        if combined.any():
            return df[combined].iloc[0], team_warn
        # Try just name
        if mask.any():
            return df[mask].iloc[0], f"Player found but not on team '{team_name}'. Using first match."

    if mask.any():
        return df[mask].iloc[0], None

    # Fuzzy: contains
    mask_fuzzy = df[name_col].astype(str).str.contains(player_name, case=False, na=False)
    if resolved_team and team_col:
        team_mask = df[team_col].astype(str).str.contains(resolved_team, case=False, na=False)
        combined = mask_fuzzy & team_mask
        if combined.any():
            warn = "Used fuzzy name matching."
            if team_warn:
                warn = f"{team_warn} {warn}"
            return df[combined].iloc[0], warn

    if mask_fuzzy.any():
        return df[mask_fuzzy].iloc[0], "Used fuzzy name matching."

    return None, f"Player '{player_name}' not found in data."


def _build_profile(row, df, all_tags=None, all_positions=None):
    """Build a player profile dict from a processed row.

    Parameters
    ----------
    row : pd.Series – one player's row from the processed DataFrame
    df  : pd.DataFrame – full league DataFrame (used for context)
    all_tags : dict | None – {index: [tag_str, ...]} from apply_tags()
    all_positions : pd.Series | None – from derive_position()
    """
    idx = row.name if hasattr(row, 'name') else None

    # Position
    if all_positions is not None and idx is not None and idx in all_positions.index:
        position = all_positions[idx]
    else:
        position = infer_position(row)  # legacy fallback

    # Tags
    if all_tags is not None and idx is not None and idx in all_tags:
        tags = all_tags[idx]  # flat list like ["Playmaker (Elite)", "Starter"]
    else:
        tags = []

    row_with_pos = row.copy()
    row_with_pos["Position"] = position
    row_with_pos["tags"] = tags

    major, minor = assign_clusters(row_with_pos)

    # Collect percentile values
    pct_cols = {c: row.get(c, np.nan) for c in row.index if c.endswith("_pct")}

    name_col = "Player" if "Player" in row.index else "Name"
    team_col = "Team" if "Team" in row.index else None

    profile = {
        "name": row.get(name_col, "Unknown"),
        "team": row.get(team_col, "Unknown") if team_col else "Unknown",
        "position": position,
        "tags": tags,
        "major_cluster": major,
        "minor_cluster": minor,
        "percentiles": pct_cols,
        "row": row,
    }
    return profile


def run_profile(file_bytes, player_name, team_name=None, min_games=5, min_mpg=0, api_key=None):
    """
    Mode A: Run percentiles + tags + clusters for a single player.
    Returns (profile_dict, markdown_str, warnings)
    """
    warnings = []

    df = load_and_clean(file_bytes)

    # Compute per-40
    df = compute_per40(df)

    pool_size = len(df)

    # Compute percentiles
    df = compute_percentiles(df, min_games=min_games, min_mpg=min_mpg)

    # v2 tags: derive position and apply archetype tags on the full pool
    df["MinPerGame"] = df["Minutes"]
    df["DerivedPosition"] = derive_position(df)
    all_tags = apply_tags(df)
    all_positions = df["DerivedPosition"]

    # Find player
    row, warn = _find_player(df, player_name, team_name)
    if warn:
        warnings.append(warn)
    if row is None:
        name_col = "Player" if "Player" in df.columns else "Name" if "Name" in df.columns else None
        avail = df[name_col].dropna().unique()[:20].tolist() if name_col else []
        return None, "", warnings + [f"Available players: {', '.join(str(x) for x in avail)}"]

    profile = _build_profile(row, df, all_tags=all_tags, all_positions=all_positions)
    md = generate_profile_markdown(profile, df, api_key=api_key)

    return profile, md, warnings


def run_similarity(file_bytes, subject_profile, target_team, position_filter=None,
                   min_games=5, min_mpg=0, api_key=None):
    """
    Mode B: Find similar players on a target team.
    Returns (results_list, markdown_str, warnings)
    """
    warnings = []

    df = load_and_clean(file_bytes)
    league_pool_size = len(df)
    df = compute_per40(df)
    df = compute_percentiles(df, min_games=min_games, min_mpg=min_mpg)
    qualifying_pool_size = len(df)

    # v2 tags: derive position and apply archetype tags on the full pool
    df["MinPerGame"] = df["Minutes"]
    df["DerivedPosition"] = derive_position(df)
    all_tags = apply_tags(df)
    all_positions = df["DerivedPosition"]

    # Filter to target team (with fuzzy matching)
    team_col = "Team" if "Team" in df.columns else None
    matched_team_display = target_team
    if team_col:
        available_teams = df[team_col].dropna().astype(str).unique().tolist()
        matched_team, match_msg = fuzzy_match_team(target_team, available_teams)

        if matched_team:
            if match_msg:
                warnings.append(match_msg)
            matched_team_display = matched_team
            team_mask = df[team_col].astype(str) == matched_team
            df_team = df[team_mask].copy()
        else:
            # Last resort: try substring contains
            team_mask = df[team_col].astype(str).str.contains(target_team, case=False, na=False)
            df_team = df[team_mask].copy()
    else:
        df_team = df.copy()
        warnings.append("No 'Team' column found -- using full league pool.")

    if df_team.empty:
        if team_col:
            available = sorted(df[team_col].dropna().astype(str).unique().tolist())
            sample = ", ".join(available[:20])
            return [], "", warnings + [f"No players found for team '{target_team}'. Available teams include: {sample}"]
        return [], "", warnings + [f"No players found for team '{target_team}'."]

    # Build profiles for ALL roster players (before position filter)
    all_roster_profiles = []
    for idx, row in df_team.iterrows():
        cand_profile = _build_profile(row, df, all_tags=all_tags, all_positions=all_positions)
        score = compute_similarity(subject_profile, cand_profile)
        cand_profile["similarity_score"] = score
        # Add MPG
        gp = row.get("Games played", np.nan)
        mins = row.get("Minutes", np.nan)
        if pd.notna(gp) and pd.notna(mins) and gp > 0:
            cand_profile["mpg"] = round(mins / gp, 1)
        else:
            cand_profile["mpg"] = 0.0
        all_roster_profiles.append(cand_profile)

    all_roster_profiles.sort(key=lambda x: x["similarity_score"], reverse=True)

    # Apply position filter for results ranking
    if position_filter:
        pos_map = {
            "Guard": ["Guard"],
            "Wing": ["Wing", "Guard"],
            "Forward": ["Forward", "Wing"],
            "Big": ["Forward", "Center"],
            "Center": ["Center"],
        }
        allowed = pos_map.get(position_filter, [position_filter])
        filtered = [p for p in all_roster_profiles if p["position"] in allowed]
        if not filtered:
            return [], "", warnings + [f"No players on '{target_team}' match position '{position_filter}'."]
        results = filtered
    else:
        results = all_roster_profiles

    # Context for markdown
    context = {
        "league_pool_size": qualifying_pool_size,
        "roster_qualifying": len(all_roster_profiles),
        "matched_team_display": matched_team_display,
        "min_games": min_games,
        "min_mpg": min_mpg,
    }

    md = generate_similarity_markdown(
        subject_profile, results, all_roster_profiles,
        target_team, warnings, context, api_key=api_key
    )
    return results, md, warnings


# ── Markdown Generation ──────────────────────────────────────────────────────

def generate_profile_markdown(profile, df, api_key=None):
    """Generate Mode A markdown report."""
    lines = []
    prof_name = profile["name"]
    prof_team = profile["team"]
    prof_pos = profile["position"]
    lines.append(f"# Player Profile: {prof_name}")
    lines.append(f"**Team:** {prof_team}  ")
    lines.append(f"**Position (inferred):** {prof_pos}")
    major = profile["major_cluster"]
    minor = profile["minor_cluster"]
    lines.append(f"**Major Cluster:** {major} | **Minor:** {minor}")
    lines.append("")

    # Stat table
    lines.append("## Stat Table")
    lines.append("")
    lines.append("| Stat | Per 40 | Percentile |")
    lines.append("|---|---|---|")

    row = profile["row"]
    # Key stats to show
    key_stats = [
        "Points", "Assists", "Rebounds", "Offensive rebounds", "Defensive rebounds",
        "Steals", "Blocks", "Turnovers",
        "Assists to turnovers", "Assist to turnover",
        "Usage Percentage", "Usage percentage",
        "Field goals made", "Field goals attempted", "Field goals, %",
        "3-pt field goals made", "3-pt field goals attempted", "3-pt field goals, %",
        "2-pt field goals made", "2-pt field goals attempted", "2-pt field goals, %",
        "Free throws made", "Free throws attempted",
        "Catch and shoot made", "Cuts made", "Catch and drive made",
        "Drives with shot", "Screens off attempted", "Transitions made",
        "Isolations made", "Secondary Assist", "Points off assists",
    ]

    shown_stat_names = set()  # track to avoid duplicate variant names
    for stat in key_stats:
        per40_val = row.get(f"{stat}_per40", None)
        pct_val = row.get(f"{stat}_pct", row.get(f"{stat}_per40_pct", None))
        raw_val = row.get(stat, None)

        # Check if this stat has ANY data at all
        has_per40 = per40_val is not None and not (isinstance(per40_val, float) and pd.isna(per40_val))
        has_pct = pct_val is not None and not isinstance(pct_val, str) and not (isinstance(pct_val, float) and pd.isna(pct_val))
        has_raw = raw_val is not None and not (isinstance(raw_val, float) and pd.isna(raw_val))

        if not has_per40 and not has_pct and not has_raw:
            continue  # skip entirely empty stats (avoids duplicate variants)

        # Skip if we already showed a variant of this stat name
        # e.g., "Assists to turnovers" and "Assist to turnover"
        stat_key = stat.lower().replace(" ", "").replace("_", "")
        if stat_key in shown_stat_names:
            continue
        shown_stat_names.add(stat_key)

        # Format per-40 value
        if has_per40:
            display_val = round(float(per40_val), 2)
        elif has_raw and isinstance(raw_val, (int, float)):
            display_val = round(float(raw_val), 2)
        else:
            display_val = "--"

        pct_display = _pct_display(pct_val)

        if display_val != "--" or pct_display != "--":
            lines.append(f"| {stat} | {display_val} | {pct_display} |")

    lines.append("")

    # Tags
    lines.append("## Tags")
    lines.append("")
    tags_list = profile.get("tags", [])
    if isinstance(tags_list, list) and tags_list:
        # Group by category for display
        shooting = [t for t in tags_list if any(k in t for k in ["Shooter", "Spot-Up", "Movement"])]
        scoring = [t for t in tags_list if any(k in t for k in ["Scorer", "Finisher", "Driver", "Transition", "Cutter", "Isolation"])]
        playmaking = [t for t in tags_list if any(k in t for k in ["Playmaker", "Creator", "PnR Ball", "Ball Security", "Handoff", "Catch-and-Drive"])]
        bigs = [t for t in tags_list if any(k in t for k in ["Roll Man", "Pick-n-Pop", "Offensive Rebounder", "Post Scorer"])]
        defense = [t for t in tags_list if any(k in t for k in ["Rim Protector", "Steals", "Defensive", "Defender"])]
        role = [t for t in tags_list if t in ("Starter", "Rotation", "Bench", "High Usage", "Low Usage")]

        categories = [
            ("Shooting", shooting), ("Scoring", scoring), ("Playmaking", playmaking),
            ("Big Man", bigs), ("Defensive", defense), ("Role", role),
        ]
        for cat_name, cat_tags in categories:
            if cat_tags:
                lines.append(f"**{cat_name}:** {', '.join(cat_tags)}")

        # Check for uncategorized
        all_categorized = set(shooting + scoring + playmaking + bigs + defense + role)
        uncategorized = [t for t in tags_list if t not in all_categorized]
        if uncategorized:
            lines.append(f"**Other:** {', '.join(uncategorized)}")
    elif isinstance(tags_list, dict):
        # Legacy dict format fallback
        for category, tag_vals in tags_list.items():
            if tag_vals:
                lines.append(f"**{category}:** {', '.join(tag_vals)}")
        if not any(tags_list.values()):
            lines.append("*No tags assigned — player did not meet any tag thresholds.*")
    else:
        lines.append("*No tags assigned — player did not meet any archetype thresholds.*")

    lines.append("")

    # Clusters
    lines.append("## Cluster Assignment")
    lines.append("")
    prof_major = profile["major_cluster"]
    prof_minor = profile["minor_cluster"]
    lines.append(f"**Major Cluster:** {prof_major}")
    lines.append(f"**Minor Cluster:** {prof_minor}")
    lines.append("")

    # Key cluster percentiles
    lines.append("### Key Percentiles (cluster inputs)")
    lines.append("")
    lines.append("| Metric | Percentile |")
    lines.append("|---|---|")

    # Include extra keys beyond cluster inputs
    extra_pct_keys = [
        "Assists to turnovers_pct", "Assist to turnover_pct",
        "Usage Percentage_pct", "Usage percentage_pct",
        "Points off assists_pct",
    ]
    all_pct_keys = list(CLUSTER_PERCENTILE_KEYS) + extra_pct_keys

    shown_pct_labels = set()
    for key in all_pct_keys:
        val = profile["percentiles"].get(key, None)
        display = _pct_display(val)
        if display != "--":
            label = key.replace("_pct", "").replace("_per40", "")
            label_key = label.lower().replace(" ", "").replace("_", "")
            if label_key in shown_pct_labels:
                continue
            shown_pct_labels.add(label_key)
            # Bold high percentiles
            if val is not None and not pd.isna(val) and val >= 0.75:
                lines.append(f"| {label} | **{display}** |")
            else:
                lines.append(f"| {label} | {display} |")

    lines.append("")

    # Profile summary via Claude API
    if api_key:
        subj_tags = _get_all_tags_flat(profile.get("tags", {}))
        pct_summary = ", ".join(
            k.replace("_pct", "").replace("_per40", "") + ": " + _pct_display(v)
            for k, v in profile["percentiles"].items()
            if v is not None and not pd.isna(v) and k in all_pct_keys
        )
        summary_prompt = f"""You are a basketball scout. Write a 2-sentence profile summary for this player. Be specific and analytical.

Player: {prof_name}
Team: {prof_team}
Position: {prof_pos}
Major Cluster: {major}
Minor Cluster: {minor}
Tags: {', '.join(subj_tags)}
Key percentiles: {pct_summary}

Write ONLY the 2-sentence summary, no headers or formatting."""
        summary = _call_claude_narrative(api_key, summary_prompt, max_tokens=200)
        if summary:
            lines.append("## Profile Summary")
            lines.append("")
            lines.append(summary)

    return "\n".join(lines)


def _get_all_tags_flat(tags):
    """Flatten tags into a single list (handles both v2 flat list and legacy dict)."""
    if isinstance(tags, list):
        return tags
    if isinstance(tags, dict):
        out = []
        for cat_tags in tags.values():
            out.extend(cat_tags)
        return out
    return []


def _cluster_match_label(subj, cand):
    """Describe how clusters match between subject and candidate."""
    if subj.get("major_cluster") == cand.get("major_cluster"):
        return "Same Major"
    elif subj.get("major_cluster") == cand.get("minor_cluster"):
        return "Major match"
    elif subj.get("minor_cluster") == cand.get("major_cluster"):
        return "Minor match"
    return "--"


def _call_claude_narrative(api_key, prompt, max_tokens=600):
    """Call Claude API for narrative generation. Returns text or empty string on failure."""
    if not api_key:
        return ""
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()
    except Exception as e:
        return f"*(Narrative generation failed: {e})*"


def generate_similarity_markdown(subject, ranked_results, all_roster,
                                  target_team, warnings, context, api_key=None):
    """Generate Mode B markdown report matching the full PlayerLynk format."""
    lines = []
    from datetime import datetime

    # ── Header ──
    lines.append("# PlayerLynk Scouting Report -- Mode B")
    lines.append("")
    subj_name = subject["name"]
    subj_team = subject["team"]
    subj_pos = subject["position"]
    subj_major = subject["major_cluster"]
    subj_minor = subject["minor_cluster"]
    lines.append(f"## Subject: {subj_name} ({subj_team}) -> {target_team}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ── Section 1: Subject Player Reference ──
    lines.append("## 1. Subject Player Reference")
    lines.append("")
    lines.append("| Field | Detail |")
    lines.append("|---|---|")
    lines.append(f"| **Player** | {subj_name} |")
    lines.append(f"| **Current team** | {subj_team} |")
    lines.append(f"| **Position (inferred)** | {subj_pos} |")
    lines.append(f"| **Major Cluster** | {subj_major} |")
    lines.append(f"| **Minor Cluster** | {subj_minor} |")

    subj_tags = _get_all_tags_flat(subject.get("tags", {}))
    lines.append(f"| **Key Tags** | {', '.join(subj_tags) if subj_tags else 'None'} |")
    lines.append("")

    # Subject's key percentiles
    lines.append("### Subject's Key Percentiles")
    lines.append("")
    lines.append("| Metric | Percentile |")
    lines.append("|---|---|")

    # Show the cluster percentile keys plus some extra useful ones
    extra_keys = [
        "Assists to turnovers_pct", "Assist to turnover_pct",
        "Usage Percentage_pct", "Usage percentage_pct",
        "Points off assists_pct",
    ]
    shown_keys = list(CLUSTER_PERCENTILE_KEYS)
    for ek in extra_keys:
        if ek not in shown_keys:
            shown_keys.append(ek)

    for key in shown_keys:
        val = subject["percentiles"].get(key)
        display = _pct_display(val)
        if display != "--":
            label = key.replace("_pct", "").replace("_per40", "")
            # Bold high percentiles
            if val is not None and not pd.isna(val) and val >= 0.75:
                lines.append(f"| {label} | **{display}** |")
            else:
                lines.append(f"| {label} | {display} |")

    lines.append("")

    # Profile summary via Claude API
    if api_key:
        subj_key_pcts = ", ".join(
            k.replace("_pct", "").replace("_per40", "") + ": " + _pct_display(v)
            for k, v in subject["percentiles"].items()
            if v is not None and not pd.isna(v) and k in shown_keys
        )
        summary_prompt = f"""You are a basketball scout. Write a 2-sentence profile summary for this player based on their data. Be specific and analytical, not generic.

Player: {subj_name}
Team: {subj_team}
Position: {subj_pos}
Major Cluster: {subj_major}
Minor Cluster: {subj_minor}
Tags: {', '.join(subj_tags)}
Key percentiles: {subj_key_pcts}

Write ONLY the 2-sentence summary, no headers or formatting."""
        summary = _call_claude_narrative(api_key, summary_prompt, max_tokens=200)
        if summary:
            lines.append(f"**Profile summary:** {summary}")
            lines.append("")

    lines.append("---")
    lines.append("")

    # ── Section 2: Target Roster Overview ──
    matched_display = context.get("matched_team_display", target_team)
    pool_size = context.get("league_pool_size", "?")
    roster_count = context.get("roster_qualifying", len(all_roster))

    lines.append(f"## 2. Target Roster -- {matched_display}")
    lines.append("")
    lines.append(f"**League pool:** {pool_size} qualifying players")
    lines.append(f"**Roster qualifying:** {roster_count} players")
    lines.append("")

    lines.append("### Full Roster Overview")
    lines.append("")
    lines.append("| Player | Pos (inf.) | MPG | Cluster | Key Tags |")
    lines.append("|---|---|---|---|---|")

    # Sort roster by MPG descending for the overview
    roster_by_mpg = sorted(all_roster, key=lambda x: x.get("mpg", 0), reverse=True)
    for p in roster_by_mpg:
        p_name = p["name"]
        p_pos = p["position"]
        p_mpg = p.get("mpg", "--")
        p_major = p["major_cluster"]
        p_minor = p["minor_cluster"]
        ptags = _get_all_tags_flat(p.get("tags", {}))
        tag_str = ", ".join(ptags[:4]) if ptags else "--"
        cluster_str = f"{p_major} / {p_minor}"
        lines.append(f"| {p_name} | {p_pos} | {p_mpg} | {cluster_str} | {tag_str} |")

    lines.append("")
    lines.append("---")
    lines.append("")

    # ── Section 3: Playstyle Overlap Ranking ──
    lines.append("## 3. Playstyle Overlap Ranking")
    lines.append("")
    lines.append(f"Players ranked by composite similarity to {subj_name}, factoring cluster match (25%), tag overlap (35%), statistical profile distance (25%), and position match (15%).")
    lines.append("")
    lines.append("| Rank | Player | Score | Cluster Match | Tag Overlap | Stat Similarity |")
    lines.append("|---|---|---|---|---|---|")

    for i, r in enumerate(ranked_results, 1):
        cand_tags = _get_all_tags_flat(r.get("tags", {}))
        shared = set(subj_tags) & set(cand_tags)
        total_tags = set(subj_tags) | set(cand_tags)
        tag_overlap_str = f"{len(shared)}/{len(total_tags)} shared" if total_tags else "--"
        cluster_label = _cluster_match_label(subject, r)

        # Compute stat similarity component for display
        subj_vec, cand_vec = [], []
        for key in CLUSTER_PERCENTILE_KEYS:
            sv = subject["percentiles"].get(key, 0.5)
            cv = r["percentiles"].get(key, 0.5)
            subj_vec.append(sv if pd.notna(sv) else 0.5)
            cand_vec.append(cv if pd.notna(cv) else 0.5)
        dist = euclidean(subj_vec, cand_vec)
        max_dist = np.sqrt(len(CLUSTER_PERCENTILE_KEYS))
        stat_sim = round(1 - dist / max_dist, 3)

        score = r["similarity_score"]
        r_name = r["name"]
        if i <= 2:
            lines.append(f"| **{i}** | **{r_name}** | **{score}** | {cluster_label} | {tag_overlap_str} | {stat_sim} |")
        elif i <= 5:
            lines.append(f"| {i} | {r_name} | {score} | {cluster_label} | {tag_overlap_str} | {stat_sim} |")

    if len(ranked_results) > 5:
        remaining = ranked_results[5:]
        remaining_names = ", ".join(r["name"] for r in remaining[:4])
        if len(remaining) > 4:
            remaining_names += ", ..."
        cutoff_score = ranked_results[5]["similarity_score"] if ranked_results[5:] else "--"
        lines.append(f"| 6--{len(ranked_results)} | {remaining_names} | <{cutoff_score} | -- | -- | -- |")

    lines.append("")
    lines.append("---")
    lines.append("")

    # ── Section 4: Top 2 Detailed Comparison ──
    lines.append("## 4. Top 2 Matches -- Detailed Comparison")
    lines.append("")

    top2 = ranked_results[:2]
    for idx, match in enumerate(top2, 1):
        m_name = match["name"]
        m_pos = match["position"]
        m_score = match["similarity_score"]
        m_major = match["major_cluster"]
        m_minor = match["minor_cluster"]
        mpg = match.get("mpg", "--")
        lines.append(f"### #{idx}: {m_name} ({m_pos}, {mpg} mpg) -- Score: {m_score}")
        lines.append("")
        lines.append(f"**Cluster:** {m_major} / {m_minor}")
        match_tags = _get_all_tags_flat(match.get("tags", {}))
        lines.append(f"**Tags:** {', '.join(match_tags) if match_tags else 'None'}")
        lines.append("")

        # Comparison table with raw per-40 + percentile
        lines.append(f"| Stat | {m_name} | {subj_name} | Delta |")
        lines.append("|---|---|---|---|")

        compare_stats = [
            ("Points", "Points"),
            ("Assists", "Assists"),
            ("Rebounds", "Rebounds"),
            ("Steals", "Steals"),
            ("Blocks", "Blocks"),
            ("Field goals, %", "FG%"),
            ("3-pt field goals, %", "3PT%"),
            ("Free throws made", "FTM"),
        ]
        for stat_key, label in compare_stats:
            # Try per40 first for counting stats, then raw for percentages
            m_per40 = match["row"].get(f"{stat_key}_per40", np.nan)
            s_per40 = subject.get("row", pd.Series()).get(f"{stat_key}_per40", np.nan) if "row" in subject else np.nan
            m_raw = match["row"].get(stat_key, np.nan)
            s_raw = subject.get("row", pd.Series()).get(stat_key, np.nan) if "row" in subject else np.nan

            m_val = m_per40 if pd.notna(m_per40) else m_raw
            s_val = s_per40 if pd.notna(s_per40) else s_raw

            m_pct = match["percentiles"].get(f"{stat_key}_pct", match["percentiles"].get(f"{stat_key}_per40_pct", np.nan))
            s_pct = subject["percentiles"].get(f"{stat_key}_pct", subject["percentiles"].get(f"{stat_key}_per40_pct", np.nan))

            # Format values
            if pd.notna(m_val):
                if "%" in stat_key:
                    m_str = f"{m_val:.1f}%"
                else:
                    m_str = f"{m_val:.1f}"
                if pd.notna(m_pct):
                    m_str += f" ({_pct_display(m_pct)})"
            else:
                m_str = "--"

            if pd.notna(s_val):
                if "%" in stat_key:
                    s_str = f"{s_val:.1f}%"
                else:
                    s_str = f"{s_val:.1f}"
                if pd.notna(s_pct):
                    s_str += f" ({_pct_display(s_pct)})"
            else:
                s_str = "--"

            # Delta
            if pd.notna(m_val) and pd.notna(s_val):
                diff = s_val - m_val
                if abs(diff) > 5:
                    higher_label = "Subject" if diff > 0 else m_name
                    delta = f"**{higher_label} higher**"
                else:
                    delta = "Comparable"
            else:
                delta = "--"

            lines.append(f"| {label} | {m_str} | {s_str} | {delta} |")

        lines.append("")

        # Narrative via Claude API
        if api_key:
            shared_tags = set(subj_tags) & set(match_tags)
            m_team = match["team"]
            subj_stats_str = ", ".join(
                k.replace("_pct", "").replace("_per40", "") + ": " + _pct_display(v)
                for k, v in subject["percentiles"].items()
                if v is not None and not pd.isna(v) and k in CLUSTER_PERCENTILE_KEYS
            )
            match_stats_str = ", ".join(
                k.replace("_pct", "").replace("_per40", "") + ": " + _pct_display(v)
                for k, v in match["percentiles"].items()
                if v is not None and not pd.isna(v) and k in CLUSTER_PERCENTILE_KEYS
            )
            shared_tags_str = ", ".join(shared_tags) if shared_tags else "None"
            idx_label = "" if idx == 1 else idx
            narrative_prompt = f"""You are a basketball scout writing a scouting report. Write two paragraphs for this player comparison:

1. "Why he's the #{idx} match" (3-4 sentences explaining the statistical and stylistic connection)
2. "Key differences" (3-4 sentences on where they diverge)

Subject: {subj_name} ({subj_team})
  Cluster: {subj_major} / {subj_minor}
  Tags: {', '.join(subj_tags)}
  Key stats: {subj_stats_str}

Match: {m_name} ({m_team}, {m_pos}, {mpg} mpg)
  Cluster: {m_major} / {m_minor}
  Tags: {', '.join(match_tags)}
  Similarity score: {m_score}
  Shared tags: {shared_tags_str}
  Key stats: {match_stats_str}

Write ONLY the two paragraphs with bold headers "**Why he's the #{idx_label} match:**" and "**Key differences:**". Be specific with numbers. No other formatting."""
            narrative = _call_claude_narrative(api_key, narrative_prompt, max_tokens=500)
            if narrative:
                lines.append(narrative)
                lines.append("")

        if idx < len(top2):
            lines.append("---")
            lines.append("")

    lines.append("")
    lines.append("---")
    lines.append("")

    # ── Section 5: Roster Fit Analysis ──
    lines.append("## 5. Roster Fit Analysis")
    lines.append("")

    if api_key:
        roster_lines = []
        for p in roster_by_mpg:
            rp_name = p["name"]
            rp_pos = p["position"]
            rp_mpg = p.get("mpg", "?")
            rp_major = p["major_cluster"]
            rp_minor = p["minor_cluster"]
            rp_tags = ", ".join(_get_all_tags_flat(p.get("tags", {}))[:5])
            roster_lines.append(f"- {rp_name} ({rp_pos}, {rp_mpg} mpg): {rp_major} / {rp_minor}. Tags: {rp_tags}")
        roster_summary = "\n".join(roster_lines)

        top1_name = ranked_results[0]["name"] if ranked_results else "N/A"
        top1_score = ranked_results[0]["similarity_score"] if ranked_results else "N/A"
        top2_name = ranked_results[1]["name"] if len(ranked_results) > 1 else "N/A"
        top2_score = ranked_results[1]["similarity_score"] if len(ranked_results) > 1 else "N/A"
        key_pcts = ", ".join(
            k.replace("_pct", "").replace("_per40", "") + ": " + _pct_display(v)
            for k, v in subject["percentiles"].items()
            if v is not None and not pd.isna(v) and k in CLUSTER_PERCENTILE_KEYS
        )

        fit_prompt = f"""You are a basketball scout. Write a 3-paragraph roster fit analysis for a player transferring to a new team.

Subject: {subj_name} ({subj_team}, {subj_pos})
  Cluster: {subj_major} / {subj_minor}
  Tags: {', '.join(subj_tags)}
  Key percentiles: {key_pcts}

Target team: {target_team}
Current roster:
{roster_summary}

Top 2 matches: {top1_name} (score: {top1_score}), {top2_name} (score: {top2_score})

Write:
Paragraph 1: What role does the subject fill that the roster currently splits or lacks?
Paragraph 2: "What {subj_name} adds that the roster lacks:" — list 3 specific bullet points with stats
Paragraph 3: "Projected role:" — predicted minutes, lineup slot, which current players he plays alongside/replaces

Be specific. Reference actual roster players by name and cite percentile numbers. No headers -- just the three paragraphs with the bullet list in paragraph 2."""
        fit_text = _call_claude_narrative(api_key, fit_prompt, max_tokens=600)
        if fit_text:
            lines.append(fit_text)
        else:
            lines.append(f"*{subj_name} fills a role on {target_team} based on the similarity analysis above. See the ranking table for detailed overlap.*")
    else:
      lines.append(f"*Add an Anthropic API key to generate a detailed roster fit analysis for {subj_name} on {target_team}.*")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(f"*Report generated from league data. Percentiles computed across {pool_size} qualifying players. Positions inferred from statistical profiles.*")

    return "\n".join(lines)
