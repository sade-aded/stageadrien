import pandas as pd
import numpy as np


def safe_float(v):
    """Convert cell value to float, handling '-' and NaN."""
    if pd.isna(v) or v == '-' or v == '':
        return 0.0
    try:
        return float(v)
    except (ValueError, TypeError):
        return 0.0


# Column mappings for offense playtypes
OFFENSE_PLAYTYPES = {
    "Pick'n'rolls Handler": {
        "made": "PnR Handlers made",
        "attempted": "PnR Handlers attempted",
        "pct": "PR Handler, %",
    },
    "Isolation": {
        "made": "Isolations made",
        "attempted": "Isolations attempted",
        "pct": "Isolation, %",
    },
    "Transitions": {
        "made": "Transitions made",
        "attempted": "Transitions attempted",
        "pct": "Transition attacks, %",
    },
    "Catch and shoots": {
        "made": "Catch and shoot made",
        "attempted": "Catch and shoot attempted",
        "pct": "Catch and shoot shots made, %",
    },
    "Catch and drives": {
        "made": "Catch and drive made",
        "attempted": "Catch and drive attempted",
        "pct": "Catch and drive shots made, %",
    },
    "Pick'n'rolls Roller": {
        "made": "PnR Rollers made",
        "attempted": "PnR Rollers attempted",
        "pct": "PR Roller, %",
    },
    "Hand offs": {
        "made": "Hand off made",
        "attempted": "Hand off attempted",
        "pct": "Hand off, %",
    },
    "Screen offs": {
        "made": "Screens off made",
        "attempted": "Screens off attempted",
        "pct": "Screens off, %",
    },
    "Cuts": {
        "made": "Cuts made",
        "attempted": "Cuts attempted",
        "pct": "Cuts, %",
    },
    "Pick'n'pops": {
        "made": "PnP made",
        "attempted": "PnP attempted",
        "pct": "Pick-n-pops, %",
    },
    "Post ups": {
        "made": "Posts up made",
        "attempted": "Posts up attempted",
        "pct": "Post up, %",
    },
}

# Column mappings for defense playtypes
DEFENSE_PLAYTYPES = {
    "Pick'n'rolls Handler": {
        "made": "Opp Pick-n-roll shots made",
        "attempted": "Opp Pick-n-roll shots",
        "pct": "Opponent Pick-n-roll shots made, %",
    },
    "Catch and shoots": {
        "made": "Opp catch and shoot shots made",
        "attempted": "Opp catch and shoot shots",
        "pct": "Opp Catch and shoot shots made, %",
    },
    "Hand offs": {
        "made": "Opp Hand off shots made",
        "attempted": "Opp Hand off shots",
        "pct": "Opponent Hand off shots made, %",
    },
    "Transitions": {
        "made": "Opp Transition shots made",
        "attempted": "Opp Transition shots",
        "pct": "Opponent Transition shots made, %",
    },
    "Catch and drives": {
        "made": "Opp catch and drive shots made",
        "attempted": "Opp Catch and drive shots",
        "pct": "Opp Catch and drive shots made, %",
    },
    "Screen offs": {
        "made": "Opp Screens off shots made",
        "attempted": "Opp Screens off shots",
        "pct": "Opponent Screens off shots made, %",
    },
    "Cuts": {
        "made": "Opp Cuts shots made",
        "attempted": "Opp Cuts shots",
        "pct": "Opponent Cuts shots made, %",
    },
    "Isolation": {
        "made": "Opp Isolations shots made",
        "attempted": "Opp Isolations shots",
        "pct": "Opponent Isolation shots made, %",
    },
    "Pick'n'pops": {
        "made": "Opp Pick-n-Pop shots made",
        "attempted": "Opp Pick-n-Pop shots",
        "pct": "Opponent Pick-n-Pop shots made, %",
    },
    "Post ups": {
        "made": "Opp Post up shots made",
        "attempted": "Opp Post up shots",
        "pct": "Opponent Post up shots made, %",
    },
}


def _fuzzy_find_player(df, player_name):
    """
    Find a player row in a DataFrame using fuzzy matching.
    Returns the matched row (Series) or raises ValueError.
    """
    from difflib import SequenceMatcher

    search_lower = player_name.lower().strip()
    search_parts = search_lower.split()
    best_score, best_idx = 0, None

    for i, p in enumerate(df["Player"]):
        p_lower = str(p).lower().strip()
        p_parts = p_lower.split()
        score = 0

        # Full containment
        if search_lower in p_lower or p_lower in search_lower:
            score = max(score, 80)
        # All words present
        if all(sw in p_parts for sw in search_parts):
            score = max(score, 85)
        # First + last name match
        if len(search_parts) >= 2 and len(p_parts) >= 2:
            if search_parts[0] == p_parts[0] and search_parts[-1] == p_parts[-1]:
                score = max(score, 90)
        # Sequence similarity
        ratio = SequenceMatcher(None, search_lower, p_lower).ratio()
        score = max(score, int(ratio * 100))

        if score > best_score:
            best_score, best_idx = score, i

    if best_score >= 55 and best_idx is not None:
        return df.iloc[best_idx]

    raise ValueError(f"Player '{player_name}' not found. Available: {df['Player'].tolist()}")


def extract_playtypes(df: pd.DataFrame, player_name: str) -> dict:
    """
    Extract playtype breakdown for a player from an InStat-exported Excel file.

    Args:
        df: DataFrame from the Players Excel export (box score sheet)
        player_name: name as it appears in the 'Player' column (fuzzy matched)

    Returns:
        dict with keys 'offense' and 'defense', each containing a list of
        playtype dicts sorted by play share descending. Each dict has:
            - name: playtype label
            - ps: play share % (based on FGA)
            - fga: field goal attempts per game
            - fgm: field goals made per game
            - fg_pct: field goal percentage (string like '45.1%')
            - est_pts: estimated points per game from this playtype
            - est_pppp: estimated points per play (using FGA as proxy for possessions)
    """
    # Try exact match first, then fuzzy
    player_row = df[df["Player"] == player_name]
    if player_row.empty:
        row = _fuzzy_find_player(df, player_name)
    else:
        row = player_row.iloc[0]

    total_fga = safe_float(row["Field goals attempted"])
    fg2m = safe_float(row["2-pt field goals made"])
    fg3m = safe_float(row["3-pt field goals made"])
    avg_pts_per_fgm = (fg2m * 2 + fg3m * 3) / safe_float(row["Field goals made"]) if safe_float(row["Field goals made"]) > 0 else 2.0

    def build_side(playtype_map, total_attempts):
        results = []
        for name, cols in playtype_map.items():
            fgm = safe_float(row.get(cols["made"], 0))
            fga = safe_float(row.get(cols["attempted"], 0))

            raw_pct = row.get(cols["pct"], None)
            if pd.notna(raw_pct) and raw_pct != '-':
                fg_pct_str = str(raw_pct) if '%' in str(raw_pct) else f"{safe_float(raw_pct):.1f}%"
                fg_pct_val = safe_float(str(raw_pct).replace('%', ''))
            elif fga > 0:
                fg_pct_val = round(fgm / fga * 100, 1)
                fg_pct_str = f"{fg_pct_val}%"
            else:
                fg_pct_val = 0.0
                fg_pct_str = "-"

            ps = round(fga / total_attempts * 100, 1) if total_attempts > 0 else 0.0
            est_pts = round(fgm * avg_pts_per_fgm, 1) if playtype_map is OFFENSE_PLAYTYPES else round(fgm * 2.0, 1)
            est_pppp = round(est_pts / fga, 2) if fga > 0 else 0.0

            results.append({
                "name": name,
                "ps": ps,
                "fga": fga,
                "fgm": fgm,
                "fg_pct": fg_pct_str,
                "fg_pct_val": fg_pct_val,
                "est_pts": est_pts,
                "est_pppp": est_pppp,
            })
        results.sort(key=lambda x: x["ps"], reverse=True)
        return results

    # Defense total attempts
    def_total = sum(
        safe_float(row.get(cols["attempted"], 0))
        for cols in DEFENSE_PLAYTYPES.values()
    )

    return {
        "player": row.get("Player", player_name),
        "games": safe_float(row.get("Games played", 0)),
        "total_fga": total_fga,
        "avg_pts_per_fgm": round(avg_pts_per_fgm, 2),
        "offense": build_side(OFFENSE_PLAYTYPES, total_fga),
        "defense": build_side(DEFENSE_PLAYTYPES, def_total),
    }


def playtypes_to_dataframes(data: dict) -> tuple:
    """
    Convert the extract_playtypes output to two DataFrames (offense, defense)
    ready for st.dataframe() display.
    """
    def to_df(side_data):
        rows = []
        for pt in side_data:
            rows.append({
                "Play type": pt["name"],
                "PS": f"{pt['ps']}%",
                "FGA": pt["fga"],
                "FGM": pt["fgm"],
                "FG%": pt["fg_pct"],
                "~PTS": pt["est_pts"],
                "~PPPP": pt["est_pppp"],
            })
        return pd.DataFrame(rows)

    return to_df(data["offense"]), to_df(data["defense"])


# ── Usage example ──────────────────────────────────────────────────
if __name__ == "__main__":
    df = pd.read_excel("Players_-_Brose_Bamberg__22-May-2026.xlsx")
    data = extract_playtypes(df, "Cobe Williams")

    print(f"\n{'='*60}")
    print(f"  PLAYTYPE BREAKDOWN — {data['player']}")
    print(f"  {int(data['games'])} games | {data['total_fga']} FGA/g | ~{data['avg_pts_per_fgm']} pts per make")
    print(f"{'='*60}")

    for side in ["offense", "defense"]:
        print(f"\n  {side.upper()}")
        print(f"  {'Play type':<22} {'PS':>6} {'FGA':>5} {'FGM':>5} {'FG%':>7} {'~PTS':>6} {'~PPPP':>6}")
        print(f"  {'-'*58}")
        for pt in data[side]:
            print(f"  {pt['name']:<22} {pt['ps']:>5.1f}% {pt['fga']:>5.1f} {pt['fgm']:>5.1f} {pt['fg_pct']:>7} {pt['est_pts']:>6.1f} {pt['est_pppp']:>6.2f}")

    # Streamlit usage:
    # off_df, def_df = playtypes_to_dataframes(data)
    # st.subheader("Offense")
    # st.dataframe(off_df, hide_index=True)
    # st.subheader("Defense")
    # st.dataframe(def_df, hide_index=True)
