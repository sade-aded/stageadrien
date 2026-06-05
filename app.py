"""
PlayerLynk Report Tool — Wizard-Style Streamlit App
Single-page guided flow: upload → auto-chain P1 → P2A → P2B → P3 → combined report.
"""

import streamlit as st
import os
import sys
from datetime import datetime

# Force fresh imports (avoid stale .pyc cache)
for mod_name in list(sys.modules.keys()):
    if mod_name.startswith(("pipeline_", "extract_")):
        del sys.modules[mod_name]

# ── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="PlayerLynk Report Tool",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .stApp {
        max-width: 1100px;
        margin: 0 auto;
    }
    .pipeline-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
    }
    .pipeline-header h1 {
        color: white !important;
        margin-bottom: 0.3rem;
    }
    .pipeline-header p {
        color: #b0b0b0;
        margin: 0;
    }
    .step-card {
        background: #f0f4ff;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
        border-left: 5px solid #4285f4;
    }
    .step-done {
        border-left-color: #34a853;
        background: #f0faf0;
    }
    .step-active {
        border-left-color: #fbbc04;
        background: #fffbf0;
    }
    .step-waiting {
        border-left-color: #ccc;
        background: #fafafa;
        opacity: 0.6;
    }
    .ai-message {
        background: linear-gradient(135deg, #e8eaf6 0%, #f3e5f5 100%);
        border-radius: 10px;
        padding: 1rem 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #7c4dff;
    }
    div[data-testid="stSidebar"] {
        background: #1a1a2e;
    }
    div[data-testid="stSidebar"] .stMarkdown h1,
    div[data-testid="stSidebar"] .stMarkdown h2,
    div[data-testid="stSidebar"] .stMarkdown h3,
    div[data-testid="stSidebar"] .stMarkdown p,
    div[data-testid="stSidebar"] .stMarkdown span {
        color: white !important;
    }
    .similar-player-card {
        background: white;
        border: 2px solid #4285f4;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.08);
    }
</style>
""", unsafe_allow_html=True)


# ── Session state init ───────────────────────────────────────────────────────

DEFAULTS = {
    "wizard_step": 1,           # 1=input, 2=running, 3=playtype upload, 4=done
    "p1_md": None,
    "p1_duo_df": None,
    "p2a_md": None,
    "p2a_profile": None,
    "p2b_md": None,
    "p2b_results": None,
    "p3_md": None,
    "p5_md": None,
    "p6_md": None,
    "all_warnings": [],
    "similar_players": [],
    "run_complete_12ab": False,
    "combined_md": None,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── Playtype helpers ────────────────────────────────────────────────────────

def _playtypes_to_md(data, label):
    """Convert extract_playtypes output to markdown tables."""
    import pandas as pd
    lines = []
    lines.append(f"## {label}")
    lines.append("")
    lines.append(f"*{int(data['games'])} games | {data['total_fga']} FGA/g*")
    lines.append("")

    for side_name, side_key in [("Offense", "offense"), ("Defense", "defense")]:
        top6 = data[side_key][:6]
        if not top6:
            continue
        lines.append(f"### {side_name} (Top 6)")
        lines.append("")
        lines.append("| Playtype | PS | FGA | FGM | FG% | ~PTS | ~PPPP |")
        lines.append("|---|---|---|---|---|---|---|")
        for pt in top6:
            lines.append(f"| {pt['name']} | {pt['ps']}% | {pt['fga']} | {pt['fgm']} | {pt['fg_pct']} | {pt['est_pts']} | {pt['est_pppp']} |")
        lines.append("")

    return "\n".join(lines)


def _match_player_in_df(search_name, available_players):
    """
    Fuzzy-match a player name against available players in a DataFrame.
    Uses a scoring approach: exact > full-name-match > first+last combo > difflib.
    Never returns a random player — requires minimum quality match.
    """
    if not search_name or not available_players:
        return None

    from difflib import SequenceMatcher

    search_lower = search_name.lower().strip()
    search_parts = search_lower.split()

    # 1) Exact match
    for p in available_players:
        if str(p).lower().strip() == search_lower:
            return p

    # 2) Score all candidates and pick the best
    #    Build a score for each player: higher = better match
    candidates = []
    for p in available_players:
        p_lower = str(p).lower().strip()
        p_parts = p_lower.split()
        score = 0

        # Full name contained in either direction
        if search_lower in p_lower or p_lower in search_lower:
            score = max(score, 80)

        # Check if ALL search words appear in the player name
        if all(sw in p_parts for sw in search_parts):
            score = max(score, 85)

        # First AND last name match (both must match)
        if len(search_parts) >= 2 and len(p_parts) >= 2:
            first_match = search_parts[0] == p_parts[0]
            last_match = search_parts[-1] == p_parts[-1]
            if first_match and last_match:
                score = max(score, 90)
            elif last_match and search_parts[0][0] == p_parts[0][0]:
                # Last name matches + first initial matches (e.g. "Z. Diallo" vs "Zoom Diallo")
                score = max(score, 75)
            elif last_match:
                # Only last name matches — weak signal, many false positives
                score = max(score, 40)

        # SequenceMatcher ratio (more reliable than get_close_matches for scoring)
        ratio = SequenceMatcher(None, search_lower, p_lower).ratio()
        ratio_score = int(ratio * 100)
        score = max(score, ratio_score)

        if score >= 50:  # minimum threshold to even consider
            candidates.append((score, p))

    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        best_score, best_match = candidates[0]
        # Only return if score is reasonably high
        if best_score >= 55:
            return best_match

    return None


# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("# 🏀 PlayerLynk")
    st.markdown("### Report Tool")
    st.markdown("---")

    mode = st.radio(
        "Mode",
        ["🧙 Wizard (All-in-One)", "🔧 Single Pipeline"],
        index=0,
    )

    if mode == "🔧 Single Pipeline":
        single_pipeline = st.radio(
            "Select Pipeline",
            [
                "1 — On/Off + Duo Stats",
                "2A — Player Profile",
                "2B — Similar Players",
                "3 — Playtype Extraction",
            ],
            index=0,
        )
    else:
        single_pipeline = None

    st.markdown("---")

    api_key = st.text_input(
        "Anthropic API Key",
        type="password",
        value=os.environ.get("ANTHROPIC_API_KEY", ""),
        help="Required for Playtype Extraction (P3) and narrative analysis (P2B).",
    )

    st.markdown("---")
    if st.button("🔄 Reset Wizard", use_container_width=True):
        for k, v in DEFAULTS.items():
            st.session_state[k] = v
        st.rerun()

    st.markdown(
        "<small style='color:#666'>Built for PlayerLynk Analytics</small>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  WIZARD MODE
# ══════════════════════════════════════════════════════════════════════════════

if mode == "🧙 Wizard (All-in-One)":

    # ── Header ───────────────────────────────────────────────────────────────

    st.markdown("""
    <div class="pipeline-header">
        <h1>🧙 Scouting Report Wizard</h1>
        <p>Upload your files → We run On/Off, Profile, and Similar Players automatically →
        Upload playtype .md file → Get one combined scouting report</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Progress indicator ───────────────────────────────────────────────────

    step = st.session_state["wizard_step"]
    steps_labels = ["📋 Input", "⚙️ Analysis", "📸 Playtypes", "✅ Report"]
    cols = st.columns(4)
    for i, (col, label) in enumerate(zip(cols, steps_labels), 1):
        if i < step:
            col.markdown(f"<div class='step-card step-done'>✅ {label}</div>", unsafe_allow_html=True)
        elif i == step:
            col.markdown(f"<div class='step-card step-active'>👉 {label}</div>", unsafe_allow_html=True)
        else:
            col.markdown(f"<div class='step-card step-waiting'>⏳ {label}</div>", unsafe_allow_html=True)

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    #  STEP 1 — INPUT
    # ══════════════════════════════════════════════════════════════════════════

    if step == 1:
        st.markdown("### Step 1 — Upload Files & Player Info")

        st.markdown("""
        <div class="ai-message">
            <strong>🤖 Hey!</strong> Upload your data files and fill in the player details below.
            I'll run the full analysis pipeline for you — On/Off impact, player profile,
            and find similar players on the target team, all in one shot.<br><br>
            <strong>3 files:</strong> Lineup (P1) + Subject's league (P2A) + Target league (P2B).
            If any are the same file, just check the boxes below.
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            lineup_file = st.file_uploader(
                "📊 Lineup File (.xlsx)",
                type=["xlsx", "xls"],
                key="wiz_lineup_file",
                help="For Pipeline 1 (On/Off + Duo Stats)",
            )

        with col2:
            league_file = st.file_uploader(
                "📊 Subject's League File (.xlsx)",
                type=["xlsx", "xls"],
                key="wiz_league_file",
                help="For Pipeline 2A (Player Profile — subject's league box score)",
            )

        with col3:
            target_league_file = st.file_uploader(
                "📊 Target League File (.xlsx)",
                type=["xlsx", "xls"],
                key="wiz_target_league_file",
                help="For Pipeline 2B (Similar Players — target team's league box score)",
            )

        st.markdown("#### Optional: Projections File")
        projections_file = st.file_uploader(
            "📈 Projections File (.xlsx) — optional",
            type=["xlsx", "xls"],
            key="wiz_projections_file",
            help="For Transfer Comps — league transition comparison (e.g. Germany_to_Italy_Projections.xlsx)",
        )

        st.markdown("#### Optional: Players Files (for Playtype Extraction)")
        st.markdown("""
        <div class="ai-message">
            <strong>🆕 New!</strong> Upload InStat "Players" exports to auto-extract playtypes —
            no screenshots needed. One file for the subject's team, one for the target team.
        </div>
        """, unsafe_allow_html=True)
        pt_col1, pt_col2 = st.columns(2)
        with pt_col1:
            subject_players_file = st.file_uploader(
                "📋 Subject Team Players (.xlsx) — optional",
                type=["xlsx", "xls"],
                key="wiz_subject_players_file",
                help="InStat Players export for the subject's team (e.g. 'Players - Brose Bamberg.xlsx')",
            )
        with pt_col2:
            target_players_file = st.file_uploader(
                "📋 Target Team Players (.xlsx) — optional",
                type=["xlsx", "xls"],
                key="wiz_target_players_file",
                help="InStat Players export for the target team (e.g. 'Players - Derthona.xlsx')",
            )

        # Shortcut checkboxes
        chk_col1, chk_col2 = st.columns(2)
        with chk_col1:
            use_same_p1_p2a = st.checkbox(
                "Lineup file = Subject's league file",
                value=False,
                key="wiz_same_p1_p2a",
                help="Check if the lineup file doubles as the subject's league box score",
            )
        with chk_col2:
            use_same_p2a_p2b = st.checkbox(
                "Subject's league = Target league (same league)",
                value=False,
                key="wiz_same_p2a_p2b",
                help="Check if subject and target team are in the same league",
            )

        st.markdown("#### Player & Team Details")
        col1, col2, col3 = st.columns(3)

        with col1:
            player_name = st.text_input("Player Name", placeholder="e.g. D. Mintz", key="wiz_player")
            team_tag = st.text_input("Team Tag (for lineups)", placeholder="e.g. WB, MIL", key="wiz_tag")

        with col2:
            team_name = st.text_input("Team Full Name", placeholder="e.g. Olimpia Milano", key="wiz_team_full")
            target_team = st.text_input("Target Team (for comps)", placeholder="e.g. St. John's", key="wiz_target")

        with col3:
            position_filter = st.selectbox(
                "Position Filter (2B)",
                ["None", "Guard", "Wing", "Forward", "Big", "Center"],
                key="wiz_pos",
            )

        st.markdown("#### Advanced Settings")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            min_minutes = st.number_input("Min Player Mins (P1)", value=100, min_value=0, step=10, key="wiz_min_min")
        with col2:
            min_duo_minutes = st.number_input("Min Duo Mins (P1)", value=100, min_value=0, step=10, key="wiz_duo_min")
        with col3:
            min_games = st.number_input("Min Games (P2)", value=5, min_value=1, step=1, key="wiz_min_games")
        with col4:
            min_mpg = st.number_input("Min MPG (P2)", value=0, min_value=0, step=1, key="wiz_min_mpg")

        st.markdown("---")

        if st.button("🚀 Launch Full Analysis", type="primary", use_container_width=True):
            # Resolve file shortcuts
            effective_lineup = lineup_file
            effective_league = league_file or (lineup_file if use_same_p1_p2a else None)
            effective_target = target_league_file or (effective_league if use_same_p2a_p2b else None)

            # Also handle: if lineup not provided but league is and same checkbox
            if not effective_lineup and use_same_p1_p2a and league_file:
                effective_lineup = league_file

            # Validate
            errors = []
            if not player_name:
                errors.append("Player Name is required.")
            if not team_tag:
                errors.append("Team Tag is required.")
            if not target_team:
                errors.append("Target Team is required.")
            if not effective_lineup and not effective_league:
                errors.append("Please upload at least one data file.")
            if not effective_target and target_team:
                errors.append("Target league file is required for Pipeline 2B. Upload it or check 'same league'.")

            if errors:
                for e in errors:
                    st.error(e)
            else:
                st.session_state["wizard_step"] = 2
                # Store file bytes in session (files disappear on rerun)
                if effective_lineup:
                    st.session_state["_lineup_bytes"] = effective_lineup.getvalue()
                if effective_league:
                    st.session_state["_league_bytes"] = effective_league.getvalue()
                if effective_target:
                    st.session_state["_target_league_bytes"] = effective_target.getvalue()
                if projections_file:
                    st.session_state["_projections_bytes"] = projections_file.getvalue()
                else:
                    st.session_state["_projections_bytes"] = None

                # Store Players files for playtype extraction
                if subject_players_file:
                    st.session_state["_subject_players_bytes"] = subject_players_file.getvalue()
                else:
                    st.session_state["_subject_players_bytes"] = None
                if target_players_file:
                    st.session_state["_target_players_bytes"] = target_players_file.getvalue()
                else:
                    st.session_state["_target_players_bytes"] = None

                # Store params
                st.session_state["_player_name"] = player_name
                st.session_state["_team_tag"] = team_tag
                st.session_state["_team_full"] = team_name or team_tag
                st.session_state["_target_team"] = target_team
                st.session_state["_pos_filter"] = position_filter if position_filter != "None" else None
                st.session_state["_min_minutes"] = min_minutes
                st.session_state["_min_duo"] = min_duo_minutes
                st.session_state["_min_games"] = min_games
                st.session_state["_min_mpg"] = min_mpg
                st.rerun()


    # ══════════════════════════════════════════════════════════════════════════
    #  STEP 2 — AUTO-RUN PIPELINES 1 + 2A + 2B
    # ══════════════════════════════════════════════════════════════════════════

    elif step == 2:
        st.markdown("### Step 2 — Running Analysis Pipelines")

        player_name = st.session_state["_player_name"]
        target_team = st.session_state["_target_team"]

        # Only run pipelines once — guard with run_complete_12ab
        if not st.session_state.get("run_complete_12ab"):
            team_tag = st.session_state["_team_tag"]
            team_full = st.session_state["_team_full"]
            pos_filter = st.session_state["_pos_filter"]
            min_minutes = st.session_state["_min_minutes"]
            min_duo = st.session_state["_min_duo"]
            min_games = st.session_state["_min_games"]
            min_mpg = st.session_state["_min_mpg"]

            lineup_bytes = st.session_state.get("_lineup_bytes")
            league_bytes = st.session_state.get("_league_bytes")

            all_warnings = []
            progress = st.progress(0, text="Starting analysis...")

            # ── Pipeline 1: On/Off ───────────────────────────────────────────

            if lineup_bytes:
                progress.progress(5, text="🏀 Running Pipeline 1 — On/Off + Duo Stats...")

                try:
                    from pipeline_onoff import run_onoff_pipeline

                    df_onoff, duo_df, p1_md, p1_warnings = run_onoff_pipeline(
                        lineup_bytes, player_name, team_tag,
                        min_minutes=min_minutes, min_duo_minutes=min_duo,
                    )
                    all_warnings.extend(p1_warnings)
                    st.session_state["p1_md"] = p1_md
                    st.session_state["p1_duo_df"] = duo_df
                except Exception as e:
                    all_warnings.append(f"Pipeline 1 error: {str(e)}")
                    st.session_state["p1_md"] = None
                    st.session_state["p1_duo_df"] = None

                progress.progress(30, text="✅ Pipeline 1 complete")
            else:
                st.session_state["p1_md"] = None
                st.session_state["p1_duo_df"] = None
                progress.progress(30, text="⏭️ Pipeline 1 skipped (no lineup file)")

            # ── Pipeline 2A: Profile ─────────────────────────────────────────

            if league_bytes:
                progress.progress(35, text="📊 Running Pipeline 2A — Player Profile...")

                try:
                    from pipeline_percentiles import run_profile

                    profile, p2a_md, p2a_warnings = run_profile(
                        league_bytes, player_name,
                        team_name=team_full if team_full else None,
                        min_games=min_games, min_mpg=min_mpg,
                        api_key=api_key if api_key else None,
                    )
                    all_warnings.extend(p2a_warnings)
                    st.session_state["p2a_md"] = p2a_md
                    st.session_state["p2a_profile"] = profile
                except Exception as e:
                    all_warnings.append(f"Pipeline 2A error: {str(e)}")
                    st.session_state["p2a_md"] = None
                    st.session_state["p2a_profile"] = None

                progress.progress(55, text="✅ Pipeline 2A complete")
            else:
                st.session_state["p2a_md"] = None
                st.session_state["p2a_profile"] = None
                progress.progress(55, text="⏭️ Pipeline 2A skipped (no league file)")

            # ── Pipeline 2B: Similar Players ─────────────────────────────────

            profile = st.session_state.get("p2a_profile")
            target_league_bytes = st.session_state.get("_target_league_bytes")

            if target_league_bytes and profile and target_team:
                progress.progress(60, text="🔍 Running Pipeline 2B — Finding Similar Players...")

                try:
                    from pipeline_percentiles import run_similarity

                    results, p2b_md, p2b_warnings = run_similarity(
                        target_league_bytes, profile, target_team,
                        position_filter=pos_filter,
                        min_games=min_games, min_mpg=min_mpg,
                        api_key=api_key if api_key else None,
                    )
                    all_warnings.extend(p2b_warnings)
                    st.session_state["p2b_md"] = p2b_md
                    st.session_state["p2b_results"] = results

                    # Extract top 2 similar player names
                    if results:
                        similar_names = [r.get("name", r.get("Player", "")) for r in results[:2]]
                        st.session_state["similar_players"] = similar_names
                    else:
                        st.session_state["similar_players"] = []

                except Exception as e:
                    all_warnings.append(f"Pipeline 2B error: {str(e)}")
                    st.session_state["p2b_md"] = None
                    st.session_state["p2b_results"] = None
                    st.session_state["similar_players"] = []

                progress.progress(90, text="✅ Pipeline 2B complete")
            elif not profile:
                all_warnings.append("Pipeline 2B skipped: no profile from 2A.")
                progress.progress(90, text="⏭️ Pipeline 2B skipped (no profile)")
            elif not target_league_bytes:
                all_warnings.append("Pipeline 2B skipped: no target league file.")
                progress.progress(90, text="⏭️ Pipeline 2B skipped (no target league file)")
            else:
                progress.progress(90, text="⏭️ Pipeline 2B skipped")

            # ── Pipeline Extras: Transfer Comps + Fit Narrative ────────────

            projections_bytes = st.session_state.get("_projections_bytes")
            duo_df = st.session_state.get("p1_duo_df")
            profile = st.session_state.get("p2a_profile")
            p2b_results = st.session_state.get("p2b_results")

            if projections_bytes:
                progress.progress(92, text="📈 Extracting Transfer Comps...")
                try:
                    from pipeline_extras import parse_transfer_comps
                    p5_md = parse_transfer_comps(projections_bytes, player_name, max_comps=3)
                    st.session_state["p5_md"] = p5_md if p5_md else None
                except Exception as e:
                    all_warnings.append(f"Transfer Comps error: {str(e)}")
                    st.session_state["p5_md"] = None

            if duo_df is not None:
                progress.progress(95, text="📝 Generating Fit Narrative...")
                try:
                    from pipeline_extras import generate_fit_narrative
                    p6_md = generate_fit_narrative(duo_df, profile, p2b_results, player_name, api_key=api_key)
                    st.session_state["p6_md"] = p6_md if p6_md else None
                except Exception as e:
                    all_warnings.append(f"Fit Narrative error: {str(e)}")
                    st.session_state["p6_md"] = None

            # ── Pipeline 3 Auto: Playtype Extraction from Excel ─────────────

            subject_players_bytes = st.session_state.get("_subject_players_bytes")
            target_players_bytes = st.session_state.get("_target_players_bytes")
            similar_names = st.session_state.get("similar_players", [])

            if subject_players_bytes or target_players_bytes:
                progress.progress(97, text="📊 Extracting Playtypes from Excel...")
                try:
                    from extract_playtypes import extract_playtypes
                    import io as _io
                    import pandas as pd

                    p3_sections = []

                    # Extract for subject player from SUBJECT Players file
                    # Also try target file as fallback (user may upload only one file)
                    subject_extracted = False
                    if subject_players_bytes:
                        subj_df = pd.read_excel(_io.BytesIO(subject_players_bytes))
                        try:
                            subj_data = extract_playtypes(subj_df, player_name)
                            subj_md = _playtypes_to_md(subj_data, f"{subj_data['player']} (Primary)")
                            p3_sections.append(subj_md)
                            subject_extracted = True
                        except ValueError as ve:
                            all_warnings.append(f"Playtype extraction (subject file): {ve}")

                    # Fallback: if no subject file but target file exists, try there
                    if not subject_extracted and target_players_bytes:
                        tgt_df_check = pd.read_excel(_io.BytesIO(target_players_bytes))
                        try:
                            subj_data = extract_playtypes(tgt_df_check, player_name)
                            subj_md = _playtypes_to_md(subj_data, f"{subj_data['player']} (Primary)")
                            p3_sections.append(subj_md)
                            subject_extracted = True
                        except ValueError:
                            pass  # Player not in target file either — that's expected

                    if not subject_extracted:
                        all_warnings.append(f"Could not extract playtypes for '{player_name}'. Upload a Players file for their team.")

                    # Extract for similar players from target team file
                    if target_players_bytes and similar_names:
                        tgt_df = pd.read_excel(_io.BytesIO(target_players_bytes))
                        available = tgt_df["Player"].tolist() if "Player" in tgt_df.columns else []

                        for sim_name in similar_names[:2]:
                            # Skip if sim_name is the subject player (already extracted above)
                            if sim_name.lower().strip() == player_name.lower().strip():
                                continue
                            matched = _match_player_in_df(sim_name, available)
                            if matched:
                                try:
                                    sim_data = extract_playtypes(tgt_df, matched)
                                    target_label = st.session_state.get("_target_team", "Target")
                                    sim_md = _playtypes_to_md(sim_data, f"{matched} ({target_label})")
                                    p3_sections.append(sim_md)
                                except ValueError as ve:
                                    all_warnings.append(f"Playtype extraction ({sim_name}): {ve}")
                            else:
                                all_warnings.append(f"Playtype: '{sim_name}' not found in target Players file. Available: {', '.join(available[:10])}")

                    if p3_sections:
                        full_p3_md = "\n\n---\n\n".join(p3_sections)
                        st.session_state["p3_md"] = full_p3_md

                except Exception as e:
                    all_warnings.append(f"Playtype extraction error: {str(e)}")

            # ── Mark done ────────────────────────────────────────────────────

            st.session_state["all_warnings"] = all_warnings
            st.session_state["run_complete_12ab"] = True
            progress.progress(100, text="🎉 All analysis pipelines complete!")

        # Show warnings (from session state, works on reruns too)
        for w in st.session_state.get("all_warnings", []):
            st.warning(w)

        # Show summary
        p1_ok = bool(st.session_state["p1_md"])
        p2a_ok = bool(st.session_state["p2a_md"])
        p2b_ok = bool(st.session_state["p2b_md"])
        similar = st.session_state.get("similar_players", [])

        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        col1.metric("P1 On/Off", "✅" if p1_ok else "❌")
        col2.metric("P2A Profile", "✅" if p2a_ok else "❌")
        col3.metric("P2B Similar", "✅" if p2b_ok else "❌")

        # AI guidance message
        if similar:
            names_str = " and ".join(f"**{n}**" for n in similar)
            st.markdown(f"""
            <div class="ai-message">
                <strong>🤖 Analysis done!</strong> The top similar players on {target_team} are: {names_str}.<br><br>
                For the combined report, upload a playtype <strong>.md file</strong> with the comparison data.<br><br>
                If you don't have playtype data or want to skip, just click "Skip & Generate Report".
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="ai-message">
                <strong>🤖 Analysis done!</strong> I couldn't find similar players, but you can still upload
                a playtype .md file for the combined report, or skip to generate with what we have.
            </div>
            """, unsafe_allow_html=True)

        # Preview expandable sections
        if p1_ok:
            with st.expander("📋 Preview: On/Off Report"):
                st.markdown(st.session_state["p1_md"])
        if p2a_ok:
            with st.expander("📋 Preview: Player Profile"):
                st.markdown(st.session_state["p2a_md"])
        if p2b_ok:
            with st.expander("📋 Preview: Similar Players"):
                st.markdown(st.session_state["p2b_md"])
        if st.session_state.get("p3_md"):
            with st.expander("📊 Preview: Playtypes (auto-extracted)"):
                st.markdown(st.session_state["p3_md"])
        if st.session_state.get("p5_md"):
            with st.expander("📈 Preview: Transfer Comps"):
                st.markdown(st.session_state["p5_md"])
        if st.session_state.get("p6_md"):
            with st.expander("📝 Preview: Lineup Analytics"):
                st.markdown(st.session_state["p6_md"])

        # Action buttons
        st.markdown("---")
        has_playtypes = bool(st.session_state.get("p3_md"))

        if has_playtypes:
            # Playtypes already extracted from Excel — go straight to report
            st.success("✅ Playtypes auto-extracted from Players files!")
            if st.button("🚀 Generate Report", type="primary", use_container_width=True):
                st.session_state["wizard_step"] = 4
                st.rerun()
            if st.button("📄 Replace with manual Playtype File", use_container_width=True):
                st.session_state["wizard_step"] = 3
                st.rerun()
        else:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📄 Upload Playtype File", type="primary", use_container_width=True):
                    st.session_state["wizard_step"] = 3
                    st.rerun()
            with col2:
                if st.button("⏭️ Skip & Generate Report", use_container_width=True):
                    st.session_state["wizard_step"] = 4
                    st.rerun()

        if not api_key:
            st.info("Add your Anthropic API key in the sidebar to enable narrative analysis.")


    # ══════════════════════════════════════════════════════════════════════════
    #  STEP 3 — PLAYTYPE SCREENSHOTS
    # ══════════════════════════════════════════════════════════════════════════

    elif step == 3:
        st.markdown("### Step 3 — Upload Playtype Data")

        player_name = st.session_state["_player_name"]
        target_team = st.session_state.get("_target_team", "Team")

        st.markdown(f"""
        <div class="ai-message">
            <strong>🤖 Almost there!</strong> Upload your playtype comparison as a <strong>.md</strong> file.
            This should contain the offense and defense playtype tables for the player(s) and team.
        </div>
        """, unsafe_allow_html=True)

        playtype_md_file = st.file_uploader(
            "Playtype Markdown File (.md)",
            type=["md", "txt"],
            key="wiz_p3_md_upload",
        )

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("🚀 Use Playtype Data & Generate Report", type="primary", use_container_width=True):
                if not playtype_md_file:
                    st.error("Please upload a playtype .md file.")
                else:
                    p3_md = playtype_md_file.getvalue().decode("utf-8", errors="replace")
                    st.session_state["p3_md"] = p3_md
                    st.session_state["wizard_step"] = 4
                    st.rerun()

        with col2:
            if st.button("⏭️ Skip Playtypes", use_container_width=True):
                st.session_state["wizard_step"] = 4
                st.rerun()


    # ══════════════════════════════════════════════════════════════════════════
    #  STEP 4 — COMBINED REPORT
    # ══════════════════════════════════════════════════════════════════════════

    elif step == 4:
        st.markdown("### Step 4 — Combined Scouting Report")

        player_name = st.session_state.get("_player_name", "Player")
        team_tag = st.session_state.get("_team_tag", "")
        target_team = st.session_state.get("_target_team", "")

        # Build combined report
        sections = []
        sections.append(f"# Combined Scouting Report: {player_name}")
        sections.append(f"**Team:** {st.session_state.get('_team_full', team_tag)}")
        sections.append(f"**Target:** {target_team}")
        sections.append(f"**Date:** {datetime.now().strftime('%d %b %Y')}")
        sections.append(f"**Generated by:** PlayerLynk Report Tool")
        sections.append("")
        sections.append("---")
        sections.append("")

        # Table of contents
        toc = ["## Table of Contents", ""]
        has_p1 = bool(st.session_state["p1_md"])
        has_p2a = bool(st.session_state["p2a_md"])
        has_p2b = bool(st.session_state["p2b_md"])
        has_p3 = bool(st.session_state["p3_md"])
        has_p5 = bool(st.session_state.get("p5_md"))
        has_p6 = bool(st.session_state.get("p6_md"))

        toc_num = 1
        if has_p1:
            toc.append(f"{toc_num}. [On/Off Impact + Duo Stats](#part-{toc_num}--onoff-impact--duo-stats)")
            toc_num += 1
        if has_p2a:
            toc.append(f"{toc_num}. [Player Profile — Percentiles, Tags & Clusters](#part-{toc_num}--player-profile)")
            toc_num += 1
        if has_p2b:
            toc.append(f"{toc_num}. [Similar Players Analysis](#part-{toc_num}--similar-players-analysis)")
            toc_num += 1
        if has_p3:
            toc.append(f"{toc_num}. [Playtype Comparison](#part-{toc_num}--playtype-comparison)")
            toc_num += 1
        if has_p5:
            toc.append(f"{toc_num}. [League Transition Comparison](#part-{toc_num}--league-transition-comparison)")
            toc_num += 1
        if has_p6:
            toc.append(f"{toc_num}. [Lineup Analytics](#part-{toc_num}--lineup-analytics)")
            toc_num += 1
        toc.append("")
        sections.extend(toc)

        # Helper to strip leading header block from a pipeline's markdown
        def _strip_header(md_text, skip_prefixes=None):
            """Return lines after the first # heading block and metadata lines."""
            result = []
            skip = True
            skip_pf = skip_prefixes or []
            for line in md_text.split("\n"):
                if skip:
                    if line.startswith("# "):
                        continue
                    if any(line.startswith(p) for p in skip_pf):
                        continue
                    if line.strip() == "":
                        continue
                    skip = False
                result.append(line)
            return result

        part_num = 1

        # ── On/Off Impact ────────────────────────────────────────────────────
        if has_p1:
            sections.append("---")
            sections.append("")
            sections.append(f"## Part {part_num} — On/Off Impact + Duo Stats")
            sections.append("")
            sections.extend(_strip_header(
                st.session_state["p1_md"],
                ["**Team:**", "**Data:**", "**Date:**"],
            ))
            sections.append("")
            part_num += 1

        # ── Player Profile ───────────────────────────────────────────────────
        if has_p2a:
            sections.append("---")
            sections.append("")
            sections.append(f"## Part {part_num} — Player Profile")
            sections.append("")
            sections.extend(_strip_header(
                st.session_state["p2a_md"],
                ["**Team:**", "**League:**", "**Date:**", "**Pool:**"],
            ))
            sections.append("")
            part_num += 1

        # ── Similar Players ──────────────────────────────────────────────────
        if has_p2b:
            sections.append("---")
            sections.append("")
            sections.append(f"## Part {part_num} — Similar Players Analysis")
            sections.append("")
            sections.extend(_strip_header(
                st.session_state["p2b_md"],
                ["**Subject:**", "**Target:**", "**Date:**", "**Position"],
            ))
            sections.append("")
            part_num += 1

        # ── Playtypes ────────────────────────────────────────────────────────
        if has_p3:
            sections.append("---")
            sections.append("")
            sections.append(f"## Part {part_num} — Playtype Comparison")
            sections.append("")
            sections.extend(_strip_header(st.session_state["p3_md"]))
            sections.append("")
            part_num += 1

        # ── Transfer Comps ───────────────────────────────────────────────────
        if has_p5:
            sections.append("---")
            sections.append("")
            sections.append(f"## Part {part_num} — League Transition Comparison")
            sections.append("")
            # p5_md already has its own ## heading, strip it
            p5_lines = st.session_state["p5_md"].split("\n")
            for line in p5_lines:
                if line.startswith("## "):
                    continue  # skip the duplicate heading
                sections.append(line)
            sections.append("")
            part_num += 1

        # ── Lineup Analytics (Fit Narrative) ─────────────────────────────────
        if has_p6:
            sections.append("---")
            sections.append("")
            sections.append(f"## Part {part_num} — Lineup Analytics")
            sections.append("")
            p6_lines = st.session_state["p6_md"].split("\n")
            for line in p6_lines:
                if line.startswith("## "):
                    continue  # skip the duplicate heading
                sections.append(line)
            sections.append("")
            part_num += 1

        # ── Footer ───────────────────────────────────────────────────────────
        sections.append("---")
        sections.append("")
        sections.append(f"*Report generated {datetime.now().strftime('%d %b %Y at %H:%M')} by PlayerLynk Report Tool*")

        combined_md = "\n".join(sections)
        st.session_state["combined_md"] = combined_md

        # Show warnings
        for w in st.session_state.get("all_warnings", []):
            st.warning(w)

        # Summary metrics
        total_sections = sum([has_p1, has_p2a, has_p2b, has_p3, has_p5, has_p6])
        max_sections = 6
        col1, col2, col3 = st.columns(3)
        col1.metric("P1 On/Off", "✅" if has_p1 else "—")
        col2.metric("P2A Profile", "✅" if has_p2a else "—")
        col3.metric("P2B Similar", "✅" if has_p2b else "—")
        col4, col5, col6 = st.columns(3)
        col4.metric("P3 Playtypes", "✅" if has_p3 else "—")
        col5.metric("Transfer Comps", "✅" if has_p5 else "—")
        col6.metric("Lineup Analytics", "✅" if has_p6 else "—")

        st.markdown(f"""
        <div class="ai-message">
            <strong>🤖 Your combined scouting report for {player_name} is ready!</strong>
            It includes {total_sections} out of {max_sections} sections.
            Download the full report below, or expand each section to preview.
        </div>
        """, unsafe_allow_html=True)

        # Download button
        filename = f"{player_name.replace(' ', '_')}_ScoutingReport_{datetime.now().strftime('%Y%m%d')}.md"
        st.download_button(
            "📥 Download Combined Scouting Report (.md)",
            data=combined_md,
            file_name=filename,
            mime="text/markdown",
            use_container_width=True,
            type="primary",
        )

        st.markdown("---")

        # Full report preview
        with st.expander("📋 Full Report Preview", expanded=True):
            st.markdown(combined_md)

        # Individual section previews
        if has_p1:
            with st.expander("🏀 On/Off Impact"):
                st.markdown(st.session_state["p1_md"])
        if has_p2a:
            with st.expander("📊 Player Profile"):
                st.markdown(st.session_state["p2a_md"])
        if has_p2b:
            with st.expander("🔍 Similar Players"):
                st.markdown(st.session_state["p2b_md"])
        if has_p3:
            with st.expander("📸 Playtype Comparison"):
                st.markdown(st.session_state["p3_md"])
        if has_p5:
            with st.expander("📈 League Transition Comparison"):
                st.markdown(st.session_state["p5_md"])
        if has_p6:
            with st.expander("📝 Lineup Analytics"):
                st.markdown(st.session_state["p6_md"])


# ══════════════════════════════════════════════════════════════════════════════
#  SINGLE PIPELINE MODE (legacy)
# ══════════════════════════════════════════════════════════════════════════════

elif mode == "🔧 Single Pipeline":

    # ── Pipeline 1 ───────────────────────────────────────────────────────────

    if single_pipeline == "1 — On/Off + Duo Stats":
        st.markdown("""
        <div class="pipeline-header">
            <h1>On/Off + Duo Stats</h1>
            <p>Upload team lineup data → Get on/off four factors + duo synergy analysis</p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            uploaded_file = st.file_uploader(
                "Team Lineup File (.xlsx)",
                type=["xlsx", "xls"],
                key="p1_file",
                help="Upload the team's lineup stint data file",
            )

        with col2:
            player_name = st.text_input("Player Name", placeholder="e.g. D. Mintz", key="p1_player")
            team_tag = st.text_input("Team Tag", placeholder="e.g. WB, MIL, OMP", key="p1_tag")

        col3, col4 = st.columns(2)
        with col3:
            min_minutes = st.number_input("Min. Player Minutes", value=100, min_value=0, step=10, key="p1_min")
        with col4:
            min_duo_minutes = st.number_input("Min. Duo Minutes", value=100, min_value=0, step=10, key="p1_duo_min")

        if st.button("🚀 Run On/Off + Duo Analysis", type="primary", use_container_width=True):
            if not uploaded_file:
                st.error("Please upload a lineup file.")
            elif not player_name or not team_tag:
                st.error("Please enter both Player Name and Team Tag.")
            else:
                with st.spinner("Running On/Off + Duo analysis..."):
                    from pipeline_onoff import run_onoff_pipeline

                    file_bytes = uploaded_file.getvalue()
                    df_onoff, duo_df, md, warnings = run_onoff_pipeline(
                        file_bytes, player_name, team_tag,
                        min_minutes=min_minutes, min_duo_minutes=min_duo_minutes,
                    )

                    for w in warnings:
                        st.warning(w)

                    if md:
                        st.success("Analysis complete!")
                        st.markdown("---")
                        st.markdown(md)

                        filename = f"{player_name.replace(' ', '_')}_OnOff_{datetime.now().strftime('%Y%m%d')}.md"
                        st.download_button(
                            "📥 Download Report (.md)",
                            data=md,
                            file_name=filename,
                            mime="text/markdown",
                            use_container_width=True,
                        )
                    else:
                        st.error("Could not generate report. Check warnings above.")


    # ── Pipeline 2A ──────────────────────────────────────────────────────────

    elif single_pipeline == "2A — Player Profile":
        st.markdown("""
        <div class="pipeline-header">
            <h1>Player Profile</h1>
            <p>Upload league data → Get percentiles, tags, and cluster assignment</p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            uploaded_file = st.file_uploader(
                "League Box Score File (.xlsx)",
                type=["xlsx", "xls"],
                key="p2a_file",
            )

        with col2:
            player_name = st.text_input("Player Name", placeholder="e.g. Quinn Ellis", key="p2a_player")
            team_name = st.text_input("Team Name (optional)", placeholder="e.g. Olimpia Milano", key="p2a_team")

        col3, col4 = st.columns(2)
        with col3:
            min_games = st.number_input("Min. Games Played", value=5, min_value=1, step=1, key="p2a_games")
        with col4:
            min_mpg = st.number_input("Min. Minutes/Game", value=0, min_value=0, step=1, key="p2a_mpg")

        if st.button("🚀 Run Player Profile", type="primary", use_container_width=True):
            if not uploaded_file:
                st.error("Please upload a league file.")
            elif not player_name:
                st.error("Please enter a Player Name.")
            else:
                with st.spinner("Computing percentiles, tags, and clusters..."):
                    from pipeline_percentiles import run_profile

                    file_bytes = uploaded_file.getvalue()
                    profile, md, warnings = run_profile(
                        file_bytes, player_name,
                        team_name=team_name if team_name else None,
                        min_games=min_games, min_mpg=min_mpg,
                        api_key=api_key if api_key else None,
                    )

                    for w in warnings:
                        st.warning(w)

                    if md:
                        st.success("Profile complete!")
                        st.session_state["last_profile"] = profile
                        st.info("💡 This profile is saved — switch to '2B' to use it.")
                        st.markdown("---")
                        st.markdown(md)

                        filename = f"{player_name.replace(' ', '_')}_Profile_{datetime.now().strftime('%Y%m%d')}.md"
                        st.download_button(
                            "📥 Download Report (.md)",
                            data=md,
                            file_name=filename,
                            mime="text/markdown",
                            use_container_width=True,
                        )
                    else:
                        st.error("Could not generate profile. Check warnings above.")


    # ── Pipeline 2B ──────────────────────────────────────────────────────────

    elif single_pipeline == "2B — Similar Players":
        st.markdown("""
        <div class="pipeline-header">
            <h1>Find Similar Players</h1>
            <p>Use a player profile to find stylistic matches on a target team</p>
        </div>
        """, unsafe_allow_html=True)

        has_profile = "last_profile" in st.session_state and st.session_state["last_profile"] is not None

        if has_profile:
            profile = st.session_state["last_profile"]
            st.success(f"Using profile: **{profile['name']}** ({profile['major_cluster']} / {profile['minor_cluster']})")

            all_tags = profile.get("tags", [])
            if isinstance(all_tags, dict):
                flat = []
                for cat_tags in all_tags.values():
                    flat.extend(cat_tags)
                all_tags = flat
            if all_tags:
                st.markdown(f"**Tags:** {', '.join(all_tags)}")
        else:
            st.warning("No profile loaded. Run Pipeline 2A first, or enter details manually below.")

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            uploaded_file = st.file_uploader(
                "Target League Box Score (.xlsx)",
                type=["xlsx", "xls"],
                key="p2b_file",
            )

        with col2:
            target_team = st.text_input("Target Team", placeholder="e.g. St. John's Red Storm", key="p2b_team")
            position_filter = st.selectbox(
                "Position Filter",
                ["None", "Guard", "Wing", "Forward", "Big", "Center"],
                key="p2b_pos",
            )

        col3, col4 = st.columns(2)
        with col3:
            min_games = st.number_input("Min. Games Played", value=5, min_value=1, step=1, key="p2b_games")
        with col4:
            min_mpg = st.number_input("Min. Minutes/Game", value=0, min_value=0, step=1, key="p2b_mpg")

        if not has_profile:
            st.markdown("### Manual Subject Profile")
            subj_name = st.text_input("Subject Player Name", key="p2b_subj_name")
            subj_team = st.text_input("Subject Team", key="p2b_subj_team")
            subj_cluster = st.text_input("Subject Major Cluster", key="p2b_subj_cluster")
            subj_minor = st.text_input("Subject Minor Cluster", key="p2b_subj_minor")
            subj_tags_str = st.text_input("Subject Tags (comma-separated)", key="p2b_subj_tags")

        if st.button("🚀 Find Similar Players", type="primary", use_container_width=True):
            if not uploaded_file:
                st.error("Please upload a target league file.")
            elif not target_team:
                st.error("Please enter a Target Team name.")
            else:
                if has_profile:
                    subject = st.session_state["last_profile"]
                else:
                    if not subj_name:
                        st.error("Please enter a Subject Player Name.")
                        st.stop()
                    tags_list = [t.strip() for t in subj_tags_str.split(",") if t.strip()] if subj_tags_str else []
                    subject = {
                        "name": subj_name,
                        "team": subj_team or "Unknown",
                        "major_cluster": subj_cluster or "Unknown",
                        "minor_cluster": subj_minor or "Unknown",
                        "tags": tags_list,
                        "percentiles": {},
                    }

                with st.spinner("Finding similar players..."):
                    from pipeline_percentiles import run_similarity

                    file_bytes = uploaded_file.getvalue()
                    pos = position_filter if position_filter != "None" else None

                    results, md, warnings = run_similarity(
                        file_bytes, subject, target_team,
                        position_filter=pos,
                        min_games=min_games, min_mpg=min_mpg,
                        api_key=api_key if api_key else None,
                    )

                    for w in warnings:
                        st.warning(w)

                    if md:
                        st.success(f"Found {len(results)} candidates — showing top matches!")
                        st.markdown("---")
                        st.markdown(md)

                        filename = f"{subject['name'].replace(' ', '_')}_to_{target_team.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.md"
                        st.download_button(
                            "📥 Download Report (.md)",
                            data=md,
                            file_name=filename,
                            mime="text/markdown",
                            use_container_width=True,
                        )
                    else:
                        st.error("Could not find matches. Check warnings above.")


    # ── Pipeline 3 ───────────────────────────────────────────────────────────

    elif single_pipeline == "3 — Playtype Extraction":
        st.markdown("""
        <div class="pipeline-header">
            <h1>Playtype Extraction</h1>
            <p>Upload playtype screenshots → Get structured comparison tables</p>
        </div>
        """, unsafe_allow_html=True)

        if not api_key:
            st.error("Please enter your Anthropic API key in the sidebar to use this pipeline.")
            st.stop()

        format_type = st.selectbox(
            "Format",
            ["Overseas (PS | POSS | PT | PPPP | SF | TOV)", "College (Playtype | Usage | PPP)"],
            key="p3_format",
        )
        fmt = "overseas" if "Overseas" in format_type else "college"

        st.markdown("### Upload Screenshots")
        num_entities = st.number_input("Number of players/teams", value=2, min_value=1, max_value=6, step=1)

        screenshots = []
        entity_names = []

        for i in range(int(num_entities)):
            col1, col2 = st.columns([1, 2])
            with col1:
                name = st.text_input(f"Name #{i+1}", key=f"p3_name_{i}", placeholder="Player or Team name")
                entity_names.append(name)
            with col2:
                img = st.file_uploader(f"Screenshot #{i+1}", type=["png", "jpg", "jpeg"], key=f"p3_img_{i}")
                screenshots.append(img)

        primary_idx = st.number_input(
            "Primary player index",
            value=1, min_value=1, max_value=int(num_entities), step=1,
        ) - 1

        model_choice = st.selectbox(
            "Claude Model",
            ["claude-sonnet-4-20250514", "claude-haiku-4-5-20251001"],
            key="p3_model",
        )

        if st.button("🚀 Extract Playtypes", type="primary", use_container_width=True):
            valid_pairs = [(entity_names[i], screenshots[i]) for i in range(int(num_entities))
                           if entity_names[i] and screenshots[i]]

            if not valid_pairs:
                st.error("Please provide at least one name + screenshot pair.")
            else:
                names = [p[0] for p in valid_pairs]
                imgs = [p[1].getvalue() for p in valid_pairs]

                with st.spinner("Sending screenshots to Claude API for extraction..."):
                    from pipeline_playtypes import run_playtype_pipeline

                    md, results, warnings = run_playtype_pipeline(
                        api_key, imgs, names,
                        format_type=fmt,
                        primary_index=min(primary_idx, len(valid_pairs) - 1),
                        model=model_choice,
                    )

                    for w in warnings:
                        st.warning(w)

                    if md:
                        st.success("Extraction complete!")
                        st.markdown("---")
                        st.markdown(md)

                        filename = f"Playtypes_{'_vs_'.join(names)}_{datetime.now().strftime('%Y%m%d')}.md"
                        st.download_button(
                            "📥 Download Report (.md)",
                            data=md,
                            file_name=filename,
                            mime="text/markdown",
                            use_container_width=True,
                        )

                        with st.expander("🔍 Raw API Responses"):
                            for r in results:
                                st.markdown(f"**{r['name']}:**")
                                st.code(r["raw"])
                    else:
                        st.error("Extraction failed. Check warnings above.")
