"""
Pipeline 3 — Playtype Extraction from Screenshots via Claude API
Sends screenshots to Claude Vision to extract structured playtype tables.
"""

import anthropic
import base64
import re


OVERSEAS_COLUMNS = ["Playtype", "PS", "POSS", "PT", "PPPP", "SF", "TOV"]
COLLEGE_COLUMNS = ["Playtype", "Usage", "PPP"]

SYSTEM_PROMPT = """You are a basketball analytics data extractor. You will receive a screenshot
of a player's or team's playtype breakdown page from InStat or a similar stats platform.

Your job is to extract ALL playtypes for both OFFENSE and DEFENSE from the screenshot.

CRITICAL READING INSTRUCTIONS:
- The page typically shows OFFENSE on the LEFT side and DEFENSE on the RIGHT side.
- Alternatively, offense and defense may be stacked vertically.
- You MUST read EVERY SINGLE ROW in the table. There are typically 8-12 playtypes per side.
- Common playtype names (read ALL that appear, not just some):
  Pick'n'rolls Handler, Catch and shoots, Transitions, Catch and drives, Post ups,
  Putbacks, Hand offs, Cuts, Screen offs, Pick'n'rolls Roller, Isolation, Pick'n'pops
- Each row has columns: PS (shown as X.X %), POSS, PT, PPPP, SF, TOV
- Read the offense columns separately from the defense columns — do NOT mix them up.
- Go through the table TOP TO BOTTOM and extract EVERY row you see.

Rules:
- Extract ALL playtypes that appear, not just a subset
- Use "—" only for values that are genuinely blank or unreadable
- Keep playtype names exactly as they appear
- Numbers should be extracted as-is from the screenshot
- Do NOT stop after reading a few rows — continue until you've read every row in the table

Return the data in this exact format (include ALL rows, not just 6):

OFFENSE:
1. [Playtype] | [PS] | [POSS] | [PT] | [PPPP] | [SF] | [TOV]
2. [Playtype] | [PS] | [POSS] | [PT] | [PPPP] | [SF] | [TOV]
... (continue for ALL playtypes shown)

DEFENSE:
1. [Playtype] | [PS] | [POSS] | [PT] | [PPPP] | [SF] | [TOV]
2. [Playtype] | [PS] | [POSS] | [PT] | [PPPP] | [SF] | [TOV]
... (continue for ALL playtypes shown)
"""


def extract_playtypes_from_screenshot(api_key, image_bytes, entity_name, format_type="overseas", model="claude-sonnet-4-20250514"):
    """
    Send a screenshot to Claude API and extract playtype data.

    Args:
        api_key: Anthropic API key
        image_bytes: raw image bytes (PNG/JPG)
        entity_name: player or team name
        format_type: "overseas" or "college"
        model: Claude model to use

    Returns:
        (offense_table, defense_table, raw_response)
        where each table is a list of dicts
    """
    client = anthropic.Anthropic(api_key=api_key)

    # Encode image
    b64_image = base64.standard_b64encode(image_bytes).decode("utf-8")

    # Detect media type
    if image_bytes[:8] == b'\x89PNG\r\n\x1a\n':
        media_type = "image/png"
    elif image_bytes[:2] == b'\xff\xd8':
        media_type = "image/jpeg"
    else:
        media_type = "image/png"  # default

    if format_type == "overseas":
        col_instruction = "Columns to extract: PS (possession share %), POSS (possessions), PT (points), PPPP (points per possession), SF (shooting fouls), TOV (turnovers)"
    else:
        col_instruction = "Columns to extract: Usage (usage %), PPP (points per possession)"

    user_msg = f"""Extract ALL playtype data for: {entity_name}

{col_instruction}

IMPORTANT: Read the ENTIRE table — every single row from top to bottom.
The table may show offense on the left and defense on the right side-by-side.
There are typically 8-12 playtype rows per side. Extract ALL of them, not just a few.

Go row by row:
- Row 1: read the playtype name, then PS, POSS, PT, PPPP, SF, TOV for offense
- Row 2: same thing
- Continue until you've read every row
Then do the same for defense.

Do NOT stop early. Do NOT skip any rows. Return ALL playtypes you can see."""

    response = client.messages.create(
        model=model,
        max_tokens=4000,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": b64_image,
                        },
                    },
                    {
                        "type": "text",
                        "text": user_msg,
                    },
                ],
            }
        ],
    )

    raw_text = response.content[0].text

    # Parse response into structured tables (gets ALL playtypes)
    offense_table, defense_table = parse_playtype_response(raw_text, format_type)

    # Sort by PS% descending and keep top 6
    offense_table = _sort_and_top6(offense_table)
    defense_table = _sort_and_top6(defense_table)

    return offense_table, defense_table, raw_text


def _parse_ps_value(ps_str):
    """Parse a PS string like '20.7 %' or '20.7%' into a float for sorting."""
    if not ps_str or ps_str == "—":
        return 0.0
    s = str(ps_str).strip().replace("%", "").strip()
    try:
        return float(s)
    except ValueError:
        return 0.0


def _sort_and_top6(table):
    """Sort a playtype table by PS% descending and return top 6."""
    if not table:
        return table
    sorted_table = sorted(table, key=lambda row: _parse_ps_value(row.get("PS", "0")), reverse=True)
    return sorted_table[:6]


def parse_playtype_response(raw_text, format_type="overseas"):
    """Parse the Claude response into offense and defense tables."""
    offense = []
    defense = []

    current_section = None
    lines = raw_text.strip().split("\n")

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if "OFFENSE" in line.upper() and ":" in line:
            current_section = "offense"
            continue
        elif "DEFENSE" in line.upper() and ":" in line:
            current_section = "defense"
            continue

        # Try to parse a numbered line: "1. Playtype | val1 | val2 | ..."
        m = re.match(r"^\d+\.\s*(.+)$", line)
        if m and current_section:
            parts = [p.strip() for p in m.group(1).split("|")]
            if len(parts) >= 2:
                entry = {"Playtype": parts[0]}

                if format_type == "overseas":
                    cols = ["PS", "POSS", "PT", "PPPP", "SF", "TOV"]
                    for i, col in enumerate(cols):
                        entry[col] = parts[i + 1] if i + 1 < len(parts) else "—"
                else:
                    cols = ["Usage", "PPP"]
                    for i, col in enumerate(cols):
                        entry[col] = parts[i + 1] if i + 1 < len(parts) else "—"

                if current_section == "offense":
                    offense.append(entry)
                else:
                    defense.append(entry)

    return offense, defense


def reorder_table_to_match(reference_table, target_table):
    """
    Reorder target_table to match the playtype ordering of reference_table.
    Missing playtypes get "—" for all value columns.
    """
    if not reference_table or not target_table:
        return target_table

    # Build lookup from target
    target_lookup = {row["Playtype"].lower().strip(): row for row in target_table}

    reordered = []
    value_cols = [k for k in target_table[0].keys() if k != "Playtype"] if target_table else []

    for ref_row in reference_table:
        playtype = ref_row["Playtype"]
        match = target_lookup.get(playtype.lower().strip())
        if match:
            reordered.append(match)
        else:
            empty_row = {"Playtype": playtype}
            for col in value_cols:
                empty_row[col] = "—"
            reordered.append(empty_row)

    return reordered


def table_to_markdown(table, format_type="overseas"):
    """Convert a playtype table to markdown."""
    if not table:
        return "*No data extracted*"

    if format_type == "overseas":
        cols = OVERSEAS_COLUMNS
    else:
        cols = COLLEGE_COLUMNS

    lines = []
    header = " | ".join(cols)
    lines.append(f"| {header} |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")

    for row in table:
        vals = [str(row.get(c, "—")) for c in cols]
        lines.append(f"| {' | '.join(vals)} |")

    return "\n".join(lines)


def run_playtype_pipeline(api_key, screenshots, entity_names, format_type="overseas",
                           primary_index=0, model="claude-sonnet-4-20250514"):
    """
    Run the full playtype extraction pipeline.

    Args:
        api_key: Anthropic API key
        screenshots: list of (name, image_bytes) tuples
        entity_names: list of entity names corresponding to screenshots
        format_type: "overseas" or "college"
        primary_index: index of the primary player (sets ordering)
        model: Claude model to use

    Returns:
        (markdown_str, all_results, warnings)
    """
    warnings = []
    all_results = []

    for i, (name, img_bytes) in enumerate(zip(entity_names, screenshots)):
        try:
            off_table, def_table, raw = extract_playtypes_from_screenshot(
                api_key, img_bytes, name, format_type, model
            )
            all_results.append({
                "name": name,
                "offense": off_table,
                "defense": def_table,
                "raw": raw,
            })
        except Exception as e:
            warnings.append(f"Error extracting playtypes for {name}: {str(e)}")
            all_results.append({
                "name": name,
                "offense": [],
                "defense": [],
                "raw": str(e),
            })

    # Reorder team/secondary tables to match primary player ordering
    if len(all_results) > 1 and all_results[primary_index]["offense"]:
        primary_off = all_results[primary_index]["offense"]
        primary_def = all_results[primary_index]["defense"]

        for i, result in enumerate(all_results):
            if i == primary_index:
                continue
            # Check if this is a team entry (typically the last one, or has "team" in name)
            is_team = any(kw in result["name"].lower() for kw in ["team", "club", "fc", "bc"])
            if is_team or i > primary_index:
                result["offense"] = reorder_table_to_match(primary_off, result["offense"])
                result["defense"] = reorder_table_to_match(primary_def, result["defense"])

    # Generate markdown
    md = generate_playtype_markdown(all_results, format_type, warnings)

    return md, all_results, warnings


def generate_playtype_markdown(results, format_type, warnings):
    """Generate the playtype comparison markdown report."""
    lines = []

    names = [r["name"] for r in results]
    lines.append(f"# Playtype Comparison: {' vs. '.join(names)}")
    lines.append("")

    if warnings:
        for w in warnings:
            lines.append(f"> **Warning:** {w}")
        lines.append("")

    for result in results:
        lines.append(f"## {result['name']}")
        lines.append("")

        lines.append("### Offense (Top 6)")
        lines.append("")
        lines.append(table_to_markdown(result["offense"], format_type))
        lines.append("")

        lines.append("### Defense (Top 6)")
        lines.append("")
        lines.append(table_to_markdown(result["defense"], format_type))
        lines.append("")

    lines.append("---")
    lines.append(f"*Format: {format_type.title()}*")

    return "\n".join(lines)
