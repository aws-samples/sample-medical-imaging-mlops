"""Parse TODO.md into structured data for the dashboard Coming Soon section."""
import re
from pathlib import Path
from typing import List, Dict

TODO_PATH = Path("TODO.md")

# Icons for each section heading
SECTION_ICONS = {
    "training": ("fas fa-brain", "#8e44ad"),
    "model registry": ("fas fa-layer-group", "#1a6fa0"),
    "models": ("fas fa-cubes", "#e67e22"),
    "infrastructure": ("fas fa-server", "#555"),
    "explainability": ("fas fa-eye", "#8e44ad"),
    "model evaluation": ("fas fa-chart-bar", "#c0392b"),
    "model monitoring": ("fas fa-heartbeat", "#1e8449"),
    "ui / ux": ("fas fa-desktop", "#4B9CD3"),
}


def parse_todo() -> List[Dict]:
    """Parse TODO.md into groups of items.

    Returns a list of dicts:
        [{"section": "Training", "icon": "fas fa-brain", "color": "#8e44ad",
          "todos": [{"title": "...", "desc": "...", "priority": "P1"}]}]
    """
    if not TODO_PATH.exists():
        return []

    content = TODO_PATH.read_text()
    groups = []
    current_section = None
    current_items = []

    for line in content.split("\n"):
        # Section header: ## Training
        header_match = re.match(r'^## (.+)', line)
        if header_match:
            # Save previous section
            if current_section and current_items:
                groups.append(_make_group(current_section, current_items))
            section_name = re.sub(r'[🔧🏋️🚀🔬📊📈⚖️📝🔄\s]+', '', header_match.group(1)).strip()
            section_name = header_match.group(1).strip()
            # Skip "In Progress" section
            if "in progress" in section_name.lower():
                current_section = None
                current_items = []
            else:
                current_section = section_name
                current_items = []
            continue

        # Todo item: - [ ] **Title** — Description
        # Also supports [P1]/[P2]/[P3] tags: - [ ] [P1] **Title** — Description
        item_match = re.match(
            r'^- \[[ x]\]\s*(?:\[(P[123])\]\s*)?(?:\*\*(.+?)\*\*)\s*[-—–]\s*(.+)',
            line
        )
        if item_match and current_section:
            priority = item_match.group(1) or None
            title = item_match.group(2).strip()
            desc = item_match.group(3).strip()
            current_items.append({
                "title": title,
                "desc": desc,
                "priority": priority,
            })

    # Save last section
    if current_section and current_items:
        groups.append(_make_group(current_section, current_items))

    # Assign default priorities if not specified (first=P1, etc.)
    for group in groups:
        for i, item in enumerate(group["todos"]):
            if not item["priority"]:
                if i == 0:
                    item["priority"] = "P1"
                elif i < 3:
                    item["priority"] = "P2"
                else:
                    item["priority"] = "P3"

    return groups


def _make_group(section_name: str, items: list) -> dict:
    key = section_name.lower().lstrip("🔧🏋️🚀🔬📊📈⚖️📝🔄 ")
    # Remove emoji prefix for lookup
    clean_key = re.sub(r'^[^\w]+', '', key).strip()
    icon, color = SECTION_ICONS.get(clean_key, ("fas fa-tasks", "#666"))
    return {
        "section": section_name.lstrip("🔧🏋️🚀🔬📊📈⚖️📝🔄 ").strip(),
        "icon": icon,
        "color": color,
        "todos": items,
    }
