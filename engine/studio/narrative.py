"""engine/studio/narrative.py
Narrative & Character Writer Agent — Feature 18.
Generates character sheets, cutscene scripts, reactive dialogue barks, and branching dialogue graphs.
Performs dialogue graph analysis (cycle detection, orphan detection) and exports narrative.json.
Enforces Rule 4 Rationale-First Epistemic Architecture (deliberation strictly field 0).
Zero occurrences of legacy branding tokens.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import uuid

from engine.studio.models import (
    CharacterSheet,
    CutsceneBeat,
    CutsceneScript,
    DialogueBarkSpec,
    DialogueChoice,
    DialogueNode,
    DialogueTreeSpec,
    GenesisWorldSpec,
    NarrativeBible,
    StudioAgentError,
)

logger = logging.getLogger("NarrativeCharacterWriterAgent")


def validate_dialogue_tree(tree: DialogueTreeSpec) -> List[str]:
    """Validates dialogue graph reachability and cycle structure."""
    errors: List[str] = []
    if not tree.nodes:
        return ["Dialogue tree contains 0 nodes"]

    root_id = tree.root_node_id
    if root_id not in tree.nodes:
        return [f"Root node '{root_id}' does not exist in nodes dictionary"]

    # Reachability analysis (BFS)
    visited: Set[str] = set()
    queue: List[str] = [root_id]
    visited.add(root_id)

    while queue:
        curr_id = queue.pop(0)
        node = tree.nodes.get(curr_id)
        if not node:
            continue
        for choice in node.choices:
            target = choice.target_node_id
            if target not in tree.nodes:
                errors.append(f"Node '{curr_id}' targets non-existent node '{target}'")
            elif target not in visited:
                visited.add(target)
                queue.append(target)

    # Detect orphan nodes
    all_node_ids = set(tree.nodes.keys())
    orphans = all_node_ids - visited
    if orphans:
        errors.append(f"Orphan (unreachable) dialogue nodes detected: {sorted(list(orphans))}")

    # Detect tight 2-node cycles without escape (e.g. node1 -> node2 -> node1)
    for n_id, n_obj in tree.nodes.items():
        targets = [c.target_node_id for c in n_obj.choices]
        for t in targets:
            if t in tree.nodes and t != n_id:
                back_targets = [c.target_node_id for c in tree.nodes[t].choices]
                if n_id in back_targets and len(n_obj.choices) == 1 and len(tree.nodes[t].choices) == 1:
                    errors.append(f"Closed 2-node infinite loop detected between '{n_id}' and '{t}'")

    return errors


class NarrativeCharacterWriterAgent:
    """Specialist Agent authoring character bibles, cutscenes, and dialogue scripts."""
    deliberation: str = "Narrative character writer agent structuring story bibles and dialogue graphs."

    def __init__(self, client: Optional[Any] = None, model_name: str = "gemini-3.8-flash"):
        self.client = client
        self.model_name = model_name

    def generate_narrative_bible(self, world_spec: GenesisWorldSpec) -> NarrativeBible:
        """Generates a cohesive NarrativeBible matching the world setting and conflict."""
        if self.client is None:
            raise StudioAgentError("NarrativeCharacterWriterAgent requires an active google.genai Client (Rule 5: Zero Silent Fallbacks)")

        try:
            from google.genai import types
            prompt = (
                f"You are the Lead Narrative Writer AI.\n"
                f"Write a full narrative bible for:\n"
                f"- Title: {world_spec.game_title}\n"
                f"- Genre: {world_spec.core_genre}\n"
                f"- World lore: {world_spec.world_lore}\n"
                f"- Tone: {world_spec.conflict_tone}\n"
                f"Create character sheets (hero, rival/companion, boss), reactive dialogue barks "
                f"for combat/low-health/boss encounters, and cutscenes. "
                f"You MUST provide deliberation first."
            )
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=NarrativeBible,
                ),
            )
            if not response.text:
                raise StudioAgentError("Empty response returned from Gemini narrative writing call")
            data = json.loads(response.text)
            return NarrativeBible(**data)
        except Exception as exc:
            logger.error("Gemini narrative writing call failed: %s", exc)
            raise StudioAgentError(f"NarrativeCharacterWriter failed to generate narrative bible: {exc}") from exc


    def export_narrative_json(self, bible: NarrativeBible, output_path: Path) -> Path:
        """Exports bible to narrative.json for cartridge runtime consumption."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data = bible.to_cartridge_narrative_dict()
        output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return output_path
