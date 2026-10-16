"""qa/profiles.py
Synthetic playtester behavioural profiles for dual-archetype simulation and testing.
Strictly Rationale-First with deliberation declared first.
"""
from __future__ import annotations
from typing import Any, Dict
from pydantic import BaseModel, Field


class PlaytesterProfile(BaseModel):
    """Configuration profile dictating synthetic bot kinetic and combat tactics."""
    deliberation: str = Field(
        default="",
        description="Epistemic rationale explaining the behavioral biases of this synthetic profile."
    )
    name: str = Field(default="rusher", description="Profile archetype identifier.")
    dash_frequency_bias: float = Field(default=0.7, description="Eagerness to dash (0.0 to 1.0).")
    combat_aggression: float = Field(default=0.8, description="Priority given to combat engagement vs sprinting.")
    weapon_range_px: float = Field(default=450.0, description="Maximum engagement distance for weapon firing.")
    standoff_distance_px: float = Field(default=160.0, description="Preferred distance from melee enemies.")
    hazard_caution_distance_px: float = Field(default=48.0, description="Distance threshold to trigger hazard evasion.")
    allow_diagonal: bool = Field(default=True, description="Whether A* pathfinding allows diagonal steps.")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "deliberation": self.deliberation,
            "name": self.name,
            "dash_frequency_bias": self.dash_frequency_bias,
            "combat_aggression": self.combat_aggression,
            "weapon_range_px": self.weapon_range_px,
            "standoff_distance_px": self.standoff_distance_px,
            "hazard_caution_distance_px": self.hazard_caution_distance_px,
            "allow_diagonal": self.allow_diagonal,
        }


RUSHER_PROFILE = PlaytesterProfile(
    deliberation="Aggressive high-mobility rusher: continuously dashes on cooldown, closes distance, and fires aggressively.",
    name="rusher",
    dash_frequency_bias=0.85,
    combat_aggression=0.90,
    weapon_range_px=500.0,
    standoff_distance_px=110.0,
    hazard_caution_distance_px=40.0,
    allow_diagonal=True,
)

CAUTIOUS_PROFILE = PlaytesterProfile(
    deliberation="Methodical cautious player: avoids unnecessary dashes, maintains standoff distance from threats, and kites around hazards.",
    name="cautious",
    dash_frequency_bias=0.20,
    combat_aggression=0.45,
    weapon_range_px=400.0,
    standoff_distance_px=260.0,
    hazard_caution_distance_px=75.0,
    allow_diagonal=True,
)
