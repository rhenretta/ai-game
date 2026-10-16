"""engine/combat/hitbox.py
Hitbox and Hurtbox components for faction-aware collision detection.
"""
from __future__ import annotations
from typing import Optional
from engine.combat.damage import DamagePacket, Faction
from engine.spatial.aabb import AABB


class Hurtbox:
    """Receiver volume for incoming damage packets."""

    def __init__(self, owner_id: str, faction: Faction, bounds: AABB, is_active: bool = True):
        self.owner_id = owner_id
        self.faction = faction
        self.bounds = bounds
        self.is_active = is_active


class Hitbox:
    """Offensive volume delivering damage packets to opposing hurtboxes."""

    def __init__(self, owner_id: str, faction: Faction, bounds: AABB, damage_packet: DamagePacket, is_active: bool = True):
        self.owner_id = owner_id
        self.faction = faction
        self.bounds = bounds
        self.damage_packet = damage_packet
        self.is_active = is_active


def check_hitbox_hurtbox(hitbox: Hitbox, hurtbox: Hurtbox) -> bool:
    """Returns True if active hitbox intersects opposing active hurtbox without friendly fire."""
    if not hitbox.is_active or not hurtbox.is_active:
        return False
    if hitbox.owner_id == hurtbox.owner_id:
        return False
    # Player cannot damage Player; Enemy cannot damage Enemy
    if hitbox.faction == hurtbox.faction and hitbox.faction != Faction.NEUTRAL:
        return False
    return hitbox.bounds.intersects(hurtbox.bounds)
