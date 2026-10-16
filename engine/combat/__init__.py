"""engine/combat package - hitboxes, hurtboxes, damage, and weapons."""
from engine.combat.damage import DamagePacket, Faction, KnockbackState, calculate_damage
from engine.combat.hitbox import Hitbox, Hurtbox, check_hitbox_hurtbox
from engine.combat.weapon import BlasterWeapon, SpreadWeapon, Weapon

__all__ = [
    "Faction",
    "DamagePacket",
    "KnockbackState",
    "calculate_damage",
    "Hitbox",
    "Hurtbox",
    "check_hitbox_hurtbox",
    "Weapon",
    "BlasterWeapon",
    "SpreadWeapon",
]
