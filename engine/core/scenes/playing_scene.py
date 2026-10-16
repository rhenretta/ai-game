"""engine/core/scenes/playing_scene.py
Active combat gameplay scene with full telemetry instrumentation,
reactive dialogue overlay with character portraits and voice line audio,
and background studio promotion hot-reload monitoring with HUD notification.
"""
from __future__ import annotations
import json
import logging
import math
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid

from pydantic import BaseModel, Field
import pygame

import config
from engine.core.scene import Scene
from engine.entities.enemies import BaseEnemy
from engine.entities.player import Player
from engine.entities.portal import Portal
from engine.entities.projectile import ProjectileManager
from engine.spatial.aabb import AABB, Vector2

logger = logging.getLogger("PlayingScene")


class DialogueBark(BaseModel):
    """A reactive, spoken dialogue line triggered during gameplay.
    Rationale-First epistemic architecture: deliberation declared first.
    """
    deliberation: str = Field(default="", description="Epistemic rationale for dialogue bark timing and dramatic context")
    bark_id: str = Field(..., description="Unique identifier for the bark")
    speaker_id: str = Field(..., description="Speaker identifier, e.g. 'hero', 'valkyrie', 'boss'")
    speaker_name: str = Field(..., description="Display name for the character")
    text: str = Field(..., description="Dialogue text displayed in the overlay subtitle")
    portrait_asset: Optional[str] = Field(default=None, description="Path to portrait PNG (relative to cartridge root)")
    audio_asset: Optional[str] = Field(default=None, description="Path to voice line WAV (relative to cartridge root)")
    duration: float = Field(default=4.5, description="Duration in seconds to display the dialogue overlay")
    priority: int = Field(default=1, description="Priority level (1=ambient, 3=combat/event, 5=boss/critical)")
    trigger_type: str = Field(default="ambient", description="'episode_start', 'low_health', 'boss_encounter', 'kill_streak', 'exit_unlocked'")


class CharacterProfile(BaseModel):
    """Character identity, personality, and visual/audio configuration."""
    deliberation: str = Field(default="", description="Epistemic rationale defining character archetype and voice style")
    character_id: str
    display_name: str
    role: str = "protagonist"
    theme_color: str = "cyan"
    portrait_file: Optional[str] = None
    voice_timbre: Optional[str] = None


class HotReloadSentinel:
    """Non-blocking filesystem watcher polling cartridges/active/manifest.json.
    Operates in <5 microseconds per check; runs every 60 frames (1 second).
    """

    def __init__(self, active_cartridge_dir: Optional[Path] = None):
        self.active_dir = Path(active_cartridge_dir).resolve() if active_cartridge_dir else (config.PROJECT_ROOT / "cartridges" / "active").resolve()
        self.manifest_path = self.active_dir / "manifest.json"

        self.current_mtime: float = self._get_manifest_mtime()
        self.current_version: str = ""
        self.current_target_episode: int = 1

        self.poll_timer: float = 0.0
        self.POLL_INTERVAL: float = 1.0  # 1 second
        self.has_pending_update: bool = False
        self.pending_manifest: Dict[str, Any] = {}

        self._read_initial_manifest()

    def _get_manifest_mtime(self) -> float:
        if self.manifest_path.exists():
            try:
                return os.stat(self.manifest_path).st_mtime
            except OSError:
                pass
        return 0.0

    def _read_initial_manifest(self) -> None:
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.current_version = data.get("version", "1.0.0")
                self.current_target_episode = data.get("target_episode", 1)
            except Exception:
                pass

    def check_for_promoted_update(self, dt: float) -> bool:
        """Ticks poll timer and checks for new atomic promotion in cartridges/active."""
        self.poll_timer += dt
        if self.poll_timer < self.POLL_INTERVAL:
            return self.has_pending_update

        self.poll_timer = 0.0
        if not self.manifest_path.exists():
            return self.has_pending_update

        try:
            mtime = os.stat(self.manifest_path).st_mtime
            if mtime > self.current_mtime:
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    new_manifest = json.load(f)
                new_ver = new_manifest.get("version", "")
                new_ep = new_manifest.get("target_episode", self.current_target_episode)

                if new_ver != self.current_version or new_ep > self.current_target_episode or mtime > self.current_mtime:
                    self.has_pending_update = True
                    self.pending_manifest = new_manifest
                    self.current_mtime = mtime
                    logger.info(
                        "[HotReload] New cartridge detected: %s (v%s, ep=%d)",
                        new_manifest.get("name") or new_manifest.get("title"),
                        new_ver,
                        new_ep,
                    )
        except (OSError, json.JSONDecodeError):
            pass

        return self.has_pending_update

    def acknowledge_reload(self) -> None:
        """Called when reload is executed; synchronizes state."""
        self.has_pending_update = False
        self.current_version = self.pending_manifest.get("version", self.current_version)
        self.current_target_episode = self.pending_manifest.get("target_episode", self.current_target_episode)


def render_hot_reload_toast(surface: pygame.Surface, pending_manifest: Dict[str, Any], elapsed_time: float) -> None:
    """Renders non-intrusive animated HUD toast banner alerting player to a newly promoted update."""
    cw = config.CANVAS_WIDTH
    toast_w, toast_h = 560, 48
    toast_x = (cw - toast_w) // 2
    toast_y = 12

    # Background
    bg = pygame.Surface((toast_w, toast_h), pygame.SRCALPHA)
    bg.fill((20, 25, 45, 240))
    surface.blit(bg, (toast_x, toast_y))

    # Pulsing Amber Border
    pulse = int(175 + 80 * math.sin(elapsed_time * 6.0))
    border_col = (255, max(0, min(255, pulse)), 0)
    pygame.draw.rect(surface, border_col, pygame.Rect(toast_x, toast_y, toast_w, toast_h), width=2, border_radius=6)

    try:
        font_main = pygame.font.SysFont("Helvetica", 14, bold=True)
        font_sub = pygame.font.SysFont("Courier", 13)
    except Exception:
        font_main = pygame.font.Font(None, 16)
        font_sub = pygame.font.Font(None, 15)

    ep_num = pending_manifest.get("target_episode", 2)
    ver = pending_manifest.get("version", "1.1.0")

    title_txt = font_main.render(f"[⚡ STUDIO UPDATE PROMOTED] EPISODE {ep_num} READY (v{ver})", True, config.PALETTE["yellow"])
    prompt_txt = font_sub.render("Press [R] to hot-reload now  |  or continue playing", True, config.PALETTE["white"])

    surface.blit(title_txt, (toast_x + 18, toast_y + 8))
    surface.blit(prompt_txt, (toast_x + 18, toast_y + 26))


class DialogueRuntime:
    """Manages dialogue queue, character portraits, voice audio playback on Pygame mixer Channel(0),
    and HUD rendering on the 1280x720 canvas.
    """

    def __init__(self, engine: Any, cartridge_dir: Optional[Union[str, Path]] = None):
        self.engine = engine
        self.cartridge_dir: Optional[Path] = Path(cartridge_dir).resolve() if cartridge_dir else None

        # Audio voice channel (Channel 0 reserved strictly for voice lines)
        self.voice_channel: Optional[pygame.mixer.Channel] = None
        if config.AUDIO_ENABLED and not getattr(engine, "headless", False):
            try:
                if not pygame.mixer.get_init():
                    pygame.mixer.init(
                        frequency=config.SAMPLE_RATE,
                        channels=config.AUDIO_CHANNELS,
                        buffer=config.AUDIO_BUFFER_SIZE,
                    )
                pygame.mixer.set_reserved(1)
                self.voice_channel = pygame.mixer.Channel(0)
            except Exception as e:
                logger.warning("[Dialogue] Pygame mixer channel initialization failed: %s", e)

        # Asset caches
        self.portrait_cache: Dict[str, pygame.Surface] = {}
        self.sound_cache: Dict[str, pygame.mixer.Sound] = {}

        # Narrative state
        self.characters: Dict[str, Dict[str, Any]] = {}
        self.barks: List[DialogueBark] = []
        self.current_bark: Optional[DialogueBark] = None
        self.bark_timer: float = 0.0
        self.bark_total_duration: float = 4.0
        self.fired_triggers: set[str] = set()

        self._load_narrative()

    def _load_narrative(self) -> None:
        """Loads and parses narrative data from cartridge instance or narrative.json."""
        cartridge = getattr(self.engine, "active_cartridge", None)
        narrative_data: Dict[str, Any] = {}

        if cartridge is not None and hasattr(cartridge, "get_narrative"):
            narrative_data = cartridge.get_narrative() or {}

        if not narrative_data and self.cartridge_dir and self.cartridge_dir.exists():
            narrative_path = self.cartridge_dir / "narrative.json"
            if narrative_path.exists():
                try:
                    with open(narrative_path, "r", encoding="utf-8") as f:
                        narrative_data = json.load(f)
                except Exception as e:
                    logger.warning("[Dialogue] Failed to read narrative.json: %s", e)

        if not narrative_data:
            return

        # 1. Parse Characters
        raw_chars = narrative_data.get("characters", {})
        if isinstance(raw_chars, dict):
            self.characters = raw_chars
        elif isinstance(raw_chars, list):
            self.characters = {c.get("id", f"char_{idx}"): c for idx, c in enumerate(raw_chars)}

        # 2. Parse Dialogue Barks from dialogue_barks or dialogues
        self.barks = []

        # Format A: "dialogue_barks": {trigger: [barks]} or list
        if "dialogue_barks" in narrative_data:
            db_data = narrative_data["dialogue_barks"]
            if isinstance(db_data, dict):
                for trig, b_list in db_data.items():
                    if isinstance(b_list, list):
                        for b_idx, b_item in enumerate(b_list):
                            self._add_bark_from_dict(b_item, default_trigger=trig, index=b_idx)
            elif isinstance(db_data, list):
                for b_idx, b_item in enumerate(db_data):
                    self._add_bark_from_dict(b_item, default_trigger=b_item.get("trigger", "ambient"), index=b_idx)

        # Format B: "dialogues": [{trigger, character_id, text, voice_clip}, ...]
        if "dialogues" in narrative_data:
            diag_list = narrative_data["dialogues"]
            if isinstance(diag_list, list):
                for d_idx, d_item in enumerate(diag_list):
                    char_id = d_item.get("character_id", "hero")
                    char_info = self.characters.get(char_id, {})
                    bark = DialogueBark(
                        deliberation=d_item.get("deliberation", "Reactive dialogue cue"),
                        bark_id=d_item.get("id", f"diag_{d_idx}"),
                        speaker_id=char_id,
                        speaker_name=char_info.get("name", char_id.capitalize()),
                        text=d_item.get("text", ""),
                        portrait_asset=d_item.get("portrait") or char_info.get("portrait") or char_info.get("portrait_file"),
                        audio_asset=d_item.get("voice_clip") or d_item.get("audio_file"),
                        duration=float(d_item.get("duration", 4.5)),
                        priority=int(d_item.get("priority", 1)),
                        trigger_type=d_item.get("trigger", "ambient"),
                    )
                    self.barks.append(bark)

    def _add_bark_from_dict(self, b_dict: Dict[str, Any], default_trigger: str, index: int) -> None:
        speaker_id = b_dict.get("speaker") or b_dict.get("speaker_id") or "hero"
        char_info = self.characters.get(speaker_id, {})
        speaker_name = b_dict.get("speaker_name") or char_info.get("name") or speaker_id.capitalize()
        portrait = (
            b_dict.get("portrait_file")
            or b_dict.get("portrait_asset")
            or b_dict.get("portrait")
            or char_info.get("portrait")
            or char_info.get("portrait_file")
        )
        audio = b_dict.get("audio_file") or b_dict.get("audio_asset") or b_dict.get("voice_clip")
        bark = DialogueBark(
            deliberation=b_dict.get("deliberation", "Reactive dialogue bark"),
            bark_id=b_dict.get("id") or b_dict.get("bark_id") or f"bark_{default_trigger}_{index}",
            speaker_id=speaker_id,
            speaker_name=speaker_name,
            text=b_dict.get("text", ""),
            portrait_asset=portrait,
            audio_asset=audio,
            duration=float(b_dict.get("duration", 4.5)),
            priority=int(b_dict.get("priority", 1)),
            trigger_type=b_dict.get("trigger_type") or b_dict.get("trigger") or default_trigger,
        )
        self.barks.append(bark)

    def trigger_bark(self, bark: DialogueBark) -> bool:
        """Triggers a dialogue bark. Higher priority barks preempt lower priority ones."""
        if self.current_bark and self.current_bark.priority > bark.priority:
            return False

        # Preempt active voice line
        if self.voice_channel and self.voice_channel.get_busy():
            self.voice_channel.fadeout(120)

        self.current_bark = bark
        self.bark_timer = bark.duration
        self.bark_total_duration = max(0.1, bark.duration)

        # Play spoken voice audio
        if bark.audio_asset and config.AUDIO_ENABLED and not getattr(self.engine, "headless", False):
            sound = self._get_sound(bark.audio_asset)
            if sound and self.voice_channel:
                try:
                    self.voice_channel.play(sound)
                except Exception as e:
                    logger.warning("[Dialogue] Error playing sound: %s", e)

        # Emit telemetry event
        rec = getattr(self.engine, "telemetry_recorder", None)
        if rec:
            payload = {
                "bark_id": bark.bark_id,
                "speaker": bark.speaker_name,
                "trigger_type": bark.trigger_type,
                "duration": bark.duration,
                "priority": bark.priority,
                "text": bark.text[:80],
            }
            if hasattr(rec, "emit_custom_probe"):
                rec.emit_custom_probe(
                    probe_name="dialogue_bark_played",
                    pos=(0.0, 0.0),
                    category="dialogue",
                    data=payload,
                )
            elif hasattr(rec, "emit_event"):
                try:
                    rec.emit_event(
                        event_type="CUSTOM_PROBE",
                        probe_name="dialogue_bark_played",
                        pos=(0.0, 0.0),
                        payload=payload,
                    )
                except TypeError:
                    rec.emit_event(
                        event_type="CUSTOM_PROBE",
                        pos=(0.0, 0.0),
                        payload=payload,
                    )
            if hasattr(rec, "events") and rec.events:
                last_event = rec.events[-1]
                if getattr(last_event, "probe_name", None) is None:
                    last_event.probe_name = "dialogue_bark_played"
        return True

    def trigger_by_type(self, trigger_type: str) -> bool:
        """Finds and plays the first unplayed bark matching the trigger type."""
        # Normalize aliases: episode_start <-> level_start
        matches = [trigger_type]
        if trigger_type == "episode_start":
            matches.append("level_start")
        elif trigger_type == "level_start":
            matches.append("episode_start")

        for bark in self.barks:
            if bark.trigger_type in matches and bark.bark_id not in self.fired_triggers:
                self.fired_triggers.add(bark.bark_id)
                return self.trigger_bark(bark)
        return False

    def update(self, dt: float) -> None:
        """Ticks down active bark display timer."""
        if self.current_bark:
            self.bark_timer -= dt
            if self.bark_timer <= 0.0:
                self.dismiss_current_bark()

    def dismiss_current_bark(self) -> None:
        """Dismisses the active dialogue bark and fades voice audio."""
        if self.voice_channel and self.voice_channel.get_busy():
            self.voice_channel.fadeout(150)
        self.current_bark = None
        self.bark_timer = 0.0

    def _get_sound(self, rel_path: str) -> Optional[pygame.mixer.Sound]:
        """Loads and caches a WAV voice line."""
        if rel_path in self.sound_cache:
            return self.sound_cache[rel_path]

        resolved: Optional[Path] = None
        cartridge = getattr(self.engine, "active_cartridge", None)
        if cartridge is not None and hasattr(cartridge, "resolve_asset_path"):
            resolved = cartridge.resolve_asset_path(rel_path)

        if not resolved and self.cartridge_dir:
            full_path = self.cartridge_dir / rel_path
            if full_path.exists():
                resolved = full_path

        if resolved and resolved.exists():
            try:
                sound = pygame.mixer.Sound(str(resolved))
                self.sound_cache[rel_path] = sound
                return sound
            except Exception as e:
                logger.warning("[Dialogue] Error loading audio file %s: %s", resolved, e)
        return None

    def _get_portrait(self, rel_path: Optional[str], speaker_name: str, theme_color_name: str = "cyan") -> pygame.Surface:
        """Loads and caches character portrait surface. Falls back to procedural 16-color avatar if absent."""
        cache_key = rel_path or speaker_name
        if cache_key in self.portrait_cache:
            return self.portrait_cache[cache_key]

        target_size = (84, 84)
        if rel_path:
            resolved: Optional[Path] = None
            cartridge = getattr(self.engine, "active_cartridge", None)
            if cartridge is not None and hasattr(cartridge, "resolve_asset_path"):
                resolved = cartridge.resolve_asset_path(rel_path)
            if not resolved and self.cartridge_dir:
                full_path = self.cartridge_dir / rel_path
                if full_path.exists():
                    resolved = full_path

            if resolved and resolved.exists():
                try:
                    img = pygame.image.load(str(resolved))
                    scaled = pygame.transform.smoothscale(img, target_size)
                    self.portrait_cache[cache_key] = scaled
                    return scaled
                except Exception as e:
                    logger.warning("[Dialogue] Failed to load portrait %s: %s", resolved, e)

        # Procedural 16-color fallback avatar
        theme_col = config.PALETTE.get(theme_color_name, config.PALETTE["cyan"])
        surf = pygame.Surface(target_size, pygame.SRCALPHA)
        surf.fill(config.PALETTE["dark_navy"])
        pygame.draw.rect(surf, theme_col, pygame.Rect(0, 0, target_size[0], target_size[1]), width=2, border_radius=4)

        initial = speaker_name[0].upper() if speaker_name else "?"
        try:
            f = pygame.font.SysFont("Helvetica", 36, bold=True)
        except Exception:
            f = pygame.font.Font(None, 40)
        txt = f.render(initial, True, theme_col)
        surf.blit(txt, ((target_size[0] - txt.get_width()) // 2, (target_size[1] - txt.get_height()) // 2))

        self.portrait_cache[cache_key] = surf
        return surf

    def render(self, surface: pygame.Surface) -> None:
        """Renders dialogue overlay on lower third of the 1280x720 canvas."""
        if not self.current_bark:
            return

        bark = self.current_bark
        cw, ch = config.CANVAS_WIDTH, config.CANVAS_HEIGHT
        box_w, box_h = 920, 115
        box_x = (cw - box_w) // 2
        box_y = ch - box_h - 20

        # Semi-transparent dark navy background
        bg_surface = pygame.Surface((box_w, box_h), pygame.SRCALPHA)
        bg_surface.fill((20, 25, 45, 235))
        surface.blit(bg_surface, (box_x, box_y))

        # Cyan / Theme border
        theme_col = config.PALETTE.get("cyan", (41, 173, 255))
        pygame.draw.rect(surface, theme_col, pygame.Rect(box_x, box_y, box_w, box_h), width=2, border_radius=6)

        # Character portrait
        portrait = self._get_portrait(bark.portrait_asset, bark.speaker_name, "cyan")
        port_rect = pygame.Rect(box_x + 15, box_y + 15, 84, 84)
        surface.blit(portrait, port_rect)
        pygame.draw.rect(surface, config.PALETTE["light_gray"], port_rect, width=1, border_radius=4)

        # Text layout
        text_x = box_x + 115
        try:
            font_name = pygame.font.SysFont("Helvetica", 17, bold=True)
            font_body = pygame.font.SysFont("Courier", 15)
        except Exception:
            font_name = pygame.font.Font(None, 20)
            font_body = pygame.font.Font(None, 18)

        # Speaker name & Voice transmission indicator
        name_surf = font_name.render(bark.speaker_name.upper(), True, config.PALETTE["yellow"])
        surface.blit(name_surf, (text_x, box_y + 16))

        if self.voice_channel and self.voice_channel.get_busy():
            voice_tag = font_body.render("[VOICE TRANSMISSION]", True, config.PALETTE["green"])
            surface.blit(voice_tag, (text_x + name_surf.get_width() + 16, box_y + 17))

        # Subtitle text word-wrapping
        words = bark.text.split(" ")
        lines = []
        cur_line: List[str] = []
        for word in words:
            cur_line.append(word)
            test_line = " ".join(cur_line)
            if font_body.size(test_line)[0] > (box_w - 140):
                cur_line.pop()
                lines.append(" ".join(cur_line))
                cur_line = [word]
        if cur_line:
            lines.append(" ".join(cur_line))

        line_y = box_y + 42
        for l in lines[:3]:
            l_surf = font_body.render(l, True, config.PALETTE["white"])
            surface.blit(l_surf, (text_x, line_y))
            line_y += 18

        # Progress duration bar along bottom of overlay
        progress = max(0.0, min(1.0, self.bark_timer / self.bark_total_duration))
        bar_w = int((box_w - 30) * progress)
        pygame.draw.rect(surface, config.PALETTE["cyan"], pygame.Rect(box_x + 15, box_y + box_h - 6, bar_w, 3))


class PlayingScene(Scene):
    """Active combat scene housing player, enemies, hazards, combat, and telemetry streaming."""

    def __init__(self, engine: Any):
        super().__init__(engine)
        self.session_id: str = ""
        self.episode_id: str = ""
        self.episode_number: int = 1
        self.elapsed_time: float = 0.0
        self.is_paused: bool = False

        # Gameplay entities
        self.player: Player = Player(120.0, 360.0)
        self.enemies: List[BaseEnemy] = []
        self.projectile_manager: ProjectileManager = ProjectileManager()
        self.hazards: List[Any] = []
        self.portal: Portal = Portal(1160.0, 360.0)
        self.walls: List[AABB] = []

        # Episode stats
        self.stats = {
            "dashes": 0,
            "shots_fired": 0,
            "damage_dealt": 0.0,
            "damage_taken": 0.0,
            "enemies_killed": 0,
        }

        # Runtime subsystems: Dialogue & Hot-Reload
        self.dialogue_runtime: DialogueRuntime = DialogueRuntime(self.engine)
        self.hot_reload_sentinel: HotReloadSentinel = HotReloadSentinel()
        self.reload_toast_timer: float = 0.0
        self.reload_error_msg: Optional[str] = None

    def on_enter(self, context: Dict[str, Any]) -> None:
        self.session_id = context.get("session_id", str(uuid.uuid4()))
        self.episode_number = context.get("episode_number", 1)
        self.episode_id = f"ep_{self.session_id}_{self.episode_number:03d}"
        self.elapsed_time = 0.0
        self.is_paused = False
        self.stats = {
            "dashes": 0,
            "shots_fired": 0,
            "damage_dealt": 0.0,
            "damage_taken": 0.0,
            "enemies_killed": 0,
        }

        # Initialize projectile manager with recorder
        self.projectile_manager = ProjectileManager(self.engine.telemetry_recorder)

        # Dynamic Cartridge vs Fallback Geometry
        cartridge = getattr(self.engine, "active_cartridge", None)
        booted_path = getattr(self.engine, "booted_cartridge_path", None)

        # Re-initialize dialogue runtime bound to current cartridge
        self.dialogue_runtime = DialogueRuntime(self.engine, booted_path)
        self.dialogue_runtime.trigger_by_type("episode_start")

        if cartridge is not None:
            spec = cartridge.get_level_spec(self.episode_number) if hasattr(cartridge, "get_level_spec") else None
            if spec:
                self.walls = [w if isinstance(w, AABB) else AABB(w[0], w[1], w[2], w[3]) for w in getattr(spec, "walls", [])]
                created_p = cartridge.create_player(spec.spawn_point[0], spec.spawn_point[1], self.engine.telemetry_recorder) if hasattr(cartridge, "create_player") else None
                if created_p is not None:
                    self.player = created_p
                self.enemies = cartridge.create_enemies(spec) if hasattr(cartridge, "create_enemies") else []
                self.hazards = cartridge.create_hazards(spec) if hasattr(cartridge, "create_hazards") else []
                if hasattr(spec, "exit_point") and spec.exit_point:
                    self.portal = Portal(spec.exit_point[0], spec.exit_point[1])
                else:
                    self.portal = None
                spawn_x, spawn_y = getattr(spec, "spawn_point", (120.0, 360.0))
                level_name = getattr(spec, "level_id", f"level_{self.episode_number:02d}")
            else:
                self.walls = []
                self.enemies = []
                self.hazards = []
                self.portal = None
                spawn_x, spawn_y = 120.0, 360.0
                level_name = f"level_{self.episode_number:02d}"
        else:
            self.walls = []
            self.enemies = []
            self.hazards = []
            self.portal = None
            spawn_x, spawn_y = 120.0, 360.0
            level_name = f"level_{self.episode_number:02d}"

        # Telemetry: Record Episode Start
        if self.engine.telemetry_recorder:
            self.engine.telemetry_recorder.start_episode(
                level_id=level_name,
                episode_index=self.episode_number,
                player_character="hero",
                metadata={"session_id": self.session_id}
            )
            self.engine.telemetry_recorder.emit_event(
                event_type="EPISODE_START",
                pos=(spawn_x, spawn_y),
                vel=(0.0, 0.0),
                health=100.0,
                payload={"episode_id": self.episode_id, "episode_number": self.episode_number}
            )

    def on_exit(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "episode_id": self.episode_id,
            "episode_number": self.episode_number,
            "elapsed_time": self.elapsed_time,
            "stats": self.stats,
        }

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.is_paused = not self.is_paused
            elif event.key == pygame.K_k:  # Debug defeat trigger
                self.trigger_defeat()
            elif event.key == pygame.K_v:  # Debug victory trigger
                self.trigger_victory()
            elif event.key in (pygame.K_SPACE, pygame.K_e):
                if self.dialogue_runtime.current_bark:
                    self.dialogue_runtime.dismiss_current_bark()
            elif event.key == pygame.K_r:  # Hot-reload triggered by user
                if self.hot_reload_sentinel.has_pending_update:
                    self._execute_hot_reload()

    def _execute_hot_reload(self) -> bool:
        """Hot-swaps cartridges/active in-place, emits telemetry, and restarts episode.
        Catches invalid/failing cartridges to keep current game session alive and responsive.
        """
        active_dir = config.PROJECT_ROOT / "cartridges" / "active"
        if not active_dir.exists():
            return False

        target_ep = self.hot_reload_sentinel.pending_manifest.get("target_episode", self.episode_number + 1)
        target_ver = self.hot_reload_sentinel.pending_manifest.get("version", "unknown")

        try:
            # Stop audio playback before reloading
            if pygame.mixer.get_init():
                try:
                    pygame.mixer.stop()
                except Exception:
                    pass

            # Emit custom telemetry event
            if self.engine.telemetry_recorder:
                rec = self.engine.telemetry_recorder
                player_pos = (self.player.center.x, self.player.center.y) if getattr(self, "player", None) is not None else (0.0, 0.0)
                payload = {
                    "session_id": self.session_id,
                    "from_episode": self.episode_number,
                    "to_episode": target_ep,
                    "target_version": target_ver,
                }
                if hasattr(rec, "emit_custom_probe"):
                    rec.emit_custom_probe(
                        probe_name="cartridge_hot_reload_triggered",
                        pos=player_pos,
                        category="system",
                        data=payload,
                    )
                elif hasattr(rec, "emit_event"):
                    try:
                        rec.emit_event(
                            event_type="CUSTOM_PROBE",
                            probe_name="cartridge_hot_reload_triggered",
                            pos=player_pos,
                            payload=payload,
                        )
                    except TypeError:
                        rec.emit_event(
                            event_type="CUSTOM_PROBE",
                            pos=player_pos,
                            payload=payload,
                        )
                if hasattr(rec, "events") and rec.events:
                    last_event = rec.events[-1]
                    if getattr(last_event, "probe_name", None) is None:
                        last_event.probe_name = "cartridge_hot_reload_triggered"

            # Load updated cartridge
            self.engine.load_cartridge(active_dir)
            self.hot_reload_sentinel.acknowledge_reload()

            # Re-initialize current scene with newly promoted cartridge
            self.on_enter({"session_id": self.session_id, "episode_number": target_ep})
            self.reload_toast_timer = 2.5
            self.reload_error_msg = None
            logger.info("[HotReload] Cartridge reloaded to Episode %d (v%s)", target_ep, target_ver)
            return True
        except Exception as e:
            logger.warning("[HotReload] Cartridge reload failed from %s: %s. Keeping current session.", active_dir, e)
            self.hot_reload_sentinel.has_pending_update = False
            self.reload_toast_timer = 3.0
            self.reload_error_msg = f"[HOT RELOAD FAILED] {e}"
            return False

    def update(self, dt: float) -> None:
        if self.is_paused:
            return

        self.elapsed_time += dt

        # Update Dialogue Runtime
        self.dialogue_runtime.update(dt)

        # Reactive Dialogue Triggers: Low Health Warning
        if self.player.health <= 30.0:
            self.dialogue_runtime.trigger_by_type("low_health")

        # Background Studio Sentinel Polling
        self.hot_reload_sentinel.check_for_promoted_update(dt)
        if self.reload_toast_timer > 0.0:
            self.reload_toast_timer = max(0.0, self.reload_toast_timer - dt)

        # 1. Update Hazards & Environmental Status Effects
        self.player.speed_multiplier = 1.0
        for hazard in self.hazards:
            hazard.update(dt, self.player, self.engine.telemetry_recorder)

        # 2. Update Player Input & Movement
        move_vec = self.engine.input_manager.get_movement_vector()
        dash_pressed = self.engine.input_manager.is_dash_triggered()
        fire_pressed = self.engine.input_manager.is_fire_active()
        mouse_pos = self.engine.input_manager.canvas_mouse_pos
        mouse_vec = Vector2(mouse_pos[0], mouse_pos[1])

        self.player.handle_input(
            move_input=Vector2(move_vec[0], move_vec[1]),
            mouse_canvas_pos=mouse_vec,
            dash_pressed=dash_pressed,
            fire_pressed=fire_pressed,
            projectile_manager=self.projectile_manager
        )
        self.player.update(dt, self.walls)

        if dash_pressed and self.player.is_dashing:
            self.stats["dashes"] += 1
        if fire_pressed and self.player.weapon.cooldown > (self.player.weapon.fire_rate - dt):
            self.stats["shots_fired"] += 1

        # 3. Update Enemies
        for enemy in self.enemies:
            if hasattr(enemy, "update_ai"):
                try:
                    enemy.update_ai(dt, self.player, self.walls, self.projectile_manager, self.engine.telemetry_recorder)
                except TypeError:
                    try:
                        enemy.update_ai(dt, self.player, self.walls, self.engine.telemetry_recorder)
                    except TypeError:
                        enemy.update_ai(dt, self.player, self.walls)
            elif hasattr(enemy, "update"):
                try:
                    enemy.update(dt, self.player, self.walls, self.projectile_manager, self.engine.telemetry_recorder)
                except TypeError:
                    try:
                        enemy.update(dt, self.player, self.walls)
                    except TypeError:
                        enemy.update(dt)

        self.stats["enemies_killed"] = sum(1 for e in self.enemies if not e.is_alive)

        # 4. Update Projectiles
        self.projectile_manager.update(dt, self.walls, self.player, self.enemies)

        # 5. Update Exit Portal
        if self.portal is not None:
            cleared = self.portal.update(dt, self.enemies, self.player, self.engine.telemetry_recorder)
            if cleared:
                self.trigger_victory()
                return

        # 6. Check Player Death
        if not self.player.is_alive:
            self.trigger_defeat()
            return

    def trigger_defeat(self) -> None:
        """Transitions into GameOverScene upon player death."""
        if self.engine.telemetry_recorder:
            self.engine.telemetry_recorder.emit_event(
                event_type="PLAYER_DEATH",
                pos=(self.player.center.x, self.player.center.y),
                vel=(0.0, 0.0),
                health=0.0,
                payload={"episode_id": self.episode_id, "time": self.elapsed_time}
            )
            self.engine.telemetry_recorder.end_episode(
                outcome="PLAYER_DEATH",
                exit_pos=self.player.center.to_tuple()
            )
        self.engine.scene_manager.switch_to("game_over", **self.on_exit())

    def trigger_victory(self) -> None:
        """Transitions into EpisodeVictoryScene upon clearing exit portal."""
        portal_pos = (self.portal.center.x, self.portal.center.y) if self.portal else (self.player.center.x, self.player.center.y)
        exit_pos = self.portal.center.to_tuple() if self.portal else self.player.center.to_tuple()
        if self.engine.telemetry_recorder:
            self.engine.telemetry_recorder.emit_event(
                event_type="LEVEL_EXIT_REACHED",
                pos=portal_pos,
                vel=(self.player.vel.x, self.player.vel.y),
                health=self.player.health,
                payload={"episode_id": self.episode_id, "time": self.elapsed_time}
            )
            self.engine.telemetry_recorder.end_episode(
                outcome="CLEARED",
                exit_pos=exit_pos
            )
        self.engine.scene_manager.switch_to("victory", **self.on_exit())

    def render(self, surface: pygame.Surface) -> None:
        # Draw Floor Grid Lines
        for x in range(32, config.CANVAS_WIDTH, 64):
            pygame.draw.line(surface, config.PALETTE["dark_navy"], (x, 32), (x, config.CANVAS_HEIGHT - 32), 1)
        for y in range(32, config.CANVAS_HEIGHT, 64):
            pygame.draw.line(surface, config.PALETTE["dark_navy"], (32, y), (config.CANVAS_WIDTH - 32, y), 1)

        # Draw Hazards
        for hazard in self.hazards:
            hazard.render(surface)

        # Draw Portal
        if self.portal is not None:
            self.portal.render(surface)

        # Draw Walls
        for wall in self.walls:
            if isinstance(wall, AABB):
                rect = pygame.Rect(int(wall.left), int(wall.top), int(wall.width), int(wall.height))
            else:
                rect = pygame.Rect(int(wall[0]), int(wall[1]), int(wall[2]), int(wall[3]))
            pygame.draw.rect(surface, config.PALETTE["muted_gray"], rect)
            pygame.draw.rect(surface, config.PALETTE["light_gray"], rect, width=1)

        # Draw Enemies
        for enemy in self.enemies:
            if enemy.is_alive:
                enemy.render(surface)

        # Draw Projectiles
        self.projectile_manager.render(surface)

        # Draw Player
        if self.player.is_alive:
            self.player.render(surface)

        # Draw HUD Overlays
        self.render_hud(surface)

        # Draw Hot-Reload Toast Banner if studio has promoted an update
        if self.hot_reload_sentinel.has_pending_update:
            render_hot_reload_toast(surface, self.hot_reload_sentinel.pending_manifest, self.elapsed_time)

        # Confirmation flash toast on reload
        if self.reload_toast_timer > 0.0:
            self._render_reload_confirmation(surface)

        # Draw Reactive Dialogue Overlay
        self.dialogue_runtime.render(surface)

        # Crosshair at canvas mouse position
        mx, my = self.engine.input_manager.canvas_mouse_pos
        pygame.draw.circle(surface, config.PALETTE["cyan"], (int(mx), int(my)), 6, width=2)
        pygame.draw.line(surface, config.PALETTE["cyan"], (int(mx) - 9, int(my)), (int(mx) + 9, int(my)), width=1)
        pygame.draw.line(surface, config.PALETTE["cyan"], (int(mx), int(my) - 9), (int(mx), int(my) + 9), width=1)

    def _render_reload_confirmation(self, surface: pygame.Surface) -> None:
        cw = config.CANVAS_WIDTH
        w, h = 480, 36
        x, y = (cw - w) // 2, 64
        bg = pygame.Surface((w, h), pygame.SRCALPHA)
        if getattr(self, "reload_error_msg", None):
            bg.fill((140, 20, 20, 230))
            surface.blit(bg, (x, y))
            pygame.draw.rect(surface, config.PALETTE["red"], pygame.Rect(x, y, w, h), width=2, border_radius=4)
            msg = str(self.reload_error_msg)
        else:
            bg.fill((0, 135, 81, 230))
            surface.blit(bg, (x, y))
            pygame.draw.rect(surface, config.PALETTE["green"], pygame.Rect(x, y, w, h), width=2, border_radius=4)
            msg = f"EPISODE {self.episode_number} HOT-RELOADED SUCCESSFULLY"
        try:
            font = pygame.font.SysFont("Helvetica", 14, bold=True)
        except Exception:
            font = pygame.font.Font(None, 16)
        txt = font.render(msg, True, config.PALETTE["white"])
        surface.blit(txt, (x + (w - txt.get_width()) // 2, y + 9))

    def render_hud(self, surface: pygame.Surface) -> None:
        try:
            font = pygame.font.SysFont("Courier", 15, bold=True)
        except Exception:
            font = pygame.font.Font(None, 18)

        # Top Info Bar
        remaining_enemies = sum(1 for e in self.enemies if e.is_alive)
        fps = self.engine.game_loop.running_fps if hasattr(self.engine, "game_loop") else 60.0
        hud_lines = [
            f"EPISODE: {self.episode_number}   TIME: {self.elapsed_time:.1f}s   FPS: {fps:.1f}",
            f"ENEMIES: {remaining_enemies} REMAINING   DASHES: {self.stats['dashes']}",
        ]
        for idx, line in enumerate(hud_lines):
            rendered = font.render(line, True, config.PALETTE["white"])
            surface.blit(rendered, (40, 10 + idx * 18))

        # Health Bar (Bottom Left)
        hp_pct = max(0.0, self.player.health / self.player.max_health)
        hp_bg = pygame.Rect(40, config.CANVAS_HEIGHT - 28, 160, 14)
        hp_fg = pygame.Rect(40, config.CANVAS_HEIGHT - 28, int(160 * hp_pct), 14)
        pygame.draw.rect(surface, config.PALETTE["red"], hp_bg)
        pygame.draw.rect(surface, config.PALETTE["green"], hp_fg)
        pygame.draw.rect(surface, config.PALETTE["white"], hp_bg, width=1)

        # Stamina Bar
        st_pct = max(0.0, self.player.stamina / self.player.max_stamina)
        st_bg = pygame.Rect(40, config.CANVAS_HEIGHT - 44, 160, 10)
        st_fg = pygame.Rect(40, config.CANVAS_HEIGHT - 44, int(160 * st_pct), 10)
        pygame.draw.rect(surface, config.PALETTE["dark_gray"], st_bg)
        pygame.draw.rect(surface, config.PALETTE["cyan"], st_fg)
        pygame.draw.rect(surface, config.PALETTE["light_gray"], st_bg, width=1)
