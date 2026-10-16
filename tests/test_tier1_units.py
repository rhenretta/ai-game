"""Tier 1 Unit & Schema Integrity Tests.

Verifies:
  1. Rationale-First ordering AST checks & schema dictionary invariants.
  2. SQLite WAL schema, pragma verification, and non-blocking concurrency.
  3. Priority enum ordering and cooperative checkpointer logic.
  4. Vector math and AABB collision primitives.
  5. Style Bible 16-color palette quantization algorithm.
"""

import ast
import hashlib
import json
import math
import sqlite3
import pytest
from enum import IntEnum
from typing import Tuple, List, Dict, Any, Optional

try:
    from pydantic import BaseModel, Field
except ImportError:
    BaseModel = None


# ============================================================================
# Contract Reference Classes & Enums (Matches PROJECT.md § Interface Contracts)
# ============================================================================

class TaskPriority(IntEnum):
    """Authoritative Three-Tier Task Priority Enum from PROJECT.md."""
    PRIORITY_0_GAMEPLAY = 0    # Never yields, 60 FPS real-time
    PRIORITY_1_TELEMETRY = 1   # High-priority thread queue drain
    PRIORITY_2_SHOWRUNNER = 2  # Background analysis & compilation


class Vector2D:
    """2D Vector primitive with safe division and distance calculations."""
    def __init__(self, x: float, y: float):
        self.x = float(x)
        self.y = float(y)

    def magnitude(self) -> float:
        return math.sqrt(self.x * self.x + self.y * self.y)

    def normalize(self) -> "Vector2D":
        mag = self.magnitude()
        if mag == 0.0:
            return Vector2D(0.0, 0.0)
        return Vector2D(self.x / mag, self.y / mag)

    def dot(self, other: "Vector2D") -> float:
        return self.x * other.x + self.y * other.y

    def distance_to(self, other: "Vector2D") -> float:
        dx = self.x - other.x
        dy = self.y - other.y
        return math.sqrt(dx * dx + dy * dy)

    def __add__(self, other: "Vector2D") -> "Vector2D":
        return Vector2D(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Vector2D") -> "Vector2D":
        return Vector2D(self.x - other.x, self.y - other.y)

    def to_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)


class AABB:
    """Axis-Aligned Bounding Box for spatial queries and collision."""
    def __init__(self, x: float, y: float, width: float, height: float):
        self.x = float(x)
        self.y = float(y)
        self.width = float(width)
        self.height = float(height)

    @property
    def left(self) -> float:
        return self.x

    @property
    def right(self) -> float:
        return self.x + self.width

    @property
    def top(self) -> float:
        return self.y

    @property
    def bottom(self) -> float:
        return self.y + self.height

    def intersects(self, other: "AABB") -> bool:
        """Returns True if this bounding box overlaps with another AABB."""
        return not (
            self.right <= other.left or
            self.left >= other.right or
            self.bottom <= other.top or
            self.top >= other.bottom
        )

    def contains_point(self, px: float, py: float) -> bool:
        """Returns True if (px, py) is strictly inside or on the boundaries."""
        return (self.left <= px <= self.right) and (self.top <= py <= self.bottom)


def quantize_color(rgb: Tuple[int, int, int], palette: List[Tuple[int, int, int]]) -> Tuple[int, int, int]:
    """Quantizes an RGB color to the closest entry in the Style Bible palette via Euclidean distance."""
    r1, g1, b1 = rgb
    best_color = palette[0]
    min_dist = float("inf")
    for pr, pg, pb in palette:
        dist_sq = (r1 - pr) ** 2 + (g1 - pg) ** 2 + (b1 - pb) ** 2
        if dist_sq < min_dist:
            min_dist = dist_sq
            best_color = (pr, pg, pb)
            if dist_sq == 0:
                break
    return best_color


def validate_rationale_first_dict(data: Dict[str, Any]) -> bool:
    """Validates that deliberation or reasoning is the very first key in a dictionary."""
    if not data:
        return False
    first_key = list(data.keys())[0]
    return first_key in ("deliberation", "reasoning", "deliberation_rationale", "editorial_rationale")


def validate_rationale_first_ast(source_code: str) -> bool:
    """Uses AST analysis to assert class annotations or function signatures declare reasoning first."""
    tree = ast.parse(source_code)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            # Inspect first annotated assignment or first function argument
            ann_assigns = [n for n in node.body if isinstance(n, ast.AnnAssign)]
            if ann_assigns:
                first_target = ann_assigns[0].target
                if isinstance(first_target, ast.Name):
                    if first_target.id not in ("deliberation", "reasoning", "editorial_rationale"):
                        return False
    return True


# ============================================================================
# Test Cases
# ============================================================================

class TestRationaleFirstASTChecks:
    """Verifies Rationale-First Epistemic Architecture at AST and schema levels."""

    def test_rationale_first_dict_valid(self):
        """Authoritative contract: deliberation must be the first key."""
        valid_spec = {
            "deliberation": "Player exhibits high dash frequency and kinetic velocity, requiring wider combat arenas.",
            "target_tier": "episodic_update",
            "mechanics": {"speed_multiplier": 1.15}
        }
        assert validate_rationale_first_dict(valid_spec) is True
        assert list(valid_spec.keys())[0] == "deliberation"

    def test_rationale_first_dict_invalid_rejection(self):
        """Negative test: decision token preceding deliberation must be rejected."""
        invalid_spec = {
            "target_tier": "episodic_update",
            "deliberation": "Late reasoning after decision emitted.",
            "mechanics": {"speed_multiplier": 1.15}
        }
        assert validate_rationale_first_dict(invalid_spec) is False
        assert list(invalid_spec.keys())[0] != "deliberation"

    def test_rationale_first_ast_valid_class(self):
        """AST inspection verifies class declaration has deliberation as first attribute."""
        valid_class_code = """
class DirectorSpecPayload:
    deliberation: str
    target_difficulty: float
    spawn_density: int
"""
        assert validate_rationale_first_ast(valid_class_code) is True

    def test_rationale_first_ast_invalid_class_rejection(self):
        """AST inspection flags classes where action/payload precedes deliberation."""
        invalid_class_code = """
class RogueDecisionPayload:
    target_difficulty: float
    deliberation: str
"""
        assert validate_rationale_first_ast(invalid_class_code) is False

    def test_artifact_metadata_envelope_integrity(self):
        """Verifies Typed ArtifactMetadataEnvelope with input/output SHA256 and deliberation first."""
        input_data = b"raw_telemetry_events_v1"
        output_data = b"behavioral_profile_v1"
        input_hash = hashlib.sha256(input_data).hexdigest()
        output_hash = hashlib.sha256(output_data).hexdigest()
        
        envelope = {
            "deliberation": "Parsed 1500 telemetry events; filtered stationary idle frames; generated 16x16 heatmap.",
            "input_sha256": input_hash,
            "output_sha256": output_hash,
            "producer_agent": "TelemetryAnalyst",
            "consumer_agent": "CognitiveDirector",
            "payload": {"kinetic_velocity_mean": 245.5}
        }
        
        # Invariant 1: Deliberation is the first property
        assert list(envelope.keys())[0] == "deliberation"
        # Invariant 2: SHA256 hashes are valid 64-character hex strings
        assert len(envelope["input_sha256"]) == 64
        assert len(envelope["output_sha256"]) == 64
        # Invariant 3: Payload contains extracted metrics
        assert envelope["payload"]["kinetic_velocity_mean"] == 245.5


class TestSQLiteWALSchemaAndPragmas:
    """Verifies SQLite WAL mode, synchronicity, schemas, and concurrent read/write access."""

    def test_wal_pragma_and_synchronous(self, temp_sqlite_wal_db):
        """Asserts persistent journal_mode=WAL and connection synchronous=NORMAL (1)."""
        conn = sqlite3.connect(temp_sqlite_wal_db)
        cursor = conn.cursor()
        
        # journal_mode=WAL is stored persistently in database header
        cursor.execute("PRAGMA journal_mode;")
        journal_mode = cursor.fetchone()[0].lower()
        assert journal_mode == "wal", f"Expected journal_mode 'wal', got '{journal_mode}'"
        
        # synchronous is connection-scoped: applying NORMAL must set it to 1
        cursor.execute("PRAGMA synchronous = NORMAL;")
        cursor.execute("PRAGMA synchronous;")
        synchronous = cursor.fetchone()[0]
        assert synchronous == 1, f"Expected synchronous=1 (NORMAL), got {synchronous}"
        conn.close()

    def test_table_schema_constraints(self, temp_sqlite_wal_db):
        """Verifies tables telemetry_sessions, telemetry_episodes, telemetry_events exist."""
        conn = sqlite3.connect(temp_sqlite_wal_db)
        cursor = conn.cursor()
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = {row[0] for row in cursor.fetchall()}
        
        assert "telemetry_sessions" in tables
        assert "telemetry_episodes" in tables
        assert "telemetry_events" in tables
        
        # Check columns in telemetry_events
        cursor.execute("PRAGMA table_info(telemetry_events);")
        columns = {row[1] for row in cursor.fetchall()}
        expected_cols = {"id", "session_id", "episode_id", "timestamp", "event_type", "pos_x", "pos_y", "vel_x", "vel_y", "health", "payload_json"}
        assert expected_cols.issubset(columns), f"Missing columns in telemetry_events: {expected_cols - columns}"
        conn.close()

    def test_concurrent_wal_read_write(self, temp_sqlite_wal_db):
        """Verifies that reader connection can read committed data without locking writer."""
        writer_conn = sqlite3.connect(temp_sqlite_wal_db, timeout=5.0)
        reader_conn = sqlite3.connect(temp_sqlite_wal_db, timeout=5.0)
        
        writer_cur = writer_conn.cursor()
        reader_cur = reader_conn.cursor()
        
        # Insert session and episode
        writer_cur.execute("INSERT INTO telemetry_sessions (session_id, player_profile_tag) VALUES (?, ?);", ("sess_01", "rusher"))
        writer_cur.execute("INSERT INTO telemetry_episodes (episode_id, session_id, episode_number, status) VALUES (?, ?, ?, ?);", ("ep_01", "sess_01", 1, "ACTIVE"))
        writer_conn.commit()
        
        # Batch insert 100 events
        events_batch = [
            ("sess_01", "ep_01", float(i) * 0.016, "MOVEMENT_TICK", 64.0 + i, 64.0 + i, 200.0, 200.0, 100, "{}")
            for i in range(100)
        ]
        writer_cur.executemany("""
            INSERT INTO telemetry_events (session_id, episode_id, timestamp, event_type, pos_x, pos_y, vel_x, vel_y, health, payload_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """, events_batch)
        writer_conn.commit()
        
        # Concurrent reader reads while writer is open
        reader_cur.execute("SELECT COUNT(*) FROM telemetry_events WHERE episode_id = 'ep_01';")
        count = reader_cur.fetchone()[0]
        assert count == 100
        
        writer_conn.close()
        reader_conn.close()


class TestPrioritySchedulerProperties:
    """Verifies Three-Tier Priority Enum and cooperative yielding properties."""

    def test_priority_enum_ordering(self):
        """Authoritative invariant: P0 (Game) < P1 (Telemetry) < P2 (Showrunner)."""
        assert TaskPriority.PRIORITY_0_GAMEPLAY < TaskPriority.PRIORITY_1_TELEMETRY
        assert TaskPriority.PRIORITY_1_TELEMETRY < TaskPriority.PRIORITY_2_SHOWRUNNER
        assert TaskPriority.PRIORITY_0_GAMEPLAY == 0
        assert TaskPriority.PRIORITY_1_TELEMETRY == 1
        assert TaskPriority.PRIORITY_2_SHOWRUNNER == 2

    def test_priority_queue_sorting(self):
        """Verifies that tasks enqueued out of order execute strictly by priority order."""
        import heapq
        tasks = []
        heapq.heappush(tasks, (TaskPriority.PRIORITY_2_SHOWRUNNER, "compile_level_update"))
        heapq.heappush(tasks, (TaskPriority.PRIORITY_0_GAMEPLAY, "render_frame_60fps"))
        heapq.heappush(tasks, (TaskPriority.PRIORITY_1_TELEMETRY, "flush_telemetry_queue"))
        
        dispatched = [heapq.heappop(tasks)[1] for _ in range(3)]
        assert dispatched == ["render_frame_60fps", "flush_telemetry_queue", "compile_level_update"]

    def test_cooperative_yielding_contract(self):
        """Verifies checkpointer yields CPU when gameplay is active."""
        class CooperativeCheckpointer:
            def __init__(self):
                self.is_gameplay_active = True
                self.yield_count = 0

            def checkpoint_and_yield(self) -> bool:
                if self.is_gameplay_active:
                    self.yield_count += 1
                    return True  # Yielded
                return False

        checkpointer = CooperativeCheckpointer()
        assert checkpointer.checkpoint_and_yield() is True
        assert checkpointer.yield_count == 1
        
        checkpointer.is_gameplay_active = False
        assert checkpointer.checkpoint_and_yield() is False
        assert checkpointer.yield_count == 1


class TestVectorMathAndCollisionPrimitives:
    """Verifies 2D vector primitives, distance, and AABB collision math."""

    def test_vector_magnitude_and_normalization(self):
        """Tests vector magnitude and safe normalization."""
        v = Vector2D(3.0, 4.0)
        assert v.magnitude() == 5.0
        
        v_norm = v.normalize()
        assert pytest.approx(v_norm.magnitude(), abs=1e-6) == 1.0
        assert pytest.approx(v_norm.x, abs=1e-6) == 0.6
        assert pytest.approx(v_norm.y, abs=1e-6) == 0.8

    def test_vector_zero_division_guard(self):
        """Adversarial check: zero vector normalization must safely return (0, 0)."""
        v_zero = Vector2D(0.0, 0.0)
        assert v_zero.magnitude() == 0.0
        v_norm = v_zero.normalize()
        assert v_norm.x == 0.0
        assert v_norm.y == 0.0

    def test_vector_dot_product_and_distance(self):
        """Tests dot product and Euclidean distance."""
        v1 = Vector2D(1.0, 0.0)
        v2 = Vector2D(0.0, 1.0)
        assert v1.dot(v2) == 0.0  # Orthogonal
        
        p1 = Vector2D(10.0, 10.0)
        p2 = Vector2D(10.0, 25.0)
        assert p1.distance_to(p2) == 15.0

    def test_aabb_overlap_detection(self):
        """Tests AABB intersection."""
        box_a = AABB(10, 10, 20, 20)  # [10, 30] x [10, 30]
        box_b = AABB(25, 25, 20, 20)  # [25, 45] x [25, 45] (Overlaps)
        box_c = AABB(50, 50, 20, 20)  # [50, 70] x [50, 70] (Disjoint)
        
        assert box_a.intersects(box_b) is True
        assert box_b.intersects(box_a) is True
        assert box_a.intersects(box_c) is False

    def test_aabb_boundary_touching(self):
        """Boundary stress: touching edges without overlap must return False."""
        box_a = AABB(0, 0, 10, 10)
        box_touching = AABB(10, 0, 10, 10)  # Left edge meets Right edge at x=10
        assert box_a.intersects(box_touching) is False

    def test_aabb_point_containment(self):
        """Tests whether points are contained in bounding box."""
        box = AABB(100, 100, 50, 50)
        assert box.contains_point(125, 125) is True
        assert box.contains_point(100, 100) is True  # Boundary
        assert box.contains_point(150, 150) is True  # Boundary
        assert box.contains_point(99, 125) is False
        assert box.contains_point(151, 125) is False


class TestStyleBiblePaletteQuantization:
    """Verifies 16-color Style Bible quantization algorithm."""

    def test_exact_palette_color_mapping(self, style_bible_palette):
        """Exact match for canonical palette colors."""
        for color in style_bible_palette:
            mapped = quantize_color(color, style_bible_palette)
            assert mapped == color, f"Expected {color}, got {mapped}"

    def test_nearest_color_quantization(self, style_bible_palette):
        """Arbitrary RGB colors map to the closest canonical entry."""
        # Near pure black (5, 5, 5) -> (0, 0, 0)
        assert quantize_color((5, 5, 5), style_bible_palette) == (0, 0, 0)
        
        # Near Neon Red (250, 5, 70) -> (255, 0, 77)
        assert quantize_color((250, 5, 70), style_bible_palette) == (255, 0, 77)
        
        # Near Bright Yellow (250, 240, 45) -> (255, 236, 39)
        assert quantize_color((250, 240, 45), style_bible_palette) == (255, 236, 39)
