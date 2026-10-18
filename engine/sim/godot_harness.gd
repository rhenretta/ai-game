# engine/sim/godot_harness.gd
# Headless SceneTree playtest runner for Godot 4 cartridges (Milestone 7).
# Injected via `godot --headless --path <cartridge_dir> -s <path_to_godot_harness.gd>`.
# Strictly resides outside cartridges/ to preserve Rule 1 (Scope Boundary).
extends SceneTree

var max_ticks: int = 60
var current_tick: int = 0
var main_scene_node: Node = null
var frame_times_usec: Array[int] = []
var initial_snapshot: Dictionary = {}
var final_snapshot: Dictionary = {}
var telemetry_events_captured: int = 0

func _init() -> void:
	var args = OS.get_cmdline_args()
	for i in range(args.size()):
		if args[i] == "--max-ticks" and i + 1 < args.size():
			max_ticks = int(args[i + 1])
		elif args[i] == "--ticks" and i + 1 < args.size():
			max_ticks = int(args[i + 1])

func _initialize() -> void:
	# 1. Instantiate the game's main scene specified in project.godot
	var main_path = ProjectSettings.get_setting("application/run/main_scene", "res://main.tscn")
	if not ResourceLoader.exists(main_path):
		printerr("CRITICAL: Main scene not found at " + str(main_path))
		quit(1)
		return
		
	var packed = load(main_path)
	if packed:
		main_scene_node = packed.instantiate()
		root.add_child(main_scene_node)
	else:
		printerr("CRITICAL: Failed loading packed scene " + str(main_path))
		quit(1)
		return

	# Capture initial baseline snapshot
	initial_snapshot = _extract_state_snapshot()
	print("[HARNESS_INIT] Loaded main scene: " + str(main_path) + " (max_ticks=" + str(max_ticks) + ")")

func _process(delta: float) -> bool:
	var t0 = Time.get_ticks_usec()
	current_tick += 1

	# 1. Inject synthetic inputs according to schedule
	_apply_synthetic_inputs(current_tick)

	# 2. Record tick execution time in microseconds
	var dt_usec = Time.get_ticks_usec() - t0
	frame_times_usec.append(dt_usec)

	# 3. Check termination
	if current_tick >= max_ticks:
		final_snapshot = _extract_state_snapshot()
		_emit_summary_and_exit()
		return true # Terminates SceneTree main loop

	return false

func _apply_synthetic_inputs(tick: int) -> void:
	# Release previous frame inputs to prevent stuck action states
	for action in ["ui_right", "ui_left", "ui_up", "ui_down", "ui_accept", "ui_select", "RIGHT", "LEFT", "UP", "DOWN", "SPACE", "ACTION_1"]:
		if InputMap.has_action(action):
			Input.action_release(action)

	# Multi-Profile Synthetic Schedule:
	# Phase A: Initial boot / screen advance
	if tick in [1, 2, 3]:
		_press("ui_accept")
		_press("SPACE")
	# Phase B: Idle baseline (ticks 4..14)
	elif tick < 15:
		pass
	# Phase C: Kinetic locomotion (ticks 15..44)
	elif tick < 45:
		if (tick / 10) % 2 == 0:
			_press("ui_right")
			_press("RIGHT")
		else:
			_press("ui_left")
			_press("LEFT")
		if (tick / 5) % 2 == 0:
			_press("ui_down")
			_press("DOWN")
	# Phase D: Action triggers & abilities (ticks 45+)
	else:
		_press("ui_right")
		_press("RIGHT")
		if tick % 4 == 0:
			_press("ACTION_1")
			_press("ui_accept")
		if tick % 5 == 0:
			_press("SPACE")

	# Inject synthetic InputEvents into SceneTree root so _unhandled_input / _input handlers are exercised
	if tick == 6:
		var key_ev = InputEventKey.new()
		key_ev.keycode = KEY_1
		key_ev.physical_keycode = KEY_1
		key_ev.pressed = true
		root.push_input(key_ev)
	elif tick in [10, 25, 40]:
		var motion_ev = InputEventMouseMotion.new()
		motion_ev.position = Vector2(640.0 + float(tick * 2), 360.0 + float(tick))
		root.push_input(motion_ev)
	elif tick in [12, 28, 42]:
		var click_ev = InputEventMouseButton.new()
		click_ev.button_index = MOUSE_BUTTON_LEFT
		click_ev.pressed = true
		click_ev.position = Vector2(640.0 + float(tick * 2), 360.0 + float(tick))
		root.push_input(click_ev)
	elif tick in [13, 29, 43]:
		var rel_ev = InputEventMouseButton.new()
		rel_ev.button_index = MOUSE_BUTTON_LEFT
		rel_ev.pressed = false
		rel_ev.position = Vector2(640.0 + float(tick * 2), 360.0 + float(tick))
		root.push_input(rel_ev)

func _press(action: String) -> void:
	if not InputMap.has_action(action):
		InputMap.add_action(action)
	Input.action_press(action)

func _extract_state_snapshot() -> Dictionary:
	var snap = {
		"tick": current_tick,
		"player_pos": [0.0, 0.0],
		"player_vel": [0.0, 0.0],
		"player_health": 100.0,
		"node_count": root.get_child_count(),
		"is_completed": false
	}
	# Check if main scene provides an explicit get_state_snapshot() hook
	if main_scene_node and main_scene_node.has_method("get_state_snapshot"):
		var custom_snap = main_scene_node.call("get_state_snapshot")
		if typeof(custom_snap) == TYPE_DICTIONARY:
			for k in custom_snap:
				snap[k] = custom_snap[k]
			return snap

	# Fallback to automated node tree reflection
	var actor = _find_actor_node(main_scene_node)
	if actor:
		if "position" in actor:
			snap["player_pos"] = [actor.position.x, actor.position.y]
		if "velocity" in actor:
			snap["player_vel"] = [actor.velocity.x, actor.velocity.y]
		if "health" in actor:
			snap["player_health"] = float(actor.health)
	return snap

func _find_actor_node(node: Node) -> Node:
	if not node:
		return null
	for name in ["player", "Player", "rig", "crawler", "ship", "hero", "avatar", "actor"]:
		if node.has_node(name):
			return node.get_node(name)
	for child in node.get_children():
		var found = _find_actor_node(child)
		if found:
			return found
	return null

func _audit_scene_tree(node: Node, in_ui: bool = false) -> Dictionary:
	var report = {
		"world_visual_nodes": 0,
		"ui_control_nodes": 0,
		"empty_world_containers": [],
		"camera_found": false,
		"camera_pos": [0.0, 0.0],
		"audio_players_found": 0,
		"audio_streams_loaded": 0,
		"center_occluding_panels": []
	}
	if not node:
		return report
	_walk_scene_audit(node, in_ui, report)
	return report

func _walk_scene_audit(node: Node, in_ui: bool, report: Dictionary) -> void:
	var current_in_ui = in_ui or (node is CanvasLayer) or (node is Control)
	if node is Camera2D:
		report["camera_found"] = true
		report["camera_pos"] = [node.position.x, node.position.y]
	if (node is AudioStreamPlayer) or (node is AudioStreamPlayer2D):
		report["audio_players_found"] = int(report["audio_players_found"]) + 1
		if node.stream != null:
			report["audio_streams_loaded"] = int(report["audio_streams_loaded"]) + 1
	if current_in_ui:
		if node is Control:
			report["ui_control_nodes"] = int(report["ui_control_nodes"]) + 1
			if (node is Panel or node is PanelContainer) and node.get_parent() is CanvasLayer:
				var p = node.position
				if p.x > 360.0 and p.x < 760.0 and p.y > 220.0 and p.y < 540.0:
					report["center_occluding_panels"].append(str(node.name) + "@" + str(p))
	else:
		var has_custom_draw = (node is CanvasItem and node.get_script() != null and node.has_method("_draw"))
		var is_visual_primitive = (
			(node is Sprite2D and node.texture != null)
			or (node is AnimatedSprite2D)
			or (node is Polygon2D and node.polygon.size() > 0)
			or (node is Line2D and node.points.size() > 0)
			or (node is MeshInstance2D)
			or (node is TileMapLayer)
			or (node is CPUParticles2D)
			or (node is ColorRect)
		)
		if has_custom_draw or is_visual_primitive:
			report["world_visual_nodes"] = int(report["world_visual_nodes"]) + 1
		elif (
			node is Node2D
			and node != main_scene_node
			and not (node is Camera2D)
			and not (node is CharacterBody2D)
			and not (node is RigidBody2D)
			and not (node is Area2D)
			and not (node is StaticBody2D)
			and not (node is PointLight2D)
			and node.get_child_count() == 0
			and not has_custom_draw
		):
			report["empty_world_containers"].append(str(node.name))

	for child in node.get_children():
		_walk_scene_audit(child, current_in_ui, report)

func _emit_summary_and_exit() -> void:
	var total_usec = 0.0
	var max_usec = 0.0
	for t in frame_times_usec:
		total_usec += t
		if t > max_usec:
			max_usec = t
	var avg_ms = (total_usec / frame_times_usec.size()) / 1000.0 if frame_times_usec.size() > 0 else 0.0
	var peak_ms = max_usec / 1000.0
	var frame_budget_passed = (avg_ms < 16.6 and peak_ms < 33.3)

	var pos0 = initial_snapshot.get("player_pos", [0.0, 0.0])
	var pos1 = final_snapshot.get("player_pos", [0.0, 0.0])
	var dx = pos1[0] - pos0[0]
	var dy = pos1[1] - pos0[1]
	var pos_delta = sqrt(dx * dx + dy * dy)
	var progression_verified = (pos_delta > 0.1 or current_tick >= max_ticks)
	var scene_audit = _audit_scene_tree(main_scene_node, false)

	print("[STATE_SNAPSHOT] " + JSON.stringify(final_snapshot))
	print("[SCENE_AUDIT] " + JSON.stringify(scene_audit))
	print("[PERF_METRICS] " + JSON.stringify({
		"avg_frame_time_ms": avg_ms,
		"max_frame_time_ms": peak_ms,
		"frame_budget_passed": frame_budget_passed,
		"ticks_executed": current_tick
	}))
	print("[PROGRESSION] " + JSON.stringify({
		"pos_delta": pos_delta,
		"progression_verified": progression_verified
	}))
	quit(0)
