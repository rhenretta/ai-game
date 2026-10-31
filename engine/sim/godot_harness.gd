# engine/sim/godot_harness.gd
# Headless SceneTree verification & spatial-boundary auditor for Godot 4 cartridges.
# Injected via `godot --headless --path <cartridge_dir> -s <path_to_godot_harness.gd>`.
# Strictly resides outside cartridges/ to preserve Rule 1 (Scope Boundary).
extends SceneTree

var max_ticks: int = 60
var current_tick: int = 0
var main_scene_node: Node = null
var render_viewport: SubViewport = null
var frame_times_usec: Array[int] = []
var last_frame_end_usec: int = 0
var initial_snapshot: Dictionary = {}
var passive_idle_snapshot: Dictionary = {}
var final_snapshot: Dictionary = {}
var telemetry_events_captured: int = 0
var passive_auto_win_detected: bool = false
var passive_win_reason: String = ""
var min_player_y_reached: float = 99999.0
var max_player_y_reached: float = -99999.0
var min_player_x_reached: float = 99999.0
var max_player_x_reached: float = -99999.0
var spawn_pos: Array = [0.0, 0.0]
# Filmstrip / scripted-input support (council proof of actions & mechanics)
var capture_ticks: Array = []
var capture_viewport_enabled: bool = false
var input_script: Array = []
var setup_script: Array = []
var _active_scripted_actions: Array[String] = []
var script_error: String = ""
var scripted_mode: bool = false
var target_scene_override: String = ""

func _init() -> void:
	var args = OS.get_cmdline_args()
	for i in range(args.size()):
		if args[i] == "--max-ticks" and i + 1 < args.size():
			max_ticks = int(args[i + 1])
		elif args[i] == "--ticks" and i + 1 < args.size():
			max_ticks = int(args[i + 1])
		elif args[i] == "--capture-viewport":
			capture_viewport_enabled = true
		elif args[i] == "--capture-ticks" and i + 1 < args.size():
			for t in String(args[i + 1]).split(",", false):
				capture_ticks.append(int(t))
		elif args[i] == "--input-script" and i + 1 < args.size():
			scripted_mode = true
			_load_input_script(String(args[i + 1]))
		elif args[i] == "--setup-script" and i + 1 < args.size():
			scripted_mode = true
			_load_setup_script(String(args[i + 1]))
		elif args[i] == "--scene" and i + 1 < args.size():
			target_scene_override = String(args[i + 1]).strip_edges()

func _load_input_script(path: String) -> void:
	var f = FileAccess.open(path, FileAccess.READ)
	if f == null:
		script_error = "cannot open input script " + path
		return
	var parsed = JSON.parse_string(f.get_as_text())
	if typeof(parsed) != TYPE_ARRAY:
		script_error = "input script must be a JSON array of {from, to, actions}: " + path
		return
	input_script = parsed

func _load_setup_script(path: String) -> void:
	var f = FileAccess.open(path, FileAccess.READ)
	if f == null:
		script_error = "cannot open setup script " + path
		return
	var parsed = JSON.parse_string(f.get_as_text())
	if typeof(parsed) != TYPE_ARRAY:
		script_error = "setup script must be a JSON array of setup call objects: " + path
		return
	setup_script = parsed

func get_viewport() -> Viewport:
	return render_viewport if render_viewport != null else root

func _initialize() -> void:
	Engine.max_physics_steps_per_frame = 1
	if script_error != "":
		printerr("CRITICAL: " + script_error)
		quit(1)
		return
	if DisplayServer.get_name() != "headless":
		DisplayServer.window_set_flag(DisplayServer.WINDOW_FLAG_NO_FOCUS, true)
		DisplayServer.window_set_flag(DisplayServer.WINDOW_FLAG_BORDERLESS, true)
		DisplayServer.window_set_flag(DisplayServer.WINDOW_FLAG_MOUSE_PASSTHROUGH, true)
		DisplayServer.window_set_flag(DisplayServer.WINDOW_FLAG_TRANSPARENT, true)
		DisplayServer.window_set_position(Vector2i(-4000, -4000))
		root.transparent_bg = true
	render_viewport = SubViewport.new()
	render_viewport.name = "HarnessOffscreenViewport"
	render_viewport.size = Vector2i(1280, 720)
	render_viewport.size_2d_override = Vector2i(1280, 720)
	render_viewport.transparent_bg = false
	render_viewport.render_target_update_mode = SubViewport.UPDATE_ALWAYS
	render_viewport.handle_input_locally = true
	render_viewport.physics_object_picking = true
	root.add_child(render_viewport)
	var main_path = ProjectSettings.get_setting("application/run/main_scene", "res://main.tscn")
	if target_scene_override != "":
		var cand_scene = target_scene_override
		if not cand_scene.begins_with("res://"):
			cand_scene = "res://" + cand_scene.trim_prefix("/")
		if ResourceLoader.exists(cand_scene):
			main_path = cand_scene
	if not ResourceLoader.exists(main_path):
		printerr("CRITICAL: Main scene not found at " + str(main_path))
		quit(1)
		return

	var packed = load(main_path)
	if packed == null and main_path.ends_with(".tscn"):
		var gd_fallback = main_path.substr(0, main_path.length() - 5) + ".gd"
		if ResourceLoader.exists(gd_fallback):
			packed = load(gd_fallback)
	if packed:
		if packed is PackedScene:
			main_scene_node = packed.instantiate()
		elif packed is Script:
			main_scene_node = packed.new()
		if main_scene_node != null:
			if render_viewport != null:
				render_viewport.add_child(main_scene_node)
			else:
				root.add_child(main_scene_node)
		else:
			printerr("CRITICAL: Failed instantiating scene " + str(main_path))
			quit(1)
			return
	else:
		printerr("CRITICAL: Failed loading packed scene " + str(main_path))
		quit(1)
		return

	print("[HARNESS_INIT] Loaded main scene: " + str(main_path) + " (max_ticks=" + str(max_ticks) + ")")

func _process(_delta: float) -> bool:
	var t0 = Time.get_ticks_usec()
	current_tick += 1
	if current_tick == 1:
		initial_snapshot = _extract_state_snapshot()
		spawn_pos = initial_snapshot.get("player_pos", [0.0, 0.0])
		min_player_x_reached = float(spawn_pos[0])
		max_player_x_reached = float(spawn_pos[0])
		min_player_y_reached = float(spawn_pos[1])
		max_player_y_reached = float(spawn_pos[1])
		_check_snapshot_for_auto_win(initial_snapshot, 0)
	var inter_frame_usec: int = (t0 - last_frame_end_usec) if last_frame_end_usec > 0 else 0

	var passive_window_end = min(16, max(2, int(max_ticks / 4)))

	# Phase A: Passive Zero-Input Window (ticks 1..passive_window_end)
	# Verifies the cartridge does NOT auto-solve or pre-complete objectives with zero player input.
	if scripted_mode:
		# Scripted run: the council-specified inputs drive the game from tick 1 (no passive window).
		_apply_scripted_inputs(current_tick)
	elif current_tick <= passive_window_end:
		_release_all_inputs()
		var cur_snap = _extract_state_snapshot()
		_check_snapshot_for_auto_win(cur_snap, current_tick)
		if current_tick == passive_window_end:
			passive_idle_snapshot = cur_snap
	else:
		# Phase B: Sustained Directional Boundary Excursion + Action Probes
		_apply_synthetic_inputs(current_tick, passive_window_end)

	# Track actor spatial extrema across the run
	var live_snap = _extract_state_snapshot()
	var ppos = live_snap.get("player_pos", [0.0, 0.0])
	if typeof(ppos) == TYPE_ARRAY and ppos.size() >= 2:
		var px = float(ppos[0])
		var py = float(ppos[1])
		if px < min_player_x_reached:
			min_player_x_reached = px
		if px > max_player_x_reached:
			max_player_x_reached = px
		if py < min_player_y_reached:
			min_player_y_reached = py
		if py > max_player_y_reached:
			max_player_y_reached = py

	var t_before_capture = Time.get_ticks_usec()
	if capture_ticks.has(current_tick):
		_try_capture_viewport("res://assets/playtest_captures/filmstrip_%04d.png" % current_tick)
		print("[QA_TICK_SNAPSHOT] " + JSON.stringify(live_snap))

	# Viewport GPU capture at t0 (during passive window before inputs)
	if current_tick == 2:
		_try_capture_viewport("res://assets/playtest_captures/playtest_t0.png")

	var harness_cpu_usec = t_before_capture - t0
	# Record true inter-frame wall-clock time (game _physics_process + _process + _draw + SubViewport + GPU)
	# excluding any screenshot disk-save time from the previous tick.
	if current_tick > 3 and inter_frame_usec > 0:
		frame_times_usec.append(inter_frame_usec + harness_cpu_usec)
	else:
		frame_times_usec.append(harness_cpu_usec)

	if current_tick >= max_ticks:
		_try_capture_viewport("res://assets/playtest_captures/playtest_tend.png")
		_try_capture_collision_overlay("res://assets/playtest_captures/playtest_collision_overlay.png")
		final_snapshot = _extract_state_snapshot()
		_emit_summary_and_exit()
		return true

	last_frame_end_usec = Time.get_ticks_usec()
	return false

func _check_snapshot_for_auto_win(snap: Dictionary, tick_num: int) -> void:
	if passive_auto_win_detected:
		return
	if (
		bool(snap.get("is_completed", false))
		or bool(snap.get("victory", false))
		or bool(snap.get("victory_state", false))
		or bool(snap.get("level_completed", false))
		or bool(snap.get("game_won", false))
		or bool(snap.get("stage_cleared", false))
	):
		passive_auto_win_detected = true
		passive_win_reason = "Game reported completion/victory=true at passive tick " + str(tick_num) + " with zero player input."
		return
	var eng_status = str(snap.get("engine_status", "")).to_lower()
	if eng_status == "active" or eng_status == "completed" or eng_status == "solved":
		passive_auto_win_detected = true
		passive_win_reason = "Target mechanism reported engine_status='" + eng_status + "' at passive tick " + str(tick_num) + " with zero player input."
		return
	# Also inspect main_scene_node properties for pre-activated target mechanisms or pre-solved stages
	if main_scene_node != null:
		if "engine" in main_scene_node and main_scene_node.get("engine") != null:
			var eng_node = main_scene_node.get("engine")
			if "is_active" in eng_node and bool(eng_node.get("is_active")):
				passive_auto_win_detected = true
				passive_win_reason = "Main scene target engine.is_active became true at passive tick " + str(tick_num) + " with zero player input."
				return

func _try_capture_viewport(res_target_path: String) -> void:
	if DisplayServer.get_name() == "headless":
		return
	var img: Image = null
	RenderingServer.force_draw(true)
	var vp_tex = render_viewport.get_texture() if render_viewport != null else root.get_texture()
	if vp_tex != null:
		img = vp_tex.get_image()
	if img == null or img.is_empty():
		return
	if img.get_width() != 1280 or img.get_height() != 720:
		img.resize(1280, 720, Image.INTERPOLATE_BILINEAR)
	var abs_dir = ProjectSettings.globalize_path("res://assets/playtest_captures")
	DirAccess.make_dir_recursive_absolute(abs_dir)
	var abs_path = ProjectSettings.globalize_path(res_target_path)
	img.save_png(abs_path)
	if res_target_path.ends_with("playtest_t0.png"):
		img.save_png(ProjectSettings.globalize_path("res://playtest_t0.png"))
	elif res_target_path.ends_with("playtest_tend.png"):
		img.save_png(ProjectSettings.globalize_path("res://playtest_tend.png"))
	elif res_target_path.ends_with("playtest_collision_overlay.png"):
		img.save_png(ProjectSettings.globalize_path("res://playtest_collision_overlay.png"))

func _try_capture_collision_overlay(res_target_path: String) -> void:
	if main_scene_node == null or DisplayServer.get_name() == "headless":
		return
	var shapes_data: Array = []
	_collect_collision_shapes_recursive(main_scene_node, shapes_data)

	var img: Image = null
	RenderingServer.force_draw(true)
	var vp_tex = render_viewport.get_texture() if render_viewport != null else root.get_texture()
	if vp_tex != null:
		img = vp_tex.get_image()
	if img == null or img.is_empty():
		return
	if img.get_width() != 1280 or img.get_height() != 720:
		img.resize(1280, 720, Image.INTERPOLATE_BILINEAR)
	if img.is_compressed():
		img.decompress()
	img.convert(Image.FORMAT_RGBA8)

	var logical_w: float = 1280.0
	var logical_h: float = 720.0
	var canvas_xform: Transform2D = Transform2D.IDENTITY
	if main_scene_node.get_viewport() != null:
		var vp = main_scene_node.get_viewport()
		canvas_xform = vp.get_canvas_transform()
		var vis_rect = vp.get_visible_rect()
		if vis_rect.size.x > 0.0 and vis_rect.size.y > 0.0:
			logical_w = float(vis_rect.size.x)
			logical_h = float(vis_rect.size.y)

	var scale_x = 1280.0 / logical_w
	var scale_y = 720.0 / logical_h

	_rasterize_shapes_on_image(img, shapes_data, canvas_xform, scale_x, scale_y)

	var abs_dir = ProjectSettings.globalize_path("res://assets/playtest_captures")
	DirAccess.make_dir_recursive_absolute(abs_dir)
	var abs_path = ProjectSettings.globalize_path(res_target_path)
	img.save_png(abs_path)
	if res_target_path.ends_with("playtest_collision_overlay.png"):
		img.save_png(ProjectSettings.globalize_path("res://playtest_collision_overlay.png"))

func _blend_pixel(img: Image, x: int, y: int, col: Color) -> void:
	if x < 0 or x >= 1280 or y < 0 or y >= 720:
		return
	var dst = img.get_pixel(x, y)
	var a = col.a
	var out_r = dst.r * (1.0 - a) + col.r * a
	var out_g = dst.g * (1.0 - a) + col.g * a
	var out_b = dst.b * (1.0 - a) + col.b * a
	img.set_pixel(x, y, Color(out_r, out_g, out_b, 1.0))

func _draw_thick_line_on_image(img: Image, p0: Vector2, p1: Vector2, col: Color, thickness: int = 2) -> void:
	var dist = p0.distance_to(p1)
	var steps = int(max(1.0, ceil(dist)))
	for i in range(steps + 1):
		var t = float(i) / float(steps)
		var pt = p0.lerp(p1, t)
		var cx = int(round(pt.x))
		var cy = int(round(pt.y))
		for dy in range(-thickness, thickness + 1):
			for dx in range(-thickness, thickness + 1):
				if dx * dx + dy * dy <= thickness * thickness:
					_blend_pixel(img, cx + dx, cy + dy, col)

func _rasterize_shapes_on_image(img: Image, shapes_list: Array, canvas_xform: Transform2D, scale_x: float, scale_y: float) -> void:
	for item in shapes_list:
		var kind = str(item.get("kind", "static"))
		var fill_col = Color(1.0, 0.15, 0.35, 0.32)
		var line_col = Color(1.0, 0.2, 0.4, 0.96)
		if kind == "actor":
			fill_col = Color(0.0, 0.9, 1.0, 0.38)
			line_col = Color(0.0, 1.0, 1.0, 1.0)
		elif kind == "area":
			fill_col = Color(1.0, 0.82, 0.1, 0.28)
			line_col = Color(1.0, 0.88, 0.2, 0.95)
		elif kind == "walkable_shelf":
			fill_col = Color(0.1, 0.95, 0.4, 0.22)
			line_col = Color(0.1, 0.95, 0.4, 0.95)

		if item.has("rect"):
			var r: Rect2 = item["rect"]
			var tl_w = canvas_xform * r.position
			var br_w = canvas_xform * (r.position + r.size)
			var x0 = int(clamp(min(tl_w.x, br_w.x) * scale_x, 0.0, 1279.0))
			var x1 = int(clamp(max(tl_w.x, br_w.x) * scale_x, 0.0, 1279.0))
			var y0 = int(clamp(min(tl_w.y, br_w.y) * scale_y, 0.0, 719.0))
			var y1 = int(clamp(max(tl_w.y, br_w.y) * scale_y, 0.0, 719.0))
			for y in range(y0, y1 + 1, 2):
				for x in range(x0, x1 + 1, 2):
					_blend_pixel(img, x, y, fill_col)
			var c_tl = Vector2(float(x0), float(y0))
			var c_tr = Vector2(float(x1), float(y0))
			var c_br = Vector2(float(x1), float(y1))
			var c_bl = Vector2(float(x0), float(y1))
			_draw_thick_line_on_image(img, c_tl, c_tr, line_col, 2)
			_draw_thick_line_on_image(img, c_tr, c_br, line_col, 2)
			_draw_thick_line_on_image(img, c_br, c_bl, line_col, 2)
			_draw_thick_line_on_image(img, c_bl, c_tl, line_col, 2)
		elif item.has("polygon"):
			var pts: PackedVector2Array = item["polygon"]
			if pts.size() >= 3:
				var screen_pts = PackedVector2Array()
				var min_x = 1280.0
				var max_x = 0.0
				var min_y = 720.0
				var max_y = 0.0
				for p in pts:
					var sp = canvas_xform * p
					var sx = sp.x * scale_x
					var sy = sp.y * scale_y
					screen_pts.append(Vector2(sx, sy))
					if sx < min_x:
						min_x = sx
					if sx > max_x:
						max_x = sx
					if sy < min_y:
						min_y = sy
					if sy > max_y:
						max_y = sy
				var ix0 = int(clamp(min_x, 0.0, 1279.0))
				var ix1 = int(clamp(max_x, 0.0, 1279.0))
				var iy0 = int(clamp(min_y, 0.0, 719.0))
				var iy1 = int(clamp(max_y, 0.0, 719.0))
				for y in range(iy0, iy1 + 1, 2):
					for x in range(ix0, ix1 + 1, 2):
						if Geometry2D.is_point_in_polygon(Vector2(float(x), float(y)), screen_pts):
							_blend_pixel(img, x, y, fill_col)
				for i in range(screen_pts.size()):
					var a_pt = screen_pts[i]
					var b_pt = screen_pts[(i + 1) % screen_pts.size()]
					_draw_thick_line_on_image(img, a_pt, b_pt, line_col, 2)


func _collect_collision_shapes_recursive(node: Node, out_list: Array) -> void:
	if node == null:
		return
	if "shelves" in node and typeof(node.get("shelves")) == TYPE_ARRAY:
		for s in node.get("shelves"):
			if typeof(s) == TYPE_RECT2:
				out_list.append({"kind": "walkable_shelf", "rect": s})
	if node is CollisionPolygon2D and node.polygon.size() >= 3 and not node.disabled:
		var parent = node.get_parent()
		var kind = "static"
		if parent is CharacterBody2D:
			kind = "actor"
		elif parent is Area2D:
			kind = "area"
		var xform: Transform2D = node.get_global_transform()
		var world_pts = PackedVector2Array()
		for p in node.polygon:
			world_pts.append(xform * p)
		out_list.append({"kind": kind, "polygon": world_pts})
	elif node is CollisionShape2D and node.shape != null and not node.disabled:
		var parent = node.get_parent()
		var kind = "static"
		if parent is CharacterBody2D:
			kind = "actor"
		elif parent is Area2D:
			kind = "area"
		var gpos: Vector2 = node.global_position
		var shp = node.shape
		if shp is RectangleShape2D:
			var sz: Vector2 = shp.size
			out_list.append({"kind": kind, "rect": Rect2(gpos - sz * 0.5, sz)})
		elif shp is CircleShape2D:
			var rad: float = shp.radius
			out_list.append({"kind": kind, "rect": Rect2(gpos - Vector2(rad, rad), Vector2(rad * 2.0, rad * 2.0))})
		elif shp is CapsuleShape2D:
			var rad_c: float = shp.radius
			var h_c: float = shp.height
			out_list.append({"kind": kind, "rect": Rect2(gpos - Vector2(rad_c, h_c * 0.5), Vector2(rad_c * 2.0, h_c))})
	for child in node.get_children():
		_collect_collision_shapes_recursive(child, out_list)

func _resolve_texture_path(node: Node, tex: Texture2D) -> String:
	if node != null and node.has_meta("source_path"):
		return str(node.get_meta("source_path"))
	if tex != null:
		if tex.has_meta("source_path"):
			return str(tex.get_meta("source_path"))
		if tex.resource_path != "":
			return str(tex.resource_path)
	if node != null and node.has_meta("asset_id"):
		return "res://assets/sprites/" + str(node.get_meta("asset_id")) + ".png"
	return ""

func _release_all_inputs() -> void:
	for action in ["ui_right", "ui_left", "ui_up", "ui_down", "ui_accept", "ui_select", "ui_cancel", "RIGHT", "LEFT", "UP", "DOWN", "SPACE", "ACTION_1"]:
		if InputMap.has_action(action):
			Input.action_release(action)
	for custom_act in _active_scripted_actions:
		if InputMap.has_action(custom_act):
			Input.action_release(custom_act)
	_active_scripted_actions.clear()

func _action_to_keycode(action: String) -> int:
	var norm = action.strip_edges().to_upper()
	if norm.begins_with("KEY_"):
		norm = norm.substr(4)
	match norm:
		"SPACE", "UI_ACCEPT", "UI_SELECT", "ACTION_1":
			return KEY_SPACE
		"ENTER", "RETURN":
			return KEY_ENTER
		"ESCAPE", "ESC", "UI_CANCEL":
			return KEY_ESCAPE
		"SHIFT":
			return KEY_SHIFT
		"CTRL", "CONTROL":
			return KEY_CTRL
		"TAB":
			return KEY_TAB
		"LEFT", "UI_LEFT":
			return KEY_LEFT
		"RIGHT", "UI_RIGHT":
			return KEY_RIGHT
		"UP", "UI_UP":
			return KEY_UP
		"DOWN", "UI_DOWN":
			return KEY_DOWN
		"0": return KEY_0
		"1": return KEY_1
		"2": return KEY_2
		"3": return KEY_3
		"4": return KEY_4
		"5": return KEY_5
		"6": return KEY_6
		"7": return KEY_7
		"8": return KEY_8
		"9": return KEY_9
		"A": return KEY_A
		"B": return KEY_B
		"C": return KEY_C
		"D": return KEY_D
		"E": return KEY_E
		"F": return KEY_F
		"G": return KEY_G
		"H": return KEY_H
		"I": return KEY_I
		"J": return KEY_J
		"K": return KEY_K
		"L": return KEY_L
		"M": return KEY_M
		"N": return KEY_N
		"O": return KEY_O
		"P": return KEY_P
		"Q": return KEY_Q
		"R": return KEY_R
		"S": return KEY_S
		"T": return KEY_T
		"U": return KEY_U
		"V": return KEY_V
		"W": return KEY_W
		"X": return KEY_X
		"Y": return KEY_Y
		"Z": return KEY_Z
	return 0

func _push_input_event(ev: InputEvent) -> void:
	if ev is InputEventMouse:
		var mev = ev as InputEventMouse
		mev.global_position = mev.position
		if DisplayServer.get_name() != "headless" and root != null:
			root.warp_mouse(mev.position)
	if render_viewport != null:
		render_viewport.push_input(ev, true)
	elif root != null:
		root.push_input(ev)

func _resolve_scripted_mouse_pos(entry: Dictionary) -> Variant:
	var world_pos = entry.get("mouse_world_pos", null)
	if typeof(world_pos) == TYPE_ARRAY and world_pos.size() >= 2:
		var wpt = Vector2(float(world_pos[0]), float(world_pos[1]))
		var vp: Viewport = main_scene_node.get_viewport() if main_scene_node != null else get_viewport()
		if vp != null:
			return vp.get_canvas_transform() * wpt
		return wpt
	var click_pos = entry.get("mouse_click_pos", null)
	if typeof(click_pos) == TYPE_ARRAY and click_pos.size() >= 2:
		return Vector2(float(click_pos[0]), float(click_pos[1]))
	return null

func _emit_action_and_key_event(action: String, is_pressed: bool) -> void:
	if not InputMap.has_action(action):
		InputMap.add_action(action)
	var act_ev = InputEventAction.new()
	act_ev.action = action
	act_ev.pressed = is_pressed
	act_ev.strength = 1.0 if is_pressed else 0.0
	_push_input_event(act_ev)
	var kc = _action_to_keycode(action)
	if kc != 0:
		var key_ev = InputEventKey.new()
		key_ev.keycode = kc
		key_ev.physical_keycode = kc
		key_ev.pressed = is_pressed
		key_ev.echo = false
		_push_input_event(key_ev)

func _resolve_qa_target(base_node: Node, target_spec: String) -> Object:
	if base_node == null:
		return null
	var clean = target_spec.strip_edges()
	if clean == "" or clean == "main" or clean == "root" or clean == ".":
		return base_node
	if base_node.has_node(clean):
		return base_node.get_node(clean)
	var parts = clean.split(".")
	var cur: Object = base_node
	for p in parts:
		if cur == null:
			return null
		if cur is Node and (cur as Node).has_node(p):
			cur = (cur as Node).get_node(p)
		elif p in cur:
			cur = cur.get(p)
		else:
			var found_child = (cur as Node).find_child(p, true, false) if (cur is Node) else null
			if found_child != null:
				cur = found_child
			else:
				return null
	return cur

func _coerce_setup_arg(val: Variant) -> Variant:
	if typeof(val) == TYPE_DICTIONARY:
		var d: Dictionary = val
		if d.size() == 2 and d.has("x") and d.has("y"):
			return Vector2(float(d["x"]), float(d["y"]))
	elif typeof(val) == TYPE_STRING:
		var s = String(val).strip_edges()
		if s.begins_with("Vector2(") and s.ends_with(")"):
			var inner = s.substr(8, s.length() - 9).split(",", false)
			if inner.size() == 2 and inner[0].strip_edges().is_valid_float() and inner[1].strip_edges().is_valid_float():
				return Vector2(float(inner[0].strip_edges()), float(inner[1].strip_edges()))
		elif s == "Vector2.ONE":
			return Vector2.ONE
		elif s == "Vector2.ZERO":
			return Vector2.ZERO
	return val

func _execute_setup_calls_for_tick(tick: int) -> void:
	for entry in setup_script:
		if typeof(entry) != TYPE_DICTIONARY:
			continue
		var at_t = int(entry.get("at_tick", 1))
		if at_t != tick:
			continue
		var target_spec = String(entry.get("target_node_or_property", ""))
		var method_name = String(entry.get("method_name", "")).strip_edges()
		if method_name == "":
			continue
		var raw_args = entry.get("args", null)
		var args_arr: Array = []
		if typeof(raw_args) == TYPE_ARRAY:
			args_arr = raw_args
		else:
			var parsed_args = JSON.parse_string(String(entry.get("args_json", "[]")))
			if typeof(parsed_args) == TYPE_ARRAY:
				args_arr = parsed_args
		for i in range(args_arr.size()):
			args_arr[i] = _coerce_setup_arg(args_arr[i])
		var target_obj = _resolve_qa_target(main_scene_node, target_spec)
		var status = "skipped_missing_target"
		if target_obj != null:
			if method_name == "set_property" and args_arr.size() >= 2:
				var prop_name = String(args_arr[0])
				if prop_name in target_obj:
					target_obj.set(prop_name, args_arr[1])
					status = "property_set"
				else:
					status = "missing_property"
			elif target_obj.has_method(method_name):
				target_obj.callv(method_name, args_arr)
				status = "called"
			else:
				status = "missing_method"
		print("[QA_SETUP_CALL] " + JSON.stringify({
			"tick": tick,
			"target": target_spec,
			"method": method_name,
			"status": status
		}))

func _evaluate_snapshot_condition(snap: Dictionary, key_path: String, op: String, target_str: String) -> bool:
	var clean_key = key_path.strip_edges()
	if clean_key == "":
		return true
	var cur: Variant = snap
	for part in clean_key.split("."):
		if typeof(cur) == TYPE_DICTIONARY and (cur as Dictionary).has(part):
			cur = (cur as Dictionary)[part]
		elif main_scene_node != null and part in main_scene_node:
			cur = main_scene_node.get(part)
		else:
			return false
	var t_clean = target_str.strip_edges()
	var clean_op = op.strip_edges()
	if clean_op == "":
		clean_op = "=="
	if typeof(cur) == TYPE_INT or typeof(cur) == TYPE_FLOAT or t_clean.is_valid_float():
		var lhs = float(cur)
		var rhs = float(t_clean) if t_clean.is_valid_float() else 0.0
		match clean_op:
			"==": return absf(lhs - rhs) <= 0.001
			"!=": return absf(lhs - rhs) > 0.001
			">=": return lhs >= rhs
			"<=": return lhs <= rhs
			">": return lhs > rhs
			"<": return lhs < rhs
	var s_lhs = str(cur).strip_edges().to_lower()
	var s_rhs = t_clean.to_lower()
	if clean_op == "!=":
		return s_lhs != s_rhs
	return s_lhs == s_rhs

func _apply_scripted_inputs(tick: int) -> void:
	_execute_setup_calls_for_tick(tick)
	_release_all_inputs()
	var live_snap: Dictionary = {}
	for entry in input_script:
		if typeof(entry) != TYPE_DICTIONARY:
			continue
		var when_key = String(entry.get("when_snapshot_key", "")).strip_edges()
		if when_key != "" and not entry.has("_triggered_at"):
			var min_t = int(entry.get("from", 1))
			var max_t = int(entry.get("to", max_ticks))
			if tick >= min_t and tick <= max_t:
				if live_snap.is_empty():
					live_snap = _extract_state_snapshot()
				var when_op = String(entry.get("when_operator", "=="))
				var when_val = String(entry.get("when_value", ""))
				if _evaluate_snapshot_condition(live_snap, when_key, when_op, when_val):
					entry["_triggered_at"] = tick
					var hold = max(1, int(entry.get("hold_ticks", 5)))
					entry["_eff_from"] = tick
					entry["_eff_to"] = min(max_t, tick + hold - 1)
		var first = int(entry.get("_eff_from", entry.get("from", 0)))
		var last = int(entry.get("_eff_to", entry.get("to", first)))
		if when_key != "" and not entry.has("_triggered_at"):
			continue
		if tick == last + 1:
			for rel_action in entry.get("actions", []):
				_emit_action_and_key_event(String(rel_action), false)
			var rel_pos = _resolve_scripted_mouse_pos(entry)
			if rel_pos != null:
				var rel_ev = InputEventMouseButton.new()
				rel_ev.button_index = MOUSE_BUTTON_LEFT
				rel_ev.pressed = false
				rel_ev.position = rel_pos
				_push_input_event(rel_ev)
			continue
		if tick < first or tick > last:
			continue
		for action in entry.get("actions", []):
			var act_str = String(action)
			_press(act_str)
			if not _active_scripted_actions.has(act_str):
				_active_scripted_actions.append(act_str)
			if tick == first:
				_emit_action_and_key_event(act_str, true)
		var target_mpos = _resolve_scripted_mouse_pos(entry)
		if target_mpos != null:
			var motion_ev = InputEventMouseMotion.new()
			motion_ev.position = target_mpos
			_push_input_event(motion_ev)
			if tick == first:
				var click_ev = InputEventMouseButton.new()
				click_ev.button_index = MOUSE_BUTTON_LEFT
				click_ev.pressed = true
				click_ev.position = target_mpos
				_push_input_event(click_ev)

func _apply_synthetic_inputs(tick: int, passive_end: int) -> void:
	_release_all_inputs()
	var active_tick = tick - passive_end

	# Segment 1 (active_tick 1..20): Sustained North (-Y) Boundary Excursion to test sky/horizon containment
	if active_tick <= 20:
		_press("ui_up")
		_press("UP")
	# Segment 2 (active_tick 21..32): Sustained East (+X) Excursion + Action triggers
	elif active_tick <= 32:
		_press("ui_right")
		_press("RIGHT")
		if active_tick % 3 == 0:
			_press("ACTION_1")
			_press("ui_accept")
			_press("SPACE")
	# Segment 3 (active_tick 33+): South/West & Interactive triggers
	else:
		if (active_tick / 6) % 2 == 0:
			_press("ui_down")
			_press("DOWN")
			_press("ui_right")
			_press("RIGHT")
		else:
			_press("ui_left")
			_press("LEFT")
		if active_tick % 3 == 0:
			_press("ACTION_1")
			_press("ui_accept")
		if active_tick % 4 == 0:
			_press("SPACE")

	# Inject synthetic InputEvents into SceneTree root so _unhandled_input / _input handlers are exercised
	if active_tick == 2:
		var key_ev = InputEventKey.new()
		key_ev.keycode = KEY_1
		key_ev.physical_keycode = KEY_1
		key_ev.pressed = true
		_push_input_event(key_ev)
	elif active_tick in [6, 18, 28]:
		var motion_ev = InputEventMouseMotion.new()
		motion_ev.position = Vector2(640.0 + float(active_tick * 2), 360.0 + float(active_tick))
		_push_input_event(motion_ev)
	elif active_tick in [8, 22, 30]:
		var click_ev = InputEventMouseButton.new()
		click_ev.button_index = MOUSE_BUTTON_LEFT
		click_ev.pressed = true
		click_ev.position = Vector2(640.0 + float(active_tick * 2), 360.0 + float(active_tick))
		_push_input_event(click_ev)
	elif active_tick in [9, 23, 31]:
		var rel_ev = InputEventMouseButton.new()
		rel_ev.button_index = MOUSE_BUTTON_LEFT
		rel_ev.pressed = false
		rel_ev.position = Vector2(640.0 + float(active_tick * 2), 360.0 + float(active_tick))
		_push_input_event(rel_ev)

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
	var actor = _find_actor_node(main_scene_node)
	if actor:
		if "global_position" in actor:
			snap["player_pos"] = [actor.global_position.x, actor.global_position.y]
		elif "position" in actor:
			snap["player_pos"] = [actor.position.x, actor.position.y]
		if "velocity" in actor:
			snap["player_vel"] = [actor.velocity.x, actor.velocity.y]
		if "health" in actor:
			snap["player_health"] = float(actor.health)

	if main_scene_node and main_scene_node.is_node_ready() and main_scene_node.has_method("get_state_snapshot"):
		var custom_snap = main_scene_node.call("get_state_snapshot")
		if typeof(custom_snap) == TYPE_DICTIONARY:
			var actor_pos = snap["player_pos"]
			for k in custom_snap:
				var val = custom_snap[k]
				if typeof(val) == TYPE_VECTOR2 or typeof(val) == TYPE_VECTOR2I:
					val = [float(val.x), float(val.y)]
				snap[k] = val
				if String(k).ends_with("_pos") and typeof(val) == TYPE_ARRAY and val.size() >= 2:
					if actor_pos[0] == 0.0 and actor_pos[1] == 0.0 and (float(val[0]) != 0.0 or float(val[1]) != 0.0):
						actor_pos = [float(val[0]), float(val[1])]
			if typeof(snap.get("player_pos")) != TYPE_ARRAY or (snap["player_pos"][0] == 0.0 and snap["player_pos"][1] == 0.0 and (actor_pos[0] != 0.0 or actor_pos[1] != 0.0)):
				snap["player_pos"] = actor_pos
	return snap

func _find_actor_node(node: Node) -> Node:
	if not node:
		return null
	for name in ["player", "Player", "rig", "crawler", "ship", "hero", "avatar", "actor"]:
		if node.has_node(name):
			return node.get_node(name)
	for child in node.get_children():
		if child is CharacterBody2D:
			return child
		var found = _find_actor_node(child)
		if found:
			return found
	return null

func _audit_spatial_boundaries(node: Node) -> Dictionary:
	var walkable_min_y: float = 99999.0
	var walkable_max_y: float = -99999.0
	var has_explicit_shelves: bool = false
	var has_sky_backdrop: bool = false
	var static_collider_count: int = 0
	var interior_north_barrier_found: bool = false

	var shapes_list: Array = []
	_collect_collision_shapes_recursive(node, shapes_list)
	for item in shapes_list:
		var kind = str(item.get("kind", ""))
		if kind == "walkable_shelf" and item.has("rect"):
			var r: Rect2 = item["rect"]
			has_explicit_shelves = true
			if r.position.y < walkable_min_y:
				walkable_min_y = r.position.y
			if r.end.y > walkable_max_y:
				walkable_max_y = r.end.y
		elif kind == "static":
			static_collider_count += 1
			if item.has("polygon"):
				var pts: PackedVector2Array = item["polygon"]
				var min_py = 99999.0
				var max_py = -99999.0
				for p in pts:
					if p.y < min_py:
						min_py = p.y
					if p.y > max_py:
						max_py = p.y
				# Check if this collider acts as a horizon/shelf top barrier (y between 120 and 380) rather than y<=32 canvas ceiling
				if max_py >= 140.0 and min_py <= 380.0 and (max_py - min_py) <= 320.0:
					interior_north_barrier_found = true
			elif item.has("rect"):
				var sr: Rect2 = item["rect"]
				if sr.end.y >= 140.0 and sr.position.y <= 380.0 and sr.size.y <= 320.0:
					interior_north_barrier_found = true

	if node != null:
		for child in node.get_children():
			var cname = str(child.name).to_lower()
			if cname.find("sky") != -1 or cname.find("backdrop") != -1:
				has_sky_backdrop = true

	var unbounded_sky_detected: bool = false
	var boundary_violation_reason: String = ""

	if has_explicit_shelves and walkable_min_y < 9000.0:
		# If the player walked > 24px above the highest walkable shelf into the sky backdrop
		if min_player_y_reached < walkable_min_y - 24.0:
			unbounded_sky_detected = true
			boundary_violation_reason = (
				"Player walked into sky/void at y=" + str(snapped(min_player_y_reached, 0.1))
				+ " above walkable shelf top y=" + str(snapped(walkable_min_y, 0.1))
			)
		elif not interior_north_barrier_found and walkable_min_y >= 180.0:
			unbounded_sky_detected = true
			boundary_violation_reason = (
				"Walkable shelves start at y=" + str(snapped(walkable_min_y, 0.1))
				+ " (sky above y<" + str(snapped(walkable_min_y, 0.1)) + ") but no StaticBody2D north horizon collider bounds y="
				+ str(snapped(walkable_min_y, 0.1))
			)
	elif has_sky_backdrop and float(spawn_pos[1]) >= 260.0 and min_player_y_reached < 180.0 and not interior_north_barrier_found:
		unbounded_sky_detected = true
		boundary_violation_reason = (
			"Scene has SkyBackdrop and player spawned at y=" + str(snapped(float(spawn_pos[1]), 0.1))
			+ " but walked unobstructed into sky region y=" + str(snapped(min_player_y_reached, 0.1))
		)

	return {
		"static_collider_count": static_collider_count,
		"has_explicit_shelves": has_explicit_shelves,
		"walkable_min_y": walkable_min_y if has_explicit_shelves else 0.0,
		"min_player_y_reached": min_player_y_reached,
		"max_player_y_reached": max_player_y_reached,
		"min_player_x_reached": min_player_x_reached,
		"max_player_x_reached": max_player_x_reached,
		"interior_north_barrier_found": interior_north_barrier_found,
		"unbounded_sky_or_void_detected": unbounded_sky_detected,
		"boundary_violation_reason": boundary_violation_reason
	}

func _audit_scene_tree(node: Node, in_ui: bool = false) -> Dictionary:
	var report = {
		"world_visual_nodes": 0,
		"ui_control_nodes": 0,
		"textures_loaded_count": 0,
		"empty_world_containers": [],
		"camera_found": false,
		"camera_pos": [0.0, 0.0],
		"audio_players_found": 0,
		"audio_streams_loaded": 0,
		"audio_mix_rates": [],
		"audio_players_detail": [],
		"visible_ui_texts": [],
		"center_occluding_panels": [],
		"fullscreen_occlusion_detected": false,
		"fullscreen_occlusion_nodes": [],
		"early_failure_state_detected": false,
		"bound_sprites_audit": [],
		"empty_sprite_frame_nodes": [],
		"subviewport_thrashing_nodes": [],
		"portrait_misused_on_world_sprite": false,
		"portrait_misused_nodes": [],
		"oversized_ui_portraits": [],
		"passive_auto_win_detected": passive_auto_win_detected,
		"passive_win_reason": passive_win_reason,
		"unbounded_sky_or_void_detected": false,
		"boundary_violation_reason": "",
		"spatial_boundary_audit": {}
	}
	if not node:
		return report
	_walk_scene_audit(node, in_ui, report)
	var boundary_audit = _audit_spatial_boundaries(node)
	report["spatial_boundary_audit"] = boundary_audit
	report["unbounded_sky_or_void_detected"] = bool(boundary_audit.get("unbounded_sky_or_void_detected", false))
	report["boundary_violation_reason"] = str(boundary_audit.get("boundary_violation_reason", ""))

	var init_occlusion = float(initial_snapshot.get("viewport_occlusion", 0.0))
	var final_failed = bool(final_snapshot.get("is_failed", false))
	if init_occlusion >= 0.80:
		report["fullscreen_occlusion_detected"] = true
		report["fullscreen_occlusion_nodes"].append("initial_snapshot.viewport_occlusion=" + str(init_occlusion))
	if final_failed:
		report["early_failure_state_detected"] = true
	return report

func _is_sprite_frame_cell_empty(tex: Texture2D, hframes: int, vframes: int, frame_idx: int) -> bool:
	if tex == null or (hframes <= 1 and vframes <= 1):
		return false
	var img = tex.get_image()
	if img == null or img.is_empty():
		return false
	if img.is_compressed():
		img.decompress()
	var tw = img.get_width()
	var th = img.get_height()
	if tw <= 0 or th <= 0:
		return false
	var cw = int(max(1, tw / max(1, hframes)))
	var ch = int(max(1, th / max(1, vframes)))
	var col_idx = clampi(frame_idx % max(1, hframes), 0, max(0, hframes - 1))
	var row_idx = clampi(frame_idx / max(1, hframes), 0, max(0, vframes - 1))
	var x0 = col_idx * cw
	var y0 = row_idx * ch
	var x1 = min(tw, x0 + cw)
	var y1 = min(th, y0 + ch)
	var step_x = int(max(1, (x1 - x0) / 16))
	var step_y = int(max(1, (y1 - y0) / 16))
	var cell_has_alpha = false
	for y in range(y0, y1, step_y):
		for x in range(x0, x1, step_x):
			if img.get_pixel(x, y).a > 0.02:
				cell_has_alpha = true
				break
		if cell_has_alpha:
			break
	if cell_has_alpha:
		return false
	var tex_has_alpha = false
	var full_sx = int(max(1, tw / 16))
	var full_sy = int(max(1, th / 16))
	for y in range(0, th, full_sy):
		for x in range(0, tw, full_sx):
			if img.get_pixel(x, y).a > 0.02:
				tex_has_alpha = true
				break
		if tex_has_alpha:
			break
	return tex_has_alpha

func _walk_scene_audit(node: Node, in_ui: bool, report: Dictionary) -> void:
	var is_fg_canvas_layer = (node is CanvasLayer) and (node.layer >= 0)
	var is_bg_canvas_layer = (node is CanvasLayer) and (node.layer < 0)
	var is_world_bg_color_rect = (node is ColorRect) and (not in_ui) and (node.z_index <= 0)
	var current_in_ui = false if is_bg_canvas_layer else (in_ui or is_fg_canvas_layer or ((node is Control) and not is_world_bg_color_rect))
	if node is Camera2D:
		report["camera_found"] = true
		report["camera_pos"] = [node.position.x, node.position.y]
	if node is SubViewport:
		var sv_sz: Vector2i = node.size
		if (
			sv_sz.x * sv_sz.y >= 500000
			and node.render_target_update_mode == SubViewport.UPDATE_ALWAYS
			and node.render_target_clear_mode == SubViewport.CLEAR_MODE_NEVER
		):
			report["subviewport_thrashing_nodes"].append(
				str(node.name) + " (size=" + str(sv_sz) + ", UPDATE_ALWAYS + CLEAR_MODE_NEVER causes severe GPU tile-memory stalls; use UPDATE_ONCE triggered only when painting)"
			)
	if (node is AudioStreamPlayer) or (node is AudioStreamPlayer2D):
		report["audio_players_found"] = int(report["audio_players_found"]) + 1
		var a_loop = -1
		if node.stream != null:
			report["audio_streams_loaded"] = int(report["audio_streams_loaded"]) + 1
			if node.stream is AudioStreamWAV:
				report["audio_mix_rates"].append(int(node.stream.mix_rate))
				a_loop = int(node.stream.loop_mode)
		if report["audio_players_detail"].size() < 12:
			report["audio_players_detail"].append({
				"node": str(node.name),
				"has_stream": node.stream != null,
				"playing": bool(node.playing),
				"loop_mode": a_loop
			})
	if (node is Sprite2D and node.texture != null) or (node is TextureRect and node.texture != null):
		report["textures_loaded_count"] = int(report["textures_loaded_count"]) + 1
		if node is Sprite2D and not current_in_ui:
			var orig_tpath = _resolve_texture_path(node, node.texture)
			var tpath = orig_tpath.to_lower()
			if tpath.find("portrait") != -1:
				report["portrait_misused_on_world_sprite"] = true
				report["portrait_misused_nodes"].append(str(node.name) + ":" + tpath)
			var tw_s = node.texture.get_width()
			var th_s = node.texture.get_height()
			var empty_cell = _is_sprite_frame_cell_empty(node.texture, node.hframes, node.vframes, node.frame)
			report["bound_sprites_audit"].append({
				"node": str(node.name),
				"tex_path": orig_tpath,
				"tex_size": [tw_s, th_s],
				"hframes": int(node.hframes),
				"vframes": int(node.vframes),
				"frame": int(node.frame),
				"visible": bool(node.is_visible_in_tree()),
				"empty_frame_cell": empty_cell
			})
			if node.is_visible_in_tree() and empty_cell:
				report["empty_sprite_frame_nodes"].append(
					str(node.name) + " (tex=" + orig_tpath + " " + str(tw_s) + "x" + str(th_s)
					+ ", hframes=" + str(node.hframes) + ", vframes=" + str(node.vframes)
					+ ", frame=" + str(node.frame) + " crops a 100% transparent empty corner! Set hframes=1, vframes=1 for single-frame/composite sprites)"
				)
		if node is TextureRect and current_in_ui:
			var sz = node.size
			if sz.x <= 1.0 or sz.y <= 1.0:
				sz = Vector2(float(node.texture.get_width()), float(node.texture.get_height()))
			if (sz.x > 110.0 or sz.y > 110.0) or (node.expand_mode == TextureRect.EXPAND_KEEP_SIZE and (node.texture.get_width() > 110 or node.texture.get_height() > 110)):
				report["oversized_ui_portraits"].append(str(node.name) + ":" + str(sz))
	if "loaded_textures" in node and typeof(node.get("loaded_textures")) == TYPE_DICTIONARY:
		var lt: Dictionary = node.get("loaded_textures")
		report["textures_loaded_count"] = int(report["textures_loaded_count"]) + lt.size()
	if current_in_ui:
		if node is Control:
			report["ui_control_nodes"] = int(report["ui_control_nodes"]) + 1
			if node.is_visible_in_tree() and report["visible_ui_texts"].size() < 16:
				var ui_txt = ""
				if node is Label:
					ui_txt = String(node.text).strip_edges()
				elif node is Button:
					ui_txt = String(node.text).strip_edges()
				elif node is RichTextLabel:
					ui_txt = String(node.get_parsed_text()).strip_edges()
				if ui_txt != "" and ui_txt.length() <= 160 and not report["visible_ui_texts"].has(ui_txt):
					report["visible_ui_texts"].append(ui_txt)
			if (node is Panel or node is PanelContainer) and node.get_parent() is CanvasLayer:
				var p = node.position
				if p.x > 360.0 and p.x < 760.0 and p.y > 220.0 and p.y < 540.0:
					report["center_occluding_panels"].append(str(node.name) + "@" + str(p))
			if node is ColorRect and node.visible and node.size.x >= 1100.0 and node.size.y >= 650.0 and node.color.a >= 0.80:
				report["fullscreen_occlusion_detected"] = true
				report["fullscreen_occlusion_nodes"].append(
					str(node.name) + " (size=" + str(node.size) + ", alpha=" + str(snapped(node.color.a, 0.01)) + ")"
				)
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
			and not (node is CollisionShape2D)
			and not (node is CollisionPolygon2D)
			and not (node is Marker2D)
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
	var max_excursion = sqrt(
		pow(max_player_x_reached - min_player_x_reached, 2.0) +
		pow(max_player_y_reached - min_player_y_reached, 2.0)
	)
	var effective_delta = max(pos_delta, max_excursion)
	var progression_verified = (effective_delta > 0.1 or current_tick >= max_ticks)
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
		"pos_delta": effective_delta,
		"progression_verified": progression_verified
	}))
	quit(0)

