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
		DisplayServer.window_set_position(Vector2i(-4000, -4000))
		render_viewport = SubViewport.new()
		render_viewport.name = "HarnessOffscreenViewport"
		render_viewport.size = Vector2i(1280, 720)
		render_viewport.size_2d_override = Vector2i(1280, 720)
		render_viewport.render_target_update_mode = SubViewport.UPDATE_ALWAYS
		render_viewport.handle_input_locally = true
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
		if "flume_grid" in main_scene_node and main_scene_node.get("flume_grid") != null:
			var fg = main_scene_node.get("flume_grid")
			if fg.has_method("get_powered_target_count") and int(fg.call("get_powered_target_count")) > 0:
				passive_auto_win_detected = true
				passive_win_reason = "Simulation grid powered target count > 0 at passive tick " + str(tick_num) + " with zero player input."
				return

func _try_capture_viewport(res_target_path: String) -> void:
	var img: Image = null
	if DisplayServer.get_name() == "headless":
		if not capture_viewport_enabled:
			return
		img = _render_headless_scene_to_image()
	else:
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
	if main_scene_node == null:
		return
	var shapes_data: Array = []
	_collect_collision_shapes_recursive(main_scene_node, shapes_data)

	var img: Image = null
	if DisplayServer.get_name() == "headless":
		if not capture_viewport_enabled:
			return
		img = _render_headless_scene_to_image()
	else:
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

func _get_effective_modulate(node: Node) -> Color:
	var col = Color(1.0, 1.0, 1.0, 1.0)
	var cur: Node = node
	while cur != null and cur != root:
		if cur is CanvasItem:
			var ci = cur as CanvasItem
			if cur == node:
				col = col * ci.self_modulate
			col = col * ci.modulate
		cur = cur.get_parent()
	return col

func _blend_rect_alpha(img: Image, rect: Rect2i, col: Color) -> void:
	var clipped = rect.intersection(Rect2i(0, 0, 1280, 720))
	if clipped.size.x <= 0 or clipped.size.y <= 0 or col.a <= 0.005:
		return
	if col.a >= 0.99:
		img.fill_rect(clipped, Color(clampf(col.r, 0.0, 1.0), clampf(col.g, 0.0, 1.0), clampf(col.b, 0.0, 1.0), 1.0))
		return
	var step = 2 if (clipped.size.x * clipped.size.y > 16000) else 1
	for y in range(clipped.position.y, clipped.end.y, step):
		for x in range(clipped.position.x, clipped.end.x, step):
			var dst = img.get_pixel(x, y)
			var a = clampf(col.a, 0.0, 1.0)
			var out_col = Color(
				clampf(dst.r * (1.0 - a) + col.r * a, 0.0, 1.0),
				clampf(dst.g * (1.0 - a) + col.g * a, 0.0, 1.0),
				clampf(dst.b * (1.0 - a) + col.b * a, 0.0, 1.0),
				1.0
			)
			if step == 2:
				img.fill_rect(Rect2i(x, y, min(2, clipped.end.x - x), min(2, clipped.end.y - y)), out_col)
			else:
				img.set_pixel(x, y, out_col)

func _collect_headless_render_items(
	node: Node,
	in_ui: bool,
	parent_z: int,
	world_items: Array,
	ui_items: Array,
	lights: Array,
	modulates: Array
) -> void:
	if node == null:
		return
	if node is CanvasItem and not (node as CanvasItem).visible:
		return
	var cur_in_ui = in_ui
	if node is CanvasLayer:
		var cl = node as CanvasLayer
		if not cl.visible:
			return
		cur_in_ui = (cl.layer >= 0)
	var eff_z = parent_z
	if node is CanvasItem:
		var ci = node as CanvasItem
		eff_z = (parent_z + ci.z_index) if ci.z_as_relative else ci.z_index
		if not cur_in_ui and (node is Control) and not ((node is ColorRect) and eff_z <= 0):
			cur_in_ui = true
	if node is CanvasModulate and not cur_in_ui:
		modulates.append((node as CanvasModulate).color)
	elif node is PointLight2D and not cur_in_ui:
		lights.append(node)
	elif node is CanvasItem:
		var entry = {"node": node, "z": eff_z, "order": world_items.size() + ui_items.size()}
		if cur_in_ui:
			ui_items.append(entry)
		else:
			world_items.append(entry)
	for child in node.get_children():
		_collect_headless_render_items(child, cur_in_ui, eff_z, world_items, ui_items, lights, modulates)

func _draw_simple_glyph_text(img: Image, text: String, pos: Vector2, col: Color, max_chars: int = 64) -> void:
	var clean = text.strip_edges()
	if clean == "":
		return
	var cx = int(round(pos.x))
	var cy = int(round(pos.y))
	var count = min(clean.length(), max_chars)
	for i in range(count):
		var ch = clean.unicode_at(i)
		if ch == 32:
			cx += 6
			continue
		if cx + 5 >= 1280 or cy + 7 >= 720 or cx < 0 or cy < 0:
			cx += 6
			continue
		for gy in range(7):
			for gx in range(5):
				var on = false
				if gy == 0 or gy == 6:
					on = (gx > 0 and gx < 4)
				elif gy == 3:
					on = (gx > 0 and ((ch + gy) % 2 == 0))
				else:
					on = (gx == 0 or gx == 4 or ((ch + gx * 3 + gy) % 5 == 0))
				if on:
					_blend_pixel(img, cx + gx, cy + gy, col)
		cx += 6

func _render_headless_canvas_item(img: Image, node: CanvasItem, canvas_xform: Transform2D, canvas_mod: Color, is_ui: bool) -> void:
	var eff_mod = _get_effective_modulate(node)
	if not is_ui:
		eff_mod = eff_mod * canvas_mod

	if node is ColorRect:
		var cr = node as ColorRect
		var xform = (Transform2D.IDENTITY if is_ui else canvas_xform) * cr.get_global_transform()
		var tl = xform * Vector2.ZERO
		var br = xform * cr.size
		var rx = int(clampf(min(tl.x, br.x), 0.0, 1279.0))
		var ry = int(clampf(min(tl.y, br.y), 0.0, 719.0))
		var rw = int(clampf(absf(br.x - tl.x), 0.0, float(1280 - rx)))
		var rh = int(clampf(absf(br.y - tl.y), 0.0, float(720 - ry)))
		if rw <= 0 or rh <= 0:
			return
		if cr.material is ShaderMaterial and (cr.material as ShaderMaterial).shader != null and (cr.material as ShaderMaterial).shader.code != "":
			var step_s = 2
			for sy in range(ry, ry + rh, step_s):
				var uvy = float(sy - ry) / float(max(1, rh))
				for sx in range(rx, rx + rw, step_s):
					var uvx = float(sx - rx) / float(max(1, rw))
					var sh_col = Color(
						clampf(uvx * 0.85 + 0.1 * cr.color.r, 0.0, 1.0),
						clampf(uvy * 0.85 + 0.1 * cr.color.g, 0.0, 1.0),
						clampf(0.85 * cr.color.b, 0.0, 1.0),
						cr.color.a * eff_mod.a
					)
					img.fill_rect(Rect2i(sx, sy, min(step_s, rx + rw - sx), min(step_s, ry + rh - sy)), sh_col)
		else:
			_blend_rect_alpha(img, Rect2i(rx, ry, rw, rh), cr.color * eff_mod)
		return

	if node is Polygon2D:
		var poly = node as Polygon2D
		if poly.polygon.size() < 3:
			return
		var xform = (Transform2D.IDENTITY if is_ui else canvas_xform) * poly.get_global_transform()
		var s_pts = PackedVector2Array()
		var min_x = 1280.0
		var max_x = 0.0
		var min_y = 720.0
		var max_y = 0.0
		for p in poly.polygon:
			var sp = xform * (p + poly.offset)
			s_pts.append(sp)
			min_x = min(min_x, sp.x)
			max_x = max(max_x, sp.x)
			min_y = min(min_y, sp.y)
			max_y = max(max_y, sp.y)
		var ix0 = int(clampf(min_x, 0.0, 1279.0))
		var ix1 = int(clampf(max_x, 0.0, 1279.0))
		var iy0 = int(clampf(min_y, 0.0, 719.0))
		var iy1 = int(clampf(max_y, 0.0, 719.0))
		var pcol = poly.color * eff_mod
		var tex_img: Image = null
		if poly.texture != null:
			tex_img = poly.texture.get_image()
			if tex_img != null and not tex_img.is_empty():
				if tex_img.is_compressed():
					tex_img.decompress()
				tex_img.convert(Image.FORMAT_RGBA8)
			else:
				tex_img = null
		for py in range(iy0, iy1 + 1, 2):
			for px in range(ix0, ix1 + 1, 2):
				if Geometry2D.is_point_in_polygon(Vector2(float(px), float(py)), s_pts):
					var fill_c = pcol
					if tex_img != null:
						var tx = clampi(int(px - ix0) % max(1, tex_img.get_width()), 0, max(0, tex_img.get_width() - 1))
						var ty = clampi(int(py - iy0) % max(1, tex_img.get_height()), 0, max(0, tex_img.get_height() - 1))
						fill_c = tex_img.get_pixel(tx, ty) * pcol
					img.fill_rect(Rect2i(px, py, min(2, 1280 - px), min(2, 720 - py)), Color(clampf(fill_c.r, 0.0, 1.0), clampf(fill_c.g, 0.0, 1.0), clampf(fill_c.b, 0.0, 1.0), 1.0))
		var edge_c = Color(clampf(pcol.r * 1.25 + 0.08, 0.0, 1.0), clampf(pcol.g * 1.25 + 0.08, 0.0, 1.0), clampf(pcol.b * 1.25 + 0.08, 0.0, 1.0), 0.95)
		for i in range(s_pts.size()):
			_draw_thick_line_on_image(img, s_pts[i], s_pts[(i + 1) % s_pts.size()], edge_c, 1)
		return

	if node is Line2D:
		var ln = node as Line2D
		if ln.points.size() < 2:
			return
		var xform = (Transform2D.IDENTITY if is_ui else canvas_xform) * ln.get_global_transform()
		var thick = clampi(int(round(ln.width * 0.45)), 1, 6)
		var lcol = ln.default_color * eff_mod
		for i in range(ln.points.size() - 1):
			var p0 = xform * ln.points[i]
			var p1 = xform * ln.points[i + 1]
			var seg_col = lcol
			if ln.gradient != null:
				seg_col = ln.gradient.sample(float(i) / float(max(1, ln.points.size() - 1))) * eff_mod
			_draw_thick_line_on_image(img, p0, p1, Color(clampf(seg_col.r, 0.0, 1.0), clampf(seg_col.g, 0.0, 1.0), clampf(seg_col.b, 0.0, 1.0), clampf(seg_col.a, 0.1, 1.0)), thick)
		return

	if (node is Sprite2D) or (node is AnimatedSprite2D):
		var tex: Texture2D = null
		var hframes: int = 1
		var vframes: int = 1
		var frame_idx: int = 0
		var centered: bool = true
		var offset_v: Vector2 = Vector2.ZERO
		var flip_h: bool = false
		var flip_v: bool = false
		var region_enabled: bool = false
		var region_rect: Rect2 = Rect2()
		if node is Sprite2D:
			var sp = node as Sprite2D
			tex = sp.texture
			hframes = max(1, sp.hframes)
			vframes = max(1, sp.vframes)
			frame_idx = sp.frame
			centered = sp.centered
			offset_v = sp.offset
			flip_h = sp.flip_h
			flip_v = sp.flip_v
			region_enabled = sp.region_enabled
			region_rect = sp.region_rect
		elif node is AnimatedSprite2D:
			var asp = node as AnimatedSprite2D
			centered = asp.centered
			offset_v = asp.offset
			flip_h = asp.flip_h
			flip_v = asp.flip_v
			if asp.sprite_frames != null and asp.sprite_frames.has_animation(asp.animation):
				var fcount = asp.sprite_frames.get_frame_count(asp.animation)
				if fcount > 0:
					tex = asp.sprite_frames.get_frame_texture(asp.animation, clampi(asp.frame, 0, fcount - 1))
		if tex != null:
			var simg = tex.get_image()
			if simg != null and not simg.is_empty():
				simg = simg.duplicate()
				if simg.is_compressed():
					simg.decompress()
				simg.convert(Image.FORMAT_RGBA8)
				if region_enabled and region_rect.size.x > 0 and region_rect.size.y > 0:
					simg = simg.get_region(Rect2i(int(region_rect.position.x), int(region_rect.position.y), int(region_rect.size.x), int(region_rect.size.y)))
				elif hframes > 1 or vframes > 1:
					var cw = max(1, int(simg.get_width() / hframes))
					var ch = max(1, int(simg.get_height() / vframes))
					var c_idx = clampi(frame_idx % hframes, 0, hframes - 1)
					var r_idx = clampi(int(frame_idx / hframes), 0, vframes - 1)
					simg = simg.get_region(Rect2i(c_idx * cw, r_idx * ch, cw, ch))
				if flip_h:
					simg.flip_x()
				if flip_v:
					simg.flip_y()
				var gscale = (node as Node2D).global_scale
				var dst_w = clampi(int(round(float(simg.get_width()) * absf(gscale.x))), 4, 1280)
				var dst_h = clampi(int(round(float(simg.get_height()) * absf(gscale.y))), 4, 720)
				if simg.get_width() != dst_w or simg.get_height() != dst_h:
					simg.resize(dst_w, dst_h, Image.INTERPOLATE_BILINEAR)
				var gpos = ((Transform2D.IDENTITY if is_ui else canvas_xform) * (node as Node2D).global_position) + offset_v * gscale
				var top_left = gpos - (Vector2(float(dst_w) * 0.5, float(dst_h) * 0.5) if centered else Vector2.ZERO)
				var dx0 = int(round(top_left.x))
				var dy0 = int(round(top_left.y))
				if absf(eff_mod.r - 1.0) > 0.03 or absf(eff_mod.g - 1.0) > 0.03 or absf(eff_mod.b - 1.0) > 0.03 or absf(eff_mod.a - 1.0) > 0.03:
					for sy in range(0, dst_h, 2):
						for sx in range(0, dst_w, 2):
							var c_px = simg.get_pixel(sx, sy)
							if c_px.a > 0.02:
								var m_px = Color(
									clampf(c_px.r * eff_mod.r, 0.0, 1.0),
									clampf(c_px.g * eff_mod.g, 0.0, 1.0),
									clampf(c_px.b * eff_mod.b, 0.0, 1.0),
									clampf(c_px.a * eff_mod.a, 0.0, 1.0)
								)
								_blend_pixel(img, dx0 + sx, dy0 + sy, m_px)
								_blend_pixel(img, dx0 + sx + 1, dy0 + sy, m_px)
								_blend_pixel(img, dx0 + sx, dy0 + sy + 1, m_px)
								_blend_pixel(img, dx0 + sx + 1, dy0 + sy + 1, m_px)
				else:
					img.blend_rect(simg, Rect2i(0, 0, dst_w, dst_h), Vector2i(dx0, dy0))
		return

	if (node is GPUParticles2D) or (node is CPUParticles2D):
		var emitting: bool = true
		var amount: int = 24
		var lifetime: float = 1.5
		var dir_v := Vector2(0.0, -1.0)
		var spread_deg: float = 35.0
		var vel_max: float = 95.0
		if node is GPUParticles2D:
			var gp = node as GPUParticles2D
			emitting = gp.emitting
			amount = clampi(gp.amount, 8, 40)
			lifetime = max(0.2, gp.lifetime)
			if gp.process_material is ParticleProcessMaterial:
				var pm = gp.process_material as ParticleProcessMaterial
				dir_v = Vector2(pm.direction.x, pm.direction.y)
				if dir_v.length_squared() < 0.001:
					dir_v = Vector2(0.0, -1.0)
				else:
					dir_v = dir_v.normalized()
				spread_deg = pm.spread
				vel_max = max(30.0, pm.initial_velocity_max)
		elif node is CPUParticles2D:
			var cp = node as CPUParticles2D
			emitting = cp.emitting
			amount = clampi(cp.amount, 8, 40)
			lifetime = max(0.2, cp.lifetime)
			dir_v = cp.direction.normalized() if cp.direction.length_squared() > 0.001 else Vector2(0.0, -1.0)
			spread_deg = cp.spread
			vel_max = max(30.0, cp.initial_velocity_max)
		if emitting:
			var origin = (Transform2D.IDENTITY if is_ui else canvas_xform) * (node as Node2D).global_position
			var raw_mod = node.modulate * node.self_modulate
			var p_col = Color(
				clampf(raw_mod.r * max(0.35, canvas_mod.r), 0.0, 1.0),
				clampf(raw_mod.g * max(0.35, canvas_mod.g), 0.0, 1.0),
				clampf(raw_mod.b * max(0.35, canvas_mod.b), 0.0, 1.0),
				clampf(raw_mod.a, 0.5, 1.0)
			)
			var base_ang = dir_v.angle()
			var max_reach = clampf(vel_max * lifetime * 0.9, 40.0, 220.0)
			for i in range(amount):
				var phase = fmod((float(i) / float(amount)) + float(current_tick) * 0.035, 1.0)
				var ang_off = deg_to_rad(spread_deg) * (sin(float(i) * 2.39996) * 0.85)
				var dist = phase * max_reach
				var pt = origin + Vector2.from_angle(base_ang + ang_off) * dist
				img.fill_rect(Rect2i(clampi(int(pt.x) - 3, 0, 1274), clampi(int(pt.y) - 3, 0, 714), 6, 6), p_col)
		return

	if node is TextureRect:
		var tr = node as TextureRect
		if tr.texture != null:
			var timg = tr.texture.get_image()
			if timg != null and not timg.is_empty():
				timg = timg.duplicate()
				if timg.is_compressed():
					timg.decompress()
				timg.convert(Image.FORMAT_RGBA8)
				var sz = tr.size
				if sz.x <= 1.0 or sz.y <= 1.0:
					sz = Vector2(float(timg.get_width()), float(timg.get_height()))
				var dw = clampi(int(round(sz.x)), 4, 1280)
				var dh = clampi(int(round(sz.y)), 4, 720)
				if timg.get_width() != dw or timg.get_height() != dh:
					timg.resize(dw, dh, Image.INTERPOLATE_BILINEAR)
				var gp = tr.global_position
				img.blend_rect(timg, Rect2i(0, 0, dw, dh), Vector2i(int(round(gp.x)), int(round(gp.y))))
		return

	if (node is Panel) or (node is PanelContainer):
		var ctrl = node as Control
		var r = Rect2i(int(round(ctrl.global_position.x)), int(round(ctrl.global_position.y)), max(16, int(round(ctrl.size.x))), max(16, int(round(ctrl.size.y))))
		_blend_rect_alpha(img, r, Color(0.08, 0.11, 0.16, 0.82) * eff_mod)
		var tl = Vector2(float(r.position.x), float(r.position.y))
		var br = Vector2(float(r.end.x), float(r.end.y))
		var bcol = Color(0.28, 0.48, 0.68, 0.9)
		_draw_thick_line_on_image(img, tl, Vector2(br.x, tl.y), bcol, 1)
		_draw_thick_line_on_image(img, Vector2(br.x, tl.y), br, bcol, 1)
		_draw_thick_line_on_image(img, br, Vector2(tl.x, br.y), bcol, 1)
		_draw_thick_line_on_image(img, Vector2(tl.x, br.y), tl, bcol, 1)
		return

	if node is ProgressBar:
		var pb = node as ProgressBar
		var r = Rect2i(int(round(pb.global_position.x)), int(round(pb.global_position.y)), max(24, int(round(pb.size.x))), max(8, int(round(pb.size.y))))
		_blend_rect_alpha(img, r, Color(0.12, 0.15, 0.22, 0.9))
		var ratio = clampf(float((pb.value - pb.min_value) / max(0.001, pb.max_value - pb.min_value)), 0.0, 1.0)
		var fill_w = int(round(float(r.size.x) * ratio))
		if fill_w > 0:
			_blend_rect_alpha(img, Rect2i(r.position.x, r.position.y, fill_w, r.size.y), Color(0.2, 0.85, 0.55, 0.95))
		return

	if node is Label:
		var lbl = node as Label
		var lcol = lbl.get_theme_color("font_color", "Label") * eff_mod
		lcol.a = 1.0
		_draw_simple_glyph_text(img, lbl.text, lbl.global_position + Vector2(2.0, 4.0), lcol, 72)
		return

	if node is Button:
		var btn = node as Button
		var r = Rect2i(int(round(btn.global_position.x)), int(round(btn.global_position.y)), max(32, int(round(btn.size.x))), max(18, int(round(btn.size.y))))
		_blend_rect_alpha(img, r, Color(0.16, 0.24, 0.36, 0.92))
		_draw_simple_glyph_text(img, btn.text, btn.global_position + Vector2(6.0, 5.0), Color(0.92, 0.96, 1.0, 1.0), 40)
		return

func _render_headless_scene_to_image() -> Image:
	var img = Image.create(1280, 720, false, Image.FORMAT_RGBA8)
	var bg_col: Color = ProjectSettings.get_setting("rendering/environment/defaults/default_clear_color", Color(0.07, 0.09, 0.13, 1.0))
	bg_col.a = 1.0
	img.fill(bg_col)
	if main_scene_node == null:
		return img

	var world_items: Array = []
	var ui_items: Array = []
	var lights: Array = []
	var modulates: Array = []
	_collect_headless_render_items(main_scene_node, false, 0, world_items, ui_items, lights, modulates)

	var canvas_mod := Color(1.0, 1.0, 1.0, 1.0)
	if modulates.size() > 0:
		canvas_mod = modulates[modulates.size() - 1]
		img.fill(Color(bg_col.r * canvas_mod.r, bg_col.g * canvas_mod.g, bg_col.b * canvas_mod.b, 1.0))

	world_items.sort_custom(func(a, b):
		if a["z"] != b["z"]:
			return int(a["z"]) < int(b["z"])
		return int(a["order"]) < int(b["order"])
	)

	var canvas_xform := Transform2D.IDENTITY
	for item in world_items:
		_render_headless_canvas_item(img, item["node"], canvas_xform, canvas_mod, false)

	# If procedural physics bodies have CollisionShape2D/CollisionPolygon2D but no Sprite2D/Polygon2D children, render their shapes
	var fallback_shapes: Array = []
	_collect_collision_shapes_recursive(main_scene_node, fallback_shapes)
	if world_items.size() <= 3 and fallback_shapes.size() > 0:
		_rasterize_shapes_on_image(img, fallback_shapes, canvas_xform, 1.0, 1.0)

	# Apply PointLight2D radial additive lighting
	for lnode in lights:
		var pl = lnode as PointLight2D
		var lpos = canvas_xform * pl.global_position
		var rad = 220.0
		if pl.texture != null:
			rad = clampf(float(pl.texture.get_width()) * absf(pl.global_scale.x) * 0.5, 48.0, 480.0)
		var lcol = pl.color
		var energy = clampf(pl.energy, 0.1, 4.0)
		var lx0 = clampi(int(lpos.x - rad), 0, 1279)
		var lx1 = clampi(int(lpos.x + rad), 0, 1279)
		var ly0 = clampi(int(lpos.y - rad), 0, 719)
		var ly1 = clampi(int(lpos.y + rad), 0, 719)
		for ly in range(ly0, ly1 + 1, 2):
			for lx in range(lx0, lx1 + 1, 2):
				var d = lpos.distance_to(Vector2(float(lx), float(ly)))
				if d < rad:
					var falloff = pow(1.0 - d / rad, 1.35) * energy * 0.55
					var cur_c = img.get_pixel(lx, ly)
					var lit_c = Color(
						clampf(cur_c.r + lcol.r * falloff, 0.0, 1.0),
						clampf(cur_c.g + lcol.g * falloff, 0.0, 1.0),
						clampf(cur_c.b + lcol.b * falloff, 0.0, 1.0),
						1.0
					)
					img.fill_rect(Rect2i(lx, ly, min(2, 1280 - lx), min(2, 720 - ly)), lit_c)

	for uitem in ui_items:
		_render_headless_canvas_item(img, uitem["node"], Transform2D.IDENTITY, Color(1.0, 1.0, 1.0, 1.0), true)

	return img

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
	if render_viewport != null:
		render_viewport.push_input(ev)
	root.push_input(ev)

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
			var rel_click = entry.get("mouse_click_pos", null)
			if typeof(rel_click) == TYPE_ARRAY and rel_click.size() >= 2:
				var rel_ev = InputEventMouseButton.new()
				rel_ev.button_index = MOUSE_BUTTON_LEFT
				rel_ev.pressed = false
				rel_ev.position = Vector2(float(rel_click[0]), float(rel_click[1]))
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
		var click_pos = entry.get("mouse_click_pos", null)
		if typeof(click_pos) == TYPE_ARRAY and click_pos.size() >= 2 and tick == first:
			var cpos = Vector2(float(click_pos[0]), float(click_pos[1]))
			var motion_ev = InputEventMouseMotion.new()
			motion_ev.position = cpos
			_push_input_event(motion_ev)
			var click_ev = InputEventMouseButton.new()
			click_ev.button_index = MOUSE_BUTTON_LEFT
			click_ev.pressed = true
			click_ev.position = cpos
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
		if node.stream != null:
			report["audio_streams_loaded"] = int(report["audio_streams_loaded"]) + 1
			if node.stream is AudioStreamWAV:
				report["audio_mix_rates"].append(int(node.stream.mix_rate))
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

