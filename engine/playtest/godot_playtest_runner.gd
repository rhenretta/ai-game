# engine/playtest/godot_playtest_runner.gd
# Interactive Opaque-Box Blind Playtester Runner for Godot 4.3 cartridges.
# Resides strictly outside cartridges/ to preserve Rule 1 (Scope Boundary).
# Features:
#   - Silent deterministic 60 FPS offscreen SubViewport + dedicated World2D + AudioEffectCapture
#   - 1:1 desktop mouse parity: Viewport.get_mouse_position(), CanvasItem.get_global_mouse_position(),
#     Control GUI hover/clicks, _input, _unhandled_input, Input singleton polling, and 2D physics picking
#   - Smooth cubic ease-out cursor trajectory toward screen targets + standard desktop pointer arrow cursor
#   - Post-observation human reaction window (default 15 ticks = 250ms) before holding on TCP socket
#   - Live 60 FPS RGB24 video frame streaming + 16kHz mono PCM audio slicing
extends SceneTree

const VIEWPORT_W: int = 1280
const VIEWPORT_H: int = 720
const TARGET_AUDIO_HZ: int = 16000

const PHASE_BOOT_SETTLE: int = 0
const PHASE_REACTION_WINDOW: int = 1
const PHASE_ACTION_BURST: int = 2

const CURSOR_ARROW_ROWS: Array[String] = [
	"B...........",
	"BB..........",
	"BWB.........",
	"BWWB........",
	"BWWWB.......",
	"BWWWWB......",
	"BWWWWWB.....",
	"BWWWWWWB....",
	"BWWWWWWWB...",
	"BWWWWWWWWB..",
	"BWWWWWWWWWB.",
	"BWWWWWWBBBB.",
	"BWWBWWB.....",
	"BWB.BWWB....",
	"BB..BWWB....",
	"B....BWWB...",
	".....BWWB...",
	"......BB....",
]

var ipc_port: int = 0
var video_port: int = 0
var capture_dir: String = ""
var reaction_ticks_cfg: int = 15
var initial_settle_ticks: int = 30

var ctrl_peer: StreamPeerTCP = null
var vid_peer: StreamPeerTCP = null

var render_viewport: SubViewport = null
var main_scene_node: Node = null

var audio_capture: AudioEffectCapture = null
var audio_mix_rate: float = 48000.0
var audio_resample_pos: float = 0.0

var current_tick: int = 0
var observation_index: int = 0
var phase: int = PHASE_BOOT_SETTLE
var reaction_ticks_remaining: int = 0

# Persistent player input & picking state across reaction windows & turns
var cursor_pos: Vector2 = Vector2(640.0, 360.0)
var held_keycodes: Dictionary = {}
var mouse_left_held: bool = false
var transient_mouse_btn_down: int = 0
var click_flash_ticks: int = 0
var hovered_pickables: Dictionary = {}

# Current burst execution state
var burst_total_ticks: int = 0
var burst_tick_idx: int = 0
var burst_has_pre_drag_move: bool = false
var burst_pre_drag_start: Vector2 = Vector2(640.0, 360.0)
var burst_pre_drag_target: Vector2 = Vector2(640.0, 360.0)
var burst_pre_drag_ticks: int = 0
var burst_cursor_start: Vector2 = Vector2(640.0, 360.0)
var burst_cursor_target: Vector2 = Vector2(640.0, 360.0)
var burst_move_start_tick: int = 1
var burst_move_end_tick: int = 1
var burst_mouse_action: String = "none"
var burst_tap_keycodes: Array[int] = []
var burst_baseline_luma: PackedFloat32Array = PackedFloat32Array()
var burst_interrupted_early: bool = false

func _init() -> void:
	var all_args: PackedStringArray = OS.get_cmdline_args()
	for uarg in OS.get_cmdline_user_args():
		all_args.append(uarg)
	for i in range(all_args.size()):
		var arg = String(all_args[i])
		if arg == "--ipc-port" and i + 1 < all_args.size():
			ipc_port = int(all_args[i + 1])
		elif arg == "--video-port" and i + 1 < all_args.size():
			video_port = int(all_args[i + 1])
		elif arg == "--capture-dir" and i + 1 < all_args.size():
			capture_dir = String(all_args[i + 1]).strip_edges()
		elif arg == "--reaction-ticks" and i + 1 < all_args.size():
			reaction_ticks_cfg = clampi(int(all_args[i + 1]), 1, 120)
		elif arg == "--initial-settle-ticks" and i + 1 < all_args.size():
			initial_settle_ticks = clampi(int(all_args[i + 1]), 2, 180)

func _initialize() -> void:
	Engine.max_physics_steps_per_frame = 1
	if ipc_port <= 0 or capture_dir == "":
		printerr("[PLAYTEST_RUNNER_FATAL] Missing required --ipc-port or --capture-dir arguments.")
		quit(1)
		return
	if DisplayServer.get_name() == "headless":
		printerr("[PLAYTEST_RUNNER_FATAL] Blind Playtester requires GPU/offscreen viewport rendering (DisplayServer is headless).")
		quit(1)
		return

	DirAccess.make_dir_recursive_absolute(capture_dir)

	# Configure silent, invisible offscreen host window
	DisplayServer.window_set_flag(DisplayServer.WINDOW_FLAG_NO_FOCUS, true)
	DisplayServer.window_set_flag(DisplayServer.WINDOW_FLAG_BORDERLESS, true)
	DisplayServer.window_set_flag(DisplayServer.WINDOW_FLAG_MOUSE_PASSTHROUGH, true)
	DisplayServer.window_set_flag(DisplayServer.WINDOW_FLAG_TRANSPARENT, true)
	DisplayServer.window_set_position(Vector2i(-4000, -4000))
	root.transparent_bg = true

	# Attach standalone SubViewport directly under root (NOT SubViewportContainer, which would
	# query DisplayServer OS mouse coordinates relative to (-4000, -4000)).
	render_viewport = SubViewport.new()
	render_viewport.name = "PlaytestSubViewport"
	render_viewport.size = Vector2i(VIEWPORT_W, VIEWPORT_H)
	render_viewport.size_2d_override = Vector2i(VIEWPORT_W, VIEWPORT_H)
	render_viewport.world_2d = World2D.new()
	render_viewport.transparent_bg = false
	render_viewport.render_target_update_mode = SubViewport.UPDATE_ALWAYS
	render_viewport.handle_input_locally = true
	render_viewport.physics_object_picking = false
	render_viewport.gui_embed_subwindows = true
	root.add_child(render_viewport)

	# Attach AudioEffectCapture to Master bus (Bus 0) for silent tick-locked game audio perception
	audio_mix_rate = float(AudioServer.get_mix_rate())
	if audio_mix_rate <= 0.0:
		audio_mix_rate = 48000.0
	audio_capture = AudioEffectCapture.new()
	audio_capture.buffer_length = 20.0
	AudioServer.add_bus_effect(0, audio_capture)

	# Connect control socket (and optional rawvideo streaming socket)
	ctrl_peer = _connect_tcp_blocking("127.0.0.1", ipc_port, 8000)
	if ctrl_peer == null:
		printerr("[PLAYTEST_RUNNER_FATAL] Failed connecting to control socket 127.0.0.1:" + str(ipc_port))
		quit(1)
		return
	if video_port > 0:
		vid_peer = _connect_tcp_blocking("127.0.0.1", video_port, 8000)
		if vid_peer == null:
			printerr("[PLAYTEST_RUNNER_FATAL] Failed connecting to video stream socket 127.0.0.1:" + str(video_port))
			quit(1)
			return

	# Instantiate cartridge main scene strictly as a player would boot the game
	var main_path = String(ProjectSettings.get_setting("application/run/main_scene", "res://main.tscn"))
	if not ResourceLoader.exists(main_path):
		printerr("[PLAYTEST_RUNNER_FATAL] Main scene not found at " + main_path)
		quit(1)
		return

	var packed = load(main_path)
	if packed == null and main_path.ends_with(".tscn"):
		var gd_fallback = main_path.substr(0, main_path.length() - 5) + ".gd"
		if ResourceLoader.exists(gd_fallback):
			packed = load(gd_fallback)
	if packed == null:
		printerr("[PLAYTEST_RUNNER_FATAL] Failed loading main scene resource at " + main_path)
		quit(1)
		return

	if packed is PackedScene:
		main_scene_node = packed.instantiate()
	elif packed is Script:
		main_scene_node = packed.new()

	if main_scene_node == null:
		printerr("[PLAYTEST_RUNNER_FATAL] Failed instantiating main scene at " + main_path)
		quit(1)
		return

	render_viewport.add_child(main_scene_node)
	_emit_mouse_motion(cursor_pos, Vector2.ZERO)

	_send_ctrl_json({
		"event": "runner_ready",
		"viewport_w": VIEWPORT_W,
		"viewport_h": VIEWPORT_H,
		"audio_mix_rate": audio_mix_rate,
		"reaction_ticks": reaction_ticks_cfg,
		"initial_settle_ticks": initial_settle_ticks,
	})

func _process(_delta: float) -> bool:
	current_tick += 1
	if current_tick == 1:
		_emit_mouse_motion(cursor_pos, Vector2.ZERO)
	if click_flash_ticks > 0:
		click_flash_ticks -= 1

	if phase == PHASE_ACTION_BURST:
		_step_action_burst_inputs()

	# Render SubViewport deterministically on every tick
	RenderingServer.force_draw(true)
	var frame_img: Image = _capture_composited_frame()

	# Stream 60 FPS frame to ffmpeg H.264 encoder if active
	if vid_peer != null and frame_img != null:
		var rgb_img: Image = frame_img.duplicate()
		rgb_img.convert(Image.FORMAT_RGB8)
		vid_peer.put_data(rgb_img.get_data())

	if phase == PHASE_BOOT_SETTLE:
		if current_tick >= initial_settle_ticks:
			_emit_turn_observation(frame_img, false)
			phase = PHASE_REACTION_WINDOW
			reaction_ticks_remaining = reaction_ticks_cfg
		return false

	if phase == PHASE_ACTION_BURST:
		# Emit live UI preview every 12 ticks (5 FPS) during longer motor bursts
		if burst_tick_idx > 1 and burst_tick_idx < burst_total_ticks and (burst_tick_idx % 12 == 0):
			_emit_live_preview(frame_img)

		# Check for sudden full-screen scene transitions after initial 12 ticks of a long burst
		var scene_cut: bool = false
		if burst_tick_idx == 4 and burst_total_ticks > 24 and frame_img != null:
			burst_baseline_luma = _compute_coarse_luma(frame_img)
		elif burst_tick_idx >= 12 and burst_total_ticks > 24 and burst_tick_idx < burst_total_ticks - 2 and not burst_baseline_luma.is_empty() and frame_img != null:
			var cur_luma = _compute_coarse_luma(frame_img)
			if _luma_shift_fraction(burst_baseline_luma, cur_luma) >= 0.48:
				scene_cut = true
				burst_interrupted_early = true

		if burst_tick_idx >= burst_total_ticks or scene_cut:
			_finish_burst_transient_inputs()
			_emit_turn_observation(frame_img, burst_interrupted_early)
			phase = PHASE_REACTION_WINDOW
			reaction_ticks_remaining = reaction_ticks_cfg
		return false

	if phase == PHASE_REACTION_WINDOW:
		reaction_ticks_remaining -= 1
		if reaction_ticks_remaining <= 0:
			var should_quit = _await_and_apply_command()
			if should_quit:
				_shutdown_runner()
				quit(0)
				return true
		return false

	return false

func _capture_composited_frame() -> Image:
	if render_viewport == null:
		return null
	var vp_tex = render_viewport.get_texture()
	if vp_tex == null:
		return null
	var img = vp_tex.get_image()
	if img == null or img.is_empty():
		return null
	if img.is_compressed():
		img.decompress()
	if img.get_width() != VIEWPORT_W or img.get_height() != VIEWPORT_H:
		img.resize(VIEWPORT_W, VIEWPORT_H, Image.INTERPOLATE_BILINEAR)
	img.convert(Image.FORMAT_RGBA8)
	_draw_cursor_overlay(img, cursor_pos, click_flash_ticks, mouse_left_held)
	return img

func _draw_cursor_overlay(img: Image, pos: Vector2, flash_ticks: int, left_down: bool) -> void:
	var mm: int = Input.mouse_mode
	if mm == Input.MOUSE_MODE_CAPTURED or mm == Input.MOUSE_MODE_HIDDEN:
		return

	var cx: int = clampi(int(round(pos.x)), 0, VIEWPORT_W - 1)
	var cy: int = clampi(int(round(pos.y)), 0, VIEWPORT_H - 1)

	# Subtle click ring behind the pointer tip during button press/flash
	if flash_ticks > 0:
		var radius: float = 5.0 + float(10 - flash_ticks) * 1.2
		var ring_col := Color(1.0, 0.84, 0.24, clampf(float(flash_ticks) / 11.0, 0.18, 0.82))
		var steps: int = 28
		for s in range(steps):
			var ang: float = TAU * float(s) / float(steps)
			var rx: int = int(round(float(cx) + cos(ang) * radius))
			var ry: int = int(round(float(cy) + sin(ang) * radius))
			_blend_px(img, rx, ry, ring_col)

	var shadow_col := Color(0.0, 0.0, 0.0, 0.28)
	var border_col := Color(0.05, 0.06, 0.08, 0.98)
	var fill_col := Color(1.0, 1.0, 1.0, 0.99) if not (left_down or flash_ticks > 6) else Color(1.0, 0.93, 0.72, 0.99)

	# Soft 1px drop shadow down-right for contrast over bright UI surfaces
	for ry in range(CURSOR_ARROW_ROWS.size()):
		var row_str: String = CURSOR_ARROW_ROWS[ry]
		for rx in range(row_str.length()):
			if row_str[rx] != ".":
				_blend_px(img, cx + rx + 1, cy + ry + 1, shadow_col)

	# Standard desktop white arrow pointer with black border (tip at (cx, cy))
	for ry in range(CURSOR_ARROW_ROWS.size()):
		var row_str: String = CURSOR_ARROW_ROWS[ry]
		for rx in range(row_str.length()):
			var ch: String = row_str[rx]
			if ch == "B":
				_blend_px(img, cx + rx, cy + ry, border_col)
			elif ch == "W":
				_blend_px(img, cx + rx, cy + ry, fill_col)

func _blend_px(img: Image, x: int, y: int, col: Color) -> void:
	if x < 0 or x >= VIEWPORT_W or y < 0 or y >= VIEWPORT_H:
		return
	var dst: Color = img.get_pixel(x, y)
	var a: float = clampf(col.a, 0.0, 1.0)
	img.set_pixel(
		x,
		y,
		Color(
			dst.r * (1.0 - a) + col.r * a,
			dst.g * (1.0 - a) + col.g * a,
			dst.b * (1.0 - a) + col.b * a,
			1.0
		)
	)

func _compute_coarse_luma(img: Image) -> PackedFloat32Array:
	var out := PackedFloat32Array()
	var gx: int = 32
	var gy: int = 18
	out.resize(gx * gy)
	var idx: int = 0
	for iy in range(gy):
		var py: int = clampi(int((float(iy) + 0.5) * float(VIEWPORT_H) / float(gy)), 0, VIEWPORT_H - 1)
		for ix in range(gx):
			var px: int = clampi(int((float(ix) + 0.5) * float(VIEWPORT_W) / float(gx)), 0, VIEWPORT_W - 1)
			var c: Color = img.get_pixel(px, py)
			out[idx] = 0.299 * c.r + 0.587 * c.g + 0.114 * c.b
			idx += 1
	return out

func _luma_shift_fraction(a: PackedFloat32Array, b: PackedFloat32Array) -> float:
	if a.size() == 0 or a.size() != b.size():
		return 0.0
	var changed: int = 0
	for i in range(a.size()):
		if absf(a[i] - b[i]) > 0.22:
			changed += 1
	return float(changed) / float(a.size())

func _emit_live_preview(img: Image) -> void:
	if img == null:
		return
	var preview_path: String = capture_dir.path_join("live_preview.jpg")
	img.save_jpg(preview_path, 0.82)
	_send_ctrl_json({
		"event": "live_frame",
		"tick": current_tick,
		"cursor_x": snapped(cursor_pos.x, 0.1),
		"cursor_y": snapped(cursor_pos.y, 0.1),
		"preview_path": preview_path,
	})

func _emit_turn_observation(img: Image, interrupted_early: bool) -> void:
	observation_index += 1
	var png_name: String = "turn_%04d.png" % observation_index
	var png_path: String = capture_dir.path_join(png_name)
	var preview_path: String = capture_dir.path_join("live_preview.jpg")
	if img != null:
		img.save_png(png_path)
		img.save_jpg(preview_path, 0.85)
	var pcm_name: String = "turn_%04d.pcm" % observation_index
	var pcm_path: String = capture_dir.path_join(pcm_name)
	var audio_info: Dictionary = _drain_audio_to_pcm16(pcm_path)
	_send_ctrl_json({
		"event": "observation_ready",
		"observation_index": observation_index,
		"tick": current_tick,
		"sim_time_sec": snapped(float(current_tick) / 60.0, 0.001),
		"cursor_x": snapped(cursor_pos.x, 0.1),
		"cursor_y": snapped(cursor_pos.y, 0.1),
		"png_path": png_path,
		"preview_path": preview_path,
		"pcm_path": pcm_path,
		"audio_samples_16k": int(audio_info.get("samples", 0)),
		"audio_rms": float(audio_info.get("rms", 0.0)),
		"audio_peak": float(audio_info.get("peak", 0.0)),
		"interrupted_early": interrupted_early,
	})

func _drain_audio_to_pcm16(out_path: String) -> Dictionary:
	if audio_capture == null:
		return {"samples": 0, "rms": 0.0, "peak": 0.0}
	var avail: int = audio_capture.get_frames_available()
	if avail <= 0:
		var empty_f = FileAccess.open(out_path, FileAccess.WRITE)
		if empty_f != null:
			empty_f.close()
		return {"samples": 0, "rms": 0.0, "peak": 0.0}

	var raw_frames: PackedVector2Array = audio_capture.get_buffer(avail)
	var step_ratio: float = audio_mix_rate / float(TARGET_AUDIO_HZ)
	var out_bytes := PackedByteArray()
	var sum_sq: float = 0.0
	var peak: float = 0.0
	var out_samples: int = 0

	var pos: float = audio_resample_pos
	var n_in: int = raw_frames.size()
	while int(pos) < n_in:
		var idx: int = int(pos)
		var frame: Vector2 = raw_frames[idx]
		var mono: float = clampf((frame.x + frame.y) * 0.5, -1.0, 1.0)
		var abs_m: float = absf(mono)
		if abs_m > peak:
			peak = abs_m
		sum_sq += mono * mono
		var s16: int = clampi(int(round(mono * 32767.0)), -32768, 32767)
		out_bytes.append(s16 & 0xFF)
		out_bytes.append((s16 >> 8) & 0xFF)
		out_samples += 1
		pos += step_ratio

	audio_resample_pos = pos - float(n_in)
	if audio_resample_pos < 0.0:
		audio_resample_pos = 0.0

	var f = FileAccess.open(out_path, FileAccess.WRITE)
	if f != null:
		f.store_buffer(out_bytes)
		f.close()

	var rms: float = sqrt(sum_sq / float(out_samples)) if out_samples > 0 else 0.0
	return {
		"samples": out_samples,
		"rms": snapped(rms, 0.0001),
		"peak": snapped(peak, 0.0001),
	}

func _await_and_apply_command() -> bool:
	while true:
		var msg: Dictionary = _recv_ctrl_json_blocking()
		if msg.is_empty():
			return true
		var cmd: String = String(msg.get("cmd", "")).strip_edges().to_lower()
		if cmd == "quit":
			return true
		elif cmd == "ping":
			_send_ctrl_json({"event": "pong", "tick": current_tick})
			continue
		elif cmd == "step":
			_configure_action_burst(msg)
			phase = PHASE_ACTION_BURST
			return false
	return true

func _configure_action_burst(msg: Dictionary) -> void:
	var dur_ms: int = clampi(int(msg.get("duration_ms", 550)), 100, 5000)
	burst_total_ticks = maxi(6, int(round(float(dur_ms) * 60.0 / 1000.0)))
	burst_tick_idx = 0
	burst_interrupted_early = false
	burst_baseline_luma = PackedFloat32Array()
	burst_has_pre_drag_move = false
	burst_pre_drag_ticks = 0

	# Synchronize held keyboard keys (keep keys that stay held; release keys no longer held)
	var next_hold_map: Dictionary = {}
	var raw_hold = msg.get("hold_keys", [])
	if typeof(raw_hold) == TYPE_ARRAY:
		for k_name in raw_hold:
			var kc: int = _key_name_to_keycode(String(k_name))
			if kc != 0:
				next_hold_map[kc] = true

	for existing_kc in held_keycodes.keys():
		if not next_hold_map.has(existing_kc):
			_emit_key_event(int(existing_kc), false)
	for new_kc in next_hold_map.keys():
		if not held_keycodes.has(new_kc):
			_emit_key_event(int(new_kc), true)
	held_keycodes = next_hold_map

	# Prepare tapped keys (pressed on burst tick 1, released on burst tick 4)
	burst_tap_keycodes.clear()
	var raw_tap = msg.get("tap_keys", [])
	if typeof(raw_tap) == TYPE_ARRAY:
		for t_name in raw_tap:
			var tkc: int = _key_name_to_keycode(String(t_name))
			if tkc != 0 and not held_keycodes.has(tkc) and not burst_tap_keycodes.has(tkc):
				burst_tap_keycodes.append(tkc)

	# Resolve target cursor position (primary: screen coordinates mouse_x/mouse_y; fallback: polar distance/angle)
	burst_mouse_action = String(msg.get("mouse_button_action", "none")).strip_edges().to_lower()
	var dest_pos: Vector2 = cursor_pos
	var raw_mx = msg.get("mouse_x", null)
	var raw_my = msg.get("mouse_y", null)
	if raw_mx != null or raw_my != null:
		var tx: float = float(raw_mx) if raw_mx != null else cursor_pos.x
		var ty: float = float(raw_my) if raw_my != null else cursor_pos.y
		dest_pos = Vector2(
			clampf(tx, 2.0, float(VIEWPORT_W - 2)),
			clampf(ty, 2.0, float(VIEWPORT_H - 2))
		)
	else:
		var dist_px: float = clampf(float(msg.get("mouse_distance_px", 0.0)), 0.0, 1600.0)
		if dist_px > 0.5:
			var rad: float = deg_to_rad(float(msg.get("mouse_angle_deg", 0.0)))
			var raw_dest: Vector2 = cursor_pos + Vector2(cos(rad), sin(rad)) * dist_px
			dest_pos = Vector2(
				clampf(raw_dest.x, 2.0, float(VIEWPORT_W - 2)),
				clampf(raw_dest.y, 2.0, float(VIEWPORT_H - 2))
			)

	# Optional drag_from_x / drag_from_y pre-move when dragging an object not currently under the cursor
	if burst_mouse_action == "drag_left":
		var raw_fx = msg.get("drag_from_x", null)
		var raw_fy = msg.get("drag_from_y", null)
		if raw_fx != null or raw_fy != null:
			var fx: float = float(raw_fx) if raw_fx != null else cursor_pos.x
			var fy: float = float(raw_fy) if raw_fy != null else cursor_pos.y
			var from_pt := Vector2(
				clampf(fx, 2.0, float(VIEWPORT_W - 2)),
				clampf(fy, 2.0, float(VIEWPORT_H - 2))
			)
			if cursor_pos.distance_to(from_pt) > 1.0:
				burst_has_pre_drag_move = true
				burst_pre_drag_start = cursor_pos
				burst_pre_drag_target = from_pt
				burst_pre_drag_ticks = maxi(2, mini(int(burst_total_ticks / 3), burst_total_ticks - 4))
				burst_cursor_start = from_pt
				burst_cursor_target = dest_pos
				burst_move_start_tick = burst_pre_drag_ticks + 2
				burst_move_end_tick = maxi(burst_move_start_tick, burst_total_ticks - 2)
				return

	burst_cursor_start = cursor_pos
	burst_cursor_target = dest_pos
	if burst_mouse_action == "drag_left":
		burst_move_start_tick = 2
		burst_move_end_tick = maxi(2, burst_total_ticks - 2)
	elif burst_mouse_action in ["left_click", "right_click", "double_click", "hold_left", "wheel_up", "wheel_down"] and burst_cursor_start.distance_to(burst_cursor_target) > 0.25:
		burst_move_start_tick = 1
		burst_move_end_tick = maxi(1, burst_total_ticks - 5)
	else:
		burst_move_start_tick = 1
		burst_move_end_tick = maxi(1, burst_total_ticks)

func _step_action_burst_inputs() -> void:
	burst_tick_idx += 1

	# Tap keys: press on tick 1, release on tick 4
	if burst_tick_idx == 1:
		for tkc in burst_tap_keycodes:
			_emit_key_event(tkc, true)
	elif burst_tick_idx == 4:
		for tkc in burst_tap_keycodes:
			_emit_key_event(tkc, false)

	# Drag gesture (optional unpressed sweep -> press -> held sweep -> release)
	if burst_mouse_action == "drag_left":
		if burst_has_pre_drag_move and burst_tick_idx <= burst_pre_drag_ticks:
			var u_pre: float = clampf(float(burst_tick_idx) / float(burst_pre_drag_ticks), 0.0, 1.0)
			var eased_pre: float = 1.0 - pow(1.0 - u_pre, 3.0)
			var next_pre: Vector2 = burst_pre_drag_start.lerp(burst_pre_drag_target, eased_pre)
			var rel_pre: Vector2 = next_pre - cursor_pos
			cursor_pos = next_pre
			_emit_mouse_motion(cursor_pos, rel_pre)
		elif burst_tick_idx == (burst_pre_drag_ticks + 1 if burst_has_pre_drag_move else 1):
			if not mouse_left_held:
				mouse_left_held = true
				_emit_mouse_button(MOUSE_BUTTON_LEFT, true, false)
				click_flash_ticks = 10
		elif burst_tick_idx >= burst_move_start_tick and burst_tick_idx <= burst_move_end_tick and burst_cursor_start.distance_to(burst_cursor_target) > 0.25:
			var span: int = maxi(1, burst_move_end_tick - burst_move_start_tick + 1)
			var u_drag: float = clampf(float(burst_tick_idx - burst_move_start_tick + 1) / float(span), 0.0, 1.0)
			var eased_drag: float = 1.0 - pow(1.0 - u_drag, 3.0)
			var next_drag: Vector2 = burst_cursor_start.lerp(burst_cursor_target, eased_drag)
			var rel_drag: Vector2 = next_drag - cursor_pos
			cursor_pos = next_drag
			_emit_mouse_motion(cursor_pos, rel_drag)
		elif burst_tick_idx >= burst_total_ticks - 1 and mouse_left_held:
			mouse_left_held = false
			_emit_mouse_button(MOUSE_BUTTON_LEFT, false, false)
		Input.flush_buffered_events()
		return

	# Release-left gesture
	if burst_mouse_action == "release_left" and burst_tick_idx == 1 and mouse_left_held:
		mouse_left_held = false
		_emit_mouse_button(MOUSE_BUTTON_LEFT, false, false)

	# Interpolate cursor along smooth human cubic ease-out trajectory: 1 - (1 - u)^3
	if burst_cursor_start.distance_to(burst_cursor_target) > 0.25 and burst_tick_idx <= burst_move_end_tick:
		var u: float = clampf(float(burst_tick_idx) / float(burst_move_end_tick), 0.0, 1.0)
		var eased: float = 1.0 - pow(1.0 - u, 3.0)
		var next_pos: Vector2 = burst_cursor_start.lerp(burst_cursor_target, eased)
		var rel: Vector2 = next_pos - cursor_pos
		cursor_pos = next_pos
		_emit_mouse_motion(cursor_pos, rel)

	# Execute clicks / scroll once cursor reaches target (or immediately on tick 1 when stationary)
	var click_start_tick: int = (burst_move_end_tick + 1) if burst_cursor_start.distance_to(burst_cursor_target) > 0.25 else 1
	if burst_mouse_action == "left_click":
		if burst_tick_idx == click_start_tick:
			transient_mouse_btn_down = MOUSE_BUTTON_LEFT
			_emit_mouse_button(MOUSE_BUTTON_LEFT, true, false)
			click_flash_ticks = 10
		elif burst_tick_idx == click_start_tick + 2:
			transient_mouse_btn_down = 0
			_emit_mouse_button(MOUSE_BUTTON_LEFT, false, false)
	elif burst_mouse_action == "right_click":
		if burst_tick_idx == click_start_tick:
			transient_mouse_btn_down = MOUSE_BUTTON_RIGHT
			_emit_mouse_button(MOUSE_BUTTON_RIGHT, true, false)
			click_flash_ticks = 10
		elif burst_tick_idx == click_start_tick + 2:
			transient_mouse_btn_down = 0
			_emit_mouse_button(MOUSE_BUTTON_RIGHT, false, false)
	elif burst_mouse_action == "double_click":
		if burst_tick_idx == click_start_tick:
			transient_mouse_btn_down = MOUSE_BUTTON_LEFT
			_emit_mouse_button(MOUSE_BUTTON_LEFT, true, false)
			click_flash_ticks = 10
		elif burst_tick_idx == click_start_tick + 1:
			transient_mouse_btn_down = 0
			_emit_mouse_button(MOUSE_BUTTON_LEFT, false, false)
		elif burst_tick_idx == click_start_tick + 2:
			transient_mouse_btn_down = MOUSE_BUTTON_LEFT
			_emit_mouse_button(MOUSE_BUTTON_LEFT, true, true)
			click_flash_ticks = 10
		elif burst_tick_idx == click_start_tick + 3:
			transient_mouse_btn_down = 0
			_emit_mouse_button(MOUSE_BUTTON_LEFT, false, false)
	elif burst_mouse_action == "hold_left":
		if burst_tick_idx == click_start_tick and not mouse_left_held:
			mouse_left_held = true
			_emit_mouse_button(MOUSE_BUTTON_LEFT, true, false)
			click_flash_ticks = 10
	elif burst_mouse_action == "wheel_up":
		if burst_tick_idx == click_start_tick:
			_emit_mouse_button(MOUSE_BUTTON_WHEEL_UP, true, false)
		elif burst_tick_idx == click_start_tick + 1:
			_emit_mouse_button(MOUSE_BUTTON_WHEEL_UP, false, false)
	elif burst_mouse_action == "wheel_down":
		if burst_tick_idx == click_start_tick:
			_emit_mouse_button(MOUSE_BUTTON_WHEEL_DOWN, true, false)
		elif burst_tick_idx == click_start_tick + 1:
			_emit_mouse_button(MOUSE_BUTTON_WHEEL_DOWN, false, false)

	Input.flush_buffered_events()

func _finish_burst_transient_inputs() -> void:
	# Ensure any short tap keys or mid-click transient presses are cleanly released if burst ended early
	if burst_tick_idx < 4:
		for tkc in burst_tap_keycodes:
			_emit_key_event(tkc, false)
	if transient_mouse_btn_down != 0:
		_emit_mouse_button(transient_mouse_btn_down, false, false)
		transient_mouse_btn_down = 0
	if burst_mouse_action == "drag_left" and mouse_left_held:
		mouse_left_held = false
		_emit_mouse_button(MOUSE_BUTTON_LEFT, false, false)
	Input.flush_buffered_events()

func _emit_key_event(keycode: int, pressed: bool) -> void:
	var kev := InputEventKey.new()
	kev.keycode = keycode as Key
	kev.physical_keycode = keycode as Key
	if pressed and keycode >= 32 and keycode <= 126:
		kev.unicode = keycode
	kev.pressed = pressed
	kev.echo = false
	Input.parse_input_event(kev)
	Input.flush_buffered_events()
	if render_viewport != null:
		render_viewport.push_input(kev, true)

func _emit_mouse_motion(pos: Vector2, rel: Vector2) -> void:
	var mev := InputEventMouseMotion.new()
	mev.position = pos
	mev.global_position = pos
	mev.relative = rel
	mev.button_mask = MOUSE_BUTTON_MASK_LEFT if mouse_left_held else 0
	_dispatch_mouse_to_viewport(mev)

func _emit_mouse_button(button_idx: int, pressed: bool, is_double: bool) -> void:
	var bev := InputEventMouseButton.new()
	bev.button_index = button_idx as MouseButton
	bev.pressed = pressed
	bev.double_click = is_double
	bev.factor = 1.0
	bev.position = cursor_pos
	bev.global_position = cursor_pos
	var mask: int = MOUSE_BUTTON_MASK_LEFT if mouse_left_held else 0
	if pressed:
		if button_idx == MOUSE_BUTTON_LEFT:
			mask |= MOUSE_BUTTON_MASK_LEFT
		elif button_idx == MOUSE_BUTTON_RIGHT:
			mask |= MOUSE_BUTTON_MASK_RIGHT
	bev.button_mask = mask
	_dispatch_mouse_to_viewport(bev)

func _dispatch_mouse_to_viewport(ev: InputEventMouse) -> void:
	Input.parse_input_event(ev)
	Input.flush_buffered_events()
	if render_viewport == null:
		return
	render_viewport.push_input(ev, true)
	if render_viewport.is_input_handled():
		# GUI Control consumed this mouse event -> unhover world pickables and do not pick through HUD
		_clear_hovered_pickables()
		return
	_dispatch_2d_physics_picking(ev)

func _clear_hovered_pickables() -> void:
	if hovered_pickables.is_empty():
		return
	for prev_iid in hovered_pickables.keys():
		var prev_obj = instance_from_id(int(prev_iid))
		if prev_obj != null and prev_obj is CollisionObject2D:
			prev_obj.mouse_exited.emit()
			if prev_obj.has_method("_mouse_exit"):
				prev_obj._mouse_exit()
	hovered_pickables.clear()

func _dispatch_2d_physics_picking(ev: InputEventMouse) -> void:
	var w2d: World2D = render_viewport.find_world_2d()
	if w2d == null:
		return
	var space_state: PhysicsDirectSpaceState2D = w2d.direct_space_state
	if space_state == null:
		return
	var world_pt: Vector2 = render_viewport.get_canvas_transform().affine_inverse() * ev.position
	var query := PhysicsPointQueryParameters2D.new()
	query.position = world_pt
	query.collide_with_areas = true
	query.collide_with_bodies = true
	var hits: Array[Dictionary] = space_state.intersect_point(query, 32)

	var next_hovered: Dictionary = {}
	for h in hits:
		var col = h.get("collider")
		if col is CollisionObject2D and col.input_pickable:
			next_hovered[col.get_instance_id()] = col

	for prev_iid in hovered_pickables.keys():
		if not next_hovered.has(prev_iid):
			var prev_obj = instance_from_id(int(prev_iid))
			if prev_obj != null and prev_obj is CollisionObject2D:
				prev_obj.mouse_exited.emit()
				if prev_obj.has_method("_mouse_exit"):
					prev_obj._mouse_exit()

	for cur_iid in next_hovered.keys():
		if not hovered_pickables.has(cur_iid):
			var cur_obj: CollisionObject2D = next_hovered[cur_iid]
			cur_obj.mouse_entered.emit()
			if cur_obj.has_method("_mouse_enter"):
				cur_obj._mouse_enter()

	hovered_pickables = next_hovered

	var delivered_iids: Dictionary = {}
	for h in hits:
		var col = h.get("collider")
		var shape_idx: int = int(h.get("shape", 0))
		if col is CollisionObject2D and col.input_pickable:
			var iid: int = col.get_instance_id()
			if delivered_iids.has(iid):
				continue
			delivered_iids[iid] = true
			col.input_event.emit(render_viewport, ev, shape_idx)
			if col.has_method("_input_event"):
				col._input_event(render_viewport, ev, shape_idx)

func _shutdown_runner() -> void:
	for kc in held_keycodes.keys():
		_emit_key_event(int(kc), false)
	held_keycodes.clear()
	if transient_mouse_btn_down != 0:
		_emit_mouse_button(transient_mouse_btn_down, false, false)
		transient_mouse_btn_down = 0
	if mouse_left_held:
		mouse_left_held = false
		_emit_mouse_button(MOUSE_BUTTON_LEFT, false, false)
	_clear_hovered_pickables()
	Input.flush_buffered_events()

	var tail_pcm_path: String = capture_dir.path_join("final_tail.pcm")
	var tail_info: Dictionary = _drain_audio_to_pcm16(tail_pcm_path)
	_send_ctrl_json({
		"event": "runner_stopping",
		"tick": current_tick,
		"tail_pcm_path": tail_pcm_path,
		"tail_audio_samples_16k": int(tail_info.get("samples", 0)),
	})
	if vid_peer != null:
		vid_peer.disconnect_from_host()
		vid_peer = null
	if ctrl_peer != null:
		ctrl_peer.disconnect_from_host()
		ctrl_peer = null

func _connect_tcp_blocking(host: String, port: int, timeout_ms: int) -> StreamPeerTCP:
	var peer := StreamPeerTCP.new()
	peer.big_endian = true
	var err = peer.connect_to_host(host, port)
	if err != OK:
		return null
	var deadline: int = Time.get_ticks_msec() + timeout_ms
	while Time.get_ticks_msec() < deadline:
		peer.poll()
		var st = peer.get_status()
		if st == StreamPeerTCP.STATUS_CONNECTED:
			peer.set_no_delay(true)
			return peer
		if st == StreamPeerTCP.STATUS_ERROR or st == StreamPeerTCP.STATUS_NONE:
			return null
		OS.delay_msec(10)
	return null

func _send_ctrl_json(payload: Dictionary) -> void:
	if ctrl_peer == null:
		return
	ctrl_peer.poll()
	if ctrl_peer.get_status() != StreamPeerTCP.STATUS_CONNECTED:
		return
	var raw: PackedByteArray = JSON.stringify(payload).to_utf8_buffer()
	ctrl_peer.put_u32(raw.size())
	ctrl_peer.put_data(raw)

func _recv_ctrl_json_blocking() -> Dictionary:
	if ctrl_peer == null:
		return {}
	# Read 4-byte big-endian length header while paused cleanly on socket
	while true:
		ctrl_peer.poll()
		var st = ctrl_peer.get_status()
		if st != StreamPeerTCP.STATUS_CONNECTED:
			return {}
		if ctrl_peer.get_available_bytes() >= 4:
			break
		OS.delay_msec(8)
	var msg_len: int = ctrl_peer.get_u32()
	if msg_len <= 0 or msg_len > 1048576:
		return {}
	var buf := PackedByteArray()
	while buf.size() < msg_len:
		ctrl_peer.poll()
		if ctrl_peer.get_status() != StreamPeerTCP.STATUS_CONNECTED:
			return {}
		var avail: int = ctrl_peer.get_available_bytes()
		if avail > 0:
			var to_read: int = mini(avail, msg_len - buf.size())
			var chunk_res = ctrl_peer.get_data(to_read)
			if chunk_res[0] != OK:
				return {}
			buf.append_array(chunk_res[1])
		else:
			OS.delay_msec(4)
	var parsed = JSON.parse_string(buf.get_string_from_utf8())
	if typeof(parsed) == TYPE_DICTIONARY:
		return parsed
	return {}

func _key_name_to_keycode(raw_name: String) -> int:
	var k: String = raw_name.strip_edges().to_upper()
	match k:
		"W": return KEY_W
		"A": return KEY_A
		"S": return KEY_S
		"D": return KEY_D
		"E": return KEY_E
		"Q": return KEY_Q
		"R": return KEY_R
		"F": return KEY_F
		"C": return KEY_C
		"X": return KEY_X
		"Z": return KEY_Z
		"V": return KEY_V
		"B": return KEY_B
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
		"T": return KEY_T
		"U": return KEY_U
		"Y": return KEY_Y
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
		"SPACE", "SPACEBAR": return KEY_SPACE
		"ENTER", "RETURN": return KEY_KP_ENTER if k == "KP_ENTER" else KEY_ENTER
		"ESC", "ESCAPE": return KEY_ESCAPE
		"TAB": return KEY_TAB
		"SHIFT", "LSHIFT", "LEFT_SHIFT": return KEY_SHIFT
		"CTRL", "CONTROL", "LCTRL": return KEY_CTRL
		"ALT", "LALT": return KEY_ALT
		"BACKSPACE": return KEY_BACKSPACE
		"UP", "ARROW_UP": return KEY_UP
		"DOWN", "ARROW_DOWN": return KEY_DOWN
		"LEFT", "ARROW_LEFT": return KEY_LEFT
		"RIGHT", "ARROW_RIGHT": return KEY_RIGHT
		"F1": return KEY_F1
		"F2": return KEY_F2
		"F3": return KEY_F3
		"F4": return KEY_F4
		"F5": return KEY_F5
		"F6": return KEY_F6
		"F7": return KEY_F7
		"F8": return KEY_F8
		"F9": return KEY_F9
		"F10": return KEY_F10
		"F11": return KEY_F11
		"F12": return KEY_F12
	return 0
