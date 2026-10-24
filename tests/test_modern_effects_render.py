"""Empirical check that modern 2D effects (lights, GPU particles, shaders) actually render in the capture path.

Skipped when no display / xvfb is available (viewport capture requires a real GL context).
"""
from __future__ import annotations

import os
import platform
import shutil
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
PIL_Image = pytest.importorskip("PIL.Image")

_HAS_DISPLAY = (
    platform.system() == "Darwin"
    or bool(os.environ.get("DISPLAY"))
    or bool(os.environ.get("WAYLAND_DISPLAY"))
    or (platform.system() == "Linux" and shutil.which("xvfb-run") is not None)
)

PROJECT = '''config_version=5
[application]
config/name="probe"
run/main_scene="res://main.tscn"
config/features=PackedStringArray("4.3", "GL Compatibility")
[display]
window/size/viewport_width=1280
window/size/viewport_height=720
[rendering]
renderer/rendering_method="gl_compatibility"
'''

SCENE = '''[gd_scene load_steps=2 format=3]
[ext_resource type="Script" path="res://main.gd" id="1"]
[node name="Main" type="Node2D"]
script = ExtResource("1")
'''

SCRIPT = '''extends Node2D
func _ready() -> void:
	var cm := CanvasModulate.new()
	cm.color = Color(0.12, 0.12, 0.2)
	add_child(cm)
	var bg := ColorRect.new()
	bg.size = Vector2(1280, 720)
	bg.color = Color(0.5, 0.5, 0.6)
	add_child(bg)
	var tex := GradientTexture2D.new()
	var g := Gradient.new()
	g.set_color(0, Color(1, 1, 1, 1))
	g.set_color(1, Color(1, 1, 1, 0))
	tex.gradient = g
	tex.fill = GradientTexture2D.FILL_RADIAL
	tex.fill_from = Vector2(0.5, 0.5)
	tex.fill_to = Vector2(1.0, 0.5)
	tex.width = 512
	tex.height = 512
	var light := PointLight2D.new()
	light.texture = tex
	light.color = Color(0.3, 0.9, 1.0)
	light.energy = 2.0
	light.position = Vector2(640, 360)
	add_child(light)
	var p := GPUParticles2D.new()
	var pm := ParticleProcessMaterial.new()
	pm.direction = Vector3(0, -1, 0)
	pm.initial_velocity_min = 60.0
	pm.initial_velocity_max = 120.0
	pm.spread = 40.0
	pm.scale_min = 0.03
	pm.scale_max = 0.06
	p.process_material = pm
	p.amount = 200
	p.lifetime = 2.0
	p.position = Vector2(300, 600)
	p.modulate = Color(3.0, 1.5, 0.5, 1.0)
	p.texture = tex
	add_child(p)
	var sp := ColorRect.new()
	sp.size = Vector2(300, 200)
	sp.position = Vector2(900, 450)
	var sh := Shader.new()
	sh.code = "shader_type canvas_item; void fragment(){ COLOR = vec4(UV.x, UV.y, 0.9, 1.0); }"
	var mat := ShaderMaterial.new()
	mat.shader = sh
	sp.material = mat
	add_child(sp)
'''


@pytest.mark.skipif(not _HAS_DISPLAY, reason="viewport capture needs a display or xvfb")
def test_lights_particles_and_shaders_render_in_capture_path():
    from engine.sim.godot_verifier import GodotHeadlessSimulator

    probe = Path(__file__).parent / "_modern_effects_probe"
    shutil.rmtree(probe, ignore_errors=True)
    probe.mkdir(parents=True)
    try:
        (probe / "project.godot").write_text(PROJECT)
        (probe / "main.tscn").write_text(SCENE)
        (probe / "main.gd").write_text(SCRIPT)
        GodotHeadlessSimulator(probe, capture_viewport=True, max_ticks=90).run_until_complete()
        img = np.asarray(PIL_Image.open(probe / "playtest_tend.png").convert("RGB")).astype(float)
    finally:
        shutil.rmtree(probe, ignore_errors=True)

    background = img[20:120, 20:120].mean()            # far corner: dim ambient only
    light_core = img[330:390, 610:670].mean()          # PointLight2D centre
    particle_zone_max = img[350:620, 200:420].max()    # column above the emitter
    shader_patch = img[470:630, 920:1180]

    assert light_core > background + 60, "PointLight2D did not light the scene"
    assert particle_zone_max > background + 40, "GPUParticles2D produced no visible particles"
    assert shader_patch.std() > 5, "canvas_item shader produced a flat region"
