//! Graphics helpers — color spaces (`math.color`), noise (`math.noise`), random
//! distributions (`math.rand`), easing (`math.ease`), splines (`math.curve`).
//! Run: `zig build example-graphics`

const std = @import("std");
const math = @import("math");
const print = std.debug.print;

pub fn main() void {
    colorSpaces();
    noiseFields();
    randomDistributions();
    easeCurves();
    curveSplines();
}

fn colorSpaces() void {
    const rgb = math.Vec3.init(0.2, 0.6, 0.4);
    print("rgb->hsv     = {f}\n", .{math.color.rgbToHsv(rgb)});
    print("rgb->hsl     = {f}\n", .{math.color.rgbToHsl(rgb)});
    print("linear->srgb = {f}\n", .{math.color.linearToSrgb(rgb)});
    print("luminance    = {d}\n", .{math.color.luminance(rgb)});
}

fn noiseFields() void {
    // Deterministic gradient/simplex noise, output ~ [-1, 1].
    print("perlin2(1.5,2.5)  = {d:.4}\n", .{math.noise.perlin2(math.Vec2.init(1.5, 2.5))});
    print("simplex3(...)     = {d:.4}\n", .{math.noise.simplex3(math.Vec3.init(1.5, 2.5, 3.5))});
    print("pnoise2 tiled     = {d:.4}\n", .{math.noise.pnoise2(math.Vec2.init(1.3, 2.1), math.Vec2.splat(5))});
    // Multi-octave fractal variants (clouds, fire, mountains).
    const p = math.Vec2.init(1.5, 2.5);
    print("fbm               = {d:.4}\n", .{math.noise.fbm(p, .{ .octaves = 6 })});
    print("turbulence        = {d:.4}\n", .{math.noise.turbulence(p, .{ .octaves = 6 })});
    print("ridged            = {d:.4}\n", .{math.noise.ridged(p, .{ .octaves = 6 })});
}

fn randomDistributions() void {
    // Pass an explicit std.Random source (no hidden global).
    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();
    print("uniform [-1,1]   = {d:.4}\n", .{math.rand.uniform(rng, @as(f32, -1), 1)});
    print("normal(0,1)      = {d:.4}\n", .{math.rand.normal(rng, @as(f32, 0), 1)});
    print("onSphere |r|     = {d:.4}\n", .{math.rand.onSphere(rng, 2).length()});
}

fn easeCurves() void {
    // Robert Penner easing, input/output in [0, 1].
    print("bounceOut(0.7)   = {d:.4}\n", .{math.ease.bounceOut(@as(f32, 0.7))});
    print("cubicInOut(0.25) = {d:.4}\n", .{math.ease.cubicInOut(@as(f32, 0.25))});
}

fn curveSplines() void {
    const p0 = math.Vec2.init(0, 0);
    const p1 = math.Vec2.init(1, 2);
    const p2 = math.Vec2.init(2, 2);
    const p3 = math.Vec2.init(3, 0);
    print("bezier(0.5)      = {f}\n", .{math.curve.bezier(p0, p1, p2, p3, @as(f32, 0.5))});
    print("catmullRom(0.5)  = {f}\n", .{math.curve.catmullRom(p0, p1, p2, p3, @as(f32, 0.5))});
}
