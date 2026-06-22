//! Animation & data — easing + spring smoothing (`math.ease`), splines
//! (`math.curve`), and running statistics (`math.stats`).
//! Run: `zig build example-animation`

const std = @import("std");
const math = @import("math");
const print = std.debug.print;

pub fn main() void {
    easeCurves();
    bezierCurves();
    splineCurves();
    springSmoothing();
    runningStats();
}

fn easeCurves() void {
    print("bounceOut(0.7)   = {d:.4}\n", .{math.ease.bounceOut(@as(f32, 0.7))});
    print("elasticOut(0.5)  = {d:.4}\n", .{math.ease.elasticOut(@as(f32, 0.5))});
}

fn bezierCurves() void {
    const p0 = math.Vec2.init(0, 0);
    const p1 = math.Vec2.init(1, 2);
    const p2 = math.Vec2.init(2, 2);
    const p3 = math.Vec2.init(3, 0);
    // Quadratic + cubic, with tangents (derivatives) and arc length.
    print("bezierQuad(0.5)  = {f}\n", .{math.curve.bezierQuad(p0, p1, p2, @as(f32, 0.5))});
    print("bezier(0.5)      = {f}\n", .{math.curve.bezier(p0, p1, p2, p3, @as(f32, 0.5))});
    print("tangent @0.5     = {f}\n", .{math.curve.bezierDerivative(p0, p1, p2, p3, @as(f32, 0.5))});
    print("arc length       = {d:.4}\n", .{math.curve.bezierArcLength(p0, p1, p2, p3, 64)});
}

fn splineCurves() void {
    const p0 = math.Vec2.init(0, 0);
    const p1 = math.Vec2.init(1, 2);
    const p2 = math.Vec2.init(2, 2);
    const p3 = math.Vec2.init(3, 0);
    print("catmullRom(0.5)  = {f}  (passes through points)\n", .{math.curve.catmullRom(p0, p1, p2, p3, @as(f32, 0.5))});
    print("bspline(0.5)     = {f}  (smooths, C²)\n", .{math.curve.bspline(p0, p1, p2, p3, @as(f32, 0.5))});
}

fn springSmoothing() void {
    // Frame-rate-independent follow toward a target.
    var pos = math.Vec3.splat(0);
    var vel = math.Vec3.splat(0);
    var t: usize = 0;
    while (t < 90) : (t += 1) pos = math.ease.smoothDamp(pos, math.Vec3.init(10, 0, 0), &vel, 0.5, 1.0 / 60.0);
    print("smoothDamp ~1.5s = {f}\n", .{pos});
    print("expDecay 1 step  = {d:.4}\n", .{math.ease.expDecay(@as(f32, 0), 1, 10, 1.0 / 60.0)});
}

fn runningStats() void {
    var s = math.stats.Stats{};
    s.addSlice(&.{ 2, 4, 4, 4, 5, 5, 7, 9 });
    print("mean={d} stddev={d} min={d} max={d}\n", .{ s.mean, s.stddev(), s.min, s.max });
}
