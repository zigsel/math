//! Scalar/vector math — the flat genType layer (works on scalars OR vectors),
//! constants, fast approximations, and the comptime `math.meta` helpers.
//! Run: `zig build example-scalar-math`

const std = @import("std");
const math = @import("math");
const print = std.debug.print;

pub fn main() void {
    mathGenType();
    mathTrig();
    mathConstants();
    mathFast();
    mathMeta();
}

fn mathGenType() void {
    // The same free function accepts a scalar or a vector, and broadcasts
    // scalar arguments component-wise.
    print("clamp scalar = {d}\n", .{math.clamp(@as(f32, 9), 0, 5)});
    print("clamp vector = {f}\n", .{math.clamp(math.Vec3.init(-1, 2, 9), 0, 5)});
    print("mix          = {f}\n", .{math.mix(math.Vec3.splat(0), math.Vec3.splat(10), 0.25)});
    print("smoothstep   = {d}\n", .{math.smoothstep(@as(f32, 0), 1, 0.5)});
    print("smootherstep = {d}\n", .{math.smootherstep(@as(f32, 0), 1, 0.5)});
    // Angle helpers wrap across ±π: halfway from 170° to -170° is 180°, not 0°.
    print("lerpAngle    = {d}\n", .{math.degrees(math.lerpAngle(math.radians(@as(f32, 170)), math.radians(@as(f32, -170)), 0.5))});
    print("abs/sign     = {f} {f}\n", .{ math.abs(math.Vec2.init(-3, 4)), math.sign(math.Vec2.init(-3, 4)) });
    print("fmod vs mod  = {d} {d}\n", .{ math.fmod(@as(f32, -1), 3), math.mod(@as(f32, -1), 3) });
}

fn mathTrig() void {
    print("radians(180) = {d}\n", .{math.radians(@as(f32, 180))});
    print("sin vector   = {f}\n", .{math.sin(math.Vec3.init(0, std.math.pi / 2.0, std.math.pi))});
    print("atan2        = {d}\n", .{math.atan2(@as(f32, 1), 1)});
    print("pow2/pow3    = {d} {d}\n", .{ math.pow2(@as(f32, 3)), math.pow3(@as(f32, 3)) });
}

fn mathConstants() void {
    // Convenience f32 constants, plus precision-generic functions.
    print("PI / TAU     = {d} {d}\n", .{ math.constants.PI, math.constants.TAU });
    print("golden ratio = {d}\n", .{math.constants.GOLDEN_RATIO});
    print("pi(f64)      = {d}\n", .{math.constants.pi(f64)});
}

fn mathFast() void {
    // Approximate variants trade accuracy for speed.
    print("fast.sqrt(9)     ~= {d}\n", .{math.fast.sqrt(9)});
    print("fast.inverseSqrt ~= {d}\n", .{math.fast.inverseSqrt(4)});
    print("fast.sin(1.0)    ~= {d} (std {d})\n", .{ math.fast.sin(1.0), @sin(@as(f32, 1.0)) });
}

fn mathMeta() void {
    // Comptime type predicates used to build genType functions.
    print("isVec(Vec3)   = {}\n", .{math.meta.isVec(math.Vec3)});
    print("isMat(Mat4)   = {}\n", .{math.meta.isMat(math.Mat4)});
    print("Element(Vec3) = {s}\n", .{@typeName(math.meta.Element(math.Vec3))});
}
