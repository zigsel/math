//! Fast (approximate) math — `math.fast`. Trades accuracy for speed.

const std = @import("std");
const vec = @import("vec.zig");
const pi = std.math.pi;

// --- square root family (Quake-style) ---------------------------------------

pub fn inverseSqrt(x: f32) f32 {
    const half = x * 0.5;
    var i: i32 = @bitCast(x);
    i = 0x5f3759df - (i >> 1);
    var y: f32 = @bitCast(i);
    y = y * (1.5 - half * y * y);
    return y;
}
pub fn sqrt(x: f32) f32 {
    if (x == 0) return 0;
    return x * inverseSqrt(x);
}
pub fn length(v: anytype) f32 {
    return sqrt(@floatCast(v.lengthSq()));
}
pub fn distance(a: anytype, b: @TypeOf(a)) f32 {
    return length(b.sub(a));
}
pub fn normalize(v: anytype) @TypeOf(v) {
    return v.scale(inverseSqrt(@floatCast(v.lengthSq())));
}

// --- trigonometry (Bhaskara sin/cos; polynomial inverse) --------------------

pub fn wrapAngle(x: f32) f32 {
    const tau = 2.0 * pi;
    return x - tau * @round(x / tau);
}
pub fn sin(x: f32) f32 {
    const a = wrapAngle(x);
    const pi2 = pi * pi;
    if (a < 0) {
        const b = -a;
        return -(16.0 * b * (pi - b)) / (5.0 * pi2 - 4.0 * b * (pi - b));
    }
    return (16.0 * a * (pi - a)) / (5.0 * pi2 - 4.0 * a * (pi - a));
}
pub fn cos(x: f32) f32 {
    return sin(x + pi / 2.0);
}
pub fn tan(x: f32) f32 {
    return sin(x) / cos(x);
}
pub fn acos(x: f32) f32 {
    const negate: f32 = if (x < 0) 1 else 0;
    const ax = @abs(x);
    var ret: f32 = -0.0187293;
    ret = ret * ax + 0.0742610;
    ret = ret * ax - 0.2121144;
    ret = ret * ax + 1.5707288;
    ret *= @sqrt(1.0 - ax);
    ret -= 2 * negate * ret;
    return negate * pi + ret;
}
pub fn asin(x: f32) f32 {
    return pi / 2.0 - acos(x);
}
pub fn atan(x: f32) f32 {
    const pi_4 = pi / 4.0;
    if (@abs(x) <= 1) return pi_4 * x - x * (@abs(x) - 1) * (0.2447 + 0.0663 * @abs(x));
    const r = 1.0 / x;
    const a = pi_4 * r - r * (@abs(r) - 1) * (0.2447 + 0.0663 * @abs(r));
    const half: f32 = if (x > 0) pi / 2.0 else -pi / 2.0;
    return half - a;
}

// --- exponential (Mineiro fastapprox) ---------------------------------------

pub fn exp2(p: f32) f32 {
    const clipp: f32 = if (p < -126) -126 else p;
    const offset: f32 = if (clipp < 0) 1 else 0;
    const w: f32 = @trunc(clipp);
    const z = clipp - w + offset;
    const val = (1 << 23) * (clipp + 121.2740575 + 27.7280233 / (4.84252568 - z) - 1.49012907 * z);
    const i: u32 = @intFromFloat(val);
    return @bitCast(i);
}
pub fn log2(x: f32) f32 {
    const vxi: u32 = @bitCast(x);
    const mxi: u32 = (vxi & 0x007FFFFF) | 0x3f000000;
    const mxf: f32 = @bitCast(mxi);
    var y: f32 = @floatFromInt(vxi);
    y *= 1.1920928955078125e-7;
    return y - 124.22551499 - 1.498030302 * mxf - 1.72587999 / (0.3520887068 + mxf);
}
pub fn exp(x: f32) f32 {
    return exp2(1.442695040 * x);
}
pub fn log(x: f32) f32 {
    return 0.6931471805599453 * log2(x);
}
pub fn pow(x: f32, y: f32) f32 {
    return exp2(y * log2(x));
}

const testing = std.testing;
const Vec3 = vec.Vec3;
test "fast math" {
    try testing.expectApproxEqRel(@as(f32, 3), sqrt(9), 4e-3);
    try testing.expectApproxEqRel(@as(f32, 5), length(Vec3.init(3, 4, 0)), 4e-3);
    try testing.expectApproxEqRel(@as(f32, 8), pow(2, 3), 3e-2);
    var x: f32 = -3.0;
    while (x < 3.0) : (x += 0.2) {
        try testing.expectApproxEqAbs(@sin(x), sin(x), 2e-3);
        try testing.expectApproxEqAbs(std.math.atan(x), atan(x), 1e-2);
    }
}
