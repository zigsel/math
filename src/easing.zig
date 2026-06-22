//! Easing & smoothing — `math.ease`. Robert Penner easing functions (input/
//! output in `[0, 1]`) plus frame-rate-independent spring/damping helpers.
//! Generic over float type.

const std = @import("std");
const sc = @import("meta.zig");
const num = @import("num.zig");
const pi = std.math.pi;

pub fn linear(a: anytype) @TypeOf(a) {
    return a;
}

pub fn quadraticIn(a: anytype) @TypeOf(a) {
    return a * a;
}
pub fn quadraticOut(a: anytype) @TypeOf(a) {
    return -(a * (a - 2));
}
pub fn quadraticInOut(a: anytype) @TypeOf(a) {
    return if (a < 0.5) 2 * a * a else (-2 * a * a) + (4 * a) - 1;
}

pub fn cubicIn(a: anytype) @TypeOf(a) {
    return a * a * a;
}
pub fn cubicOut(a: anytype) @TypeOf(a) {
    const f = a - 1;
    return f * f * f + 1;
}
pub fn cubicInOut(a: anytype) @TypeOf(a) {
    if (a < 0.5) return 4 * a * a * a;
    const f = (2 * a) - 2;
    return 0.5 * f * f * f + 1;
}

pub fn quarticIn(a: anytype) @TypeOf(a) {
    return a * a * a * a;
}
pub fn quarticOut(a: anytype) @TypeOf(a) {
    const f = a - 1;
    return f * f * f * (1 - a) + 1;
}
pub fn quarticInOut(a: anytype) @TypeOf(a) {
    if (a < 0.5) return 8 * a * a * a * a;
    const f = a - 1;
    return -8 * f * f * f * f + 1;
}

pub fn quinticIn(a: anytype) @TypeOf(a) {
    return a * a * a * a * a;
}
pub fn quinticOut(a: anytype) @TypeOf(a) {
    const f = a - 1;
    return f * f * f * f * f + 1;
}
pub fn quinticInOut(a: anytype) @TypeOf(a) {
    if (a < 0.5) return 16 * a * a * a * a * a;
    const f = (2 * a) - 2;
    return 0.5 * f * f * f * f * f + 1;
}

pub fn sineIn(a: anytype) @TypeOf(a) {
    return @sin((a - 1) * (pi / 2.0)) + 1;
}
pub fn sineOut(a: anytype) @TypeOf(a) {
    return @sin(a * (pi / 2.0));
}
pub fn sineInOut(a: anytype) @TypeOf(a) {
    return 0.5 * (1 - @cos(a * pi));
}

pub fn circularIn(a: anytype) @TypeOf(a) {
    return 1 - @sqrt(1 - a * a);
}
pub fn circularOut(a: anytype) @TypeOf(a) {
    return @sqrt((2 - a) * a);
}
pub fn circularInOut(a: anytype) @TypeOf(a) {
    if (a < 0.5) return 0.5 * (1 - @sqrt(1 - 4 * a * a));
    return 0.5 * (@sqrt(-((2 * a - 3) * (2 * a - 1))) + 1);
}

pub fn exponentialIn(a: anytype) @TypeOf(a) {
    const T = @TypeOf(a);
    return if (a == 0) 0 else std.math.pow(T, 2, 10 * (a - 1));
}
pub fn exponentialOut(a: anytype) @TypeOf(a) {
    const T = @TypeOf(a);
    return if (a == 1) 1 else 1 - std.math.pow(T, 2, -10 * a);
}
pub fn exponentialInOut(a: anytype) @TypeOf(a) {
    const T = @TypeOf(a);
    if (a == 0) return 0;
    if (a == 1) return 1;
    if (a < 0.5) return 0.5 * std.math.pow(T, 2, (20 * a) - 10);
    return -0.5 * std.math.pow(T, 2, (-20 * a) + 10) + 1;
}

pub fn elasticIn(a: anytype) @TypeOf(a) {
    const T = @TypeOf(a);
    return @sin(13 * (pi / 2.0) * a) * std.math.pow(T, 2, 10 * (a - 1));
}
pub fn elasticOut(a: anytype) @TypeOf(a) {
    const T = @TypeOf(a);
    return @sin(-13 * (pi / 2.0) * (a + 1)) * std.math.pow(T, 2, -10 * a) + 1;
}
pub fn elasticInOut(a: anytype) @TypeOf(a) {
    const T = @TypeOf(a);
    if (a < 0.5) return 0.5 * @sin(13 * (pi / 2.0) * (2 * a)) * std.math.pow(T, 2, 10 * ((2 * a) - 1));
    return 0.5 * (@sin(-13 * (pi / 2.0) * ((2 * a - 1) + 1)) * std.math.pow(T, 2, -10 * (2 * a - 1)) + 2);
}

pub fn backIn(a: anytype) @TypeOf(a) {
    return a * a * a - a * @sin(a * pi);
}
pub fn backOut(a: anytype) @TypeOf(a) {
    const f = 1 - a;
    return 1 - (f * f * f - f * @sin(f * pi));
}
pub fn backInOut(a: anytype) @TypeOf(a) {
    if (a < 0.5) {
        const f = 2 * a;
        return 0.5 * (f * f * f - f * @sin(f * pi));
    }
    const f = 1 - (2 * a - 1);
    return 0.5 * (1 - (f * f * f - f * @sin(f * pi))) + 0.5;
}

pub fn bounceOut(a: anytype) @TypeOf(a) {
    if (a < 4.0 / 11.0) return (121 * a * a) / 16.0;
    if (a < 8.0 / 11.0) return (363.0 / 40.0 * a * a) - (99.0 / 10.0 * a) + 17.0 / 5.0;
    if (a < 9.0 / 10.0) return (4356.0 / 361.0 * a * a) - (35442.0 / 1805.0 * a) + 16061.0 / 1805.0;
    return (54.0 / 5.0 * a * a) - (513.0 / 25.0 * a) + 268.0 / 25.0;
}
pub fn bounceIn(a: anytype) @TypeOf(a) {
    return 1 - bounceOut(1 - a);
}
pub fn bounceInOut(a: anytype) @TypeOf(a) {
    if (a < 0.5) return 0.5 * bounceIn(a * 2);
    return 0.5 * bounceOut(a * 2 - 1) + 0.5;
}

// --- spring / damping (frame-rate independent) ------------------------------

/// Exponential decay toward `target`. `decay` ≈ 1..25 — higher converges
/// faster. Equivalent to `mix(current, target, 1 - exp(-decay·dt))`. genType.
pub fn expDecay(
    current: anytype,
    target: @TypeOf(current),
    decay: sc.Element(@TypeOf(current)),
    dt: sc.Element(@TypeOf(current)),
) @TypeOf(current) {
    const t = 1.0 - @exp(-decay * dt);
    return num.mix(current, target, t);
}

/// Critically-damped spring smoothing (exact integration). `velocity` is mutable
/// state kept between frames; `smooth_time` is roughly the time to reach the
/// target (the natural frequency is `ω = 2/smooth_time`). genType: a float
/// scalar or a `Vec`. Returns the new position.
pub fn smoothDamp(
    current: anytype,
    target: @TypeOf(current),
    velocity: *@TypeOf(current),
    smooth_time: sc.Element(@TypeOf(current)),
    dt: sc.Element(@TypeOf(current)),
) @TypeOf(current) {
    const C = @TypeOf(current);
    const omega = 2.0 / @max(smooth_time, 1e-4);
    const e = @exp(-omega * dt); // exact decay term
    if (comptime sc.isVec(C)) {
        const delta = current.sub(target);
        const temp = velocity.add(delta.scale(omega)).scale(dt);
        velocity.* = velocity.sub(temp.scale(omega)).scale(e);
        return target.add(delta.add(temp).scale(e));
    } else {
        const delta = current - target;
        const temp = (velocity.* + omega * delta) * dt;
        velocity.* = (velocity.* - omega * temp) * e;
        return target + (delta + temp) * e;
    }
}

const testing = std.testing;
const Vec3 = @import("vec.zig").Vec3;
test "spring decays toward the target" {
    var p = Vec3.splat(0);
    var v = Vec3.splat(0);
    var i: usize = 0;
    while (i < 300) : (i += 1) p = smoothDamp(p, Vec3.splat(5), &v, 0.3, 1.0 / 60.0);
    try testing.expect(p.approxEql(Vec3.splat(5), 1e-2));
    var s: f32 = 0;
    i = 0;
    while (i < 200) : (i += 1) s = expDecay(s, 10.0, 8.0, 1.0 / 60.0);
    try testing.expectApproxEqAbs(@as(f32, 10), s, 1e-2);
}

test "easing endpoints" {
    inline for (.{
        quadraticIn, quadraticOut,   cubicInOut,
        quarticOut,  quinticInOut,    sineInOut,
        circularOut, exponentialInOut, bounceOut,
        elasticOut,
    }) |f| {
        try testing.expectApproxEqAbs(@as(f32, 0), f(@as(f32, 0)), 1e-4);
        try testing.expectApproxEqAbs(@as(f32, 1), f(@as(f32, 1)), 1e-4);
    }
}
