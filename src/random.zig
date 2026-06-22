//! Random distributions (GLM `gtc/random`). Idiomatic Zig: pass an explicit
//! `std.Random` source rather than relying on a hidden global.

const std = @import("std");
const vec = @import("vec.zig");
const Vec2 = vec.Vec2;
const Vec3 = vec.Vec3;

/// Uniform distribution in `[min, max]`.
pub fn uniform(rng: std.Random, min: anytype, max: @TypeOf(min)) @TypeOf(min) {
    const T = @TypeOf(min);
    return min + rng.float(T) * (max - min);
}

/// Normal (Gaussian) distribution, via Box–Muller.
pub fn normal(rng: std.Random, mean: anytype, deviation: @TypeOf(mean)) @TypeOf(mean) {
    const T = @TypeOf(mean);
    const un1 = @max(rng.float(T), std.math.floatMin(T));
    const un2 = rng.float(T);
    const r = @sqrt(-2.0 * @log(un1));
    return mean + deviation * r * @cos(2.0 * std.math.pi * un2);
}

/// Uniform point on a circle of `radius`.
pub fn onCircle(rng: std.Random, radius: f32) Vec2 {
    const a = rng.float(f32) * 2.0 * std.math.pi;
    return Vec2.init(@cos(a) * radius, @sin(a) * radius);
}
/// Uniform point inside a disk of `radius`.
pub fn inDisk(rng: std.Random, radius: f32) Vec2 {
    const r = radius * @sqrt(rng.float(f32));
    const a = rng.float(f32) * 2.0 * std.math.pi;
    return Vec2.init(@cos(a) * r, @sin(a) * r);
}
/// Uniform point on a sphere of `radius`.
pub fn onSphere(rng: std.Random, radius: f32) Vec3 {
    const z = rng.float(f32) * 2.0 - 1.0;
    const a = rng.float(f32) * 2.0 * std.math.pi;
    const r = @sqrt(1.0 - z * z);
    return Vec3.init(r * @cos(a) * radius, r * @sin(a) * radius, z * radius);
}
/// Uniform point inside a ball of `radius`.
pub fn inBall(rng: std.Random, radius: f32) Vec3 {
    const u = std.math.cbrt(rng.float(f32));
    return onSphere(rng, radius * u);
}

const testing = std.testing;
test "random ranges" {
    var prng = std.Random.DefaultPrng.init(12345);
    const rng = prng.random();
    var i: usize = 0;
    while (i < 200) : (i += 1) {
        const x = uniform(rng, @as(f32, -3), 5);
        try testing.expect(x >= -3 and x <= 5);
        try testing.expectApproxEqAbs(@as(f32, 2), onCircle(rng, 2).length(), 1e-4);
        try testing.expectApproxEqAbs(@as(f32, 3), onSphere(rng, 3).length(), 1e-4);
        try testing.expect(inBall(rng, 1).length() <= 1.0001);
    }
}
