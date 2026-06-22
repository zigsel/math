//! Mathematical constants — the GLM `gtc/constants` set, plus epsilon helpers.
//!
//! Each constant is a generic function of the float type so you get the right
//! precision for `f16`/`f32`/`f64`/`f128`. Convenience `f32` aliases (uppercase)
//! are provided for the common case.

const std = @import("std");

pub fn epsilon(comptime T: type) T {
    return std.math.floatEps(T);
}
pub fn zero(comptime T: type) T {
    return 0;
}
pub fn one(comptime T: type) T {
    return 1;
}
pub fn pi(comptime T: type) T {
    return std.math.pi;
}
pub fn twoPi(comptime T: type) T {
    return 2.0 * std.math.pi;
}
pub fn halfPi(comptime T: type) T {
    return 0.5 * std.math.pi;
}
pub fn threeOverTwoPi(comptime T: type) T {
    return 3.0 * std.math.pi / 2.0;
}
pub fn quarterPi(comptime T: type) T {
    return 0.25 * std.math.pi;
}
pub fn oneOverPi(comptime T: type) T {
    return 1.0 / std.math.pi;
}
pub fn oneOverTwoPi(comptime T: type) T {
    return 1.0 / (2.0 * std.math.pi);
}
pub fn twoOverPi(comptime T: type) T {
    return 2.0 / std.math.pi;
}
pub fn fourOverPi(comptime T: type) T {
    return 4.0 / std.math.pi;
}
pub fn twoOverRootPi(comptime T: type) T {
    return 2.0 / @sqrt(@as(T, std.math.pi));
}
pub fn oneOverRootTwo(comptime T: type) T {
    return 1.0 / std.math.sqrt2;
}
pub fn rootPi(comptime T: type) T {
    return @sqrt(@as(T, std.math.pi));
}
pub fn rootHalfPi(comptime T: type) T {
    return @sqrt(@as(T, std.math.pi / 2.0));
}
pub fn rootTwoPi(comptime T: type) T {
    return @sqrt(@as(T, 2.0 * std.math.pi));
}
pub fn e(comptime T: type) T {
    return std.math.e;
}
pub fn euler(comptime T: type) T {
    return 0.577215664901532860606; // Euler–Mascheroni constant γ
}
pub fn rootTwo(comptime T: type) T {
    return std.math.sqrt2;
}
pub fn rootThree(comptime T: type) T {
    return @sqrt(@as(T, 3.0));
}
pub fn rootFive(comptime T: type) T {
    return @sqrt(@as(T, 5.0));
}
pub fn lnTwo(comptime T: type) T {
    return std.math.ln2;
}
pub fn lnTen(comptime T: type) T {
    return std.math.ln10;
}
pub fn lnLnTwo(comptime T: type) T {
    return @log(@as(T, std.math.ln2));
}
pub fn third(comptime T: type) T {
    return 1.0 / 3.0;
}
pub fn twoThirds(comptime T: type) T {
    return 2.0 / 3.0;
}
pub fn goldenRatio(comptime T: type) T {
    return (1.0 + @sqrt(@as(T, 5.0))) / 2.0;
}
pub fn rootLnFour(comptime T: type) T {
    return @sqrt(@log(@as(T, 4.0)));
}
pub fn cosOneOverTwo(comptime T: type) T {
    return @cos(@as(T, 0.5));
}

// f32 convenience aliases (the common case).
pub const PI: f32 = pi(f32);
pub const TAU: f32 = twoPi(f32);
pub const HALF_PI: f32 = halfPi(f32);
pub const QUARTER_PI: f32 = quarterPi(f32);
pub const E: f32 = e(f32);
pub const SQRT2: f32 = rootTwo(f32);
pub const SQRT3: f32 = rootThree(f32);
pub const GOLDEN_RATIO: f32 = goldenRatio(f32);
pub const EPSILON: f32 = epsilon(f32);

test "constants" {
    try std.testing.expectApproxEqAbs(@as(f32, 3.14159265), PI, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f64, 6.283185307), twoPi(f64), 1e-9);
    try std.testing.expectApproxEqAbs(@as(f32, 1.618033988), GOLDEN_RATIO, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.41421356), SQRT2, 1e-6);
}
