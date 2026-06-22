//! Everyday scalar/vector numeric math — re-exported flat on `math`. GLSL
//! common/exponential/trigonometric builtins, GTC reciprocal trig + epsilon
//! compare, and GTX numeric helpers (component-wise, powers, log base, wrap,
//! extended min/max, integer rounding, multiples, ULP).

const std = @import("std");
const sc = @import("meta.zig");
const vec = @import("vec.zig");
const Vec = vec.Vec;

// === common (GLSL) ===

/// Broadcast `y` (scalar or vector) to the SIMD shape of vector type `X`.
inline fn likeVec(comptime X: type, y: anytype) @Vector(X.dim, X.Element) {
    if (comptime sc.isVec(@TypeOf(y))) return y.simd();
    return @splat(y);
}

fn BoolLike(comptime T: type) type {
    return if (sc.isVec(T)) Vec(T.dim, bool) else bool;
}

pub fn abs(v: anytype) @TypeOf(v) {
    const T = @TypeOf(v);
    if (comptime sc.isVec(T)) return T.fromSimd(@abs(v.simd()));
    return @abs(v);
}
pub fn sign(v: anytype) @TypeOf(v) {
    return sc.map1(v, std.math.sign);
}
pub fn floor(v: anytype) @TypeOf(v) {
    const T = @TypeOf(v);
    if (comptime sc.isVec(T)) return T.fromSimd(@floor(v.simd()));
    return @floor(v);
}
pub fn ceil(v: anytype) @TypeOf(v) {
    const T = @TypeOf(v);
    if (comptime sc.isVec(T)) return T.fromSimd(@ceil(v.simd()));
    return @ceil(v);
}
pub fn trunc(v: anytype) @TypeOf(v) {
    const T = @TypeOf(v);
    if (comptime sc.isVec(T)) return T.fromSimd(@trunc(v.simd()));
    return @trunc(v);
}
pub fn round(v: anytype) @TypeOf(v) {
    const T = @TypeOf(v);
    if (comptime sc.isVec(T)) return T.fromSimd(@round(v.simd()));
    return @round(v);
}
pub fn fract(v: anytype) @TypeOf(v) {
    const T = @TypeOf(v);
    if (comptime sc.isVec(T)) return T.fromSimd(v.simd() - @floor(v.simd()));
    return v - @floor(v);
}
/// GLSL `mod`: `x - y * floor(x/y)` (note: differs from C `fmod`).
pub fn mod(x: anytype, y: anytype) @TypeOf(x) {
    const X = @TypeOf(x);
    if (comptime sc.isVec(X)) {
        const yv = likeVec(X, y);
        return X.fromSimd(x.simd() - yv * @floor(x.simd() / yv));
    }
    return x - y * @floor(x / y);
}
/// Truncated modulo (C `fmod`) — unlike GLSL `mod`, which floors. genType.
pub fn fmod(x: anytype, y: @TypeOf(x)) @TypeOf(x) {
    const T = @TypeOf(x);
    if (comptime sc.isVec(T)) return T.fromSimd(x.simd() - y.simd() * @trunc(x.simd() / y.simd()));
    return x - y * @trunc(x / y);
}
pub fn min(a: anytype, b: anytype) @TypeOf(a) {
    const A = @TypeOf(a);
    if (comptime sc.isVec(A)) return A.fromSimd(@min(a.simd(), likeVec(A, b)));
    return @min(a, b);
}
pub fn max(a: anytype, b: anytype) @TypeOf(a) {
    const A = @TypeOf(a);
    if (comptime sc.isVec(A)) return A.fromSimd(@max(a.simd(), likeVec(A, b)));
    return @max(a, b);
}
pub fn clamp(x: anytype, lo: anytype, hi: anytype) @TypeOf(x) {
    return max(min(x, hi), lo);
}
/// Clamp to `[0, 1]`.
pub fn saturate(x: anytype) @TypeOf(x) {
    return clamp(x, 0, 1);
}
/// Linear interpolation `x*(1-a) + y*a`. If `a` is a `bool`/boolean-vector,
/// performs a component-wise select (GLSL `mix(x, y, bvec)` — picks `y` where true).
pub fn mix(x: anytype, y: @TypeOf(x), a: anytype) @TypeOf(x) {
    const X = @TypeOf(x);
    const A = @TypeOf(a);
    if (comptime A == bool) return if (a) y else x;
    if (comptime sc.isVec(A) and A.Element == bool) {
        return X.fromSimd(@select(X.Element, a.simd(), y.simd(), x.simd()));
    }
    if (comptime sc.isVec(X)) {
        const av = likeVec(X, a);
        const one: @Vector(X.dim, X.Element) = @splat(1);
        return X.fromSimd(x.simd() * (one - av) + y.simd() * av);
    }
    return x * (1 - a) + y * a;
}
/// `0` where `x < edge`, else `1`.
pub fn step(edge: anytype, x: anytype) @TypeOf(x) {
    const X = @TypeOf(x);
    if (comptime sc.isVec(X)) {
        const ev = likeVec(X, edge);
        const one: @Vector(X.dim, X.Element) = @splat(1);
        const zero: @Vector(X.dim, X.Element) = @splat(0);
        return X.fromSimd(@select(X.Element, x.simd() < ev, zero, one));
    }
    return if (x < edge) 0 else 1;
}
/// Smooth Hermite interpolation between `e0` and `e1`.
pub fn smoothstep(e0: anytype, e1: anytype, x: anytype) @TypeOf(x) {
    const X = @TypeOf(x);
    if (comptime sc.isVec(X)) {
        const e0v = likeVec(X, e0);
        const e1v = likeVec(X, e1);
        const one: @Vector(X.dim, X.Element) = @splat(1);
        const zero: @Vector(X.dim, X.Element) = @splat(0);
        const three: @Vector(X.dim, X.Element) = @splat(3);
        const two: @Vector(X.dim, X.Element) = @splat(2);
        var t = (x.simd() - e0v) / (e1v - e0v);
        t = @min(@max(t, zero), one);
        return X.fromSimd(t * t * (three - two * t));
    }
    const t = std.math.clamp((x - e0) / (e1 - e0), 0, 1);
    return t * t * (3 - 2 * t);
}
/// Ken Perlin's 6th-order smootherstep (zero 1st *and* 2nd derivatives at the ends).
pub fn smootherstep(e0: anytype, e1: anytype, x: anytype) @TypeOf(x) {
    const X = @TypeOf(x);
    if (comptime sc.isVec(X)) {
        const e0v = likeVec(X, e0);
        const e1v = likeVec(X, e1);
        const one: @Vector(X.dim, X.Element) = @splat(1);
        const zero: @Vector(X.dim, X.Element) = @splat(0);
        var t = (x.simd() - e0v) / (e1v - e0v);
        t = @min(@max(t, zero), one);
        const c6: @Vector(X.dim, X.Element) = @splat(6);
        const c15: @Vector(X.dim, X.Element) = @splat(15);
        const c10: @Vector(X.dim, X.Element) = @splat(10);
        return X.fromSimd(t * t * t * (t * (t * c6 - c15) + c10));
    }
    const t = std.math.clamp((x - e0) / (e1 - e0), 0, 1);
    return t * t * t * (t * (t * 6 - 15) + 10);
}

// === angles (radians) ===

/// Wrap an angle into `(-π, π]`.
pub fn wrapAngle(a: anytype) @TypeOf(a) {
    const T = @TypeOf(a);
    const pi: T = std.math.pi;
    return @mod(a + pi, 2 * pi) - pi;
}
/// Normalize an angle into `[0, 2π)`.
pub fn normalizeAngle(a: anytype) @TypeOf(a) {
    const T = @TypeOf(a);
    return @mod(a, 2 * @as(T, std.math.pi));
}
/// Shortest signed difference `b - a`, in `(-π, π]`.
pub fn deltaAngle(a: anytype, b: @TypeOf(a)) @TypeOf(a) {
    return wrapAngle(b - a);
}
/// Interpolate from angle `a` toward `b` along the shortest arc.
pub fn lerpAngle(a: anytype, b: @TypeOf(a), t: @TypeOf(a)) @TypeOf(a) {
    return a + deltaAngle(a, b) * t;
}
/// Fused multiply-add `a*b + c`.
pub fn fma(a: anytype, b: @TypeOf(a), c: @TypeOf(a)) @TypeOf(a) {
    const T = @TypeOf(a);
    if (comptime sc.isVec(T)) return T.fromSimd(@mulAdd(@Vector(T.dim, T.Element), a.simd(), b.simd(), c.simd()));
    return @mulAdd(T, a, b, c);
}
pub fn isnan(v: anytype) BoolLike(@TypeOf(v)) {
    const T = @TypeOf(v);
    if (comptime sc.isVec(T)) {
        var r: @Vector(T.dim, bool) = undefined;
        const s = v.simd();
        inline for (0..T.dim) |i| r[i] = std.math.isNan(s[i]);
        return Vec(T.dim, bool).fromSimd(r);
    }
    return std.math.isNan(v);
}
pub fn isinf(v: anytype) BoolLike(@TypeOf(v)) {
    const T = @TypeOf(v);
    if (comptime sc.isVec(T)) {
        var r: @Vector(T.dim, bool) = undefined;
        const s = v.simd();
        inline for (0..T.dim) |i| r[i] = std.math.isInf(s[i]);
        return Vec(T.dim, bool).fromSimd(r);
    }
    return std.math.isInf(v);
}
pub fn isfinite(v: anytype) BoolLike(@TypeOf(v)) {
    const T = @TypeOf(v);
    if (comptime sc.isVec(T)) {
        var r: @Vector(T.dim, bool) = undefined;
        const s = v.simd();
        inline for (0..T.dim) |i| r[i] = std.math.isFinite(s[i]);
        return Vec(T.dim, bool).fromSimd(r);
    }
    return std.math.isFinite(v);
}
/// True for a non-zero subnormal (denormal) float.
pub fn isdenormal(x: anytype) bool {
    const T = @TypeOf(x);
    return x != 0 and @abs(x) < std.math.floatMin(T);
}

/// Round to nearest even on ties (GLSL `roundEven` / banker's rounding).
pub fn roundEven(v: anytype) @TypeOf(v) {
    return sc.map1(v, struct {
        fn f(x: anytype) @TypeOf(x) {
            const fl = @floor(x);
            const d = x - fl;
            if (d < 0.5) return fl;
            if (d > 0.5) return fl + 1;
            return if (@mod(fl, 2) == 0) fl else fl + 1;
        }
    }.f);
}

fn IntLike(comptime T: type, comptime Scalar: type) type {
    return if (sc.isVec(T)) Vec(T.dim, Scalar) else Scalar;
}

/// Split into integer (`whole`) and fractional (`fract`) parts (GLSL `modf`).
pub fn modf(v: anytype) struct { fract: @TypeOf(v), whole: @TypeOf(v) } {
    const T = @TypeOf(v);
    if (comptime sc.isVec(T)) {
        const w = T.fromSimd(@trunc(v.simd()));
        return .{ .whole = w, .fract = v.sub(w) };
    }
    const w = @trunc(v);
    return .{ .whole = w, .fract = v - w };
}

/// Decompose into significand ∈ [0.5, 1) and exponent (GLSL `frexp`).
pub fn frexp(v: anytype) struct { significand: @TypeOf(v), exponent: IntLike(@TypeOf(v), i32) } {
    const T = @TypeOf(v);
    if (comptime sc.isVec(T)) {
        var sig = v.simd();
        var exps: @Vector(T.dim, i32) = undefined;
        inline for (0..T.dim) |i| {
            const r = std.math.frexp(sig[i]);
            sig[i] = r.significand;
            exps[i] = @intCast(r.exponent);
        }
        return .{ .significand = T.fromSimd(sig), .exponent = Vec(T.dim, i32).fromSimd(exps) };
    }
    const r = std.math.frexp(v);
    return .{ .significand = r.significand, .exponent = @intCast(r.exponent) };
}

/// `significand * 2^exponent` (GLSL `ldexp`).
pub fn ldexp(v: anytype, exponent: anytype) @TypeOf(v) {
    const T = @TypeOf(v);
    if (comptime sc.isVec(T)) {
        var r = v.simd();
        const e = exponent.simd();
        inline for (0..T.dim) |i| r[i] = std.math.ldexp(r[i], @intCast(e[i]));
        return T.fromSimd(r);
    }
    return std.math.ldexp(v, @intCast(exponent));
}

fn bitCastEach(v: anytype, comptime Scalar: type) IntLike(@TypeOf(v), Scalar) {
    const T = @TypeOf(v);
    if (comptime sc.isVec(T)) {
        var r: @Vector(T.dim, Scalar) = undefined;
        const s = v.simd();
        inline for (0..T.dim) |i| r[i] = @bitCast(s[i]);
        return Vec(T.dim, Scalar).fromSimd(r);
    }
    return @bitCast(v);
}

pub fn floatBitsToInt(v: anytype) IntLike(@TypeOf(v), i32) {
    return bitCastEach(v, i32);
}
pub fn floatBitsToUint(v: anytype) IntLike(@TypeOf(v), u32) {
    return bitCastEach(v, u32);
}
pub fn intBitsToFloat(v: anytype) IntLike(@TypeOf(v), f32) {
    return bitCastEach(v, f32);
}
pub fn uintBitsToFloat(v: anytype) IntLike(@TypeOf(v), f32) {
    return bitCastEach(v, f32);
}

// === trigonometric (GLSL) ===

pub fn radians(deg: anytype) @TypeOf(deg) {
    const T = @TypeOf(deg);
    const k = std.math.pi / 180.0;
    if (comptime sc.isVec(T)) return deg.scale(k);
    return deg * k;
}
pub fn degrees(rad: anytype) @TypeOf(rad) {
    const T = @TypeOf(rad);
    const k = 180.0 / std.math.pi;
    if (comptime sc.isVec(T)) return rad.scale(k);
    return rad * k;
}

pub fn sin(v: anytype) @TypeOf(v) {
    const T = @TypeOf(v);
    if (comptime sc.isVec(T)) return T.fromSimd(@sin(v.simd()));
    return @sin(v);
}
pub fn cos(v: anytype) @TypeOf(v) {
    const T = @TypeOf(v);
    if (comptime sc.isVec(T)) return T.fromSimd(@cos(v.simd()));
    return @cos(v);
}
pub fn tan(v: anytype) @TypeOf(v) {
    const T = @TypeOf(v);
    if (comptime sc.isVec(T)) return T.fromSimd(@sin(v.simd()) / @cos(v.simd()));
    return @sin(v) / @cos(v);
}
pub fn asin(v: anytype) @TypeOf(v) {
    return sc.map1(v, std.math.asin);
}
pub fn acos(v: anytype) @TypeOf(v) {
    return sc.map1(v, std.math.acos);
}
pub fn atan(v: anytype) @TypeOf(v) {
    return sc.map1(v, std.math.atan);
}
/// Two-argument arctangent (GLSL `atan(y, x)`).
pub fn atan2(y: anytype, x: @TypeOf(y)) @TypeOf(y) {
    return sc.map2(y, x, struct {
        fn f(a: anytype, b: @TypeOf(a)) @TypeOf(a) {
            return std.math.atan2(a, b);
        }
    }.f);
}
pub fn sinh(v: anytype) @TypeOf(v) {
    return sc.map1(v, std.math.sinh);
}
pub fn cosh(v: anytype) @TypeOf(v) {
    return sc.map1(v, std.math.cosh);
}
pub fn tanh(v: anytype) @TypeOf(v) {
    return sc.map1(v, std.math.tanh);
}
pub fn asinh(v: anytype) @TypeOf(v) {
    return sc.map1(v, std.math.asinh);
}
pub fn acosh(v: anytype) @TypeOf(v) {
    return sc.map1(v, std.math.acosh);
}
pub fn atanh(v: anytype) @TypeOf(v) {
    return sc.map1(v, std.math.atanh);
}

// === exponential (GLSL) ===

pub fn pow(x: anytype, y: @TypeOf(x)) @TypeOf(x) {
    const T = @TypeOf(x);
    if (comptime sc.isVec(T)) {
        var xs = x.simd();
        const ys = y.simd();
        inline for (0..T.dim) |i| xs[i] = std.math.pow(T.Element, xs[i], ys[i]);
        return T.fromSimd(xs);
    }
    return std.math.pow(T, x, y);
}
pub fn exp(v: anytype) @TypeOf(v) {
    const T = @TypeOf(v);
    if (comptime sc.isVec(T)) return T.fromSimd(@exp(v.simd()));
    return @exp(v);
}
pub fn log(v: anytype) @TypeOf(v) {
    const T = @TypeOf(v);
    if (comptime sc.isVec(T)) return T.fromSimd(@log(v.simd()));
    return @log(v);
}
pub fn exp2(v: anytype) @TypeOf(v) {
    const T = @TypeOf(v);
    if (comptime sc.isVec(T)) return T.fromSimd(@exp2(v.simd()));
    return @exp2(v);
}
pub fn log2(v: anytype) @TypeOf(v) {
    const T = @TypeOf(v);
    if (comptime sc.isVec(T)) return T.fromSimd(@log2(v.simd()));
    return @log2(v);
}
pub fn sqrt(v: anytype) @TypeOf(v) {
    const T = @TypeOf(v);
    if (comptime sc.isVec(T)) return T.fromSimd(@sqrt(v.simd()));
    return @sqrt(v);
}
/// `1 / sqrt(x)` (GLSL `inversesqrt`).
pub fn inverseSqrt(v: anytype) @TypeOf(v) {
    const T = @TypeOf(v);
    if (comptime sc.isVec(T)) {
        const one: @Vector(T.dim, T.Element) = @splat(1);
        return T.fromSimd(one / @sqrt(v.simd()));
    }
    return 1.0 / @sqrt(v);
}

// === epsilon compare (GTC) ===

pub fn epsilonEqual(a: anytype, b: @TypeOf(a), eps: sc.Element(@TypeOf(a))) BoolLike(@TypeOf(a)) {
    const T = @TypeOf(a);
    if (comptime sc.isVec(T)) {
        const e: @Vector(T.dim, T.Element) = @splat(eps);
        return Vec(T.dim, bool).fromSimd(@abs(a.simd() - b.simd()) <= e);
    }
    return @abs(a - b) <= eps;
}

pub fn epsilonNotEqual(a: anytype, b: @TypeOf(a), eps: sc.Element(@TypeOf(a))) BoolLike(@TypeOf(a)) {
    const T = @TypeOf(a);
    if (comptime sc.isVec(T)) {
        const e: @Vector(T.dim, T.Element) = @splat(eps);
        return Vec(T.dim, bool).fromSimd(@abs(a.simd() - b.simd()) > e);
    }
    return @abs(a - b) > eps;
}

// === reciprocal trig (GTC) ===

pub fn sec(v: anytype) @TypeOf(v) {
    return sc.map1(v, struct {
        fn f(x: anytype) @TypeOf(x) {
            return 1.0 / @cos(x);
        }
    }.f);
}
pub fn csc(v: anytype) @TypeOf(v) {
    return sc.map1(v, struct {
        fn f(x: anytype) @TypeOf(x) {
            return 1.0 / @sin(x);
        }
    }.f);
}
pub fn cot(v: anytype) @TypeOf(v) {
    return sc.map1(v, struct {
        fn f(x: anytype) @TypeOf(x) {
            return @cos(x) / @sin(x);
        }
    }.f);
}
pub fn asec(v: anytype) @TypeOf(v) {
    return sc.map1(v, struct {
        fn f(x: anytype) @TypeOf(x) {
            return std.math.acos(1.0 / x);
        }
    }.f);
}
pub fn acsc(v: anytype) @TypeOf(v) {
    return sc.map1(v, struct {
        fn f(x: anytype) @TypeOf(x) {
            return std.math.asin(1.0 / x);
        }
    }.f);
}
pub fn acot(v: anytype) @TypeOf(v) {
    return sc.map1(v, struct {
        fn f(x: anytype) @TypeOf(x) {
            return std.math.atan(1.0 / x);
        }
    }.f);
}
pub fn sech(v: anytype) @TypeOf(v) {
    return sc.map1(v, struct {
        fn f(x: anytype) @TypeOf(x) {
            return 1.0 / std.math.cosh(x);
        }
    }.f);
}
pub fn csch(v: anytype) @TypeOf(v) {
    return sc.map1(v, struct {
        fn f(x: anytype) @TypeOf(x) {
            return 1.0 / std.math.sinh(x);
        }
    }.f);
}
pub fn coth(v: anytype) @TypeOf(v) {
    return sc.map1(v, struct {
        fn f(x: anytype) @TypeOf(x) {
            return std.math.cosh(x) / std.math.sinh(x);
        }
    }.f);
}

pub fn asech(v: anytype) @TypeOf(v) {
    return sc.map1(v, struct {
        fn f(x: anytype) @TypeOf(x) {
            return std.math.acosh(1.0 / x);
        }
    }.f);
}
pub fn acsch(v: anytype) @TypeOf(v) {
    return sc.map1(v, struct {
        fn f(x: anytype) @TypeOf(x) {
            return std.math.asinh(1.0 / x);
        }
    }.f);
}
pub fn acoth(v: anytype) @TypeOf(v) {
    return sc.map1(v, struct {
        fn f(x: anytype) @TypeOf(x) {
            return std.math.atanh(1.0 / x);
        }
    }.f);
}

// === component-wise (GTX) ===

/// Normalize an integer vector to floats: unsigned → [0,1], signed → [-1,1].
pub fn compNormalize(comptime Float: type, v: anytype) Vec(@TypeOf(v).dim, Float) {
    const T = @TypeOf(v);
    const E = T.Element;
    var r: @Vector(T.dim, Float) = undefined;
    const s = v.simd();
    const maxv: Float = @floatFromInt(std.math.maxInt(E));
    inline for (0..T.dim) |i| {
        const f: Float = @floatFromInt(s[i]);
        r[i] = if (comptime @typeInfo(E).int.signedness == .signed) @max(f / maxv, -1.0) else f / maxv;
    }
    return Vec(T.dim, Float).fromSimd(r);
}
/// Inverse of `compNormalize`: scale a normalized float vector to integers.
pub fn compScale(comptime Int: type, v: anytype) Vec(@TypeOf(v).dim, Int) {
    const T = @TypeOf(v);
    var r: @Vector(T.dim, Int) = undefined;
    const s = v.simd();
    const maxv: T.Element = @floatFromInt(std.math.maxInt(Int));
    const lo: T.Element = if (comptime @typeInfo(Int).int.signedness == .signed) -1.0 else 0.0;
    inline for (0..T.dim) |i| r[i] = @intFromFloat(@round(std.math.clamp(s[i], lo, 1.0) * maxv));
    return Vec(T.dim, Int).fromSimd(r);
}

/// NaN-aware component minimum.
pub fn fcompMin(v: anytype) sc.Element(@TypeOf(v)) {
    const T = @TypeOf(v);
    const s = v.simd();
    var m: T.Element = s[0];
    inline for (1..T.dim) |i| {
        m = if (std.math.isNan(m)) s[i] else if (std.math.isNan(s[i])) m else @min(m, s[i]);
    }
    return m;
}
/// NaN-aware component maximum.
pub fn fcompMax(v: anytype) sc.Element(@TypeOf(v)) {
    const T = @TypeOf(v);
    const s = v.simd();
    var m: T.Element = s[0];
    inline for (1..T.dim) |i| {
        m = if (std.math.isNan(m)) s[i] else if (std.math.isNan(s[i])) m else @max(m, s[i]);
    }
    return m;
}

// === optimum pow (GTX) ===

pub fn pow2(x: anytype) @TypeOf(x) {
    const T = @TypeOf(x);
    if (comptime sc.isVec(T)) return x.mul(x);
    return x * x;
}
pub fn pow3(x: anytype) @TypeOf(x) {
    const T = @TypeOf(x);
    if (comptime sc.isVec(T)) return x.mul(x).mul(x);
    return x * x * x;
}
pub fn pow4(x: anytype) @TypeOf(x) {
    return pow2(pow2(x));
}

// === log base (GTX) ===

pub fn logBase(x: anytype, base: sc.Element(@TypeOf(x))) @TypeOf(x) {
    const T = @TypeOf(x);
    const inv = 1.0 / @log(base);
    if (comptime sc.isVec(T)) return log(x).scale(inv);
    return @log(x) * inv;
}

// === wrap (GTX) ===

/// Wrap into `[0, 1)` (GL_REPEAT).
pub fn repeat(t: anytype) @TypeOf(t) {
    return fract(t);
}
/// Clamp into `[0, 1]` (GL_CLAMP_TO_EDGE).
pub fn wrapClamp(t: anytype) @TypeOf(t) {
    return clamp(t, 0, 1);
}
/// Triangle wave into `[0, 1]` (GL_MIRRORED_REPEAT).
pub fn mirrorRepeat(t: anytype) @TypeOf(t) {
    const T = @TypeOf(t);
    if (comptime sc.isVec(T)) {
        const two: @Vector(T.dim, T.Element) = @splat(2);
        const one: @Vector(T.dim, T.Element) = @splat(1);
        const m = t.simd() - two * @floor(t.simd() / two);
        return T.fromSimd(one - @abs(one - m));
    }
    const m = t - 2 * @floor(t / 2);
    return 1 - @abs(1 - m);
}

/// `fract(abs(t))` — mirror toward 0 then wrap into [0, 1).
pub fn mirrorClamp(t: anytype) @TypeOf(t) {
    const T = @TypeOf(t);
    if (comptime sc.isVec(T)) {
        const a = @abs(t.simd());
        return T.fromSimd(a - @floor(a));
    }
    const a = @abs(t);
    return a - @floor(a);
}

// === gauss (GTX) ===

/// Gaussian: `exp(-(x-mean)² / (2·sd²))`.
pub fn gauss(x: anytype, mean: @TypeOf(x), sd: @TypeOf(x)) @TypeOf(x) {
    const d = x - mean;
    return @exp(-(d * d) / (2 * sd * sd));
}

// === extended min/max (GTX) ===

fn fminScalar(a: anytype, b: @TypeOf(a)) @TypeOf(a) {
    if (std.math.isNan(a)) return b;
    if (std.math.isNan(b)) return a;
    return @min(a, b);
}
fn fmaxScalar(a: anytype, b: @TypeOf(a)) @TypeOf(a) {
    if (std.math.isNan(a)) return b;
    if (std.math.isNan(b)) return a;
    return @max(a, b);
}

/// NaN-aware minimum (returns the non-NaN argument when one is NaN).
pub fn fmin(a: anytype, b: @TypeOf(a)) @TypeOf(a) {
    const T = @TypeOf(a);
    if (comptime sc.isVec(T)) {
        var r = a.simd();
        const bs = b.simd();
        inline for (0..T.dim) |i| r[i] = fminScalar(r[i], bs[i]);
        return T.fromSimd(r);
    }
    return fminScalar(a, b);
}
pub fn fmax(a: anytype, b: @TypeOf(a)) @TypeOf(a) {
    const T = @TypeOf(a);
    if (comptime sc.isVec(T)) {
        var r = a.simd();
        const bs = b.simd();
        inline for (0..T.dim) |i| r[i] = fmaxScalar(r[i], bs[i]);
        return T.fromSimd(r);
    }
    return fmaxScalar(a, b);
}
pub fn fclamp(x: anytype, lo: anytype, hi: anytype) @TypeOf(x) {
    const X = @TypeOf(x);
    if (comptime sc.isVec(X)) {
        const lov = if (comptime sc.isVec(@TypeOf(lo))) lo else X.splat(lo);
        const hiv = if (comptime sc.isVec(@TypeOf(hi))) hi else X.splat(hi);
        return fmin(fmax(x, lov), hiv);
    }
    return fminScalar(fmaxScalar(x, lo), hi);
}

pub fn min3(a: anytype, b: @TypeOf(a), c: @TypeOf(a)) @TypeOf(a) {
    return min(min(a, b), c);
}
pub fn min4(a: anytype, b: @TypeOf(a), c: @TypeOf(a), d: @TypeOf(a)) @TypeOf(a) {
    return min(min(a, b), min(c, d));
}
pub fn max3(a: anytype, b: @TypeOf(a), c: @TypeOf(a)) @TypeOf(a) {
    return max(max(a, b), c);
}
pub fn max4(a: anytype, b: @TypeOf(a), c: @TypeOf(a), d: @TypeOf(a)) @TypeOf(a) {
    return max(max(a, b), max(c, d));
}

// === integer rounding / multiples / ULP ===

// --- round float to integer type --------------------------------------------

/// Round to the nearest `i32` (or `i32` vector, component-wise).
pub fn iround(v: anytype) IntLike(@TypeOf(v), i32) {
    const T = @TypeOf(v);
    if (comptime sc.isVec(T)) {
        var r: @Vector(T.dim, i32) = undefined;
        const s = v.simd();
        inline for (0..T.dim) |i| r[i] = @intFromFloat(@round(s[i]));
        return Vec(T.dim, i32).fromSimd(r);
    }
    return @intFromFloat(@round(v));
}
/// Round to the nearest `u32` (or `u32` vector, component-wise).
pub fn uround(v: anytype) IntLike(@TypeOf(v), u32) {
    const T = @TypeOf(v);
    if (comptime sc.isVec(T)) {
        var r: @Vector(T.dim, u32) = undefined;
        const s = v.simd();
        inline for (0..T.dim) |i| r[i] = @intFromFloat(@round(s[i]));
        return Vec(T.dim, u32).fromSimd(r);
    }
    return @intFromFloat(@round(v));
}

// --- multiples --------------------------------------------------------------

/// True if `value` is an exact multiple of `multiple`.
pub fn isMultiple(value: anytype, multiple: @TypeOf(value)) bool {
    return @mod(value, multiple) == 0;
}
/// Smallest multiple of `multiple` that is >= `value`.
pub fn ceilMultiple(value: anytype, multiple: @TypeOf(value)) @TypeOf(value) {
    if (multiple == 0) return value;
    const rem = @mod(value, multiple);
    return if (rem == 0) value else value + (multiple - rem);
}
/// Largest multiple of `multiple` that is <= `value`.
pub fn floorMultiple(value: anytype, multiple: @TypeOf(value)) @TypeOf(value) {
    if (multiple == 0) return value;
    return value - @mod(value, multiple);
}
/// Nearest multiple of `multiple` to `value`.
pub fn roundMultiple(value: anytype, multiple: @TypeOf(value)) @TypeOf(value) {
    const lo = floorMultiple(value, multiple);
    const hi = ceilMultiple(value, multiple);
    return if ((value - lo) < (hi - value)) lo else hi;
}

// --- ULP (unit in last place) -----------------------------------------------

fn UintOf(comptime T: type) type {
    return std.meta.Int(.unsigned, @bitSizeOf(T));
}

/// Next representable float toward +infinity.
pub fn nextFloat(x: anytype) @TypeOf(x) {
    const T = @TypeOf(x);
    const U = UintOf(T);
    if (std.math.isNan(x) or x == std.math.inf(T)) return x;
    if (x == 0) return std.math.floatTrueMin(T);
    var bits: U = @bitCast(x);
    if (x > 0) bits += 1 else bits -= 1;
    return @bitCast(bits);
}

/// Next representable float toward -infinity.
pub fn prevFloat(x: anytype) @TypeOf(x) {
    const T = @TypeOf(x);
    const U = UintOf(T);
    if (std.math.isNan(x) or x == -std.math.inf(T)) return x;
    if (x == 0) return -std.math.floatTrueMin(T);
    var bits: U = @bitCast(x);
    if (x > 0) bits -= 1 else bits += 1;
    return @bitCast(bits);
}

/// Count of representable floats between `a` and `b`.
pub fn floatDistance(a: anytype, b: @TypeOf(a)) i64 {
    return monotonicKey(b) - monotonicKey(a);
}
fn monotonicKey(x: anytype) i64 {
    const T = @TypeOf(x);
    const U = UintOf(T);
    const S = std.meta.Int(.signed, @bitSizeOf(T));
    const bits: U = @bitCast(x);
    const sign_mask: U = @as(U, 1) << (@bitSizeOf(T) - 1);
    // map so that integer ordering matches float ordering
    const key: U = if (bits & sign_mask != 0) ~bits else (bits | sign_mask);
    return @as(i64, @as(S, @bitCast(key)));
}

const testing = std.testing;
const Vec2 = vec.Vec2;
const Vec3 = vec.Vec3;

test "roundEven / modf / frexp / ldexp / bitcasts" {
    try testing.expectEqual(@as(f32, 2), roundEven(@as(f32, 2.5)));
    try testing.expectEqual(@as(f32, 4), roundEven(@as(f32, 3.5)));
    try testing.expectEqual(@as(f32, -2), roundEven(@as(f32, -2.5)));
    const m = modf(@as(f32, 3.75));
    try testing.expectApproxEqAbs(@as(f32, 3), m.whole, 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.75), m.fract, 1e-6);
    const fr = frexp(@as(f32, 8));
    try testing.expectApproxEqAbs(@as(f32, 0.5), fr.significand, 1e-6);
    try testing.expectEqual(@as(i32, 4), fr.exponent);
    try testing.expectApproxEqAbs(@as(f32, 8), ldexp(@as(f32, 0.5), @as(i32, 4)), 1e-6);
    try testing.expectEqual(@as(f32, 1.0), intBitsToFloat(floatBitsToInt(@as(f32, 1.0))));
    try testing.expect(floatBitsToUint(@as(f32, 1.0)) == 0x3f800000);
}

test "mix bool select" {
    try testing.expect(mix(Vec3.init(1, 2, 3), Vec3.init(4, 5, 6), vec.BVec3.init(true, false, true)).eql(Vec3.init(4, 2, 6)));
    try testing.expectEqual(@as(f32, 9), mix(@as(f32, 1), @as(f32, 9), true));
}

test "common scalar" {
    try testing.expectEqual(@as(f32, 2), abs(@as(f32, -2)));
    try testing.expectEqual(@as(f32, 3), floor(@as(f32, 3.7)));
    try testing.expectApproxEqAbs(@as(f32, 0.5), fract(@as(f32, 3.5)), 1e-6);
    try testing.expectEqual(@as(f32, 5), clamp(@as(f32, 9), @as(f32, 0), @as(f32, 5)));
    try testing.expectApproxEqAbs(@as(f32, 1.5), mix(@as(f32, 1), @as(f32, 2), @as(f32, 0.5)), 1e-6);
}

test "common vector + scalar broadcast" {
    const v = Vec3.init(-1.5, 2.5, -3.0);
    try testing.expect(abs(v).eql(Vec3.init(1.5, 2.5, 3.0)));
    try testing.expect(clamp(v, 0, 1).eql(Vec3.init(0, 1, 0)));
    try testing.expect(min(v, 0).eql(Vec3.init(-1.5, 0, -3.0)));
    const m = mix(Vec3.splat(0), Vec3.splat(10), 0.5);
    try testing.expect(m.eql(Vec3.splat(5)));
    try testing.expect(step(Vec3.splat(0), v).eql(Vec3.init(0, 1, 0)));
}

test "trig scalar" {
    try testing.expectApproxEqAbs(@as(f32, std.math.pi), radians(@as(f32, 180)), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0), sin(@as(f32, 0)), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 1), cos(@as(f32, 0)), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, std.math.pi / 4.0), atan2(@as(f32, 1), @as(f32, 1)), 1e-6);
}

test "trig vector" {
    const v = Vec3.splat(0);
    try testing.expect(sin(v).approxEql(Vec3.splat(0), 1e-6));
    try testing.expect(cos(v).approxEql(Vec3.splat(1), 1e-6));
    try testing.expect(degrees(radians(Vec3.init(30, 60, 90))).approxEql(Vec3.init(30, 60, 90), 1e-4));
}

test "exponential scalar" {
    try testing.expectApproxEqAbs(@as(f32, 8), pow(@as(f32, 2), @as(f32, 3)), 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 1), exp(@as(f32, 0)), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 3), sqrt(@as(f32, 9)), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.5), inverseSqrt(@as(f32, 4)), 1e-6);
}

test "exponential vector" {
    const v = Vec3.init(1, 4, 9);
    try testing.expect(sqrt(v).approxEql(Vec3.init(1, 2, 3), 1e-5));
    try testing.expect(pow(Vec3.splat(2), Vec3.init(1, 2, 3)).approxEql(Vec3.init(2, 4, 8), 1e-5));
}

test "epsilon compare" {
    try testing.expect(epsilonEqual(@as(f32, 1.0), @as(f32, 1.0001), 1e-3));
    try testing.expect(!epsilonEqual(@as(f32, 1.0), @as(f32, 1.1), 1e-3));
    const a = Vec3.init(1, 2, 3);
    const b = Vec3.init(1.0001, 2.5, 3.0);
    try testing.expect(epsilonEqual(a, b, 1e-3).eql(vec.BVec3.init(true, false, true)));
}

test "reciprocal" {
    try testing.expectApproxEqAbs(@as(f32, 1.0), sec(@as(f32, 0)), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 2.0), sec(@as(f32, std.math.pi / 3.0)), 1e-5);
}

test "component reductions are Vec methods" {
    const v = Vec3.init(1, 2, 4);
    try testing.expectEqual(@as(f32, 7), v.sum());
    try testing.expectEqual(@as(f32, 8), v.product());
    try testing.expectEqual(@as(f32, 1), v.minComponent());
    try testing.expectEqual(@as(f32, 4), v.maxComponent());
}

test "optimum_pow" {
    try testing.expectEqual(@as(f32, 9), pow2(@as(f32, 3)));
    try testing.expectEqual(@as(f32, 27), pow3(@as(f32, 3)));
    try testing.expect(pow2(Vec3.init(2, 3, 4)).eql(Vec3.init(4, 9, 16)));
}

test "log_base" {
    try testing.expectApproxEqAbs(@as(f32, 3), logBase(@as(f32, 8), 2), 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 2), logBase(@as(f32, 100), 10), 1e-5);
}

test "wrap" {
    try testing.expectApproxEqAbs(@as(f32, 0.25), repeat(@as(f32, 3.25)), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.75), mirrorRepeat(@as(f32, 1.25)), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 1.0), wrapClamp(@as(f32, 5.0)), 1e-6);
}

test "functions" {
    try testing.expectApproxEqAbs(@as(f32, 1.0), gauss(@as(f32, 0), 0, 1), 1e-6);
}

test "extended_min_max" {
    const nan = std.math.nan(f32);
    try testing.expectEqual(@as(f32, 3), fmin(@as(f32, 3), nan));
    try testing.expectEqual(@as(f32, 5), fmax(nan, @as(f32, 5)));
    try testing.expectEqual(@as(f32, 2), fclamp(@as(f32, 9), @as(f32, 0), @as(f32, 2)));
    try testing.expectEqual(@as(i32, 1), min3(@as(i32, 3), 1, 2));
    try testing.expectEqual(@as(i32, 9), max4(@as(i32, 3), 1, 9, 2));
}

test "num: integer rounding / multiples / ulp" {
    try testing.expectEqual(@as(i32, 3), iround(@as(f32, 2.7)));
    try testing.expect(isMultiple(@as(u32, 15), 5));
    try testing.expectEqual(@as(u32, 15), ceilMultiple(@as(u32, 11), 5));
    try testing.expectEqual(@as(u32, 10), floorMultiple(@as(u32, 11), 5));
    const x: f32 = 1.0;
    try testing.expect(nextFloat(x) > x);
    try testing.expect(prevFloat(x) < x);
    try testing.expectEqual(@as(i64, 1), floatDistance(x, nextFloat(x)));
}
