//! Comptime type predicates and generic-type (`genType`) dispatch helpers —
//! `math.meta`. Shared across the whole library.
//!
//! GLSL functions are polymorphic over "genType": they accept a scalar *or* a
//! vector and act component-wise. Zig has no overloading, so we recover that
//! polymorphism at comptime: every public math function takes `anytype` and
//! routes through the helpers here based on whether the argument is a scalar,
//! a builtin `@Vector`, or one of our `Vec(N, T)` structs.

const std = @import("std");

// ---------------------------------------------------------------------------
// Type predicates (all comptime)
// ---------------------------------------------------------------------------

/// True for our `Vec(N, T)` struct types (they carry an `is_math_vector` marker).
pub fn isVec(comptime T: type) bool {
    return @typeInfo(T) == .@"struct" and @hasDecl(T, "is_math_vector");
}

/// True for our `Mat(C, R, T)` struct types.
pub fn isMat(comptime T: type) bool {
    return @typeInfo(T) == .@"struct" and @hasDecl(T, "is_math_matrix");
}

/// True for our `Quat(T)` struct type.
pub fn isQuat(comptime T: type) bool {
    return @typeInfo(T) == .@"struct" and @hasDecl(T, "is_math_quaternion");
}

/// True for a builtin SIMD `@Vector(N, T)`.
pub fn isSimd(comptime T: type) bool {
    return @typeInfo(T) == .vector;
}

pub fn isFloat(comptime T: type) bool {
    return switch (@typeInfo(T)) {
        .float, .comptime_float => true,
        else => false,
    };
}

pub fn isInt(comptime T: type) bool {
    return switch (@typeInfo(T)) {
        .int, .comptime_int => true,
        else => false,
    };
}

/// The underlying scalar element type of any genType value type.
pub fn Element(comptime T: type) type {
    if (isVec(T) or isQuat(T)) return T.Element;
    if (isMat(T)) return T.Element;
    if (isSimd(T)) return @typeInfo(T).vector.child;
    return T; // already a scalar
}

/// The component count of a vector-like type (1 for scalars).
pub fn len(comptime T: type) comptime_int {
    if (isVec(T)) return T.dim;
    if (isSimd(T)) return @typeInfo(T).vector.len;
    return 1;
}

// ---------------------------------------------------------------------------
// Compile-time guards (produce nice error messages)
// ---------------------------------------------------------------------------

pub fn requireFloat(comptime T: type) void {
    if (!isFloat(Element(T)))
        @compileError("this operation requires a floating-point element type, got " ++ @typeName(T));
}

// ---------------------------------------------------------------------------
// Component-wise mapping over scalar / @Vector / Vec(N,T)
// ---------------------------------------------------------------------------

/// Apply scalar function `f` component-wise. Works on a scalar, a builtin
/// `@Vector`, or a `Vec(N, T)`; the return type matches the input.
///
/// Used for transcendental functions that have no vector-aware `@`-builtin
/// (e.g. `asin`, `atan`). For functions that *do* have a SIMD builtin
/// (`@sqrt`, `@sin`, `@floor`, ...) prefer the builtin directly for true SIMD.
pub inline fn map1(v: anytype, comptime f: anytype) @TypeOf(v) {
    const T = @TypeOf(v);
    if (comptime isVec(T)) {
        var s = v.simd();
        inline for (0..T.dim) |i| s[i] = f(s[i]);
        return T.fromSimd(s);
    } else if (comptime isSimd(T)) {
        var s = v;
        inline for (0..@typeInfo(T).vector.len) |i| s[i] = f(s[i]);
        return s;
    } else {
        return f(v);
    }
}

/// Apply binary scalar function `f` component-wise across two matching values.
pub inline fn map2(a: anytype, b: @TypeOf(a), comptime f: anytype) @TypeOf(a) {
    const T = @TypeOf(a);
    if (comptime isVec(T)) {
        var sa = a.simd();
        const sb = b.simd();
        inline for (0..T.dim) |i| sa[i] = f(sa[i], sb[i]);
        return T.fromSimd(sa);
    } else if (comptime isSimd(T)) {
        var sa = a;
        inline for (0..@typeInfo(T).vector.len) |i| sa[i] = f(sa[i], b[i]);
        return sa;
    } else {
        return f(a, b);
    }
}

/// Lift a value to a SIMD vector of the same shape, run `op` (which receives
/// and returns a `@Vector`), and lower back. For functions naturally written
/// in terms of vector-aware builtins.
pub inline fn lift1(v: anytype, comptime op: anytype) @TypeOf(v) {
    const T = @TypeOf(v);
    if (comptime isVec(T)) return T.fromSimd(op(v.simd()));
    return op(v);
}

test "predicates" {
    try std.testing.expect(isFloat(f32));
    try std.testing.expect(isInt(i32));
    try std.testing.expect(isSimd(@Vector(4, f32)));
    try std.testing.expect(!isVec(f32));
    try std.testing.expectEqual(@as(type, f32), Element(@Vector(3, f32)));
    try std.testing.expectEqual(@as(comptime_int, 3), len(@Vector(3, f32)));
}
