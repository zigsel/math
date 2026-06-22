//! Bit manipulation — `math.bits`. Integer bit ops, Morton codes, power-of-two,
//! and integer arithmetic helpers. Scalar unless noted component-wise.

const std = @import("std");
const sc = @import("meta.zig");

// --- population / scan / reverse --------------------------------------------

/// Count set bits (popcount). Scalar or component-wise over integer vectors.
pub fn count(v: anytype) @TypeOf(v) {
    return sc.map1(v, struct {
        fn f(x: anytype) @TypeOf(x) {
            return @popCount(x);
        }
    }.f);
}

/// Reverse bit order. Scalar or component-wise.
pub fn reverse(v: anytype) @TypeOf(v) {
    return sc.map1(v, struct {
        fn f(x: anytype) @TypeOf(x) {
            return @bitReverse(x);
        }
    }.f);
}

/// Number of leading zero bits.
pub fn leadingZeros(x: anytype) @TypeOf(x) {
    return @clz(x);
}

/// Index of the least-significant set bit, or -1 if none.
pub fn lsb(x: anytype) i32 {
    if (x == 0) return -1;
    return @intCast(@ctz(x));
}

/// Index of the most-significant set bit. GLSL semantics: for signed negatives
/// returns the MSB of `~x`; returns -1 for 0 (and for -1).
pub fn msb(x: anytype) i32 {
    const T = @TypeOf(x);
    const info = @typeInfo(T).int;
    const total = info.bits;
    if (info.signedness == .signed) {
        if (x == 0 or x == -1) return -1;
        const v = if (x < 0) ~x else x;
        return @as(i32, total - 1) - @as(i32, @clz(v));
    }
    if (x == 0) return -1;
    return @as(i32, total - 1) - @as(i32, @clz(x));
}

/// Position of the `n`-th set bit (1-indexed), or -1 if fewer than `n` are set.
pub fn nsb(x: anytype, n: i32) i32 {
    if (n <= 0) return -1;
    const total = @bitSizeOf(@TypeOf(x));
    var c: i32 = 0;
    var i: u32 = 0;
    while (i < total) : (i += 1) {
        if ((x >> @intCast(i)) & 1 != 0) {
            c += 1;
            if (c == n) return @intCast(i);
        }
    }
    return -1;
}

// --- extract / insert / mask / isolate --------------------------------------

/// Extract `n` bits starting at `offset`, right-aligned.
pub fn extract(value: anytype, offset: u6, n: u7) @TypeOf(value) {
    const T = @TypeOf(value);
    if (n == 0) return 0;
    const total = @typeInfo(T).int.bits;
    if (n >= total) return value >> @intCast(offset);
    const shifted = value >> @intCast(offset);
    const m = (@as(T, 1) << @intCast(n)) - 1;
    return shifted & m;
}

/// Insert the low `n` bits of `ins` into `base` starting at `offset`.
pub fn insert(base: anytype, ins: @TypeOf(base), offset: u6, n: u7) @TypeOf(base) {
    const T = @TypeOf(base);
    if (n == 0) return base;
    const total = @typeInfo(T).int.bits;
    const m: T = if (n >= total)
        ~@as(T, 0)
    else
        ((@as(T, 1) << @intCast(n)) - 1) << @intCast(offset);
    return (base & ~m) | ((ins << @intCast(offset)) & m);
}

/// A value with the low `n` bits set.
pub fn mask(n: anytype) @TypeOf(n) {
    const T = @TypeOf(n);
    if (n >= @bitSizeOf(T)) return ~@as(T, 0);
    return (@as(T, 1) << @intCast(n)) - 1;
}

/// Isolate the lowest set bit (`x & -x`).
pub fn lowest(x: anytype) @TypeOf(x) {
    return x & (~x +% 1);
}

/// Isolate the highest set bit.
pub fn highest(x: anytype) @TypeOf(x) {
    return floorPow2(x);
}

// --- rotate / fill ----------------------------------------------------------

pub fn rotateLeft(x: anytype, n: anytype) @TypeOf(x) {
    return std.math.rotl(@TypeOf(x), x, n);
}
pub fn rotateRight(x: anytype, n: anytype) @TypeOf(x) {
    return std.math.rotr(@TypeOf(x), x, n);
}

/// Set `n` bits starting at bit `first` to 1.
pub fn fillOne(value: anytype, first: usize, n: usize) @TypeOf(value) {
    const T = @TypeOf(value);
    var m: T = 0;
    var i: usize = 0;
    while (i < n) : (i += 1) m |= (@as(T, 1) << @intCast(first + i));
    return value | m;
}
/// Clear `n` bits starting at bit `first` to 0.
pub fn fillZero(value: anytype, first: usize, n: usize) @TypeOf(value) {
    const T = @TypeOf(value);
    var m: T = 0;
    var i: usize = 0;
    while (i < n) : (i += 1) m |= (@as(T, 1) << @intCast(first + i));
    return value & ~m;
}

// --- Morton codes (bit interleave) ------------------------------------------

/// Interleave two 16-bit values into a 32-bit Morton code.
pub fn interleave2(x: u16, y: u16) u32 {
    return spread2(x) | (spread2(y) << 1);
}
fn spread2(v: u16) u32 {
    var r: u32 = v;
    r = (r | (r << 8)) & 0x00FF00FF;
    r = (r | (r << 4)) & 0x0F0F0F0F;
    r = (r | (r << 2)) & 0x33333333;
    r = (r | (r << 1)) & 0x55555555;
    return r;
}

/// Interleave three 10-bit values into a 32-bit Morton code.
pub fn interleave3(x: u10, y: u10, z: u10) u32 {
    return spread3(x) | (spread3(y) << 1) | (spread3(z) << 2);
}
fn spread3(v: u10) u32 {
    var r: u32 = v;
    r = (r | (r << 16)) & 0x030000FF;
    r = (r | (r << 8)) & 0x0300F00F;
    r = (r | (r << 4)) & 0x030C30C3;
    r = (r | (r << 2)) & 0x09249249;
    return r;
}

/// Inverse of `interleave2`.
pub fn deinterleave2(v: u32) struct { x: u16, y: u16 } {
    return .{ .x = compact2(v), .y = compact2(v >> 1) };
}
fn compact2(v: u32) u16 {
    var r = v & 0x55555555;
    r = (r | (r >> 1)) & 0x33333333;
    r = (r | (r >> 2)) & 0x0F0F0F0F;
    r = (r | (r >> 4)) & 0x00FF00FF;
    r = (r | (r >> 8)) & 0x0000FFFF;
    return @truncate(r);
}

// --- extended integer arithmetic --------------------------------------------

/// `x + y` with the carry-out bit (generic over integer width / signedness).
pub fn addCarry(x: anytype, y: @TypeOf(x)) struct { result: @TypeOf(x), carry: u1 } {
    const r = @addWithOverflow(x, y);
    return .{ .result = r[0], .carry = r[1] };
}
/// `x - y` with the borrow-out bit.
pub fn subBorrow(x: anytype, y: @TypeOf(x)) struct { result: @TypeOf(x), borrow: u1 } {
    const r = @subWithOverflow(x, y);
    return .{ .result = r[0], .borrow = r[1] };
}
/// Full double-width product, split into low / high halves.
pub fn mulExtended(x: anytype, y: @TypeOf(x)) struct { lo: @TypeOf(x), hi: @TypeOf(x) } {
    const T = @TypeOf(x);
    const info = @typeInfo(T).int;
    const Wide = std.meta.Int(info.signedness, info.bits * 2);
    const p: Wide = @as(Wide, x) * @as(Wide, y);
    return .{ .lo = @truncate(p), .hi = @truncate(p >> info.bits) };
}

// --- power of two -----------------------------------------------------------

pub fn isPow2(x: anytype) bool {
    return x != 0 and (x & (x - 1)) == 0;
}
pub fn ceilPow2(x: anytype) @TypeOf(x) {
    const T = @TypeOf(x);
    if (x <= 1) return 1;
    return @as(T, 1) << @intCast(@bitSizeOf(T) - @clz(x - 1));
}
pub fn floorPow2(x: anytype) @TypeOf(x) {
    const T = @TypeOf(x);
    if (x == 0) return 0;
    return @as(T, 1) << @intCast(@bitSizeOf(T) - 1 - @clz(x));
}
pub fn roundPow2(x: anytype) @TypeOf(x) {
    const lo = floorPow2(x);
    const hi = ceilPow2(x);
    return if ((x - lo) < (hi - x)) lo else hi;
}

// --- integer arithmetic -----------------------------------------------------

/// Integer exponentiation by squaring.
pub fn ipow(base: anytype, exponent: u32) @TypeOf(base) {
    const T = @TypeOf(base);
    var result: T = 1;
    var b = base;
    var e = exponent;
    while (e != 0) : (e >>= 1) {
        if (e & 1 != 0) result *= b;
        if (e > 1) b *= b;
    }
    return result;
}

/// Integer square root (floor).
pub fn isqrt(x: anytype) @TypeOf(x) {
    const T = @TypeOf(x);
    if (x <= 0) return 0;
    var r: T = @intFromFloat(@sqrt(@as(f64, @floatFromInt(x))));
    while ((r + 1) * (r + 1) <= x) r += 1;
    while (r * r > x) r -= 1;
    return r;
}

pub fn factorial(n: anytype) @TypeOf(n) {
    var result: @TypeOf(n) = 1;
    var i: @TypeOf(n) = 2;
    while (i <= n) : (i += 1) result *= i;
    return result;
}

const testing = std.testing;
test "bits" {
    // scan / count / reverse
    try testing.expectEqual(@as(u32, 3), count(@as(u32, 0b1011)));
    try testing.expectEqual(@as(i32, 0), lsb(@as(u32, 0b1000_0001)));
    try testing.expectEqual(@as(i32, 7), msb(@as(u32, 0b1000_0001)));
    try testing.expectEqual(@as(i32, 5), msb(@as(i32, -64))); // ~(-64)=63 -> bit 5
    try testing.expectEqual(@as(i32, -1), msb(@as(i32, -1)));
    try testing.expectEqual(@as(i32, -1), lsb(@as(u32, 0)));
    try testing.expectEqual(@as(i32, 4), nsb(@as(u32, 0b1101_0100), 2));
    try testing.expectEqual(@as(u32, 0b011), extract(@as(u32, 0b1010_1100), 2, 3));
    // extended arithmetic
    const c = addCarry(@as(u32, 0xFFFF_FFFF), 2);
    try testing.expectEqual(@as(u32, 1), c.result);
    try testing.expectEqual(@as(u1, 1), c.carry);
    const m = mulExtended(@as(u32, 0x1000_0000), 0x10);
    try testing.expectEqual(@as(u32, 1), m.hi);
    const sm = mulExtended(@as(i32, -2), 3);
    try testing.expectEqual(@as(i32, -6), sm.lo);
    // morton
    try testing.expectEqual(@as(u32, 0b11), interleave2(1, 1));
    const code = interleave2(0xABCD, 0x1234);
    const d = deinterleave2(code);
    try testing.expectEqual(@as(u16, 0xABCD), d.x);
    try testing.expectEqual(@as(u16, 0x1234), d.y);
    // rotate / fill / mask / isolate
    try testing.expectEqual(@as(u8, 0b00011000), rotateLeft(@as(u8, 0b00000110), 2));
    try testing.expectEqual(@as(u8, 0b00011100), fillOne(@as(u8, 0), 2, 3));
    try testing.expectEqual(@as(u32, 0b111), mask(@as(u32, 3)));
    try testing.expectEqual(@as(u32, 0b100), lowest(@as(u32, 0b1100)));
    try testing.expectEqual(@as(u32, 8), highest(@as(u32, 0b1101)));
    // power of two / integer math
    try testing.expect(isPow2(@as(u32, 16)));
    try testing.expect(!isPow2(@as(u32, 17)));
    try testing.expectEqual(@as(u32, 16), ceilPow2(@as(u32, 9)));
    try testing.expectEqual(@as(u32, 8), floorPow2(@as(u32, 9)));
    try testing.expectEqual(@as(u32, 81), ipow(@as(u32, 3), 4));
    try testing.expectEqual(@as(u32, 5), isqrt(@as(u32, 30)));
    try testing.expectEqual(@as(u32, 120), factorial(@as(u32, 5)));
}
