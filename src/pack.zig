//! GLSL packing functions — pack normalized floats into 32-bit words and back.

const std = @import("std");
const sc = @import("meta.zig");
const vec = @import("vec.zig");
const Vec2 = vec.Vec2;
const Vec3 = vec.Vec3;
const Vec4 = vec.Vec4;
const UVec2 = vec.UVec2;

pub fn packUnorm2x16(v: Vec2) u32 {
    const x: u32 = @intFromFloat(@round(std.math.clamp(v.x, 0, 1) * 65535.0));
    const y: u32 = @intFromFloat(@round(std.math.clamp(v.y, 0, 1) * 65535.0));
    return x | (y << 16);
}
pub fn unpackUnorm2x16(p: u32) Vec2 {
    const x: f32 = @floatFromInt(p & 0xffff);
    const y: f32 = @floatFromInt(p >> 16);
    return Vec2.init(x / 65535.0, y / 65535.0);
}

pub fn packSnorm2x16(v: Vec2) u32 {
    const x: i16 = @intFromFloat(@round(std.math.clamp(v.x, -1, 1) * 32767.0));
    const y: i16 = @intFromFloat(@round(std.math.clamp(v.y, -1, 1) * 32767.0));
    return @as(u32, @as(u16, @bitCast(x))) | (@as(u32, @as(u16, @bitCast(y))) << 16);
}
pub fn unpackSnorm2x16(p: u32) Vec2 {
    const x: f32 = @floatFromInt(@as(i16, @bitCast(@as(u16, @truncate(p)))));
    const y: f32 = @floatFromInt(@as(i16, @bitCast(@as(u16, @truncate(p >> 16)))));
    return Vec2.init(std.math.clamp(x / 32767.0, -1, 1), std.math.clamp(y / 32767.0, -1, 1));
}

pub fn packUnorm4x8(v: Vec4) u32 {
    var out: u32 = 0;
    inline for (.{ v.x, v.y, v.z, v.w }, 0..) |comp, i| {
        const b: u32 = @intFromFloat(@round(std.math.clamp(comp, 0, 1) * 255.0));
        out |= b << @intCast(i * 8);
    }
    return out;
}
pub fn unpackUnorm4x8(p: u32) Vec4 {
    return Vec4.init(
        @as(f32, @floatFromInt((p >> 0) & 0xff)) / 255.0,
        @as(f32, @floatFromInt((p >> 8) & 0xff)) / 255.0,
        @as(f32, @floatFromInt((p >> 16) & 0xff)) / 255.0,
        @as(f32, @floatFromInt((p >> 24) & 0xff)) / 255.0,
    );
}

pub fn packSnorm4x8(v: Vec4) u32 {
    var out: u32 = 0;
    inline for (.{ v.x, v.y, v.z, v.w }, 0..) |comp, i| {
        const s: i8 = @intFromFloat(@round(std.math.clamp(comp, -1, 1) * 127.0));
        out |= @as(u32, @as(u8, @bitCast(s))) << @intCast(i * 8);
    }
    return out;
}
pub fn unpackSnorm4x8(p: u32) Vec4 {
    var out: [4]f32 = undefined;
    inline for (0..4) |i| {
        const s: i8 = @bitCast(@as(u8, @truncate(p >> @intCast(i * 8))));
        out[i] = std.math.clamp(@as(f32, @floatFromInt(s)) / 127.0, -1, 1);
    }
    return Vec4.fromArray(out);
}

/// Pack two f32 into two f16 halves.
pub fn packHalf2x16(v: Vec2) u32 {
    const x: u16 = @bitCast(@as(f16, @floatCast(v.x)));
    const y: u16 = @bitCast(@as(f16, @floatCast(v.y)));
    return @as(u32, x) | (@as(u32, y) << 16);
}
pub fn unpackHalf2x16(p: u32) Vec2 {
    const x: f16 = @bitCast(@as(u16, @truncate(p)));
    const y: f16 = @bitCast(@as(u16, @truncate(p >> 16)));
    return Vec2.init(@floatCast(x), @floatCast(y));
}

pub fn packDouble2x32(v: UVec2) f64 {
    return @bitCast(@as(u64, v.x) | (@as(u64, v.y) << 32));
}
pub fn unpackDouble2x32(d: f64) UVec2 {
    const bits: u64 = @bitCast(d);
    return UVec2.init(@truncate(bits), @truncate(bits >> 32));
}

pub fn packUnorm1x8(v: f32) u8 {
    return @intFromFloat(@round(std.math.clamp(v, 0, 1) * 255.0));
}
pub fn unpackUnorm1x8(p: u8) f32 {
    return @as(f32, @floatFromInt(p)) / 255.0;
}
pub fn packSnorm1x8(v: f32) i8 {
    return @intFromFloat(@round(std.math.clamp(v, -1, 1) * 127.0));
}
pub fn unpackSnorm1x8(p: i8) f32 {
    return std.math.clamp(@as(f32, @floatFromInt(p)) / 127.0, -1, 1);
}
pub fn packUnorm1x16(v: f32) u16 {
    return @intFromFloat(@round(std.math.clamp(v, 0, 1) * 65535.0));
}
pub fn unpackUnorm1x16(p: u16) f32 {
    return @as(f32, @floatFromInt(p)) / 65535.0;
}
pub fn packSnorm1x16(v: f32) i16 {
    return @intFromFloat(@round(std.math.clamp(v, -1, 1) * 32767.0));
}
pub fn unpackSnorm1x16(p: i16) f32 {
    return std.math.clamp(@as(f32, @floatFromInt(p)) / 32767.0, -1, 1);
}
pub fn packHalf1x16(v: f32) u16 {
    return @bitCast(@as(f16, @floatCast(v)));
}
pub fn unpackHalf1x16(p: u16) f32 {
    return @floatCast(@as(f16, @bitCast(p)));
}

// --- 10/10/10/2 -------------------------------------------------------------

pub fn packUnorm3x10_1x2(v: Vec4) u32 {
    const x: u32 = @intFromFloat(@round(std.math.clamp(v.x, 0, 1) * 1023.0));
    const y: u32 = @intFromFloat(@round(std.math.clamp(v.y, 0, 1) * 1023.0));
    const z: u32 = @intFromFloat(@round(std.math.clamp(v.z, 0, 1) * 1023.0));
    const w: u32 = @intFromFloat(@round(std.math.clamp(v.w, 0, 1) * 3.0));
    return x | (y << 10) | (z << 20) | (w << 30);
}
pub fn unpackUnorm3x10_1x2(p: u32) Vec4 {
    return Vec4.init(
        @as(f32, @floatFromInt(p & 0x3FF)) / 1023.0,
        @as(f32, @floatFromInt((p >> 10) & 0x3FF)) / 1023.0,
        @as(f32, @floatFromInt((p >> 20) & 0x3FF)) / 1023.0,
        @as(f32, @floatFromInt((p >> 30) & 0x3)) / 3.0,
    );
}

// --- R11F_G11F_B10F ---------------------------------------------------------

fn f32toUF(f: f32, comptime mant_bits: u5) u32 {
    if (f <= 0 or std.math.isNan(f)) return 0;
    const b: u32 = @bitCast(f);
    const e: i32 = @as(i32, @intCast((b >> 23) & 0xFF)) - 127 + 15;
    const m = b & 0x7FFFFF;
    const max_e: i32 = 31;
    if (e <= 0) return 0;
    if (e >= max_e) return (@as(u32, max_e) << mant_bits) | ((@as(u32, 1) << mant_bits) - 1);
    return (@as(u32, @intCast(e)) << mant_bits) | (m >> @intCast(23 - @as(u32, mant_bits)));
}
fn ufToF32(v: u32, comptime mant_bits: u5) f32 {
    const e = (v >> mant_bits) & 0x1F;
    const m = v & ((@as(u32, 1) << mant_bits) - 1);
    if (e == 0 and m == 0) return 0;
    const b: u32 = ((@as(u32, @intCast(@as(i32, @intCast(e)) - 15 + 127))) << 23) |
        (m << @intCast(23 - @as(u32, mant_bits)));
    return @bitCast(b);
}
pub fn packF2x11_1x10(v: Vec3) u32 {
    return f32toUF(v.x, 6) | (f32toUF(v.y, 6) << 11) | (f32toUF(v.z, 5) << 22);
}
pub fn unpackF2x11_1x10(p: u32) Vec3 {
    return Vec3.init(ufToF32(p & 0x7FF, 6), ufToF32((p >> 11) & 0x7FF, 6), ufToF32((p >> 22) & 0x3FF, 5));
}

// --- RGB9_E5 (shared exponent) ---------------------------------------------

pub fn packF3x9_E1x5(v: Vec3) u32 {
    const max_val: f32 = 65408.0;
    const r = std.math.clamp(v.x, 0, max_val);
    const g = std.math.clamp(v.y, 0, max_val);
    const b = std.math.clamp(v.z, 0, max_val);
    const maxc = @max(r, @max(g, b));
    var exp: i32 = -16;
    if (maxc > 1e-30) exp = @max(-16, @as(i32, @intFromFloat(@floor(@log2(maxc)))) + 1);
    const denom = std.math.pow(f32, 2.0, @floatFromInt(exp - 9));
    const rm: u32 = @intFromFloat(@round(r / denom));
    const gm: u32 = @intFromFloat(@round(g / denom));
    const bm: u32 = @intFromFloat(@round(b / denom));
    const e: u32 = @intCast(exp + 15);
    return (rm & 0x1FF) | ((gm & 0x1FF) << 9) | ((bm & 0x1FF) << 18) | (e << 27);
}
pub fn unpackF3x9_E1x5(p: u32) Vec3 {
    const e: i32 = @intCast(p >> 27);
    const scale = std.math.pow(f32, 2.0, @floatFromInt(e - 15 - 9));
    return Vec3.init(
        @as(f32, @floatFromInt(p & 0x1FF)) * scale,
        @as(f32, @floatFromInt((p >> 9) & 0x1FF)) * scale,
        @as(f32, @floatFromInt((p >> 18) & 0x1FF)) * scale,
    );
}

// --- RGBM (HDR in 4x unorm) -------------------------------------------------

pub fn packRGBM(rgb: Vec3) Vec4 {
    const c = rgb.scale(1.0 / 6.0);
    var m = std.math.clamp(@max(@max(c.x, c.y), @max(c.z, 1e-6)), 0, 1);
    m = @ceil(m * 255.0) / 255.0;
    return Vec4.init(c.x / m, c.y / m, c.z / m, m);
}
pub fn unpackRGBM(rgbm: Vec4) Vec3 {
    return Vec3.init(rgbm.x, rgbm.y, rgbm.z).scale(6.0 * rgbm.w);
}

// --- generic normalized pack/unpack (comptime int/float type) ---------------

pub fn packUnorm(comptime U: type, v: anytype) vec.Vec(@TypeOf(v).dim, U) {
    const T = @TypeOf(v);
    const maxv: T.Element = @floatFromInt(std.math.maxInt(U));
    var r: @Vector(T.dim, U) = undefined;
    const s = v.simd();
    inline for (0..T.dim) |i| r[i] = @intFromFloat(@round(std.math.clamp(s[i], 0, 1) * maxv));
    return vec.Vec(T.dim, U).fromSimd(r);
}
pub fn unpackUnorm(comptime F: type, v: anytype) vec.Vec(@TypeOf(v).dim, F) {
    const T = @TypeOf(v);
    const maxv: F = @floatFromInt(std.math.maxInt(T.Element));
    var r: @Vector(T.dim, F) = undefined;
    const s = v.simd();
    inline for (0..T.dim) |i| r[i] = @as(F, @floatFromInt(s[i])) / maxv;
    return vec.Vec(T.dim, F).fromSimd(r);
}
pub fn packSnorm(comptime I: type, v: anytype) vec.Vec(@TypeOf(v).dim, I) {
    const T = @TypeOf(v);
    const maxv: T.Element = @floatFromInt(std.math.maxInt(I));
    var r: @Vector(T.dim, I) = undefined;
    const s = v.simd();
    inline for (0..T.dim) |i| r[i] = @intFromFloat(@round(std.math.clamp(s[i], -1, 1) * maxv));
    return vec.Vec(T.dim, I).fromSimd(r);
}
pub fn unpackSnorm(comptime F: type, v: anytype) vec.Vec(@TypeOf(v).dim, F) {
    const T = @TypeOf(v);
    const maxv: F = @floatFromInt(std.math.maxInt(T.Element));
    var r: @Vector(T.dim, F) = undefined;
    const s = v.simd();
    inline for (0..T.dim) |i| r[i] = std.math.clamp(@as(F, @floatFromInt(s[i])) / maxv, -1, 1);
    return vec.Vec(T.dim, F).fromSimd(r);
}

pub fn packHalf(v: anytype) vec.Vec(@TypeOf(v).dim, u16) {
    const T = @TypeOf(v);
    var r: @Vector(T.dim, u16) = undefined;
    const s = v.simd();
    inline for (0..T.dim) |i| r[i] = @bitCast(@as(f16, @floatCast(s[i])));
    return vec.Vec(T.dim, u16).fromSimd(r);
}
pub fn unpackHalf(v: anytype) vec.Vec(@TypeOf(v).dim, f32) {
    const T = @TypeOf(v);
    var r: @Vector(T.dim, f32) = undefined;
    const s = v.simd();
    inline for (0..T.dim) |i| r[i] = @floatCast(@as(f16, @bitCast(s[i])));
    return vec.Vec(T.dim, f32).fromSimd(r);
}

// --- more normalized words --------------------------------------------------

pub fn packUnorm2x8(v: Vec2) u16 {
    const x: u16 = @intFromFloat(@round(std.math.clamp(v.x, 0, 1) * 255.0));
    const y: u16 = @intFromFloat(@round(std.math.clamp(v.y, 0, 1) * 255.0));
    return x | (y << 8);
}
pub fn unpackUnorm2x8(p: u16) Vec2 {
    return Vec2.init(@as(f32, @floatFromInt(p & 0xff)) / 255.0, @as(f32, @floatFromInt(p >> 8)) / 255.0);
}
pub fn packSnorm2x8(v: Vec2) u16 {
    const x: i8 = @intFromFloat(@round(std.math.clamp(v.x, -1, 1) * 127.0));
    const y: i8 = @intFromFloat(@round(std.math.clamp(v.y, -1, 1) * 127.0));
    return @as(u16, @as(u8, @bitCast(x))) | (@as(u16, @as(u8, @bitCast(y))) << 8);
}
pub fn unpackSnorm2x8(p: u16) Vec2 {
    const x: i8 = @bitCast(@as(u8, @truncate(p)));
    const y: i8 = @bitCast(@as(u8, @truncate(p >> 8)));
    return Vec2.init(std.math.clamp(@as(f32, @floatFromInt(x)) / 127.0, -1, 1), std.math.clamp(@as(f32, @floatFromInt(y)) / 127.0, -1, 1));
}
pub fn packUnorm4x16(v: Vec4) u64 {
    var out: u64 = 0;
    inline for (.{ v.x, v.y, v.z, v.w }, 0..) |c, i| {
        const u: u64 = @intFromFloat(@round(std.math.clamp(c, 0, 1) * 65535.0));
        out |= u << @intCast(i * 16);
    }
    return out;
}
pub fn unpackUnorm4x16(p: u64) Vec4 {
    var out: [4]f32 = undefined;
    inline for (0..4) |i| out[i] = @as(f32, @floatFromInt((p >> @intCast(i * 16)) & 0xffff)) / 65535.0;
    return Vec4.fromArray(out);
}
pub fn packSnorm4x16(v: Vec4) u64 {
    var out: u64 = 0;
    inline for (.{ v.x, v.y, v.z, v.w }, 0..) |c, i| {
        const s: i16 = @intFromFloat(@round(std.math.clamp(c, -1, 1) * 32767.0));
        out |= @as(u64, @as(u16, @bitCast(s))) << @intCast(i * 16);
    }
    return out;
}
pub fn unpackSnorm4x16(p: u64) Vec4 {
    var out: [4]f32 = undefined;
    inline for (0..4) |i| {
        const s: i16 = @bitCast(@as(u16, @truncate(p >> @intCast(i * 16))));
        out[i] = std.math.clamp(@as(f32, @floatFromInt(s)) / 32767.0, -1, 1);
    }
    return Vec4.fromArray(out);
}
pub fn packHalf4x16(v: Vec4) u64 {
    var out: u64 = 0;
    inline for (.{ v.x, v.y, v.z, v.w }, 0..) |c, i| {
        out |= @as(u64, @as(u16, @bitCast(@as(f16, @floatCast(c))))) << @intCast(i * 16);
    }
    return out;
}
pub fn unpackHalf4x16(p: u64) Vec4 {
    var out: [4]f32 = undefined;
    inline for (0..4) |i| out[i] = @floatCast(@as(f16, @bitCast(@as(u16, @truncate(p >> @intCast(i * 16))))));
    return Vec4.fromArray(out);
}

// --- sub-byte layouts (bitfields; component 0 in low bits) ------------------

pub fn packUnorm2x4(v: Vec2) u8 {
    const x: u8 = @intFromFloat(@round(std.math.clamp(v.x, 0, 1) * 15.0));
    const y: u8 = @intFromFloat(@round(std.math.clamp(v.y, 0, 1) * 15.0));
    return x | (y << 4);
}
pub fn unpackUnorm2x4(p: u8) Vec2 {
    return Vec2.init(@as(f32, @floatFromInt(p & 0xf)) / 15.0, @as(f32, @floatFromInt(p >> 4)) / 15.0);
}
pub fn packUnorm4x4(v: Vec4) u16 {
    var out: u16 = 0;
    inline for (.{ v.x, v.y, v.z, v.w }, 0..) |c, i| {
        const u: u16 = @intFromFloat(@round(std.math.clamp(c, 0, 1) * 15.0));
        out |= u << @intCast(i * 4);
    }
    return out;
}
pub fn unpackUnorm4x4(p: u16) Vec4 {
    var out: [4]f32 = undefined;
    inline for (0..4) |i| out[i] = @as(f32, @floatFromInt((p >> @intCast(i * 4)) & 0xf)) / 15.0;
    return Vec4.fromArray(out);
}
pub fn packUnorm1x5_1x6_1x5(v: Vec3) u16 {
    const x: u16 = @intFromFloat(@round(std.math.clamp(v.x, 0, 1) * 31.0));
    const y: u16 = @intFromFloat(@round(std.math.clamp(v.y, 0, 1) * 63.0));
    const z: u16 = @intFromFloat(@round(std.math.clamp(v.z, 0, 1) * 31.0));
    return x | (y << 5) | (z << 11);
}
pub fn unpackUnorm1x5_1x6_1x5(p: u16) Vec3 {
    return Vec3.init(@as(f32, @floatFromInt(p & 0x1f)) / 31.0, @as(f32, @floatFromInt((p >> 5) & 0x3f)) / 63.0, @as(f32, @floatFromInt((p >> 11) & 0x1f)) / 31.0);
}
pub fn packUnorm3x5_1x1(v: Vec4) u16 {
    const x: u16 = @intFromFloat(@round(std.math.clamp(v.x, 0, 1) * 31.0));
    const y: u16 = @intFromFloat(@round(std.math.clamp(v.y, 0, 1) * 31.0));
    const z: u16 = @intFromFloat(@round(std.math.clamp(v.z, 0, 1) * 31.0));
    const w: u16 = @intFromFloat(@round(std.math.clamp(v.w, 0, 1) * 1.0));
    return x | (y << 5) | (z << 10) | (w << 15);
}
pub fn unpackUnorm3x5_1x1(p: u16) Vec4 {
    return Vec4.init(@as(f32, @floatFromInt(p & 0x1f)) / 31.0, @as(f32, @floatFromInt((p >> 5) & 0x1f)) / 31.0, @as(f32, @floatFromInt((p >> 10) & 0x1f)) / 31.0, @as(f32, @floatFromInt((p >> 15) & 0x1)));
}
pub fn packUnorm2x3_1x2(v: Vec3) u8 {
    const x: u8 = @intFromFloat(@round(std.math.clamp(v.x, 0, 1) * 7.0));
    const y: u8 = @intFromFloat(@round(std.math.clamp(v.y, 0, 1) * 7.0));
    const z: u8 = @intFromFloat(@round(std.math.clamp(v.z, 0, 1) * 3.0));
    return x | (y << 3) | (z << 6);
}
pub fn unpackUnorm2x3_1x2(p: u8) Vec3 {
    return Vec3.init(@as(f32, @floatFromInt(p & 0x7)) / 7.0, @as(f32, @floatFromInt((p >> 3) & 0x7)) / 7.0, @as(f32, @floatFromInt((p >> 6) & 0x3)) / 3.0);
}

// --- 10/10/10/2 integer + snorm ---------------------------------------------

pub fn packU3x10_1x2(v: vec.UVec4) u32 {
    return (v.x & 0x3FF) | ((v.y & 0x3FF) << 10) | ((v.z & 0x3FF) << 20) | ((v.w & 0x3) << 30);
}
pub fn unpackU3x10_1x2(p: u32) vec.UVec4 {
    return vec.UVec4.init(p & 0x3FF, (p >> 10) & 0x3FF, (p >> 20) & 0x3FF, (p >> 30) & 0x3);
}
pub fn packI3x10_1x2(v: vec.IVec4) u32 {
    const x: u32 = @as(u10, @bitCast(@as(i10, @intCast(v.x))));
    const y: u32 = @as(u10, @bitCast(@as(i10, @intCast(v.y))));
    const z: u32 = @as(u10, @bitCast(@as(i10, @intCast(v.z))));
    const w: u32 = @as(u2, @bitCast(@as(i2, @intCast(v.w))));
    return x | (y << 10) | (z << 20) | (w << 30);
}
pub fn unpackI3x10_1x2(p: u32) vec.IVec4 {
    return vec.IVec4.init(
        @as(i10, @bitCast(@as(u10, @truncate(p)))),
        @as(i10, @bitCast(@as(u10, @truncate(p >> 10)))),
        @as(i10, @bitCast(@as(u10, @truncate(p >> 20)))),
        @as(i2, @bitCast(@as(u2, @truncate(p >> 30)))),
    );
}
pub fn packSnorm3x10_1x2(v: Vec4) u32 {
    const x: i10 = @intFromFloat(@round(std.math.clamp(v.x, -1, 1) * 511.0));
    const y: i10 = @intFromFloat(@round(std.math.clamp(v.y, -1, 1) * 511.0));
    const z: i10 = @intFromFloat(@round(std.math.clamp(v.z, -1, 1) * 511.0));
    const w: i2 = @intFromFloat(@round(std.math.clamp(v.w, -1, 1) * 1.0));
    return @as(u32, @as(u10, @bitCast(x))) | (@as(u32, @as(u10, @bitCast(y))) << 10) | (@as(u32, @as(u10, @bitCast(z))) << 20) | (@as(u32, @as(u2, @bitCast(w))) << 30);
}
pub fn unpackSnorm3x10_1x2(p: u32) Vec4 {
    const x: i10 = @bitCast(@as(u10, @truncate(p)));
    const y: i10 = @bitCast(@as(u10, @truncate(p >> 10)));
    const z: i10 = @bitCast(@as(u10, @truncate(p >> 20)));
    const w: i2 = @bitCast(@as(u2, @truncate(p >> 30)));
    return Vec4.init(
        std.math.clamp(@as(f32, @floatFromInt(x)) / 511.0, -1, 1),
        std.math.clamp(@as(f32, @floatFromInt(y)) / 511.0, -1, 1),
        std.math.clamp(@as(f32, @floatFromInt(z)) / 511.0, -1, 1),
        @as(f32, @floatFromInt(w)),
    );
}

// --- integer reinterpret packs (byte-equal to GLM memcpy) -------------------

pub fn packInt2x8(v: vec.Vec(2, i8)) i16 {
    return @bitCast(v);
}
pub fn unpackInt2x8(p: i16) vec.Vec(2, i8) {
    return @bitCast(p);
}
pub fn packUint2x8(v: vec.Vec(2, u8)) u16 {
    return @bitCast(v);
}
pub fn unpackUint2x8(p: u16) vec.Vec(2, u8) {
    return @bitCast(p);
}
pub fn packInt4x8(v: vec.Vec(4, i8)) i32 {
    return @bitCast(v);
}
pub fn unpackInt4x8(p: i32) vec.Vec(4, i8) {
    return @bitCast(p);
}
pub fn packUint4x8(v: vec.Vec(4, u8)) u32 {
    return @bitCast(v);
}
pub fn unpackUint4x8(p: u32) vec.Vec(4, u8) {
    return @bitCast(p);
}
pub fn packInt2x16(v: vec.Vec(2, i16)) i32 {
    return @bitCast(v);
}
pub fn unpackInt2x16(p: i32) vec.Vec(2, i16) {
    return @bitCast(p);
}
pub fn packUint2x16(v: vec.Vec(2, u16)) u32 {
    return @bitCast(v);
}
pub fn unpackUint2x16(p: u32) vec.Vec(2, u16) {
    return @bitCast(p);
}
pub fn packInt4x16(v: vec.Vec(4, i16)) i64 {
    return @bitCast(v);
}
pub fn unpackInt4x16(p: i64) vec.Vec(4, i16) {
    return @bitCast(p);
}
pub fn packUint4x16(v: vec.Vec(4, u16)) u64 {
    return @bitCast(v);
}
pub fn unpackUint4x16(p: u64) vec.Vec(4, u16) {
    return @bitCast(p);
}
pub fn packInt2x32(v: vec.Vec(2, i32)) i64 {
    return @bitCast(v);
}
pub fn unpackInt2x32(p: i64) vec.Vec(2, i32) {
    return @bitCast(p);
}
pub fn packUint2x32(v: vec.Vec(2, u32)) u64 {
    return @bitCast(v);
}
pub fn unpackUint2x32(p: u64) vec.Vec(2, u32) {
    return @bitCast(p);
}


// --- public namespaces: math.pack.* and math.unpack.* ---
pub const pack = struct {
    pub const unorm2x16 = packUnorm2x16;
    pub const snorm2x16 = packSnorm2x16;
    pub const unorm4x8 = packUnorm4x8;
    pub const snorm4x8 = packSnorm4x8;
    pub const half2x16 = packHalf2x16;
    pub const double2x32 = packDouble2x32;
    pub const unorm1x8 = packUnorm1x8;
    pub const snorm1x8 = packSnorm1x8;
    pub const unorm1x16 = packUnorm1x16;
    pub const snorm1x16 = packSnorm1x16;
    pub const half1x16 = packHalf1x16;
    pub const unorm3x10_1x2 = packUnorm3x10_1x2;
    pub const f2x11_1x10 = packF2x11_1x10;
    pub const f3x9_e1x5 = packF3x9_E1x5;
    pub const rgbm = packRGBM;
    pub const unorm = packUnorm;
    pub const snorm = packSnorm;
    pub const half = packHalf;
    pub const unorm2x8 = packUnorm2x8;
    pub const snorm2x8 = packSnorm2x8;
    pub const unorm4x16 = packUnorm4x16;
    pub const snorm4x16 = packSnorm4x16;
    pub const half4x16 = packHalf4x16;
    pub const unorm2x4 = packUnorm2x4;
    pub const unorm4x4 = packUnorm4x4;
    pub const unorm1x5_1x6_1x5 = packUnorm1x5_1x6_1x5;
    pub const unorm3x5_1x1 = packUnorm3x5_1x1;
    pub const unorm2x3_1x2 = packUnorm2x3_1x2;
    pub const u3x10_1x2 = packU3x10_1x2;
    pub const i3x10_1x2 = packI3x10_1x2;
    pub const snorm3x10_1x2 = packSnorm3x10_1x2;
    pub const int2x8 = packInt2x8;
    pub const uint2x8 = packUint2x8;
    pub const int4x8 = packInt4x8;
    pub const uint4x8 = packUint4x8;
    pub const int2x16 = packInt2x16;
    pub const uint2x16 = packUint2x16;
    pub const int4x16 = packInt4x16;
    pub const uint4x16 = packUint4x16;
    pub const int2x32 = packInt2x32;
    pub const uint2x32 = packUint2x32;
};
pub const unpack = struct {
    pub const unorm2x16 = unpackUnorm2x16;
    pub const snorm2x16 = unpackSnorm2x16;
    pub const unorm4x8 = unpackUnorm4x8;
    pub const snorm4x8 = unpackSnorm4x8;
    pub const half2x16 = unpackHalf2x16;
    pub const double2x32 = unpackDouble2x32;
    pub const unorm1x8 = unpackUnorm1x8;
    pub const snorm1x8 = unpackSnorm1x8;
    pub const unorm1x16 = unpackUnorm1x16;
    pub const snorm1x16 = unpackSnorm1x16;
    pub const half1x16 = unpackHalf1x16;
    pub const unorm3x10_1x2 = unpackUnorm3x10_1x2;
    pub const f2x11_1x10 = unpackF2x11_1x10;
    pub const f3x9_e1x5 = unpackF3x9_E1x5;
    pub const rgbm = unpackRGBM;
    pub const unorm = unpackUnorm;
    pub const snorm = unpackSnorm;
    pub const half = unpackHalf;
    pub const unorm2x8 = unpackUnorm2x8;
    pub const snorm2x8 = unpackSnorm2x8;
    pub const unorm4x16 = unpackUnorm4x16;
    pub const snorm4x16 = unpackSnorm4x16;
    pub const half4x16 = unpackHalf4x16;
    pub const unorm2x4 = unpackUnorm2x4;
    pub const unorm4x4 = unpackUnorm4x4;
    pub const unorm1x5_1x6_1x5 = unpackUnorm1x5_1x6_1x5;
    pub const unorm3x5_1x1 = unpackUnorm3x5_1x1;
    pub const unorm2x3_1x2 = unpackUnorm2x3_1x2;
    pub const u3x10_1x2 = unpackU3x10_1x2;
    pub const i3x10_1x2 = unpackI3x10_1x2;
    pub const snorm3x10_1x2 = unpackSnorm3x10_1x2;
    pub const int2x8 = unpackInt2x8;
    pub const uint2x8 = unpackUint2x8;
    pub const int4x8 = unpackInt4x8;
    pub const uint4x8 = unpackUint4x8;
    pub const int2x16 = unpackInt2x16;
    pub const uint2x16 = unpackUint2x16;
    pub const int4x16 = unpackInt4x16;
    pub const uint4x16 = unpackUint4x16;
    pub const int2x32 = unpackInt2x32;
    pub const uint2x32 = unpackUint2x32;
};

test "pack/unpack namespaces" {
    const t = std.testing;
    // GLSL fixed-width formats
    try t.expect(unpack.unorm2x16(pack.unorm2x16(Vec2.init(0.25, 0.75))).approxEql(Vec2.init(0.25, 0.75), 1e-4));
    try t.expect(unpack.snorm4x8(pack.snorm4x8(Vec4.init(-0.5, 0.5, 0, 1))).approxEql(Vec4.init(-0.5, 0.5, 0, 1), 1e-2));
    try t.expect(unpack.half2x16(pack.half2x16(Vec2.init(1.5, -2.25))).approxEql(Vec2.init(1.5, -2.25), 1e-3));
    // generic comptime-width normalized
    const u = Vec2.init(0.1, 0.9);
    try t.expect(unpack.unorm(f32, pack.unorm(u16, u)).approxEql(u, 1e-3));
    const s = Vec3.init(-0.3, 0.3, 1.0);
    try t.expect(unpack.snorm(f32, pack.snorm(i16, s)).approxEql(s, 1e-3));
    // hardware integer packing (exact)
    const iv = vec.Vec(2, i8).init(-7, 42);
    try t.expectEqual(iv, unpack.int2x8(pack.int2x8(iv)));
}
