//! Miscellaneous utilities — `math.util`. Pointer/array interop, hashing, string
//! formatting, gradient paint, texture mip levels, polar/Cartesian conversion,
//! and interval predicates. (PCA lives in `math.matrix`.)

const std = @import("std");
const sc = @import("meta.zig");
const vec = @import("vec.zig");
const mat = @import("mat.zig");
const quat = @import("quat.zig");
const Vec2 = vec.Vec2;
const Vec3 = vec.Vec3;
const Mat2 = mat.Mat2;
const UVec2 = vec.UVec2;
const Quat = quat.Quaternion;

// === type_ptr ===


/// Many-item const pointer to the first scalar element. Pass `&value`.
pub fn valuePtr(v: anytype) [*]const sc.Element(std.meta.Child(@TypeOf(v))) {
    return @ptrCast(v);
}
/// Mutable variant.
pub fn valuePtrMut(v: anytype) [*]sc.Element(std.meta.Child(@TypeOf(v))) {
    return @ptrCast(v);
}


// === string_cast ===


/// Render `v` into `buf`, returning the written slice.
pub fn toBuf(buf: []u8, v: anytype) ![]u8 {
    return std.fmt.bufPrint(buf, "{f}", .{v});
}


// === hash ===


pub fn hash(v: anytype) u64 {
    return std.hash.Wyhash.hash(0, std.mem.asBytes(&v));
}


// === interval predicates ===

pub fn openBounded(value: anytype, min: @TypeOf(value), max: @TypeOf(value)) bool {
    return value > min and value < max;
}
pub fn closeBounded(value: anytype, min: @TypeOf(value), max: @TypeOf(value)) bool {
    return value >= min and value <= max;
}


// === texture ===


/// Number of mipmap levels for the given texture extent (integer vector).
pub fn levels(extent: anytype) u32 {
    const m = extent.maxComponent();
    return @as(u32, @intFromFloat(@floor(@log2(@as(f64, @floatFromInt(m)))))) + 1;
}

/// Scalar form.
pub fn levelsScalar(extent: u32) u32 {
    return @as(u32, @intFromFloat(@floor(@log2(@as(f64, @floatFromInt(extent)))))) + 1;
}


// === polar_coordinates ===


/// Cartesian → (latitude, longitude, xz-distance-of-unit-vector).
pub fn polar(cart: Vec3) Vec3 {
    const length = cart.length();
    const t = cart.scale(1.0 / length);
    const xz = @sqrt(t.x * t.x + t.z * t.z);
    return Vec3.init(std.math.asin(t.y), std.math.atan2(t.x, t.z), xz);
}

/// (latitude, longitude) → unit Cartesian vector.
pub fn euclidean(p: Vec2) Vec3 {
    const lat = p.x;
    const lon = p.y;
    return Vec3.init(@cos(lat) * @sin(lon), @sin(lat), @cos(lat) * @cos(lon));
}


// === gradient_paint ===


pub fn linearGradient(point0: Vec2, point1: Vec2, position: Vec2) f32 {
    const dist = point1.sub(point0);
    return dist.dot(position.sub(point0)) / dist.dot(dist);
}

pub fn radialGradient(center: Vec2, radius: f32, focal: Vec2, position: Vec2) f32 {
    const f = focal.sub(center);
    const d = position.sub(focal);
    const radius2 = radius * radius;
    const fc2 = f.dot(f);
    const dlen2 = d.dot(d);
    const ddotf = d.dot(f);
    const cross = d.x * f.y - d.y * f.x;
    const numerator = ddotf + @sqrt(radius2 * dlen2 - cross * cross);
    return numerator / (radius2 - fc2);
}


const testing = std.testing;

test "type_ptr" {
    const v = Vec3.init(7, 8, 9);
    const p = valuePtr(&v);
    try testing.expectEqual(@as(f32, 7), p[0]);
    try testing.expectEqual(@as(f32, 9), p[2]);
    const m = Mat2.fromColumns(.{ Vec2.init(1, 2), Vec2.init(3, 4) });
    const mp = valuePtr(&m);
    try testing.expectEqual(@as(f32, 1), mp[0]); // column-major
    try testing.expectEqual(@as(f32, 3), mp[2]);
}

test "string_cast" {
    var buf: [128]u8 = undefined;
    try testing.expectEqualStrings("vec3(1, 2, 3)", try toBuf(&buf, Vec3.init(1, 2, 3)));
    try testing.expectEqualStrings("mat2x2[vec2(1, 2), vec2(3, 4)]", try toBuf(&buf, Mat2.fromColumns(.{ Vec2.init(1, 2), Vec2.init(3, 4) })));
    try testing.expectEqualStrings("quat(1, {0, 0, 0})", try toBuf(&buf, Quat.identity()));
}

test "hash" {
    const a = Vec3.init(1, 2, 3);
    const b = Vec3.init(1, 2, 3);
    const c = Vec3.init(1, 2, 4);
    try testing.expectEqual(hash(a), hash(b));
    try testing.expect(hash(a) != hash(c));
}

test "interval predicates" {
    try testing.expect(closeBounded(@as(f32, 5), 0, 5));
    try testing.expect(!openBounded(@as(f32, 5), 0, 5));
}

test "texture levels" {
    try testing.expectEqual(@as(u32, 11), levels(UVec2.init(1024, 512)));
    try testing.expectEqual(@as(u32, 1), levelsScalar(1));
}

test "polar round trip" {
    const v = Vec3.init(0.3, 0.5, -0.8).normalize();
    const p = polar(v);
    const back = euclidean(Vec2.init(p.x, p.y));
    try testing.expect(back.approxEql(v, 1e-5));
}

test "gradient_paint" {
    try testing.expectApproxEqAbs(@as(f32, 0.5), linearGradient(Vec2.init(0, 0), Vec2.init(10, 0), Vec2.init(5, 99)), 1e-6);
}


