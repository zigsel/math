//! Ray/line intersection tests — `math.intersect`. Generic over the float
//! element type (inferred from the `Vec(3, T)` arguments). Optionals replace
//! GLM's out-parameter + bool convention.
//!
//! A `Ray(T)` bundles an origin + direction and offers method forms of the same
//! tests (`ray.sphere(c, r)`, `ray.triangle(v0, v1, v2)`, …).

const std = @import("std");
const sc = @import("meta.zig");
const vec = @import("vec.zig");

fn Elem(comptime V: type) type {
    return V.Element;
}

/// Distance along `dir` to a plane, or null if parallel / behind.
pub fn rayPlane(orig: anytype, dir: @TypeOf(orig), plane_orig: @TypeOf(orig), plane_normal: @TypeOf(orig)) ?Elem(@TypeOf(orig)) {
    const denom = dir.dot(plane_normal);
    if (@abs(denom) < 1e-7) return null;
    const t = plane_orig.sub(orig).dot(plane_normal) / denom;
    return if (t >= 0) t else null;
}

/// Nearest distance along (unit) `dir` to a sphere surface, or null on a miss.
pub fn raySphere(orig: anytype, dir: @TypeOf(orig), center: @TypeOf(orig), radius: Elem(@TypeOf(orig))) ?Elem(@TypeOf(orig)) {
    const diff = center.sub(orig);
    const t0 = diff.dot(dir);
    const d2 = diff.dot(diff) - t0 * t0;
    const r2 = radius * radius;
    if (d2 > r2) return null;
    const thc = @sqrt(r2 - d2);
    const t = t0 - thc;
    if (t >= 0) return t;
    const t_far = t0 + thc;
    return if (t_far >= 0) t_far else null;
}

pub fn RayTriangleHit(comptime T: type) type {
    return struct { t: T, u: T, v: T };
}

/// Möller–Trumbore ray/triangle test. Returns distance `t` and barycentric `u,v`.
pub fn rayTriangle(orig: anytype, dir: @TypeOf(orig), v0: @TypeOf(orig), v1: @TypeOf(orig), v2: @TypeOf(orig)) ?RayTriangleHit(Elem(@TypeOf(orig))) {
    const e1 = v1.sub(v0);
    const e2 = v2.sub(v0);
    const p = dir.cross(e2);
    const det = e1.dot(p);
    if (@abs(det) < 1e-7) return null;
    const inv = 1.0 / det;
    const tvec = orig.sub(v0);
    const u = tvec.dot(p) * inv;
    if (u < 0 or u > 1) return null;
    const q = tvec.cross(e1);
    const v = dir.dot(q) * inv;
    if (v < 0 or u + v > 1) return null;
    const t = e2.dot(q) * inv;
    if (t < 0) return null;
    return .{ .t = t, .u = u, .v = v };
}

pub fn SphereHit(comptime T: type) type {
    return struct { point: vec.Vec(3, T), normal: vec.Vec(3, T) };
}

/// Ray/sphere returning the hit point and outward surface normal.
pub fn raySphereHit(orig: anytype, dir: @TypeOf(orig), center: @TypeOf(orig), radius: Elem(@TypeOf(orig))) ?SphereHit(Elem(@TypeOf(orig))) {
    const t = raySphere(orig, dir, center, radius) orelse return null;
    const point = orig.add(dir.scale(t));
    return .{ .point = point, .normal = point.sub(center).normalize() };
}

/// Line/triangle (unbounded line, both directions). Returns barycentric u,v
/// and signed distance t.
pub fn lineTriangle(orig: anytype, dir: @TypeOf(orig), v0: @TypeOf(orig), v1: @TypeOf(orig), v2: @TypeOf(orig)) ?RayTriangleHit(Elem(@TypeOf(orig))) {
    const e1 = v1.sub(v0);
    const e2 = v2.sub(v0);
    const p = dir.cross(e2);
    const det = e1.dot(p);
    if (@abs(det) < 1e-7) return null;
    const inv = 1.0 / det;
    const tvec = orig.sub(v0);
    const u = tvec.dot(p) * inv;
    if (u < 0 or u > 1) return null;
    const q = tvec.cross(e1);
    const v = dir.dot(q) * inv;
    if (v < 0 or u + v > 1) return null;
    return .{ .t = e2.dot(q) * inv, .u = u, .v = v };
}

pub fn LineSphereHit(comptime T: type) type {
    return struct {
        near_point: vec.Vec(3, T),
        near_normal: vec.Vec(3, T),
        far_point: vec.Vec(3, T),
        far_normal: vec.Vec(3, T),
    };
}

/// Line/sphere returning both intersection points and their outward normals.
pub fn lineSphere(orig: anytype, dir: @TypeOf(orig), center: @TypeOf(orig), radius: Elem(@TypeOf(orig))) ?LineSphereHit(Elem(@TypeOf(orig))) {
    const d = dir.normalize();
    const diff = center.sub(orig);
    const t0 = diff.dot(d);
    const dd = diff.dot(diff) - t0 * t0;
    const r2 = radius * radius;
    if (dd > r2) return null;
    const thc = @sqrt(r2 - dd);
    const np = orig.add(d.scale(t0 - thc));
    const fp = orig.add(d.scale(t0 + thc));
    return .{
        .near_point = np,
        .near_normal = np.sub(center).normalize(),
        .far_point = fp,
        .far_normal = fp.sub(center).normalize(),
    };
}

/// A ray (origin + direction) over `Vec(3, T)`, with method forms of the tests.
pub fn Ray(comptime T: type) type {
    comptime sc.requireFloat(T);
    return struct {
        orig: V3,
        dir: V3,

        const Self = @This();
        const V3 = vec.Vec(3, T);

        pub fn init(orig: V3, dir: V3) Self {
            return .{ .orig = orig, .dir = dir };
        }
        /// Point at parameter `t` along the ray.
        pub fn at(self: Self, t: T) V3 {
            return self.orig.add(self.dir.scale(t));
        }
        pub fn plane(self: Self, plane_orig: V3, plane_normal: V3) ?T {
            return rayPlane(self.orig, self.dir, plane_orig, plane_normal);
        }
        pub fn sphere(self: Self, center: V3, radius: T) ?T {
            return raySphere(self.orig, self.dir, center, radius);
        }
        pub fn sphereHit(self: Self, center: V3, radius: T) ?SphereHit(T) {
            return raySphereHit(self.orig, self.dir, center, radius);
        }
        pub fn triangle(self: Self, v0: V3, v1: V3, v2: V3) ?RayTriangleHit(T) {
            return rayTriangle(self.orig, self.dir, v0, v1, v2);
        }
    };
}

const testing = std.testing;
const Vec3 = vec.Vec3;

test "intersect" {
    try testing.expectApproxEqAbs(@as(f32, 5), rayPlane(Vec3.init(0, 0, 0), Vec3.init(0, 0, -1), Vec3.init(0, 0, -5), Vec3.init(0, 0, 1)).?, 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 4), raySphere(Vec3.init(0, 0, 0), Vec3.init(1, 0, 0), Vec3.init(5, 0, 0), 1).?, 1e-5);
    const hit = rayTriangle(Vec3.init(0.25, 0.25, 1), Vec3.init(0, 0, -1), Vec3.init(0, 0, 0), Vec3.init(1, 0, 0), Vec3.init(0, 1, 0)).?;
    try testing.expectApproxEqAbs(@as(f32, 1), hit.t, 1e-5);

    const sh = raySphereHit(Vec3.init(0, 0, 0), Vec3.init(1, 0, 0), Vec3.init(5, 0, 0), 1).?;
    try testing.expect(sh.point.approxEql(Vec3.init(4, 0, 0), 1e-5));
    try testing.expect(sh.normal.approxEql(Vec3.init(-1, 0, 0), 1e-5));
    const ls = lineSphere(Vec3.init(0, 0, 0), Vec3.init(1, 0, 0), Vec3.init(5, 0, 0), 1).?;
    try testing.expect(ls.near_point.approxEql(Vec3.init(4, 0, 0), 1e-5));
    try testing.expect(ls.far_point.approxEql(Vec3.init(6, 0, 0), 1e-5));
}

test "Ray method form + f64 generic" {
    const r = Ray(f32).init(Vec3.init(0, 0, 0), Vec3.init(1, 0, 0));
    try testing.expectApproxEqAbs(@as(f32, 4), r.sphere(Vec3.init(5, 0, 0), 1).?, 1e-5);
    try testing.expect(r.at(2).approxEql(Vec3.init(2, 0, 0), 1e-6));

    const V = vec.Vec(3, f64);
    try testing.expectApproxEqAbs(@as(f64, 4), raySphere(V.init(0, 0, 0), V.init(1, 0, 0), V.init(5, 0, 0), 1).?, 1e-12);
}
