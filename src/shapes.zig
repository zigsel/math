//! Geometry primitives — `math.Plane`, `math.Line`, `math.Segment`,
//! `math.Rect`, `math.Triangle` (generic builders; f32 concretes `Plane3`, …).
//! Bounding volumes (Aabb/Sphere/Obb/Capsule/Frustum) live in `math.bounds`.

const std = @import("std");
const sc = @import("meta.zig");
const vec = @import("vec.zig");
const intersect = @import("intersect.zig");

// === Plane: n·p + d = 0 =====================================================

pub fn Plane(comptime T: type) type {
    comptime sc.requireFloat(T);
    return struct {
        normal: V3,
        d: T,

        const Self = @This();
        const V3 = vec.Vec(3, T);

        pub fn init(normal: V3, d: T) Self {
            return .{ .normal = normal, .d = d };
        }
        /// Plane through `point` with the given (unit) `normal`.
        pub fn fromPointNormal(point: V3, normal: V3) Self {
            return .{ .normal = normal, .d = -normal.dot(point) };
        }
        /// Plane through three points (normal = (b-a)×(c-a), normalized).
        pub fn fromPoints(a: V3, b: V3, c: V3) Self {
            const n = b.sub(a).cross(c.sub(a)).normalize();
            return fromPointNormal(a, n);
        }
        pub fn normalized(self: Self) Self {
            const len = self.normal.length();
            return .{ .normal = self.normal.scale(1.0 / len), .d = self.d / len };
        }
        /// Signed distance from `p` to the plane (positive on the normal side).
        pub fn distance(self: Self, p: V3) T {
            return self.normal.dot(p) + self.d;
        }
        /// Orthogonal projection of `p` onto the plane.
        pub fn project(self: Self, p: V3) V3 {
            return p.sub(self.normal.scale(self.distance(p)));
        }
        /// Distance along `dir` to the plane, or null if parallel.
        pub fn intersectRay(self: Self, orig: V3, dir: V3) ?T {
            const denom = dir.dot(self.normal);
            if (@abs(denom) < 1e-7) return null;
            return -(self.normal.dot(orig) + self.d) / denom;
        }
    };
}

// === Line (infinite) & Segment ==============================================

pub fn Line(comptime T: type) type {
    comptime sc.requireFloat(T);
    return struct {
        point: V3,
        dir: V3, // need not be unit length

        const Self = @This();
        const V3 = vec.Vec(3, T);

        pub fn init(point: V3, dir: V3) Self {
            return .{ .point = point, .dir = dir };
        }
        pub fn closestPoint(self: Self, p: V3) V3 {
            const t = p.sub(self.point).dot(self.dir) / self.dir.dot(self.dir);
            return self.point.add(self.dir.scale(t));
        }
        pub fn distance(self: Self, p: V3) T {
            return p.distance(self.closestPoint(p));
        }
    };
}

pub fn Segment(comptime T: type) type {
    comptime sc.requireFloat(T);
    return struct {
        a: V3,
        b: V3,

        const Self = @This();
        const V3 = vec.Vec(3, T);

        pub fn init(a: V3, b: V3) Self {
            return .{ .a = a, .b = b };
        }
        pub fn lerp(self: Self, t: T) V3 {
            return self.a.lerp(self.b, t);
        }
        pub fn length(self: Self) T {
            return self.a.distance(self.b);
        }
        pub fn closestPoint(self: Self, p: V3) V3 {
            const ab = self.b.sub(self.a);
            const t = std.math.clamp(p.sub(self.a).dot(ab) / ab.dot(ab), 0, 1);
            return self.a.add(ab.scale(t));
        }
        pub fn distance(self: Self, p: V3) T {
            return p.distance(self.closestPoint(p));
        }
    };
}

// === Rect (2-D axis-aligned) ================================================

pub fn Rect(comptime T: type) type {
    comptime sc.requireFloat(T);
    return struct {
        min: V2,
        max: V2,

        const Self = @This();
        const V2 = vec.Vec(2, T);

        pub fn init(min: V2, max: V2) Self {
            return .{ .min = min, .max = max };
        }
        pub fn fromPosSize(pos: V2, extent: V2) Self {
            return .{ .min = pos, .max = pos.add(extent) };
        }
        pub fn center(self: Self) V2 {
            return self.min.add(self.max).scale(0.5);
        }
        pub fn size(self: Self) V2 {
            return self.max.sub(self.min);
        }
        pub fn area(self: Self) T {
            const s = self.size();
            return s.x * s.y;
        }
        pub fn contains(self: Self, p: V2) bool {
            return p.x >= self.min.x and p.x <= self.max.x and p.y >= self.min.y and p.y <= self.max.y;
        }
        pub fn intersects(a: Self, b: Self) bool {
            return a.min.x <= b.max.x and a.max.x >= b.min.x and a.min.y <= b.max.y and a.max.y >= b.min.y;
        }
        pub fn expand(self: Self, p: V2) Self {
            return .{ .min = self.min.min(p), .max = self.max.max(p) };
        }
        pub fn merge(a: Self, b: Self) Self {
            return .{ .min = a.min.min(b.min), .max = a.max.max(b.max) };
        }
    };
}

// === Triangle ===============================================================

pub fn Triangle(comptime T: type) type {
    comptime sc.requireFloat(T);
    return struct {
        a: V3,
        b: V3,
        c: V3,

        const Self = @This();
        const V3 = vec.Vec(3, T);

        pub fn init(a: V3, b: V3, c: V3) Self {
            return .{ .a = a, .b = b, .c = c };
        }
        pub fn normal(self: Self) V3 {
            return self.b.sub(self.a).cross(self.c.sub(self.a)).normalize();
        }
        pub fn area(self: Self) T {
            return self.b.sub(self.a).cross(self.c.sub(self.a)).length() * 0.5;
        }
        pub fn centroid(self: Self) V3 {
            return self.a.add(self.b).add(self.c).scale(1.0 / 3.0);
        }
        /// Barycentric coordinates `(u, v, w)` of `p` projected onto the triangle.
        pub fn barycentric(self: Self, p: V3) V3 {
            const v0 = self.b.sub(self.a);
            const v1 = self.c.sub(self.a);
            const v2 = p.sub(self.a);
            const d00 = v0.dot(v0);
            const d01 = v0.dot(v1);
            const d11 = v1.dot(v1);
            const d20 = v2.dot(v0);
            const d21 = v2.dot(v1);
            const denom = d00 * d11 - d01 * d01;
            const v = (d11 * d20 - d01 * d21) / denom;
            const w = (d00 * d21 - d01 * d20) / denom;
            return V3.init(1 - v - w, v, w);
        }
        /// Closest point on the triangle to `p` (Ericson, RTCD).
        pub fn closestPoint(self: Self, p: V3) V3 {
            const a = self.a;
            const b = self.b;
            const c = self.c;
            const ab = b.sub(a);
            const ac = c.sub(a);
            const ap = p.sub(a);
            const d1 = ab.dot(ap);
            const d2 = ac.dot(ap);
            if (d1 <= 0 and d2 <= 0) return a;
            const bp = p.sub(b);
            const d3 = ab.dot(bp);
            const d4 = ac.dot(bp);
            if (d3 >= 0 and d4 <= d3) return b;
            const vc = d1 * d4 - d3 * d2;
            if (vc <= 0 and d1 >= 0 and d3 <= 0) return a.add(ab.scale(d1 / (d1 - d3)));
            const cp = p.sub(c);
            const d5 = ab.dot(cp);
            const d6 = ac.dot(cp);
            if (d6 >= 0 and d5 <= d6) return c;
            const vb = d5 * d2 - d1 * d6;
            if (vb <= 0 and d2 >= 0 and d6 <= 0) return a.add(ac.scale(d2 / (d2 - d6)));
            const va = d3 * d6 - d5 * d4;
            if (va <= 0 and (d4 - d3) >= 0 and (d5 - d6) >= 0) {
                const w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
                return b.add(c.sub(b).scale(w));
            }
            const denom = 1.0 / (va + vb + vc);
            return a.add(ab.scale(vb * denom)).add(ac.scale(vc * denom));
        }
        /// Möller–Trumbore ray hit (distance + barycentric u,v), or null.
        pub fn intersectRay(self: Self, orig: V3, dir: V3) ?intersect.RayTriangleHit(T) {
            return intersect.rayTriangle(orig, dir, self.a, self.b, self.c);
        }
    };
}

const testing = std.testing;
const Vec3 = vec.Vec3;
const Vec2 = vec.Vec2;

test "plane" {
    const pl = Plane(f32).fromPoints(Vec3.init(0, 0, 0), Vec3.init(1, 0, 0), Vec3.init(0, 1, 0));
    try testing.expectApproxEqAbs(@as(f32, 1), pl.distance(Vec3.init(0, 0, 1)), 1e-6);
    try testing.expect(pl.project(Vec3.init(2, 3, 5)).approxEql(Vec3.init(2, 3, 0), 1e-6));
    try testing.expectApproxEqAbs(@as(f32, 4), pl.intersectRay(Vec3.init(0, 0, 4), Vec3.init(0, 0, -1)).?, 1e-6);
}
test "segment / line closest point" {
    const s = Segment(f32).init(Vec3.init(0, 0, 0), Vec3.init(10, 0, 0));
    try testing.expect(s.closestPoint(Vec3.init(3, 5, 0)).approxEql(Vec3.init(3, 0, 0), 1e-6));
    try testing.expect(s.closestPoint(Vec3.init(-5, 1, 0)).approxEql(Vec3.init(0, 0, 0), 1e-6)); // clamped
    const ln = Line(f32).init(Vec3.init(0, 0, 0), Vec3.init(1, 0, 0));
    try testing.expectApproxEqAbs(@as(f32, 5), ln.distance(Vec3.init(3, 5, 0)), 1e-6);
}
test "rect" {
    const r = Rect(f32).fromPosSize(Vec2.init(0, 0), Vec2.init(4, 2));
    try testing.expect(r.contains(Vec2.init(2, 1)));
    try testing.expect(!r.contains(Vec2.init(5, 1)));
    try testing.expectEqual(@as(f32, 8), r.area());
    try testing.expect(r.intersects(Rect(f32).init(Vec2.init(3, 1), Vec2.init(6, 6))));
}
test "triangle" {
    const t = Triangle(f32).init(Vec3.init(0, 0, 0), Vec3.init(1, 0, 0), Vec3.init(0, 1, 0));
    try testing.expect(t.normal().approxEql(Vec3.init(0, 0, 1), 1e-6));
    try testing.expectApproxEqAbs(@as(f32, 0.5), t.area(), 1e-6);
    try testing.expect(t.barycentric(t.centroid()).approxEql(Vec3.splat(1.0 / 3.0), 1e-5));
    try testing.expect(t.closestPoint(Vec3.init(-1, -1, 0)).approxEql(Vec3.init(0, 0, 0), 1e-6));
    try testing.expectApproxEqAbs(@as(f32, 1), t.intersectRay(Vec3.init(0.25, 0.25, 1), Vec3.init(0, 0, -1)).?.t, 1e-6);
}
