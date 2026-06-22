//! Bounding volumes & frustum culling — `math.bounds`.
//!
//! `Aabb(T)` / `Sphere(T)` are generic over the float element type and operate
//! on `Vec(3, T)` / `Mat(4, 4, T)`. Root exposes the f32 instances as the
//! concrete `math.Aabb` / `math.Sphere` (mirroring `math.Vec` vs `math.Vec3`).
//!
//! Frustum extraction assumes the depth-0..1 clip convention (our default `*Zo`
//! projections); a plane is a `Vec4` `(a,b,c,d)` with a point inside when
//! `dot(plane.xyz, p) + d >= 0`.

const std = @import("std");
const sc = @import("meta.zig");
const vec = @import("vec.zig");
const mat = @import("mat.zig");

// === Axis-aligned bounding box ==============================================

/// Axis-aligned bounding box over `Vec(3, T)`.
pub fn Aabb(comptime T: type) type {
    comptime sc.requireFloat(T);
    return struct {
        min: V3,
        max: V3,

        const Self = @This();
        const V3 = vec.Vec(3, T);
        const V4 = vec.Vec(4, T);
        const M4 = mat.Mat(4, 4, T);

        pub fn init(min: V3, max: V3) Self {
            return .{ .min = min, .max = max };
        }
        /// Empty box (min = +inf, max = -inf) — identity for `expand`/`merge`.
        pub fn invalid() Self {
            const inf = std.math.inf(T);
            return .{ .min = V3.splat(inf), .max = V3.splat(-inf) };
        }
        pub fn fromPoints(points: []const V3) Self {
            var b = invalid();
            for (points) |p| b = b.expand(p);
            return b;
        }
        pub fn isValid(self: Self) bool {
            return @reduce(.And, self.min.simd() <= self.max.simd());
        }
        pub fn center(self: Self) V3 {
            return self.min.add(self.max).scale(0.5);
        }
        pub fn size(self: Self) V3 {
            return self.max.sub(self.min);
        }
        pub fn halfSize(self: Self) V3 {
            return self.size().scale(0.5);
        }
        /// Radius of the enclosing bounding sphere.
        pub fn radius(self: Self) T {
            return self.halfSize().length();
        }
        pub fn expand(self: Self, p: V3) Self {
            return .{ .min = self.min.min(p), .max = self.max.max(p) };
        }
        pub fn merge(a: Self, b: Self) Self {
            return .{ .min = a.min.min(b.min), .max = a.max.max(b.max) };
        }
        pub fn crop(self: Self, clip: Self) Self {
            return .{ .min = self.min.max(clip.min), .max = self.max.min(clip.max) };
        }
        pub fn contains(self: Self, p: V3) bool {
            return @reduce(.And, p.simd() >= self.min.simd()) and @reduce(.And, p.simd() <= self.max.simd());
        }
        pub fn containsAabb(self: Self, b: Self) bool {
            return @reduce(.And, b.min.simd() >= self.min.simd()) and @reduce(.And, b.max.simd() <= self.max.simd());
        }
        pub fn intersects(a: Self, b: Self) bool {
            return @reduce(.And, a.min.simd() <= b.max.simd()) and @reduce(.And, a.max.simd() >= b.min.simd());
        }
        pub fn corners(self: Self) [8]V3 {
            const lo = self.min;
            const hi = self.max;
            return .{
                V3.init(lo.x, lo.y, lo.z), V3.init(hi.x, lo.y, lo.z),
                V3.init(hi.x, hi.y, lo.z), V3.init(lo.x, hi.y, lo.z),
                V3.init(lo.x, lo.y, hi.z), V3.init(hi.x, lo.y, hi.z),
                V3.init(hi.x, hi.y, hi.z), V3.init(lo.x, hi.y, hi.z),
            };
        }
        /// Smallest AABB enclosing this box after transforming by `m` (Arvo's method).
        pub fn transform(self: Self, m: M4) Self {
            const t = V3.init(m.at(3, 0), m.at(3, 1), m.at(3, 2));
            var newmin = t;
            var newmax = t;
            const mn = self.min.toArray();
            const mx = self.max.toArray();
            inline for (0..3) |i| {
                inline for (0..3) |j| {
                    const a = m.at(j, i) * mn[j];
                    const b = m.at(j, i) * mx[j];
                    newmin = newmin.set(i, newmin.get(i) + @min(a, b));
                    newmax = newmax.set(i, newmax.get(i) + @max(a, b));
                }
            }
            return .{ .min = newmin, .max = newmax };
        }
        /// Ray/box slab test; entry distance along `dir`, or null on a miss.
        pub fn intersectRay(self: Self, orig: V3, dir: V3) ?T {
            var tmin: T = -std.math.inf(T);
            var tmax: T = std.math.inf(T);
            const o = orig.toArray();
            const d = dir.toArray();
            const lo = self.min.toArray();
            const hi = self.max.toArray();
            inline for (0..3) |i| {
                if (@abs(d[i]) < 1e-8) {
                    if (o[i] < lo[i] or o[i] > hi[i]) return null;
                } else {
                    const inv = 1.0 / d[i];
                    var t1 = (lo[i] - o[i]) * inv;
                    var t2 = (hi[i] - o[i]) * inv;
                    if (t1 > t2) std.mem.swap(T, &t1, &t2);
                    tmin = @max(tmin, t1);
                    tmax = @min(tmax, t2);
                    if (tmin > tmax) return null;
                }
            }
            if (tmax < 0) return null;
            return if (tmin >= 0) tmin else tmax;
        }
        /// Convert to the enclosing bounding sphere.
        pub fn boundingSphere(self: Self) Sphere(T) {
            return .{ .center = self.center(), .radius = self.radius() };
        }
    };
}

// === Bounding sphere ========================================================

/// Bounding sphere over `Vec(3, T)`.
pub fn Sphere(comptime T: type) type {
    comptime sc.requireFloat(T);
    return struct {
        center: V3,
        radius: T,

        const Self = @This();
        const V3 = vec.Vec(3, T);
        const V4 = vec.Vec(4, T);
        const M4 = mat.Mat(4, 4, T);

        pub fn init(center: V3, radius: T) Self {
            return .{ .center = center, .radius = radius };
        }
        pub fn fromAabb(box: Aabb(T)) Self {
            return .{ .center = box.center(), .radius = box.radius() };
        }
        /// Approximate minimal bounding sphere (Ritter's algorithm).
        pub fn fromPoints(points: []const V3) Self {
            if (points.len == 0) return .{ .center = V3.splat(0), .radius = 0 };
            var px = points[0];
            var d2: T = 0;
            for (points) |p| {
                const dd = points[0].distanceSq(p);
                if (dd > d2) {
                    d2 = dd;
                    px = p;
                }
            }
            var py = px;
            d2 = 0;
            for (points) |p| {
                const dd = px.distanceSq(p);
                if (dd > d2) {
                    d2 = dd;
                    py = p;
                }
            }
            var c = px.add(py).scale(0.5);
            var r = px.distance(py) * 0.5;
            for (points) |p| {
                const dist_ = c.distance(p);
                if (dist_ > r) {
                    const nr = (r + dist_) * 0.5;
                    c = c.add(p.sub(c).scale((nr - r) / dist_));
                    r = nr;
                }
            }
            return .{ .center = c, .radius = r };
        }
        pub fn contains(self: Self, p: V3) bool {
            return self.center.distance(p) <= self.radius;
        }
        pub fn intersects(a: Self, b: Self) bool {
            return a.center.distance(b.center) <= a.radius + b.radius;
        }
        pub fn intersectsAabb(self: Self, box: Aabb(T)) bool {
            const closest = self.center.clamp(box.min, box.max);
            return self.center.distanceSq(closest) <= self.radius * self.radius;
        }
        pub fn merge(a: Self, b: Self) Self {
            const d = b.center.sub(a.center);
            const dist_ = d.length();
            if (dist_ + b.radius <= a.radius) return a;
            if (dist_ + a.radius <= b.radius) return b;
            const r = (a.radius + b.radius + dist_) * 0.5;
            const c = if (dist_ > 1e-8) a.center.add(d.scale((r - a.radius) / dist_)) else a.center;
            return .{ .center = c, .radius = r };
        }
        pub fn transform(self: Self, m: M4) Self {
            const c4 = m.mulVec(V4.fromVec3(self.center, 1));
            const s0 = V3.init(m.at(0, 0), m.at(0, 1), m.at(0, 2)).length();
            const s1 = V3.init(m.at(1, 0), m.at(1, 1), m.at(1, 2)).length();
            const s2 = V3.init(m.at(2, 0), m.at(2, 1), m.at(2, 2)).length();
            return .{ .center = V3.init(c4.x, c4.y, c4.z), .radius = self.radius * @max(s0, @max(s1, s2)) };
        }
    };
}

// === Frustum extraction & culling ===========================================
//
// Generic over the matrix/vector element type, inferred from the argument.

fn ElemOf(comptime M: type) type {
    return M.Element;
}

fn normalizePlane(p: anytype) @TypeOf(p) {
    const len = @sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
    return p.scale(1.0 / len);
}
fn planeDist(plane: anytype, p: anytype) @TypeOf(p).Element {
    return plane.x * p.x + plane.y * p.y + plane.z * p.z + plane.w;
}

/// Six frustum planes from a view-projection matrix (left,right,bottom,top,near,far).
pub fn frustumPlanes(vp: anytype) [6]vec.Vec(4, ElemOf(@TypeOf(vp))) {
    const r0 = vp.row(0);
    const r1 = vp.row(1);
    const r2 = vp.row(2);
    const r3 = vp.row(3);
    return .{
        normalizePlane(r3.add(r0)), normalizePlane(r3.sub(r0)),
        normalizePlane(r3.add(r1)), normalizePlane(r3.sub(r1)),
        normalizePlane(r2),         normalizePlane(r3.sub(r2)),
    };
}
/// Eight world-space frustum corners from a view-projection matrix.
pub fn frustumCorners(vp: anytype) [8]vec.Vec(3, ElemOf(@TypeOf(vp))) {
    const T = ElemOf(@TypeOf(vp));
    const V3 = vec.Vec(3, T);
    const V4 = vec.Vec(4, T);
    const inv = vp.inverse();
    const ndc = [8]V4{
        V4.init(-1, -1, 0, 1), V4.init(1, -1, 0, 1), V4.init(1, 1, 0, 1), V4.init(-1, 1, 0, 1),
        V4.init(-1, -1, 1, 1), V4.init(1, -1, 1, 1), V4.init(1, 1, 1, 1), V4.init(-1, 1, 1, 1),
    };
    var out: [8]V3 = undefined;
    inline for (0..8) |i| {
        const c = inv.mulVec(ndc[i]);
        out[i] = V3.init(c.x / c.w, c.y / c.w, c.z / c.w);
    }
    return out;
}
pub fn frustumBox(corners: anytype) Aabb(@TypeOf(corners[0]).Element) {
    return Aabb(@TypeOf(corners[0]).Element).fromPoints(&corners);
}
pub fn pointInFrustum(planes: anytype, p: anytype) bool {
    inline for (0..6) |i| {
        if (planeDist(planes[i], p) < 0) return false;
    }
    return true;
}
pub fn sphereInFrustum(planes: anytype, s: anytype) bool {
    inline for (0..6) |i| {
        if (planeDist(planes[i], s.center) < -s.radius) return false;
    }
    return true;
}
/// Conservative AABB cull (positive-vertex test): false ⇒ fully outside.
pub fn aabbInFrustum(planes: anytype, box: anytype) bool {
    const V3 = @TypeOf(box.min);
    inline for (0..6) |i| {
        const pl = planes[i];
        const pv = V3.init(
            if (pl.x >= 0) box.max.x else box.min.x,
            if (pl.y >= 0) box.max.y else box.min.y,
            if (pl.z >= 0) box.max.z else box.min.z,
        );
        if (planeDist(pl, pv) < 0) return false;
    }
    return true;
}

// === Oriented bounding box ==================================================

/// Oriented bounding box: a box with arbitrary rotation. `axes` columns are the
/// (orthonormal) local axes; `half` is the extent along each.
pub fn Obb(comptime T: type) type {
    comptime sc.requireFloat(T);
    return struct {
        center: V3,
        half: V3,
        axes: M3,

        const Self = @This();
        const V3 = vec.Vec(3, T);
        const M3 = mat.Mat(3, 3, T);

        pub fn init(center: V3, half: V3, axes: M3) Self {
            return .{ .center = center, .half = half, .axes = axes };
        }
        /// Axis-aligned box as an OBB.
        pub fn fromAabb(box: Aabb(T)) Self {
            return .{ .center = box.center(), .half = box.halfSize(), .axes = M3.identity() };
        }
        fn axis(self: Self, comptime i: usize) V3 {
            return self.axes.cols[i];
        }
        /// Local-frame coordinates of `p` relative to the box centre.
        fn toLocal(self: Self, p: V3) V3 {
            const d = p.sub(self.center);
            return V3.init(d.dot(self.axis(0)), d.dot(self.axis(1)), d.dot(self.axis(2)));
        }
        pub fn contains(self: Self, p: V3) bool {
            const l = self.toLocal(p);
            return @abs(l.x) <= self.half.x and @abs(l.y) <= self.half.y and @abs(l.z) <= self.half.z;
        }
        pub fn closestPoint(self: Self, p: V3) V3 {
            const l = self.toLocal(p).clamp(self.half.neg(), self.half);
            return self.center
                .add(self.axis(0).scale(l.x))
                .add(self.axis(1).scale(l.y))
                .add(self.axis(2).scale(l.z));
        }
        pub fn intersectsSphere(self: Self, center: V3, radius: T) bool {
            return self.closestPoint(center).distanceSq(center) <= radius * radius;
        }
        pub fn corners(self: Self) [8]V3 {
            const ex = self.axis(0).scale(self.half.x);
            const ey = self.axis(1).scale(self.half.y);
            const ez = self.axis(2).scale(self.half.z);
            var out: [8]V3 = undefined;
            inline for (0..8) |i| {
                const sx: T = if (i & 1 != 0) 1 else -1;
                const sy: T = if (i & 2 != 0) 1 else -1;
                const sz: T = if (i & 4 != 0) 1 else -1;
                out[i] = self.center.add(ex.scale(sx)).add(ey.scale(sy)).add(ez.scale(sz));
            }
            return out;
        }
        /// Separating-axis test against another OBB.
        pub fn intersects(a: Self, b: Self) bool {
            const eps = 1e-6;
            var rot: [3][3]T = undefined;
            var absr: [3][3]T = undefined;
            inline for (0..3) |i| inline for (0..3) |j| {
                rot[i][j] = a.axis(i).dot(b.axis(j));
                absr[i][j] = @abs(rot[i][j]) + eps;
            };
            const tv = b.center.sub(a.center);
            const t = V3.init(tv.dot(a.axis(0)), tv.dot(a.axis(1)), tv.dot(a.axis(2))).toArray();
            const ae = a.half.toArray();
            const be = b.half.toArray();
            // 3 axes of A, 3 axes of B
            inline for (0..3) |i| {
                const ra = ae[i];
                const rb = be[0] * absr[i][0] + be[1] * absr[i][1] + be[2] * absr[i][2];
                if (@abs(t[i]) > ra + rb) return false;
            }
            inline for (0..3) |j| {
                const ra = ae[0] * absr[0][j] + ae[1] * absr[1][j] + ae[2] * absr[2][j];
                const rb = be[j];
                const tj = @abs(t[0] * rot[0][j] + t[1] * rot[1][j] + t[2] * rot[2][j]);
                if (tj > ra + rb) return false;
            }
            // 9 cross-product axes
            inline for (0..3) |i| {
                inline for (0..3) |j| {
                    const a1 = (i + 1) % 3;
                    const a2 = (i + 2) % 3;
                    const b1 = (j + 1) % 3;
                    const b2 = (j + 2) % 3;
                    const ra = ae[a1] * absr[a2][j] + ae[a2] * absr[a1][j];
                    const rb = be[b1] * absr[i][b2] + be[b2] * absr[i][b1];
                    const tj = @abs(t[a2] * rot[a1][j] - t[a1] * rot[a2][j]);
                    if (tj > ra + rb) return false;
                }
            }
            return true;
        }
    };
}

// === Capsule (swept sphere) =================================================

pub fn Capsule(comptime T: type) type {
    comptime sc.requireFloat(T);
    return struct {
        a: V3,
        b: V3,
        radius: T,

        const Self = @This();
        const V3 = vec.Vec(3, T);

        pub fn init(a: V3, b: V3, radius: T) Self {
            return .{ .a = a, .b = b, .radius = radius };
        }
        fn closestOnSegment(self: Self, p: V3) V3 {
            const ab = self.b.sub(self.a);
            const t = std.math.clamp(p.sub(self.a).dot(ab) / ab.dot(ab), 0, 1);
            return self.a.add(ab.scale(t));
        }
        pub fn contains(self: Self, p: V3) bool {
            return self.closestOnSegment(p).distanceSq(p) <= self.radius * self.radius;
        }
        pub fn distance(self: Self, p: V3) T {
            return @max(self.closestOnSegment(p).distance(p) - self.radius, 0);
        }
        pub fn intersectsSphere(self: Self, center: V3, radius: T) bool {
            const r = self.radius + radius;
            return self.closestOnSegment(center).distanceSq(center) <= r * r;
        }
        pub fn intersects(self: Self, other: Self) bool {
            const d2 = segmentSegmentDistSq(self.a, self.b, other.a, other.b);
            const r = self.radius + other.radius;
            return d2 <= r * r;
        }
    };
}

/// Squared distance between two segments (Ericson, RTCD).
fn segmentSegmentDistSq(p1: anytype, q1: @TypeOf(p1), p2: @TypeOf(p1), q2: @TypeOf(p1)) @TypeOf(p1).Element {
    const d1 = q1.sub(p1);
    const d2 = q2.sub(p2);
    const r = p1.sub(p2);
    const a = d1.dot(d1);
    const e = d2.dot(d2);
    const f = d2.dot(r);
    var s: @TypeOf(p1).Element = 0;
    var t: @TypeOf(p1).Element = 0;
    if (a <= 1e-12 and e <= 1e-12) return r.dot(r);
    if (a <= 1e-12) {
        t = std.math.clamp(f / e, 0, 1);
    } else {
        const c = d1.dot(r);
        if (e <= 1e-12) {
            s = std.math.clamp(-c / a, 0, 1);
        } else {
            const b = d1.dot(d2);
            const denom = a * e - b * b;
            if (denom != 0) s = std.math.clamp((b * f - c * e) / denom, 0, 1);
            t = (b * s + f) / e;
            if (t < 0) {
                t = 0;
                s = std.math.clamp(-c / a, 0, 1);
            } else if (t > 1) {
                t = 1;
                s = std.math.clamp((b - c) / a, 0, 1);
            }
        }
    }
    const c1 = p1.add(d1.scale(s));
    const c2 = p2.add(d2.scale(t));
    return c1.distanceSq(c2);
}

// === Frustum (6 planes) =====================================================

pub fn Frustum(comptime T: type) type {
    comptime sc.requireFloat(T);
    return struct {
        planes: [6]V4,

        const Self = @This();
        const V3 = vec.Vec(3, T);
        const V4 = vec.Vec(4, T);
        const M4 = mat.Mat(4, 4, T);

        /// Extract the six planes from a view-projection matrix.
        pub fn fromViewProj(vp: M4) Self {
            return .{ .planes = frustumPlanes(vp) };
        }
        pub fn containsPoint(self: Self, p: V3) bool {
            return pointInFrustum(self.planes, p);
        }
        pub fn containsSphere(self: Self, s: Sphere(T)) bool {
            return sphereInFrustum(self.planes, s);
        }
        pub fn intersectsAabb(self: Self, box: Aabb(T)) bool {
            return aabbInFrustum(self.planes, box);
        }
    };
}

const testing = std.testing;
const Vec3 = vec.Vec3;
const Aabbf = Aabb(f32);
const Spheref = Sphere(f32);
const tf = @import("transform.zig").transform;
const cam = @import("transform.zig").camera;

test "aabb" {
    const pts = [_]Vec3{ Vec3.init(1, 2, 3), Vec3.init(-1, 0, 5), Vec3.init(2, -2, 1) };
    const b = Aabbf.fromPoints(&pts);
    try testing.expect(b.min.eql(Vec3.init(-1, -2, 1)));
    try testing.expect(b.max.eql(Vec3.init(2, 2, 5)));
    try testing.expect(b.contains(Vec3.init(0, 0, 3)));
    try testing.expect(b.intersects(Aabbf.init(Vec3.init(1, 1, 1), Vec3.init(3, 3, 3))));
    try testing.expectApproxEqAbs(@as(f32, 4), b.intersectRay(Vec3.init(-5, 0, 3), Vec3.init(1, 0, 0)).?, 1e-5);
}
test "sphere" {
    const pts = [_]Vec3{ Vec3.init(2, 0, 0), Vec3.init(-2, 0, 0), Vec3.init(0, 1, 0) };
    const s = Spheref.fromPoints(&pts);
    for (pts) |p| try testing.expect(s.center.distance(p) <= s.radius + 1e-4);
    try testing.expect(s.intersectsAabb(Aabbf.init(Vec3.splat(-0.5), Vec3.splat(0.5))));
}
test "frustum culling" {
    const view = cam.lookAt(Vec3.init(0, 0, 5), Vec3.splat(0), Vec3.init(0, 1, 0));
    const proj = cam.perspective(std.math.pi / 3.0, 1.0, 0.1, 100.0);
    const planes = frustumPlanes(proj.mul(view));
    try testing.expect(pointInFrustum(planes, Vec3.splat(0)));
    try testing.expect(!pointInFrustum(planes, Vec3.init(0, 0, 60)));
    try testing.expect(aabbInFrustum(planes, Aabbf.init(Vec3.splat(-1), Vec3.splat(1))));
    try testing.expect(!aabbInFrustum(planes, Aabbf.init(Vec3.splat(50), Vec3.splat(52))));
    try testing.expect(frustumBox(frustumCorners(proj.mul(view))).contains(Vec3.splat(0)));
}
test "bounds are generic over f64" {
    const V = vec.Vec(3, f64);
    const b = Aabb(f64).init(V.splat(-1), V.splat(2));
    try testing.expect(b.contains(V.splat(0)));
    try testing.expectApproxEqAbs(@as(f64, 0.5), b.center().x, 1e-12);
    const s = Sphere(f64).fromAabb(b);
    try testing.expect(s.radius > 0);
}
test "obb" {
    const M3 = mat.Mat3;
    const rot = M3.fromColumns(.{ Vec3.init(0, 1, 0), Vec3.init(-1, 0, 0), Vec3.init(0, 0, 1) }); // 90° about Z
    const o = Obb(f32).init(Vec3.splat(0), Vec3.init(2, 1, 1), rot);
    try testing.expect(o.contains(Vec3.init(0, 1.9, 0))); // long axis now points +Y
    try testing.expect(!o.contains(Vec3.init(1.5, 0, 0)));
    try testing.expect(o.intersectsSphere(Vec3.init(0, 2.5, 0), 0.7));
    // OBB vs OBB
    const a = Obb(f32).fromAabb(Aabbf.init(Vec3.splat(-1), Vec3.splat(1)));
    try testing.expect(a.intersects(Obb(f32).fromAabb(Aabbf.init(Vec3.splat(0.5), Vec3.splat(2)))));
    try testing.expect(!a.intersects(Obb(f32).fromAabb(Aabbf.init(Vec3.splat(3), Vec3.splat(4)))));
}
test "capsule" {
    const cap = Capsule(f32).init(Vec3.init(0, 0, 0), Vec3.init(0, 4, 0), 1);
    try testing.expect(cap.contains(Vec3.init(0.5, 2, 0)));
    try testing.expect(!cap.contains(Vec3.init(2, 2, 0)));
    try testing.expectApproxEqAbs(@as(f32, 1), cap.distance(Vec3.init(2, 2, 0)), 1e-6); // 2 - radius
    try testing.expect(cap.intersects(Capsule(f32).init(Vec3.init(1.5, 2, 0), Vec3.init(3, 2, 0), 0.6)));
}
test "frustum type" {
    const view = cam.lookAt(Vec3.init(0, 0, 5), Vec3.splat(0), Vec3.init(0, 1, 0));
    const proj = cam.perspective(std.math.pi / 3.0, 1.0, 0.1, 100.0);
    const f = Frustum(f32).fromViewProj(proj.mul(view));
    try testing.expect(f.containsPoint(Vec3.splat(0)));
    try testing.expect(!f.containsPoint(Vec3.init(0, 0, 60)));
    try testing.expect(f.intersectsAabb(Aabbf.init(Vec3.splat(-1), Vec3.splat(1))));
}
