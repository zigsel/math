//! Vector geometry — `math.geom`. The operations *without* a natural `Vec`
//! method form: non-Euclidean norms, angles, vector rotation, projection onto
//! vectors, triple products, and geometric queries. (`dot`/`cross`/`length`/
//! `distance`/`normalize` are `Vec` methods — `a.dot(b)`, `v.length()`.)

const std = @import("std");
const sc = @import("meta.zig");
const vec = @import("vec.zig");
const mat = @import("mat.zig");
const fast = @import("fast.zig");
const transform = @import("transform.zig").transform;

// --- GLSL geometric ---------------------------------------------------------
//
// `length`, `distance`, `dot`, `cross`, `normalize` are `Vec` *methods*
// (`a.dot(b)`, `v.length()`, …) — use those. Only the operations with no
// natural method form live here as free functions.

/// Returns `n` if `dot(nref, i) < 0`, else `-n` (orient a normal away from `i`).
pub fn faceForward(n: anytype, i: @TypeOf(n), nref: @TypeOf(n)) @TypeOf(n) {
    return if (nref.dot(i) < 0) n else n.neg();
}
/// Reflect incident vector `i` about unit normal `n`.
pub fn reflect(i: anytype, n: @TypeOf(i)) @TypeOf(i) {
    return i.sub(n.scale(2 * n.dot(i)));
}
/// Refract incident vector `i` through unit normal `n` with ratio `eta`.
/// Returns the zero vector on total internal reflection.
pub fn refract(i: anytype, n: @TypeOf(i), eta: sc.Element(@TypeOf(i))) @TypeOf(i) {
    const T = @TypeOf(i);
    const d = n.dot(i);
    const k = 1.0 - eta * eta * (1.0 - d * d);
    if (k < 0) return T.splat(0);
    return i.scale(eta).sub(n.scale(eta * d + @sqrt(k)));
}

// --- norms (gtx/norm) -------------------------------------------------------
//
// The L2 norm and squared norms are `Vec` methods (`v.length()`,
// `v.lengthSq()`, `a.distanceSq(b)`). Only the non-Euclidean norms live here.

pub fn l1Norm(v: anytype) sc.Element(@TypeOf(v)) {
    return @reduce(.Add, @abs(v.simd()));
}
pub fn lMaxNorm(v: anytype) sc.Element(@TypeOf(v)) {
    return @reduce(.Max, @abs(v.simd()));
}
pub fn lxNorm(v: anytype, depth: u32) sc.Element(@TypeOf(v)) {
    const T = @TypeOf(v);
    const E = T.Element;
    const s = v.simd();
    var acc: E = 0;
    const d: E = @floatFromInt(depth);
    inline for (0..T.dim) |i| acc += std.math.pow(E, @abs(s[i]), d);
    return std.math.pow(E, acc, 1.0 / d);
}

// --- angles (gtx/vector_angle) ----------------------------------------------

/// Unsigned angle (radians) between two vectors.
pub fn angle(a: anytype, b: @TypeOf(a)) sc.Element(@TypeOf(a)) {
    const E = sc.Element(@TypeOf(a));
    const d = a.normalize().dot(b.normalize());
    return std.math.acos(std.math.clamp(d, @as(E, -1), @as(E, 1)));
}
/// Signed angle between two 2-D vectors.
pub fn orientedAngle2(a: anytype, b: @TypeOf(a)) sc.Element(@TypeOf(a)) {
    const ang = angle(a, b);
    const c = a.x * b.y - a.y * b.x;
    return if (c < 0) -ang else ang;
}
/// Signed angle between two 3-D vectors, sign relative to `ref`.
pub fn orientedAngle3(a: anytype, b: @TypeOf(a), ref: @TypeOf(a)) sc.Element(@TypeOf(a)) {
    const ang = angle(a, b);
    return if (a.cross(b).dot(ref) < 0) -ang else ang;
}

// --- rotate vectors (gtx/rotate_vector) -------------------------------------

/// Rotation matrix that takes `up` to `normal`.
pub fn orientation(normal: vec.Vec3, up: vec.Vec3) mat.Mat4 {
    if (normal.approxEql(up, 1e-6)) return mat.Mat4.identity();
    const axis = up.cross(normal);
    const ang = std.math.acos(std.math.clamp(normal.dot(up), -1, 1));
    return transform.rotation(ang, axis);
}
/// Rotate a 3-D vector about an arbitrary axis (Rodrigues' rotation).
pub fn rotate(v: anytype, ang: sc.Element(@TypeOf(v)), axis: @TypeOf(v)) @TypeOf(v) {
    const c = @cos(ang);
    const s = @sin(ang);
    const k = axis.normalize();
    return v.scale(c).add(k.cross(v).scale(s)).add(k.scale(k.dot(v) * (1 - c)));
}
pub fn rotateX(v: anytype, ang: sc.Element(@TypeOf(v))) @TypeOf(v) {
    const T = @TypeOf(v);
    const c = @cos(ang);
    const s = @sin(ang);
    return T.init(v.x, v.y * c - v.z * s, v.y * s + v.z * c);
}
pub fn rotateY(v: anytype, ang: sc.Element(@TypeOf(v))) @TypeOf(v) {
    const T = @TypeOf(v);
    const c = @cos(ang);
    const s = @sin(ang);
    return T.init(v.x * c + v.z * s, v.y, -v.x * s + v.z * c);
}
pub fn rotateZ(v: anytype, ang: sc.Element(@TypeOf(v))) @TypeOf(v) {
    const T = @TypeOf(v);
    const c = @cos(ang);
    const s = @sin(ang);
    return T.init(v.x * c - v.y * s, v.x * s + v.y * c, v.z);
}
/// Rotate a 2-D vector.
pub fn rotate2(v: anytype, ang: sc.Element(@TypeOf(v))) @TypeOf(v) {
    const T = @TypeOf(v);
    const c = @cos(ang);
    const s = @sin(ang);
    return T.init(v.x * c - v.y * s, v.x * s + v.y * c);
}

// --- projection / perpendicular ---------------------------------------------

/// Project `a` onto `b`.
pub fn proj(a: anytype, b: @TypeOf(a)) @TypeOf(a) {
    return b.scale(a.dot(b) / b.dot(b));
}
/// Component of `a` perpendicular to `b`.
pub fn perp(a: anytype, b: @TypeOf(a)) @TypeOf(a) {
    return a.sub(proj(a, b));
}

// --- points / normals / handedness ------------------------------------------

/// Closest point to `point` on the segment `a`–`b`.
pub fn closestPointOnLine(point: anytype, a: @TypeOf(point), b: @TypeOf(point)) @TypeOf(point) {
    const E = sc.Element(@TypeOf(point));
    const ab = b.sub(a);
    const t = std.math.clamp(point.sub(a).dot(ab) / ab.dot(ab), @as(E, 0), @as(E, 1));
    return a.add(ab.scale(t));
}
/// 2-D cross product → scalar (z-component of the 3-D cross).
pub fn cross2(a: anytype, b: @TypeOf(a)) sc.Element(@TypeOf(a)) {
    return a.x * b.y - a.y * b.x;
}
pub fn triangleNormal(p1: anytype, p2: @TypeOf(p1), p3: @TypeOf(p1)) @TypeOf(p1) {
    return p2.sub(p1).cross(p3.sub(p1)).normalize();
}
pub fn rightHanded(tangent: anytype, binormal: @TypeOf(tangent), normal: @TypeOf(tangent)) bool {
    return normal.cross(tangent).dot(binormal) > 0;
}
pub fn leftHanded(tangent: anytype, binormal: @TypeOf(tangent), normal: @TypeOf(tangent)) bool {
    return normal.cross(tangent).dot(binormal) < 0;
}

// --- orthonormalize / triple products ---------------------------------------

/// Orthonormalize `x` with respect to `y` (Gram–Schmidt).
pub fn orthonormalize(x: anytype, y: @TypeOf(x)) @TypeOf(x) {
    return x.sub(y.scale(y.dot(x))).normalize();
}
/// Orthonormalize the columns of a 3x3 matrix.
pub fn orthonormalizeMat3(m: anytype) @TypeOf(m) {
    const M = @TypeOf(m);
    const c0 = m.cols[0].normalize();
    const c1 = m.cols[1].sub(c0.scale(c0.dot(m.cols[1]))).normalize();
    const c2 = c0.cross(c1);
    return M.fromColumns(.{ c0, c1, c2 });
}
/// Scalar triple product `(a × b) · c`.
pub fn mixedProduct(a: anytype, b: @TypeOf(a), c: @TypeOf(a)) sc.Element(@TypeOf(a)) {
    return a.cross(b).dot(c);
}

// --- queries (gtx/vector_query) ---------------------------------------------

/// True if both vectors are unit length and mutually orthogonal.
pub fn areOrthonormal(a: anytype, b: @TypeOf(a), eps: sc.Element(@TypeOf(a))) bool {
    return isNormalized(a, eps) and isNormalized(b, eps) and areOrthogonal(a, b, eps);
}
/// Per-component "is near zero" mask.
pub fn isCompNull(v: anytype, eps: sc.Element(@TypeOf(v))) vec.Vec(@TypeOf(v).dim, bool) {
    const T = @TypeOf(v);
    const e: @Vector(T.dim, T.Element) = @splat(eps);
    return vec.Vec(T.dim, bool).fromSimd(@abs(v.simd()) < e);
}
pub fn isNull(v: anytype, eps: sc.Element(@TypeOf(v))) bool {
    const T = @TypeOf(v);
    const e: @Vector(T.dim, T.Element) = @splat(eps);
    return @reduce(.And, @abs(v.simd()) <= e);
}
pub fn isNormalized(v: anytype, eps: sc.Element(@TypeOf(v))) bool {
    return @abs(v.lengthSq() - 1) <= eps * 2;
}
pub fn areCollinear(a: anytype, b: @TypeOf(a), eps: sc.Element(@TypeOf(a))) bool {
    return a.cross(b).lengthSq() <= eps;
}
pub fn areOrthogonal(a: anytype, b: @TypeOf(a), eps: sc.Element(@TypeOf(a))) bool {
    return @abs(a.dot(b)) <= eps;
}

// --- associated min/max + normalized dot ------------------------------------

/// Value paired with the smaller key.
pub fn associatedMin(key_a: anytype, val_a: anytype, key_b: @TypeOf(key_a), val_b: @TypeOf(val_a)) @TypeOf(val_a) {
    return if (key_a < key_b) val_a else val_b;
}
/// Value paired with the larger key.
pub fn associatedMax(key_a: anytype, val_a: anytype, key_b: @TypeOf(key_a), val_b: @TypeOf(val_a)) @TypeOf(val_a) {
    return if (key_a > key_b) val_a else val_b;
}
/// `dot(a,b) / (|a||b|)` — cosine of the angle, without normalizing first.
pub fn normalizeDot(a: anytype, b: @TypeOf(a)) sc.Element(@TypeOf(a)) {
    return a.dot(b) / @sqrt(a.lengthSq() * b.lengthSq());
}
/// Same, using the fast inverse square root.
pub fn fastNormalizeDot(a: anytype, b: @TypeOf(a)) sc.Element(@TypeOf(a)) {
    return a.dot(b) * fast.inverseSqrt(@floatCast(a.lengthSq() * b.lengthSq()));
}

const testing = std.testing;
const Vec2 = vec.Vec2;
const Vec3 = vec.Vec3;
const Vec4 = vec.Vec4;

test "geom: glsl geometric" {
    const a = Vec3.init(1, 0, 0);
    const b = Vec3.init(0, 1, 0);
    // length/distance/dot/cross/normalize are Vec methods now
    try testing.expectEqual(@as(f32, 0), a.dot(b));
    try testing.expect(a.cross(b).eql(Vec3.init(0, 0, 1)));
    try testing.expectApproxEqAbs(@as(f32, 1), a.length(), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, std.math.sqrt2), a.distance(b), 1e-6);
    try testing.expect(faceForward(a, a, a).eql(a.neg())); // nref·i >= 0 → -n
    const i = Vec3.init(1, -1, 0).normalize();
    try testing.expect(reflect(i, Vec3.init(0, 1, 0)).approxEql(Vec3.init(1, 1, 0).normalize(), 1e-6));
    try testing.expect(refract(Vec3.init(0, -1, 0), Vec3.init(0, 1, 0), 1.0).approxEql(Vec3.init(0, -1, 0), 1e-6));
}

test "geom: norms" {
    const v = Vec3.init(3, 4, 0);
    try testing.expectEqual(@as(f32, 25), v.lengthSq()); // (was geom.length2)
    try testing.expectEqual(@as(f32, 25), Vec3.splat(0).distanceSq(v));
    try testing.expectEqual(@as(f32, 7), l1Norm(v));
    try testing.expectApproxEqAbs(@as(f32, 5), v.length(), 1e-6); // (was geom.l2Norm)
    try testing.expectEqual(@as(f32, 4), lMaxNorm(v));
    try testing.expectApproxEqAbs(@as(f32, 5), lxNorm(v, 2), 1e-4);
}

test "geom: angles & rotation" {
    try testing.expectApproxEqAbs(@as(f32, std.math.pi / 2.0), angle(Vec3.init(1, 0, 0), Vec3.init(0, 1, 0)), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, -std.math.pi / 2.0), orientedAngle2(Vec2.init(0, 1), Vec2.init(1, 0)), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, std.math.pi / 2.0), orientedAngle3(Vec3.init(1, 0, 0), Vec3.init(0, 1, 0), Vec3.init(0, 0, 1)), 1e-6);
    try testing.expect(rotateZ(Vec3.init(1, 0, 0), std.math.pi / 2.0).approxEql(Vec3.init(0, 1, 0), 1e-6));
    try testing.expect(rotateX(Vec3.init(0, 1, 0), std.math.pi / 2.0).approxEql(Vec3.init(0, 0, 1), 1e-6));
    try testing.expect(rotateY(Vec3.init(0, 0, 1), std.math.pi / 2.0).approxEql(Vec3.init(1, 0, 0), 1e-6));
    try testing.expect(rotate2(Vec2.init(1, 0), std.math.pi / 2.0).approxEql(Vec2.init(0, 1), 1e-6));
    try testing.expect(rotate(Vec3.init(1, 0, 0), std.math.pi / 2.0, Vec3.init(0, 0, 1)).approxEql(Vec3.init(0, 1, 0), 1e-6));
    const o = orientation(Vec3.init(0, 0, 1), Vec3.init(0, 1, 0)); // maps up → normal
    try testing.expect(o.mulVec(Vec4.init(0, 1, 0, 0)).swizzle("xyz").approxEql(Vec3.init(0, 0, 1), 1e-5));
}

test "geom: projection / points / normals" {
    try testing.expect(proj(Vec3.init(2, 3, 0), Vec3.init(1, 0, 0)).approxEql(Vec3.init(2, 0, 0), 1e-6));
    try testing.expect(perp(Vec3.init(2, 3, 0), Vec3.init(1, 0, 0)).approxEql(Vec3.init(0, 3, 0), 1e-6));
    try testing.expect(closestPointOnLine(Vec3.init(0.5, 5, 0), Vec3.splat(0), Vec3.init(1, 0, 0)).approxEql(Vec3.init(0.5, 0, 0), 1e-6));
    try testing.expectEqual(@as(f32, 1), cross2(Vec2.init(1, 0), Vec2.init(0, 1)));
    try testing.expect(triangleNormal(Vec3.splat(0), Vec3.init(1, 0, 0), Vec3.init(0, 1, 0)).approxEql(Vec3.init(0, 0, 1), 1e-6));
    try testing.expect(rightHanded(Vec3.init(1, 0, 0), Vec3.init(0, 1, 0), Vec3.init(0, 0, 1)));
    try testing.expect(leftHanded(Vec3.init(1, 0, 0), Vec3.init(0, 1, 0), Vec3.init(0, 0, -1)));
}

test "geom: orthonormalize / triple / queries" {
    try testing.expect(orthonormalize(Vec3.init(1, 1, 0), Vec3.init(1, 0, 0)).approxEql(Vec3.init(0, 1, 0), 1e-6));
    const m = mat.Mat3.fromColumns(.{ Vec3.init(2, 0, 0), Vec3.init(1, 3, 0), Vec3.init(0, 0, 5) });
    const on = orthonormalizeMat3(m);
    try testing.expect(on.cols[0].approxEql(Vec3.init(1, 0, 0), 1e-6));
    try testing.expect(on.cols[1].approxEql(Vec3.init(0, 1, 0), 1e-6));
    try testing.expectEqual(@as(f32, 1), mixedProduct(Vec3.init(1, 0, 0), Vec3.init(0, 1, 0), Vec3.init(0, 0, 1)));
    try testing.expect(isNull(Vec3.splat(0), 1e-6));
    try testing.expect(isNormalized(Vec3.init(0, 0, 1), 1e-6));
    try testing.expect(areOrthogonal(Vec3.init(1, 0, 0), Vec3.init(0, 1, 0), 1e-6));
    try testing.expect(areOrthonormal(Vec3.init(1, 0, 0), Vec3.init(0, 1, 0), 1e-6));
    try testing.expect(areCollinear(Vec3.init(1, 2, 3), Vec3.init(2, 4, 6), 1e-4));
    const cn = isCompNull(Vec3.init(0, 1, 0), 1e-6);
    try testing.expect(cn.x and !cn.y and cn.z);
}

test "geom: associated min/max + normalized dot" {
    try testing.expectEqual(@as(u32, 7), associatedMin(@as(f32, 1.0), @as(u32, 7), @as(f32, 2.0), @as(u32, 9)));
    try testing.expectEqual(@as(u32, 9), associatedMax(@as(f32, 1.0), @as(u32, 7), @as(f32, 2.0), @as(u32, 9)));
    try testing.expectApproxEqAbs(@as(f32, 0), normalizeDot(Vec3.init(2, 0, 0), Vec3.init(0, 5, 0)), 1e-6);
    try testing.expectApproxEqRel(@as(f32, 1), fastNormalizeDot(Vec3.init(3, 0, 0), Vec3.init(9, 0, 0)), 1e-2);
}
