//! Interpolating splines (GLM `gtx/spline`). Generic over vector type.

const std = @import("std");
const sc = @import("meta.zig");

pub fn catmullRom(v1: anytype, v2: @TypeOf(v1), v3: @TypeOf(v1), v4: @TypeOf(v1), s: sc.Element(@TypeOf(v1))) @TypeOf(v1) {
    const s2 = s * s;
    const s3 = s2 * s;
    const f1 = -s3 + 2 * s2 - s;
    const f2 = 3 * s3 - 5 * s2 + 2;
    const f3 = -3 * s3 + 4 * s2 + s;
    const f4 = s3 - s2;
    return v1.scale(f1).add(v2.scale(f2)).add(v3.scale(f3)).add(v4.scale(f4)).scale(0.5);
}

pub fn hermite(v1: anytype, t1: @TypeOf(v1), v2: @TypeOf(v1), t2: @TypeOf(v1), s: sc.Element(@TypeOf(v1))) @TypeOf(v1) {
    const s2 = s * s;
    const s3 = s2 * s;
    const f1 = 2 * s3 - 3 * s2 + 1;
    const f2 = -2 * s3 + 3 * s2;
    const f3 = s3 - 2 * s2 + s;
    const f4 = s3 - s2;
    return v1.scale(f1).add(v2.scale(f2)).add(t1.scale(f3)).add(t2.scale(f4));
}

pub fn cubic(v1: anytype, v2: @TypeOf(v1), v3: @TypeOf(v1), v4: @TypeOf(v1), s: sc.Element(@TypeOf(v1))) @TypeOf(v1) {
    const s2 = s * s;
    const s3 = s2 * s;
    return v1.scale(s3).add(v2.scale(s2)).add(v3.scale(s)).add(v4);
}

/// Cubic Bézier point at `s ∈ [0,1]` from 4 control points.
pub fn bezier(p0: anytype, p1: @TypeOf(p0), p2: @TypeOf(p0), p3: @TypeOf(p0), s: sc.Element(@TypeOf(p0))) @TypeOf(p0) {
    const u = 1 - s;
    return p0.scale(u * u * u).add(p1.scale(3 * u * u * s)).add(p2.scale(3 * u * s * s)).add(p3.scale(s * s * s));
}
/// Cubic Bézier via de Casteljau's recursive-lerp construction (same result as `bezier`).
pub fn deCasteljau(p0: anytype, p1: @TypeOf(p0), p2: @TypeOf(p0), p3: @TypeOf(p0), s: sc.Element(@TypeOf(p0))) @TypeOf(p0) {
    const a = p0.lerp(p1, s);
    const b = p1.lerp(p2, s);
    const c = p2.lerp(p3, s);
    return a.lerp(b, s).lerp(b.lerp(c, s), s);
}

/// Quadratic Bézier point at `s ∈ [0,1]` from 3 control points.
pub fn bezierQuad(p0: anytype, p1: @TypeOf(p0), p2: @TypeOf(p0), s: sc.Element(@TypeOf(p0))) @TypeOf(p0) {
    const u = 1 - s;
    return p0.scale(u * u).add(p1.scale(2 * u * s)).add(p2.scale(s * s));
}
/// Tangent (1st derivative) of a quadratic Bézier at `s`.
pub fn bezierQuadDerivative(p0: anytype, p1: @TypeOf(p0), p2: @TypeOf(p0), s: sc.Element(@TypeOf(p0))) @TypeOf(p0) {
    const u = 1 - s;
    return p1.sub(p0).scale(2 * u).add(p2.sub(p1).scale(2 * s));
}
/// Tangent (1st derivative) of a cubic Bézier at `s`.
pub fn bezierDerivative(p0: anytype, p1: @TypeOf(p0), p2: @TypeOf(p0), p3: @TypeOf(p0), s: sc.Element(@TypeOf(p0))) @TypeOf(p0) {
    const u = 1 - s;
    return p1.sub(p0).scale(3 * u * u)
        .add(p2.sub(p1).scale(6 * u * s))
        .add(p3.sub(p2).scale(3 * s * s));
}
/// Approximate arc length of a cubic Bézier (sampled into `segments` chords).
pub fn bezierArcLength(p0: anytype, p1: @TypeOf(p0), p2: @TypeOf(p0), p3: @TypeOf(p0), segments: u32) sc.Element(@TypeOf(p0)) {
    const E = sc.Element(@TypeOf(p0));
    var len: E = 0;
    var prev = p0;
    var i: u32 = 1;
    while (i <= segments) : (i += 1) {
        const s: E = @as(E, @floatFromInt(i)) / @as(E, @floatFromInt(segments));
        const cur = bezier(p0, p1, p2, p3, s);
        len += cur.distance(prev);
        prev = cur;
    }
    return len;
}

/// Uniform cubic B-spline point at `s ∈ [0,1]` over 4 control points. Unlike
/// Bézier/Catmull-Rom it does *not* pass through the control points, but stays
/// C² continuous when chained.
pub fn bspline(p0: anytype, p1: @TypeOf(p0), p2: @TypeOf(p0), p3: @TypeOf(p0), s: sc.Element(@TypeOf(p0))) @TypeOf(p0) {
    const s2 = s * s;
    const s3 = s2 * s;
    const b0 = (-s3 + 3 * s2 - 3 * s + 1) / 6.0;
    const b1 = (3 * s3 - 6 * s2 + 4) / 6.0;
    const b2 = (-3 * s3 + 3 * s2 + 3 * s + 1) / 6.0;
    const b3 = s3 / 6.0;
    return p0.scale(b0).add(p1.scale(b1)).add(p2.scale(b2)).add(p3.scale(b3));
}

const testing = std.testing;
const Vec2 = @import("vec.zig").Vec2;
test "spline catmullRom passes through control points" {
    const p0 = Vec2.init(0, 0);
    const p1 = Vec2.init(1, 1);
    const p2 = Vec2.init(2, 0);
    const p3 = Vec2.init(3, 1);
    try testing.expect(catmullRom(p0, p1, p2, p3, 0).approxEql(p1, 1e-5));
    try testing.expect(catmullRom(p0, p1, p2, p3, 1).approxEql(p2, 1e-5));
}
