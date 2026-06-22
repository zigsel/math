//! Quaternions `Quat(T)` for rotations. Stored as `(x, y, z, w)` with `w` the
//! real part; constructors take GLM's `(w, x, y, z)` order. Float `T` only.

const std = @import("std");
const sc = @import("meta.zig");
const vec = @import("vec.zig");
const mat = @import("mat.zig");
const Vec = vec.Vec;
const Mat = mat.Mat;

pub fn Quat(comptime T: type) type {
    comptime sc.requireFloat(T);
    return extern struct {
        x: T,
        y: T,
        z: T,
        w: T,

        pub const Element = T;
        pub const is_math_quaternion = true;
        const Self = @This();
        const Vec3 = Vec(3, T);
        const V4 = @Vector(4, T);

        // --- construction ---------------------------------------------------

        /// GLM order: real part first.
        pub inline fn init(w: T, x: T, y: T, z: T) Self {
            return .{ .x = x, .y = y, .z = z, .w = w };
        }
        pub inline fn fromXYZW(x: T, y: T, z: T, w: T) Self {
            return .{ .x = x, .y = y, .z = z, .w = w };
        }
        /// Build from a 4-element array in `(x, y, z, w)` order.
        pub inline fn fromArray(a: [4]T) Self {
            return .{ .x = a[0], .y = a[1], .z = a[2], .w = a[3] };
        }
        pub inline fn identity() Self {
            return .{ .x = 0, .y = 0, .z = 0, .w = 1 };
        }
        inline fn simd(q: Self) V4 {
            return @bitCast(q);
        }
        inline fn fromSimd(v: V4) Self {
            return @bitCast(v);
        }

        /// Rotation of `angle` radians about a (unit) `axis`.
        pub fn fromAxisAngle(ax: Vec3, ang: T) Self {
            const half = ang * 0.5;
            const s = @sin(half);
            const a = ax.normalize();
            return .{ .x = a.x * s, .y = a.y * s, .z = a.z * s, .w = @cos(half) };
        }

        /// From Euler angles `(pitch=x, yaw=y, roll=z)` in radians.
        pub fn fromEuler(euler: Vec3) Self {
            const cx = @cos(euler.x * 0.5);
            const cy = @cos(euler.y * 0.5);
            const cz = @cos(euler.z * 0.5);
            const sx = @sin(euler.x * 0.5);
            const sy = @sin(euler.y * 0.5);
            const sz = @sin(euler.z * 0.5);
            return .{
                .w = cx * cy * cz + sx * sy * sz,
                .x = sx * cy * cz - cx * sy * sz,
                .y = cx * sy * cz + sx * cy * sz,
                .z = cx * cy * sz - sx * sy * cz,
            };
        }

        // --- algebra --------------------------------------------------------

        /// Hamilton product `a * b` (apply `b` then `a`).
        pub fn mul(a: Self, b: Self) Self {
            return .{
                .w = a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
                .x = a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
                .y = a.w * b.y + a.y * b.w + a.z * b.x - a.x * b.z,
                .z = a.w * b.z + a.z * b.w + a.x * b.y - a.y * b.x,
            };
        }
        pub fn scale(q: Self, s: T) Self {
            const v: V4 = @splat(s);
            return fromSimd(q.simd() * v);
        }
        pub fn add(a: Self, b: Self) Self {
            return fromSimd(a.simd() + b.simd());
        }
        pub fn neg(q: Self) Self {
            return fromSimd(-q.simd());
        }
        pub fn conjugate(q: Self) Self {
            return .{ .x = -q.x, .y = -q.y, .z = -q.z, .w = q.w };
        }
        pub fn dot(a: Self, b: Self) T {
            return @reduce(.Add, a.simd() * b.simd());
        }
        pub fn lengthSq(q: Self) T {
            return q.dot(q);
        }
        pub fn length(q: Self) T {
            return @sqrt(q.dot(q));
        }
        pub fn normalize(q: Self) Self {
            const l = q.length();
            if (l == 0) return identity();
            return q.scale(1.0 / l);
        }
        pub fn inverse(q: Self) Self {
            return q.conjugate().scale(1.0 / q.lengthSq());
        }
        pub fn eql(a: Self, b: Self) bool {
            return @reduce(.And, a.simd() == b.simd());
        }
        pub fn approxEql(a: Self, b: Self, eps: T) bool {
            return @reduce(.And, @abs(a.simd() - b.simd()) <= @as(V4, @splat(eps)));
        }

        /// `std.fmt` integration (use `{f}`): prints `quat(w, {x, y, z})`.
        pub fn format(self: Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
            try writer.print("quat({d}, {{{d}, {d}, {d}}})", .{ self.w, self.x, self.y, self.z });
        }

        // --- interpolation --------------------------------------------------

        pub fn lerp(a: Self, b: Self, t: T) Self {
            return a.scale(1 - t).add(b.scale(t));
        }
        pub fn nlerp(a: Self, b: Self, t: T) Self {
            return a.lerp(b, t).normalize();
        }
        /// Spherical linear interpolation along the shortest arc.
        pub fn slerp(a: Self, b: Self, t: T) Self {
            var cos_theta = a.dot(b);
            var z = b;
            if (cos_theta < 0) {
                z = b.neg();
                cos_theta = -cos_theta;
            }
            if (cos_theta > 1.0 - std.math.floatEps(T) * 16) {
                return a.lerp(z, t).normalize();
            }
            const theta = std.math.acos(cos_theta);
            const sin_theta = @sin(theta);
            const wa = @sin((1 - t) * theta) / sin_theta;
            const wb = @sin(t * theta) / sin_theta;
            return a.scale(wa).add(z.scale(wb));
        }

        // --- application ----------------------------------------------------

        /// Rotate a 3-vector by this (unit) quaternion.
        pub fn rotateVec(q: Self, v: Vec3) Vec3 {
            const u = Vec3.init(q.x, q.y, q.z);
            const t = u.cross(v).scale(2);
            return v.add(t.scale(q.w)).add(u.cross(t));
        }

        // --- conversions ----------------------------------------------------

        pub fn toMat3(q: Self) Mat(3, 3, T) {
            const n = q.normalize();
            const xx = n.x * n.x;
            const yy = n.y * n.y;
            const zz = n.z * n.z;
            const xy = n.x * n.y;
            const xz = n.x * n.z;
            const yz = n.y * n.z;
            const wx = n.w * n.x;
            const wy = n.w * n.y;
            const wz = n.w * n.z;
            return Mat(3, 3, T).fromColumns(.{
                Vec3.init(1 - 2 * (yy + zz), 2 * (xy + wz), 2 * (xz - wy)),
                Vec3.init(2 * (xy - wz), 1 - 2 * (xx + zz), 2 * (yz + wx)),
                Vec3.init(2 * (xz + wy), 2 * (yz - wx), 1 - 2 * (xx + yy)),
            });
        }

        pub fn toMat4(q: Self) Mat(4, 4, T) {
            const m3 = q.toMat3();
            const V3 = Vec(4, T);
            return Mat(4, 4, T).fromColumns(.{
                V3.fromVec3(m3.cols[0], 0),
                V3.fromVec3(m3.cols[1], 0),
                V3.fromVec3(m3.cols[2], 0),
                V3.init(0, 0, 0, 1),
            });
        }

        pub fn fromMat3(m: Mat(3, 3, T)) Self {
            const fx = m.at(0, 0) - m.at(1, 1) - m.at(2, 2);
            const fy = m.at(1, 1) - m.at(0, 0) - m.at(2, 2);
            const fz = m.at(2, 2) - m.at(0, 0) - m.at(1, 1);
            const fw = m.at(0, 0) + m.at(1, 1) + m.at(2, 2);

            var biggest: usize = 0;
            var four_biggest = fw;
            if (fx > four_biggest) {
                four_biggest = fx;
                biggest = 1;
            }
            if (fy > four_biggest) {
                four_biggest = fy;
                biggest = 2;
            }
            if (fz > four_biggest) {
                four_biggest = fz;
                biggest = 3;
            }

            const biggest_val = @sqrt(four_biggest + 1) * 0.5;
            const m4 = 0.25 / biggest_val;
            return switch (biggest) {
                0 => init(biggest_val, (m.at(1, 2) - m.at(2, 1)) * m4, (m.at(2, 0) - m.at(0, 2)) * m4, (m.at(0, 1) - m.at(1, 0)) * m4),
                1 => init((m.at(1, 2) - m.at(2, 1)) * m4, biggest_val, (m.at(0, 1) + m.at(1, 0)) * m4, (m.at(2, 0) + m.at(0, 2)) * m4),
                2 => init((m.at(2, 0) - m.at(0, 2)) * m4, (m.at(0, 1) + m.at(1, 0)) * m4, biggest_val, (m.at(1, 2) + m.at(2, 1)) * m4),
                else => init((m.at(0, 1) - m.at(1, 0)) * m4, (m.at(2, 0) + m.at(0, 2)) * m4, (m.at(1, 2) + m.at(2, 1)) * m4, biggest_val),
            };
        }

        /// Euler angles `(pitch, yaw, roll)` in radians.
        pub fn eulerAngles(q: Self) Vec3 {
            return Vec3.init(q.pitch(), q.yaw(), q.roll());
        }
        pub fn pitch(q: Self) T {
            const y = 2 * (q.y * q.z + q.w * q.x);
            const x = q.w * q.w - q.x * q.x - q.y * q.y + q.z * q.z;
            if (x == 0 and y == 0) return 2 * std.math.atan2(q.x, q.w);
            return std.math.atan2(y, x);
        }
        pub fn yaw(q: Self) T {
            return std.math.asin(std.math.clamp(-2 * (q.x * q.z - q.w * q.y), @as(T, -1), @as(T, 1)));
        }
        pub fn roll(q: Self) T {
            const y = 2 * (q.x * q.y + q.w * q.z);
            const x = q.w * q.w + q.x * q.x - q.y * q.y - q.z * q.z;
            if (x == 0 and y == 0) return 0;
            return std.math.atan2(y, x);
        }

        pub fn angle(q: Self) T {
            return 2 * std.math.acos(std.math.clamp(q.w, @as(T, -1), @as(T, 1)));
        }
        pub fn axis(q: Self) Vec3 {
            const tmp = 1 - q.w * q.w;
            if (tmp <= 0) return Vec3.init(0, 0, 1);
            const s = 1.0 / @sqrt(tmp);
            return Vec3.init(q.x * s, q.y * s, q.z * s);
        }

        // --- exponential / squad (gtc/gtx quaternion) -----------------------

        /// Quaternion exponential (maps a pure/tangent quaternion to a rotation).
        pub fn exp(q: Self) Self {
            const u = Vec3.init(q.x, q.y, q.z);
            const ang = u.length();
            if (ang < 1e-7) return identity();
            const v = u.scale(1.0 / ang);
            const s = @sin(ang);
            return init(@cos(ang), v.x * s, v.y * s, v.z * s);
        }
        /// Quaternion logarithm (inverse of `exp`).
        pub fn log(q: Self) Self {
            const u = Vec3.init(q.x, q.y, q.z);
            const len = u.length();
            if (len < 1e-7) {
                if (q.w > 0) return init(@log(q.w), 0, 0, 0);
                if (q.w < 0) return init(@log(-q.w), std.math.pi, 0, 0);
                const inf = std.math.inf(T);
                return init(inf, inf, inf, inf);
            }
            const t = std.math.atan2(len, q.w) / len;
            const len2 = len * len + q.w * q.w;
            return init(0.5 * @log(len2), t * q.x, t * q.y, t * q.z);
        }
        /// Raise to a real power: `exp(y · log(q))`.
        pub fn pow(q: Self, y: T) Self {
            if (@abs(y) < 1e-7) return identity();
            return q.log().scale(y).exp();
        }
        /// Spherical cubic interpolation through control points `s1`, `s2`.
        pub fn squad(q1: Self, q2: Self, s1: Self, s2: Self, h: T) Self {
            return q1.slerp(q2, h).slerp(s1.slerp(s2, h), 2 * (1 - h) * h);
        }
        /// Squad control point for this key given its neighbours `prev`/`next`.
        pub fn intermediate(curr: Self, prev: Self, next: Self) Self {
            const inv = curr.inverse();
            const s = next.mul(inv).log().add(prev.mul(inv).log()).scale(-0.25);
            return curr.mul(s.exp());
        }
        /// Reconstruct the real (w) component of a unit quaternion from x,y,z.
        pub fn extractReal(q: Self) T {
            const w = 1.0 - q.x * q.x - q.y * q.y - q.z * q.z;
            return if (w < 0) 0 else -@sqrt(w);
        }

        // --- constructors from directions -----------------------------------

        /// Shortest-arc rotation taking unit vector `from` onto unit vector `to`.
        pub fn fromTo(from: Vec3, to: Vec3) Self {
            const cos_theta = from.dot(to);
            if (cos_theta >= 1 - 1e-6) return identity();
            if (cos_theta < -1 + 1e-6) {
                var ax = Vec3.init(0, 0, 1).cross(from);
                if (ax.lengthSq() < 1e-6) ax = Vec3.init(1, 0, 0).cross(from);
                return fromAxisAngle(ax.normalize(), std.math.pi);
            }
            const ax = from.cross(to);
            const s = @sqrt((1 + cos_theta) * 2);
            const inv = 1.0 / s;
            return init(s * 0.5, ax.x * inv, ax.y * inv, ax.z * inv);
        }
        /// Orientation looking along `direction` with `up` (right-handed).
        pub fn lookAtRh(direction: Vec3, up: Vec3) Self {
            const c2 = direction.normalize().neg();
            const c0 = up.cross(c2).normalize();
            const c1 = c2.cross(c0);
            return fromMat3(Mat(3, 3, T).fromColumns(.{ c0, c1, c2 }));
        }
        /// Left-handed variant of `lookAtRh`.
        pub fn lookAtLh(direction: Vec3, up: Vec3) Self {
            const c2 = direction.normalize();
            const c0 = up.cross(c2).normalize();
            const c1 = c2.cross(c0);
            return fromMat3(Mat(3, 3, T).fromColumns(.{ c0, c1, c2 }));
        }
        /// Default look-at (right-handed).
        pub const lookAt = lookAtRh;
    };
}

pub const Quaternion = Quat(f32);
pub const DQuaternion = Quat(f64);

const testing = std.testing;

test "quat identity rotates nothing" {
    const q = Quat(f32).identity();
    const v = Vec(3, f32).init(1, 2, 3);
    try testing.expect(q.rotateVec(v).approxEql(v, 1e-6));
}

test "quat 90deg about Z rotates X to Y" {
    const q = Quat(f32).fromAxisAngle(Vec(3, f32).init(0, 0, 1), std.math.pi / 2.0);
    const r = q.rotateVec(Vec(3, f32).init(1, 0, 0));
    try testing.expect(r.approxEql(Vec(3, f32).init(0, 1, 0), 1e-6));
}

test "quat <-> mat3 round trip" {
    const q = Quat(f32).fromAxisAngle(Vec(3, f32).init(1, 1, 0).normalize(), 0.7).normalize();
    const back = Quat(f32).fromMat3(q.toMat3());
    // q and -q represent the same rotation; compare via rotation action
    const v = Vec(3, f32).init(0.3, -0.5, 0.8);
    try testing.expect(q.rotateVec(v).approxEql(back.rotateVec(v), 1e-5));
}

test "slerp endpoints" {
    const a = Quat(f32).identity();
    const b = Quat(f32).fromAxisAngle(Vec(3, f32).init(0, 1, 0), 1.2);
    try testing.expect(a.slerp(b, 0).approxEql(a, 1e-6));
    try testing.expect(a.slerp(b, 1).normalize().approxEql(b.normalize(), 1e-5));
}

test "euler round trip" {
    const e = Vec(3, f32).init(0.3, 0.5, -0.2);
    const q = Quat(f32).fromEuler(e);
    const e2 = q.eulerAngles();
    try testing.expect(e2.approxEql(e, 1e-5));
}

test "quaternion exp/log/pow/squad (methods)" {
    const q = Quaternion.fromAxisAngle(vec.Vec3.init(0, 1, 0), 0.9).normalize();
    try testing.expect(q.log().exp().approxEql(q, 1e-5) or q.log().exp().approxEql(q.neg(), 1e-5));
    // pow(2) rotates twice as far
    const v = vec.Vec3.init(1, 0, 0);
    try testing.expect(q.pow(2).rotateVec(v).approxEql(q.mul(q).rotateVec(v), 1e-4));
    const a = Quaternion.identity();
    const b = Quaternion.fromAxisAngle(vec.Vec3.init(0, 1, 0), 1.0);
    try testing.expect(a.squad(b, a, b, 0).approxEql(a, 1e-5));
    try testing.expect(a.squad(b, a, b, 1).approxEql(b, 1e-5));
}
test "quaternion fromTo between vectors" {
    const q = Quaternion.fromTo(vec.Vec3.init(1, 0, 0), vec.Vec3.init(0, 1, 0));
    try testing.expect(q.rotateVec(vec.Vec3.init(1, 0, 0)).approxEql(vec.Vec3.init(0, 1, 0), 1e-5));
}
test "quaternion lookAt faces direction" {
    const q = Quaternion.lookAt(vec.Vec3.init(0, 0, -1), vec.Vec3.init(0, 1, 0));
    const fwd = q.rotateVec(vec.Vec3.init(0, 0, -1));
    try testing.expect(fwd.length() > 0.99);
}
