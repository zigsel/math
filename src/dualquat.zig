//! Dual quaternions for rigid (rotation + translation) transforms
//! (GLM `gtx/dual_quaternion`).

const std = @import("std");
const vec = @import("vec.zig");
const quat = @import("quat.zig");
const mat = @import("mat.zig");

pub fn DualQuat(comptime T: type) type {
    return struct {
        real: Q,
        dual: Q,

        const Self = @This();
        const Q = quat.Quat(T);
        const Vec3 = vec.Vec(3, T);
        const Vec4 = vec.Vec(4, T);
        const Mat4 = mat.Mat(4, 4, T);
        const Mat3 = mat.Mat(3, 3, T);

        pub fn identity() Self {
            return .{ .real = Q.identity(), .dual = Q.init(0, 0, 0, 0) };
        }

        /// Build from a unit rotation quaternion and a translation.
        pub fn fromRotationTranslation(r: Q, t: Vec3) Self {
            const tq = Q.fromXYZW(t.x, t.y, t.z, 0);
            return .{ .real = r, .dual = tq.mul(r).scale(0.5) };
        }

        pub fn mul(a: Self, b: Self) Self {
            return .{
                .real = a.real.mul(b.real),
                .dual = a.real.mul(b.dual).add(a.dual.mul(b.real)),
            };
        }

        pub fn normalize(d: Self) Self {
            const len = d.real.length();
            return .{ .real = d.real.scale(1.0 / len), .dual = d.dual.scale(1.0 / len) };
        }

        pub fn getTranslation(d: Self) Vec3 {
            const t = d.dual.scale(2).mul(d.real.conjugate());
            return Vec3.init(t.x, t.y, t.z);
        }

        pub fn transformPoint(d: Self, p: Vec3) Vec3 {
            return d.real.rotateVec(p).add(d.getTranslation());
        }

        pub fn dot(a: Self, b: Self) T {
            return a.real.dot(b.real);
        }
        pub fn conjugate(d: Self) Self {
            return .{ .real = d.real.conjugate(), .dual = d.dual.conjugate() };
        }
        /// Shortest-path normalized linear interpolation of rigid transforms.
        pub fn lerp(a: Self, b: Self, t: T) Self {
            var bb = b;
            if (a.real.dot(b.real) < 0) bb = .{ .real = b.real.neg(), .dual = b.dual.neg() };
            return (Self{
                .real = a.real.scale(1 - t).add(bb.real.scale(t)),
                .dual = a.dual.scale(1 - t).add(bb.dual.scale(t)),
            }).normalize();
        }
        pub fn toMat4(d: Self) Mat4 {
            const nd = d.normalize();
            var out = nd.real.toMat4();
            const tr = nd.getTranslation();
            out.cols[3] = Vec4.init(tr.x, tr.y, tr.z, 1);
            return out;
        }

        /// Pack real + dual quaternions as a 2x4 matrix (lossless).
        pub fn toMat2x4(d: Self) mat.Mat(2, 4, T) {
            return mat.Mat(2, 4, T).fromColumns(.{
                Vec4.init(d.real.x, d.real.y, d.real.z, d.real.w),
                Vec4.init(d.dual.x, d.dual.y, d.dual.z, d.dual.w),
            });
        }
        pub fn fromMat2x4(m: mat.Mat(2, 4, T)) Self {
            return .{
                .real = Q.fromXYZW(m.at(0, 0), m.at(0, 1), m.at(0, 2), m.at(0, 3)),
                .dual = Q.fromXYZW(m.at(1, 0), m.at(1, 1), m.at(1, 2), m.at(1, 3)),
            };
        }
        /// The rigid transform as a 3x4 matrix (rows of [R|T]; used for skinning).
        pub fn toMat3x4(d: Self) mat.Mat(3, 4, T) {
            const m = d.toMat4();
            return mat.Mat(3, 4, T).fromColumns(.{
                Vec4.init(m.at(0, 0), m.at(1, 0), m.at(2, 0), m.at(3, 0)),
                Vec4.init(m.at(0, 1), m.at(1, 1), m.at(2, 1), m.at(3, 1)),
                Vec4.init(m.at(0, 2), m.at(1, 2), m.at(2, 2), m.at(3, 2)),
            });
        }
        pub fn fromMat3x4(m: mat.Mat(3, 4, T)) Self {
            const rot = Mat3.fromColumns(.{
                Vec3.init(m.at(0, 0), m.at(1, 0), m.at(2, 0)),
                Vec3.init(m.at(0, 1), m.at(1, 1), m.at(2, 1)),
                Vec3.init(m.at(0, 2), m.at(1, 2), m.at(2, 2)),
            });
            return fromRotationTranslation(Q.fromMat3(rot), Vec3.init(m.at(0, 3), m.at(1, 3), m.at(2, 3)));
        }
    };
}

const testing = std.testing;
test "dualquat mat casts round trip" {
    const Q2 = quat.Quat(f32);
    const V = vec.Vec3;
    const dq = DualQuat(f32).fromRotationTranslation(Q2.fromAxisAngle(V.init(0, 0, 1), 0.7), V.init(2, 3, 4));
    try testing.expect(DualQuat(f32).fromMat2x4(dq.toMat2x4()).real.approxEql(dq.real, 1e-5));
    const p = V.init(1, -0.5, 0.3);
    try testing.expect(DualQuat(f32).fromMat3x4(dq.toMat3x4()).transformPoint(p).approxEql(dq.transformPoint(p), 1e-4));
}

test "dual_quaternion rigid transform" {
    const Q = quat.Quat(f32);
    const V = vec.Vec3;
    const r = Q.fromAxisAngle(V.init(0, 0, 1), std.math.pi / 2.0);
    const dq = DualQuat(f32).fromRotationTranslation(r, V.init(5, 0, 0));
    const p = dq.transformPoint(V.init(1, 0, 0)); // rotate +X->+Y then translate +5X
    try testing.expect(p.approxEql(V.init(5, 1, 0), 1e-5));
}
