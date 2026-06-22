//! Euler-angle rotation matrices and extraction — `math.euler`.
//!
//! Generic over the float element type via `Euler(T)`; root exposes the f32
//! instance as `math.euler` and the generic builder as `math.Euler` (mirroring
//! `math.transform` vs `math.Transforms`). Column-major. Builders compose
//! elementary rotations; extractors are their exact inverses (round-trip tested).

const std = @import("std");
const mat = @import("mat.zig");
const vec = @import("vec.zig");

/// Euler-angle constructors/extractors parameterized by float element type `T`.
pub fn Euler(comptime T: type) type {
    const Mat4 = mat.Mat(4, 4, T);
    const Mat3 = mat.Mat(3, 3, T);
    const Mat2 = mat.Mat(2, 2, T);
    const Vec2 = vec.Vec(2, T);
    const Vec3 = vec.Vec(3, T);
    const Vec4 = vec.Vec(4, T);

    return struct {
        // --- single-axis ----------------------------------------------------

        pub fn x(angle: T) Mat4 {
            const c = @cos(angle);
            const s = @sin(angle);
            return Mat4.fromColumns(.{
                Vec4.init(1, 0, 0, 0),
                Vec4.init(0, c, s, 0),
                Vec4.init(0, -s, c, 0),
                Vec4.init(0, 0, 0, 1),
            });
        }
        pub fn y(angle: T) Mat4 {
            const c = @cos(angle);
            const s = @sin(angle);
            return Mat4.fromColumns(.{
                Vec4.init(c, 0, -s, 0),
                Vec4.init(0, 1, 0, 0),
                Vec4.init(s, 0, c, 0),
                Vec4.init(0, 0, 0, 1),
            });
        }
        pub fn z(angle: T) Mat4 {
            const c = @cos(angle);
            const s = @sin(angle);
            return Mat4.fromColumns(.{
                Vec4.init(c, s, 0, 0),
                Vec4.init(-s, c, 0, 0),
                Vec4.init(0, 0, 1, 0),
                Vec4.init(0, 0, 0, 1),
            });
        }

        // --- two-axis -------------------------------------------------------

        pub fn xy(t1: T, t2: T) Mat4 {
            return x(t1).mul(y(t2));
        }
        pub fn yx(t1: T, t2: T) Mat4 {
            return y(t1).mul(x(t2));
        }
        pub fn xz(t1: T, t2: T) Mat4 {
            return x(t1).mul(z(t2));
        }
        pub fn zx(t1: T, t2: T) Mat4 {
            return z(t1).mul(x(t2));
        }
        pub fn yz(t1: T, t2: T) Mat4 {
            return y(t1).mul(z(t2));
        }
        pub fn zy(t1: T, t2: T) Mat4 {
            return z(t1).mul(y(t2));
        }

        // --- three-axis -----------------------------------------------------

        pub fn xyz(t1: T, t2: T, t3: T) Mat4 {
            return x(t1).mul(y(t2)).mul(z(t3));
        }
        pub fn yxz(t1: T, t2: T, t3: T) Mat4 {
            return y(t1).mul(x(t2)).mul(z(t3));
        }
        pub fn xzy(t1: T, t2: T, t3: T) Mat4 {
            return x(t1).mul(z(t2)).mul(y(t3));
        }
        pub fn yzx(t1: T, t2: T, t3: T) Mat4 {
            return y(t1).mul(z(t2)).mul(x(t3));
        }
        pub fn zyx(t1: T, t2: T, t3: T) Mat4 {
            return z(t1).mul(y(t2)).mul(x(t3));
        }
        pub fn zxy(t1: T, t2: T, t3: T) Mat4 {
            return z(t1).mul(x(t2)).mul(y(t3));
        }
        pub fn xyx(t1: T, t2: T, t3: T) Mat4 {
            return x(t1).mul(y(t2)).mul(x(t3));
        }
        pub fn xzx(t1: T, t2: T, t3: T) Mat4 {
            return x(t1).mul(z(t2)).mul(x(t3));
        }
        pub fn yxy(t1: T, t2: T, t3: T) Mat4 {
            return y(t1).mul(x(t2)).mul(y(t3));
        }
        pub fn yzy(t1: T, t2: T, t3: T) Mat4 {
            return y(t1).mul(z(t2)).mul(y(t3));
        }
        pub fn zxz(t1: T, t2: T, t3: T) Mat4 {
            return z(t1).mul(x(t2)).mul(z(t3));
        }
        pub fn zyz(t1: T, t2: T, t3: T) Mat4 {
            return z(t1).mul(y(t2)).mul(z(t3));
        }
        pub fn yawPitchRoll(yaw: T, pitch: T, roll: T) Mat4 {
            return y(yaw).mul(x(pitch)).mul(z(roll));
        }

        // --- extraction (inverse of the builders above) ---------------------

        inline fn rc(m: Mat4, comptime r: usize, comptime c: usize) T {
            return m.at(c, r); // M[row][col] from column-major storage
        }

        fn extractTaitBryan(m: Mat4, comptime i: usize, comptime j: usize, comptime k: usize, comptime odd: bool) Vec3 {
            const sgn: T = if (odd) -1 else 1;
            const cy = @sqrt(rc(m, i, i) * rc(m, i, i) + rc(m, i, j) * rc(m, i, j));
            return Vec3.init(
                std.math.atan2(-sgn * rc(m, j, k), rc(m, k, k)),
                std.math.atan2(sgn * rc(m, i, k), cy),
                std.math.atan2(-sgn * rc(m, i, j), rc(m, i, i)),
            );
        }
        fn extractProper(m: Mat4, comptime i: usize, comptime j: usize, comptime k: usize, comptime odd: bool) Vec3 {
            const a2: T = if (odd) 1 else -1;
            const c2: T = if (odd) -1 else 1;
            const sy = @sqrt(rc(m, i, j) * rc(m, i, j) + rc(m, i, k) * rc(m, i, k));
            return Vec3.init(
                std.math.atan2(rc(m, j, i), a2 * rc(m, k, i)),
                std.math.atan2(sy, rc(m, i, i)),
                std.math.atan2(rc(m, i, j), c2 * rc(m, i, k)),
            );
        }

        pub fn extractXyz(m: Mat4) Vec3 {
            return extractTaitBryan(m, 0, 1, 2, false);
        }
        pub fn extractYzx(m: Mat4) Vec3 {
            return extractTaitBryan(m, 1, 2, 0, false);
        }
        pub fn extractZxy(m: Mat4) Vec3 {
            return extractTaitBryan(m, 2, 0, 1, false);
        }
        pub fn extractXzy(m: Mat4) Vec3 {
            return extractTaitBryan(m, 0, 2, 1, true);
        }
        pub fn extractYxz(m: Mat4) Vec3 {
            return extractTaitBryan(m, 1, 0, 2, true);
        }
        pub fn extractZyx(m: Mat4) Vec3 {
            return extractTaitBryan(m, 2, 1, 0, true);
        }
        pub fn extractXyx(m: Mat4) Vec3 {
            return extractProper(m, 0, 1, 2, false);
        }
        pub fn extractYzy(m: Mat4) Vec3 {
            return extractProper(m, 1, 2, 0, false);
        }
        pub fn extractZxz(m: Mat4) Vec3 {
            return extractProper(m, 2, 0, 1, false);
        }
        pub fn extractXzx(m: Mat4) Vec3 {
            return extractProper(m, 0, 2, 1, true);
        }
        pub fn extractYxy(m: Mat4) Vec3 {
            return extractProper(m, 1, 0, 2, true);
        }
        pub fn extractZyz(m: Mat4) Vec3 {
            return extractProper(m, 2, 1, 0, true);
        }

        // --- derivatives & orientate ----------------------------------------

        pub fn derivedX(angle: T, velocity: T) Mat4 {
            const c = @cos(angle) * velocity;
            const s = @sin(angle) * velocity;
            return Mat4.fromColumns(.{ Vec4.init(0, 0, 0, 0), Vec4.init(0, -s, c, 0), Vec4.init(0, -c, -s, 0), Vec4.init(0, 0, 0, 0) });
        }
        pub fn derivedY(angle: T, velocity: T) Mat4 {
            const c = @cos(angle) * velocity;
            const s = @sin(angle) * velocity;
            return Mat4.fromColumns(.{ Vec4.init(-s, 0, -c, 0), Vec4.init(0, 0, 0, 0), Vec4.init(c, 0, -s, 0), Vec4.init(0, 0, 0, 0) });
        }
        pub fn derivedZ(angle: T, velocity: T) Mat4 {
            const c = @cos(angle) * velocity;
            const s = @sin(angle) * velocity;
            return Mat4.fromColumns(.{ Vec4.init(-s, c, 0, 0), Vec4.init(-c, -s, 0, 0), Vec4.init(0, 0, 0, 0), Vec4.init(0, 0, 0, 0) });
        }

        pub fn orientate2(angle: T) Mat2 {
            const c = @cos(angle);
            const s = @sin(angle);
            return Mat2.fromColumns(.{ Vec2.init(c, s), Vec2.init(-s, c) });
        }
        /// Rotation matrix from Euler `(pitch=x, yaw=y, roll=z)`.
        pub fn orientate3(angles: Vec3) Mat3 {
            const m = yawPitchRoll(angles.z, angles.x, angles.y);
            return Mat3.fromColumns(.{
                Vec3.init(m.at(0, 0), m.at(0, 1), m.at(0, 2)),
                Vec3.init(m.at(1, 0), m.at(1, 1), m.at(1, 2)),
                Vec3.init(m.at(2, 0), m.at(2, 1), m.at(2, 2)),
            });
        }
        pub fn orientate4(angles: Vec3) Mat4 {
            return yawPitchRoll(angles.z, angles.x, angles.y);
        }
    };
}

/// f32 default-precision Euler builders.
pub const euler = Euler(f32);

const testing = std.testing;

test "euler builders rotate correctly" {
    const Vec4 = vec.Vec4;
    const r = euler.z(std.math.pi / 2.0).mulVec(Vec4.init(1, 0, 0, 1));
    try testing.expect(r.approxEql(Vec4.init(0, 1, 0, 1), 1e-6));
}

test "euler extractors are exact inverses of builders" {
    const e = euler;
    const pairs = .{
        .{ e.xyz, e.extractXyz },
        .{ e.yzx, e.extractYzx },
        .{ e.zxy, e.extractZxy },
        .{ e.xzy, e.extractXzy },
        .{ e.yxz, e.extractYxz },
        .{ e.zyx, e.extractZyx },
        .{ e.xyx, e.extractXyx },
        .{ e.yzy, e.extractYzy },
        .{ e.zxz, e.extractZxz },
        .{ e.xzx, e.extractXzx },
        .{ e.yxy, e.extractYxy },
        .{ e.zyz, e.extractZyz },
    };
    inline for (pairs) |p| {
        const m = p[0](0.3, 0.6, -0.2);
        const ex = p[1](m);
        const m2 = p[0](ex.x, ex.y, ex.z);
        try testing.expect(m2.approxEql(m, 1e-4));
    }
}

test "euler generic over f64" {
    const E = Euler(f64);
    const V = vec.Vec(4, f64);
    const r = E.z(std.math.pi / 2.0).mulVec(V.init(1, 0, 0, 1));
    try testing.expect(r.approxEql(V.init(0, 1, 0, 1), 1e-12));
}
