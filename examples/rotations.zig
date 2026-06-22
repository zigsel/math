//! Rotations — quaternions (`math.Quat`), dual quaternions (`math.DualQuat`),
//! and Euler-angle matrices (`math.euler`).
//! Run: `zig build example-rotations`

const std = @import("std");
const math = @import("math");
const print = std.debug.print;

pub fn main() void {
    quatBasics();
    quatInterpolate();
    quatConvert();
    eulerAngles();
    dualQuatRigid();
}

fn quatBasics() void {
    // 90° about Z, applied to +X -> +Y.
    const q = math.Quaternion.fromAxisAngle(math.Vec3.init(0, 0, 1), math.radians(@as(f32, 90)));
    print("q*+X      = {f}\n", .{q.rotateVec(math.Vec3.init(1, 0, 0))});
    print("angle     = {d}\n", .{math.degrees(q.angle())});
    // Composition: apply b then a via Hamilton product a.mul(b).
    const a = math.Quaternion.fromAxisAngle(math.Vec3.init(1, 0, 0), 0.5);
    print("compose*+Z= {f}\n", .{a.mul(q).rotateVec(math.Vec3.init(0, 0, 1))});
}

fn quatInterpolate() void {
    const a = math.Quaternion.identity();
    const b = math.Quaternion.fromAxisAngle(math.Vec3.init(0, 1, 0), math.radians(@as(f32, 90)));
    // Shortest-arc spherical interpolation.
    print("slerp 0.5 * +X = {f}\n", .{a.slerp(b, 0.5).rotateVec(math.Vec3.init(1, 0, 0))});
    print("nlerp 0.5 * +X = {f}\n", .{a.nlerp(b, 0.5).rotateVec(math.Vec3.init(1, 0, 0))});
}

fn quatConvert() void {
    const e = math.Vec3.init(0.3, 0.5, -0.2); // pitch, yaw, roll
    const q = math.Quaternion.fromEuler(e);
    print("euler round-trip = {f}\n", .{q.eulerAngles()});
    const m = q.toMat4(); // upload-ready rotation matrix
    print("back from mat3   = {}\n", .{math.Quaternion.fromMat3(q.toMat3()).approxEql(q, 1e-5)});
    print("toMat4 col0      = {f}\n", .{m.cols[0]});
}

fn eulerAngles() void {
    // Build a rotation matrix from an explicit axis order.
    const m = math.euler.yawPitchRoll(0.3, 0.2, 0.1);
    const back = math.euler.extractYxz(m); // exact inverse of the builder
    print("yawPitchRoll -> extractYxz = {f}\n", .{back});
    // Other precisions via the generic builder:
    const md = math.Euler(f64).z(std.math.pi / 2.0);
    print("Euler(f64).z col0.y = {d}\n", .{md.at(0, 1)});
}

fn dualQuatRigid() void {
    const Q = math.Quat(f32);
    const DQ = math.DualQuat(f32);
    // Rigid transform = rotation + translation, no scale.
    const r = Q.fromAxisAngle(math.Vec3.init(0, 0, 1), std.math.pi / 2.0);
    const dq = DQ.fromRotationTranslation(r, math.Vec3.init(5, 0, 0));
    print("dq * +X = {f}\n", .{dq.transformPoint(math.Vec3.init(1, 0, 0))}); // rotate then +5X
    print("skinning matrix col3 = {f}\n", .{dq.toMat4().cols[3]});
}
