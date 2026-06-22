//! Matrices — `math.Mat(C, R, T)` (column-major) and the `math.matrix` namespace.
//! Run: `zig build example-matrices`

const std = @import("std");
const math = @import("math");
const print = std.debug.print;

pub fn main() void {
    matConstruct();
    matProducts();
    matInverse();
    matDecompose();
    matFactor();
    matPca();
}

fn matConstruct() void {
    const i = math.Mat3.identity();
    const cols = math.Mat3.fromColumns(.{
        math.Vec3.init(1, 0, 0),
        math.Vec3.init(0, 2, 0),
        math.Vec3.init(0, 0, 3),
    });
    const diag = math.Mat3.diagonal(math.Vec3.init(1, 2, 3)); // same matrix
    print("identity   = {f}\n", .{i});
    print("at(1,1)    = {d}\n", .{cols.at(1, 1)}); // (col, row) indexing
    print("diag==cols = {}\n", .{diag.eql(cols)});
    // Row-major input is available too.
    const rm = math.Mat3.fromRows(.{
        math.Vec3.init(1, 2, 3),
        math.Vec3.init(4, 5, 6),
        math.Vec3.init(7, 8, 9),
    });
    print("fromRows row0col1 = {d}\n", .{rm.at(1, 0)});
}

fn matProducts() void {
    const a = math.Mat4.identity().scale(2); // scale all entries
    const v = math.Vec4.init(1, 2, 3, 1);
    print("M*v        = {f}\n", .{a.mulVec(v)});
    // Non-square products compose: Mat(C,R) * Mat(C2,C) -> Mat(C2,R).
    const m23 = math.Mat2x3.fromColumns(.{ math.Vec3.init(1, 2, 3), math.Vec3.init(4, 5, 6) });
    print("transpose  = {f}\n", .{m23.transpose()}); // -> Mat3x2
}

fn matInverse() void {
    const a = math.Mat3.fromColumns(.{
        math.Vec3.init(2, 0, 1),
        math.Vec3.init(1, 3, 2),
        math.Vec3.init(1, 0, 4),
    });
    print("det        = {d}\n", .{a.determinant()});
    print("A*inv(A)   = {f}\n", .{a.mul(a.inverse())});
    // Fast inverses for affine/rigid 4x4 transforms.
    var t = math.Mat4.identity();
    t.cols[3] = math.Vec4.init(5, 6, 7, 1);
    print("rigidInv ok= {}\n", .{t.mul(t.rigidInverse()).approxEql(math.Mat4.identity(), 1e-5)});
}

fn matDecompose() void {
    // Split a TRS matrix back into translation / rotation / scale.
    const m = math.transform.translate(math.Mat4.identity(), math.Vec3.init(5, 6, 7));
    const d = math.matrix.decompose(m).?;
    print("translation= {f}\n", .{d.translation});
    print("scale      = {f}\n", .{d.scale});
    // ...and rebuild it.
    print("recompose ok= {}\n", .{math.matrix.recompose(d).approxEql(m, 1e-4)});
}

fn matFactor() void {
    const a = math.Mat3.fromColumns(.{
        math.Vec3.init(2, 1, 0),
        math.Vec3.init(1, 3, 1),
        math.Vec3.init(0, 1, 4),
    });
    // Factorisation helpers are METHODS on the matrix.
    const qr = a.qr();
    print("A = Q*R    = {}\n", .{qr.q.mul(qr.r).approxEql(a, 1e-4)});
    print("isOrthogonal(Q) = {}\n", .{qr.q.isOrthogonal(1e-4)});
}

fn matPca() void {
    // Principal component analysis lives under math.matrix.
    const pts = [_]math.Vec3{
        math.Vec3.init(2, 0, 0),   math.Vec3.init(-2, 0, 0),
        math.Vec3.init(0, 0.5, 0), math.Vec3.init(0, -0.5, 0),
    };
    const cov = math.matrix.covariance(&pts);
    const e = math.matrix.eigenSymmetric(cov);
    print("eigenvalues = {f}\n", .{e.values}); // largest spread along X
}
