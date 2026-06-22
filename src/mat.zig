//! Generic matrices `Mat(C, R, T)` — `C` columns, `R` rows, element type `T`.
//!
//! **Column-major** storage (matching GLM / GLSL / OpenGL): the matrix is an
//! array of `C` column vectors, each a `Vec(R, T)`. Indexing follows GLM:
//! `at(col, row)`. Since the storage shape is uniform across all sizes, the
//! entire method surface lives in a single struct body (no per-size variants).

const std = @import("std");
const sc = @import("meta.zig");
const vec = @import("vec.zig");
const Vec = vec.Vec;
const quat = @import("quat.zig");
const transform = @import("transform.zig").transform;
const Quat = quat.Quaternion;

pub fn Mat(comptime c: comptime_int, comptime r: comptime_int, comptime T: type) type {
    return struct {
        cols: [c]ColVec,

        pub const ColVec = Vec(r, T);
        pub const RowVec = Vec(c, T);
        pub const Element = T;
        pub const cols_n: comptime_int = c;
        pub const rows_n: comptime_int = r;
        pub const is_math_matrix = true;
        const Self = @This();

        // --- construction ---------------------------------------------------

        pub fn zero() Self {
            var m: Self = undefined;
            inline for (0..c) |i| m.cols[i] = ColVec.splat(0);
            return m;
        }

        pub fn fromColumns(columns: [c]ColVec) Self {
            return .{ .cols = columns };
        }

        /// Build from a flat, column-major array of `c*r` scalars.
        pub fn fromArray(data: [c * r]T) Self {
            var m: Self = undefined;
            inline for (0..c) |ci| {
                var col: @Vector(r, T) = undefined;
                inline for (0..r) |ri| col[ri] = data[ci * r + ri];
                m.cols[ci] = ColVec.fromSimd(col);
            }
            return m;
        }

        /// Identity matrix (square only).
        pub fn identity() Self {
            comptime if (c != r) @compileError("identity requires a square matrix");
            var m = zero();
            inline for (0..c) |i| m.cols[i] = m.cols[i].set(i, 1);
            return m;
        }

        /// Diagonal matrix. Pass a scalar for a uniform square diagonal, or a
        /// vector for per-component entries (works for non-square sizes too).
        pub fn diagonal(d: anytype) Self {
            var m = zero();
            if (comptime sc.isVec(@TypeOf(d))) {
                const a = d.toArray();
                inline for (0..@min(c, r)) |i| m.cols[i] = m.cols[i].set(i, a[i]);
            } else {
                comptime if (c != r) @compileError("scalar diagonal requires a square matrix");
                inline for (0..c) |i| m.cols[i] = m.cols[i].set(i, d);
            }
            return m;
        }

        /// Build from `r` row vectors (each of length `c`) — row-major input.
        pub fn fromRows(rows: [r]RowVec) Self {
            var m: Self = undefined;
            inline for (0..c) |ci| {
                var col: [r]T = undefined;
                inline for (0..r) |ri| col[ri] = rows[ri].get(ci);
                m.cols[ci] = ColVec.fromArray(col);
            }
            return m;
        }

        /// Skew-symmetric cross-product matrix: `m.crossMatrix(v).mulVec(w) == v×w`.
        /// Square 3x3 (pure) or 4x4 (homogeneous) only.
        pub fn crossMatrix(v: Vec(3, T)) Self {
            comptime if (!((c == 3 and r == 3) or (c == 4 and r == 4)))
                @compileError("crossMatrix requires a 3x3 or 4x4 matrix");
            var m = if (c == 4) identity() else zero();
            m.cols[0] = m.cols[0].set(0, 0).set(1, v.z).set(2, -v.y);
            m.cols[1] = m.cols[1].set(0, -v.z).set(1, 0).set(2, v.x);
            m.cols[2] = m.cols[2].set(0, v.y).set(1, -v.x).set(2, 0);
            return m;
        }

        /// Classical adjugate (`det(M)·inverse(M)`); square invertible matrices.
        pub fn adjugate(self: Self) Self {
            return self.inverse().scale(self.determinant());
        }

        // --- access ---------------------------------------------------------

        /// Element at (column, row) — GLM ordering.
        pub inline fn at(self: Self, ci: usize, ri: usize) T {
            return self.cols[ci].get(ri);
        }
        pub inline fn column(self: Self, i: usize) ColVec {
            return self.cols[i];
        }
        pub fn row(self: Self, i: usize) RowVec {
            var v: @Vector(c, T) = undefined;
            inline for (0..c) |j| v[j] = self.at(j, i);
            return RowVec.fromSimd(v);
        }

        // --- element-wise ---------------------------------------------------

        pub fn add(a: Self, b: Self) Self {
            var m: Self = undefined;
            inline for (0..c) |i| m.cols[i] = a.cols[i].add(b.cols[i]);
            return m;
        }
        pub fn sub(a: Self, b: Self) Self {
            var m: Self = undefined;
            inline for (0..c) |i| m.cols[i] = a.cols[i].sub(b.cols[i]);
            return m;
        }
        pub fn scale(a: Self, s: T) Self {
            var m: Self = undefined;
            inline for (0..c) |i| m.cols[i] = a.cols[i].scale(s);
            return m;
        }
        /// Component-wise multiply (GLSL `matrixCompMult`).
        pub fn compMul(a: Self, b: Self) Self {
            var m: Self = undefined;
            inline for (0..c) |i| m.cols[i] = a.cols[i].mul(b.cols[i]);
            return m;
        }

        // --- linear-algebra products ---------------------------------------

        /// Matrix * column-vector → vector. `v` must have `C` components.
        pub fn mulVec(self: Self, v: RowVec) ColVec {
            var acc: @Vector(r, T) = @splat(0);
            const va = v.toArray();
            inline for (0..c) |k| {
                const s: @Vector(r, T) = @splat(va[k]);
                acc += self.cols[k].simd() * s;
            }
            return ColVec.fromSimd(acc);
        }

        /// Matrix product. `self` is `Mat(C, R)`, `other` is `Mat(C2, C)`,
        /// result is `Mat(C2, R)`.
        pub fn mul(self: Self, other: anytype) Mat(@TypeOf(other).cols_n, r, T) {
            const Other = @TypeOf(other);
            comptime if (Other.rows_n != c)
                @compileError("matrix product dimension mismatch: lhs cols != rhs rows");
            const C2 = Other.cols_n;
            var m: Mat(C2, r, T) = undefined;
            inline for (0..C2) |j| m.cols[j] = self.mulVec(other.cols[j]);
            return m;
        }

        /// Transpose: `Mat(C, R)` → `Mat(R, C)`.
        pub fn transpose(self: Self) Mat(r, c, T) {
            var m: Mat(r, c, T) = undefined;
            inline for (0..r) |i| {
                var v: @Vector(c, T) = undefined;
                inline for (0..c) |j| v[j] = self.at(j, i);
                m.cols[i] = Vec(c, T).fromSimd(v);
            }
            return m;
        }

        // --- determinant / inverse (square float matrices) -----------------

        pub fn determinant(self: Self) T {
            comptime if (c != r) @compileError("determinant requires a square matrix");
            comptime sc.requireFloat(Self);
            return switch (r) {
                2 => det2(self),
                3 => det3(self),
                4 => det4(self),
                else => @compileError("determinant supports 2x2, 3x3, 4x4 only"),
            };
        }

        /// Inverse (assumes the matrix is invertible; result is undefined if
        /// the determinant is zero — see `invertible`).
        pub fn inverse(self: Self) Self {
            comptime if (c != r) @compileError("inverse requires a square matrix");
            comptime sc.requireFloat(Self);
            return switch (r) {
                2 => inv2(self),
                3 => inv3(self),
                4 => inv4(self),
                else => @compileError("inverse supports 2x2, 3x3, 4x4 only"),
            };
        }

        pub fn invertible(self: Self) bool {
            return self.determinant() != 0;
        }

        /// Fast inverse of a rigid transform (orthonormal rotation + translation,
        /// no scale): `[Rᵀ | -Rᵀt]`. Much cheaper than a general inverse.
        pub fn rigidInverse(self: Self) Self {
            comptime if (c != 4 or r != 4) @compileError("rigidInverse requires a 4x4 matrix");
            comptime sc.requireFloat(Self);
            const tx = self.at(3, 0);
            const ty = self.at(3, 1);
            const tz = self.at(3, 2);
            return Self.fromColumns(.{
                ColVec.init(self.at(0, 0), self.at(1, 0), self.at(2, 0), 0),
                ColVec.init(self.at(0, 1), self.at(1, 1), self.at(2, 1), 0),
                ColVec.init(self.at(0, 2), self.at(1, 2), self.at(2, 2), 0),
                ColVec.init(
                    -(self.at(0, 0) * tx + self.at(0, 1) * ty + self.at(0, 2) * tz),
                    -(self.at(1, 0) * tx + self.at(1, 1) * ty + self.at(1, 2) * tz),
                    -(self.at(2, 0) * tx + self.at(2, 1) * ty + self.at(2, 2) * tz),
                    1,
                ),
            });
        }

        /// Product of two affine 4x4 matrices, skipping the homogeneous bottom
        /// row (assumed `0 0 0 1`). Equivalent to `mul` for affine inputs.
        pub fn mulAffine(a: Self, b: Self) Self {
            comptime if (c != 4 or r != 4) @compileError("mulAffine requires a 4x4 matrix");
            var out: Self = undefined;
            inline for (0..3) |j| {
                const bc = b.cols[j];
                out.cols[j] = a.cols[0].scale(bc.x).add(a.cols[1].scale(bc.y)).add(a.cols[2].scale(bc.z)).set(3, 0);
            }
            const bc3 = b.cols[3];
            out.cols[3] = a.cols[0].scale(bc3.x).add(a.cols[1].scale(bc3.y)).add(a.cols[2].scale(bc3.z)).add(a.cols[3]).set(3, 1);
            return out;
        }

        /// Left-to-right product of a chain of square matrices.
        pub fn mulChain(mats: []const Self) Self {
            comptime if (c != r) @compileError("mulChain requires square matrices");
            if (mats.len == 0) return identity();
            var result = mats[0];
            for (mats[1..]) |m| result = result.mul(m);
            return result;
        }

        pub fn setColumn(self: Self, i: usize, v: ColVec) Self {
            var m = self;
            m.cols[i] = v;
            return m;
        }
        pub fn setRow(self: Self, i: usize, v: RowVec) Self {
            var m = self;
            const a = v.toArray();
            inline for (0..c) |j| m.cols[j] = m.cols[j].set(i, a[j]);
            return m;
        }

        /// `transpose(inverse(M))` — the correct transform for surface normals.
        pub fn inverseTranspose(self: Self) Self {
            comptime if (c != r) @compileError("inverseTranspose requires a square matrix");
            return self.inverse().transpose();
        }

        /// Fast inverse of a 4x4 affine transform (invert the 3x3 linear part
        /// and the translation; ignores any perspective row).
        pub fn affineInverse(self: Self) Self {
            comptime if (c != 4 or r != 4) @compileError("affineInverse requires a 4x4 matrix");
            comptime sc.requireFloat(Self);
            const M3 = Mat(3, 3, T);
            const V3 = Vec(3, T);
            const V4 = Vec(4, T);
            const lin = M3.fromColumns(.{
                V3.init(self.at(0, 0), self.at(0, 1), self.at(0, 2)),
                V3.init(self.at(1, 0), self.at(1, 1), self.at(1, 2)),
                V3.init(self.at(2, 0), self.at(2, 1), self.at(2, 2)),
            });
            const inv = lin.inverse();
            const t = V3.init(self.at(3, 0), self.at(3, 1), self.at(3, 2));
            const nt = inv.mulVec(t).neg();
            return Self.fromColumns(.{
                V4.init(inv.at(0, 0), inv.at(0, 1), inv.at(0, 2), 0),
                V4.init(inv.at(1, 0), inv.at(1, 1), inv.at(1, 2), 0),
                V4.init(inv.at(2, 0), inv.at(2, 1), inv.at(2, 2), 0),
                V4.init(nt.x, nt.y, nt.z, 1),
            });
        }

        pub fn eql(a: Self, b: Self) bool {
            inline for (0..c) |i| if (!a.cols[i].eql(b.cols[i])) return false;
            return true;
        }
        pub fn approxEql(a: Self, b: Self, eps: T) bool {
            inline for (0..c) |i| if (!a.cols[i].approxEql(b.cols[i], eps)) return false;
            return true;
        }

        /// `std.fmt` integration (use `{f}`): prints `mat3x3[vec3(...), ...]`.
        pub fn format(self: Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
            try writer.print("mat{d}x{d}[", .{ c, r });
            inline for (0..c) |i| {
                if (i > 0) try writer.writeAll(", ");
                try self.cols[i].format(writer);
            }
            try writer.writeAll("]");
        }

        // --- factorisation / queries (methods) -----------------------------

        /// `A = Q·R`, Q orthonormal, R upper-triangular (modified Gram–Schmidt).
        pub fn qr(self: Self) struct { q: Self, r: Self } {
            comptime if (c != r) @compileError("qr requires a square matrix");
            var q = zero();
            var rr = zero();
            inline for (0..c) |j| {
                var v = self.cols[j];
                inline for (0..j) |i| {
                    const rij = q.cols[i].dot(self.cols[j]);
                    rr.cols[j] = rr.cols[j].set(i, rij);
                    v = v.sub(q.cols[i].scale(rij));
                }
                const nrm = v.length();
                rr.cols[j] = rr.cols[j].set(j, nrm);
                q.cols[j] = v.scale(1.0 / nrm);
            }
            return .{ .q = q, .r = rr };
        }
        /// `A = R·Q`, R upper-triangular, Q orthonormal (via QR of a flipped transpose).
        pub fn rq(self: Self) struct { r: Self, q: Self } {
            comptime if (c != r) @compileError("rq requires a square matrix");
            const a = flipRows(self).transpose();
            const d = a.qr();
            return .{
                .r = flipCols(flipRows(d.r.transpose())), // R = P·R1ᵀ·P
                .q = flipRows(d.q.transpose()), // Q = P·Q1ᵀ
            };
        }
        /// Reverse column order (flip left-right).
        pub fn fliplr(self: Self) Self {
            return flipCols(self);
        }
        /// Reverse row order (flip up-down).
        pub fn flipud(self: Self) Self {
            return flipRows(self);
        }
        /// True if every entry equals the identity within `eps`.
        pub fn isIdentity(self: Self, eps: T) bool {
            inline for (0..c) |ci| {
                inline for (0..r) |ri| {
                    const expected: T = if (ci == ri) 1 else 0;
                    if (@abs(self.at(ci, ri) - expected) > eps) return false;
                }
            }
            return true;
        }
        /// True if every column is unit length within `eps`.
        pub fn isNormalized(self: Self, eps: T) bool {
            inline for (0..c) |ci| {
                if (@abs(self.cols[ci].length() - 1) > eps) return false;
            }
            return true;
        }
        /// True if all columns are mutually orthogonal within `eps`.
        pub fn isOrthogonal(self: Self, eps: T) bool {
            inline for (0..c) |i| {
                inline for (i + 1..c) |j| {
                    if (@abs(self.cols[i].dot(self.cols[j])) > eps) return false;
                }
            }
            return true;
        }
        /// True if every entry is zero within `eps`.
        pub fn isNull(self: Self, eps: T) bool {
            inline for (0..c) |ci| {
                inline for (0..r) |ri| {
                    if (@abs(self.at(ci, ri)) > eps) return false;
                }
            }
            return true;
        }

        // --- private det/inverse implementations (GLM-derived) -------------

        fn det2(m: Self) T {
            return m.at(0, 0) * m.at(1, 1) - m.at(1, 0) * m.at(0, 1);
        }
        fn inv2(m: Self) Self {
            const ood = 1.0 / det2(m);
            var out: Self = undefined;
            out.cols[0] = ColVec.init(m.at(1, 1) * ood, -m.at(0, 1) * ood);
            out.cols[1] = ColVec.init(-m.at(1, 0) * ood, m.at(0, 0) * ood);
            return out;
        }

        fn det3(m: Self) T {
            return m.at(0, 0) * (m.at(1, 1) * m.at(2, 2) - m.at(2, 1) * m.at(1, 2)) -
                m.at(1, 0) * (m.at(0, 1) * m.at(2, 2) - m.at(2, 1) * m.at(0, 2)) +
                m.at(2, 0) * (m.at(0, 1) * m.at(1, 2) - m.at(1, 1) * m.at(0, 2));
        }
        fn inv3(m: Self) Self {
            const ood = 1.0 / det3(m);
            var out: Self = undefined;
            out.cols[0] = ColVec.init(
                (m.at(1, 1) * m.at(2, 2) - m.at(2, 1) * m.at(1, 2)) * ood,
                -(m.at(0, 1) * m.at(2, 2) - m.at(2, 1) * m.at(0, 2)) * ood,
                (m.at(0, 1) * m.at(1, 2) - m.at(1, 1) * m.at(0, 2)) * ood,
            );
            out.cols[1] = ColVec.init(
                -(m.at(1, 0) * m.at(2, 2) - m.at(2, 0) * m.at(1, 2)) * ood,
                (m.at(0, 0) * m.at(2, 2) - m.at(2, 0) * m.at(0, 2)) * ood,
                -(m.at(0, 0) * m.at(1, 2) - m.at(1, 0) * m.at(0, 2)) * ood,
            );
            out.cols[2] = ColVec.init(
                (m.at(1, 0) * m.at(2, 1) - m.at(2, 0) * m.at(1, 1)) * ood,
                -(m.at(0, 0) * m.at(2, 1) - m.at(2, 0) * m.at(0, 1)) * ood,
                (m.at(0, 0) * m.at(1, 1) - m.at(1, 0) * m.at(0, 1)) * ood,
            );
            return out;
        }

        fn det4(m: Self) T {
            const s00 = m.at(2, 2) * m.at(3, 3) - m.at(3, 2) * m.at(2, 3);
            const s01 = m.at(2, 1) * m.at(3, 3) - m.at(3, 1) * m.at(2, 3);
            const s02 = m.at(2, 1) * m.at(3, 2) - m.at(3, 1) * m.at(2, 2);
            const s03 = m.at(2, 0) * m.at(3, 3) - m.at(3, 0) * m.at(2, 3);
            const s04 = m.at(2, 0) * m.at(3, 2) - m.at(3, 0) * m.at(2, 2);
            const s05 = m.at(2, 0) * m.at(3, 1) - m.at(3, 0) * m.at(2, 1);
            const cof0 = m.at(1, 1) * s00 - m.at(1, 2) * s01 + m.at(1, 3) * s02;
            const cof1 = -(m.at(1, 0) * s00 - m.at(1, 2) * s03 + m.at(1, 3) * s04);
            const cof2 = m.at(1, 0) * s01 - m.at(1, 1) * s03 + m.at(1, 3) * s05;
            const cof3 = -(m.at(1, 0) * s02 - m.at(1, 1) * s04 + m.at(1, 2) * s05);
            return m.at(0, 0) * cof0 + m.at(0, 1) * cof1 + m.at(0, 2) * cof2 + m.at(0, 3) * cof3;
        }
        fn inv4(m: Self) Self {
            const V4 = @Vector(4, T);
            const c0 = m.cols[0].simd();
            const c1 = m.cols[1].simd();
            const c2 = m.cols[2].simd();
            const c3 = m.cols[3].simd();

            const Coef00 = c2[2] * c3[3] - c3[2] * c2[3];
            const Coef02 = c1[2] * c3[3] - c3[2] * c1[3];
            const Coef03 = c1[2] * c2[3] - c2[2] * c1[3];
            const Coef04 = c2[1] * c3[3] - c3[1] * c2[3];
            const Coef06 = c1[1] * c3[3] - c3[1] * c1[3];
            const Coef07 = c1[1] * c2[3] - c2[1] * c1[3];
            const Coef08 = c2[1] * c3[2] - c3[1] * c2[2];
            const Coef10 = c1[1] * c3[2] - c3[1] * c1[2];
            const Coef11 = c1[1] * c2[2] - c2[1] * c1[2];
            const Coef12 = c2[0] * c3[3] - c3[0] * c2[3];
            const Coef14 = c1[0] * c3[3] - c3[0] * c1[3];
            const Coef15 = c1[0] * c2[3] - c2[0] * c1[3];
            const Coef16 = c2[0] * c3[2] - c3[0] * c2[2];
            const Coef18 = c1[0] * c3[2] - c3[0] * c1[2];
            const Coef19 = c1[0] * c2[2] - c2[0] * c1[2];
            const Coef20 = c2[0] * c3[1] - c3[0] * c2[1];
            const Coef22 = c1[0] * c3[1] - c3[0] * c1[1];
            const Coef23 = c1[0] * c2[1] - c2[0] * c1[1];

            const Fac0 = V4{ Coef00, Coef00, Coef02, Coef03 };
            const Fac1 = V4{ Coef04, Coef04, Coef06, Coef07 };
            const Fac2 = V4{ Coef08, Coef08, Coef10, Coef11 };
            const Fac3 = V4{ Coef12, Coef12, Coef14, Coef15 };
            const Fac4 = V4{ Coef16, Coef16, Coef18, Coef19 };
            const Fac5 = V4{ Coef20, Coef20, Coef22, Coef23 };

            const V0 = V4{ c1[0], c0[0], c0[0], c0[0] };
            const V1 = V4{ c1[1], c0[1], c0[1], c0[1] };
            const V2 = V4{ c1[2], c0[2], c0[2], c0[2] };
            const V3 = V4{ c1[3], c0[3], c0[3], c0[3] };

            const Inv0 = V1 * Fac0 - V2 * Fac1 + V3 * Fac2;
            const Inv1 = V0 * Fac0 - V2 * Fac3 + V3 * Fac4;
            const Inv2 = V0 * Fac1 - V1 * Fac3 + V3 * Fac5;
            const Inv3 = V0 * Fac2 - V1 * Fac4 + V2 * Fac5;

            const SignA = V4{ 1, -1, 1, -1 };
            const SignB = V4{ -1, 1, -1, 1 };
            const I0 = Inv0 * SignA;
            const I1 = Inv1 * SignB;
            const I2 = Inv2 * SignA;
            const I3 = Inv3 * SignB;

            const Row0 = V4{ I0[0], I1[0], I2[0], I3[0] };
            const Dot0 = c0 * Row0;
            const det = (Dot0[0] + Dot0[1]) + (Dot0[2] + Dot0[3]);
            const ood: V4 = @splat(1.0 / det);

            var out: Self = undefined;
            out.cols[0] = ColVec.fromSimd(I0 * ood);
            out.cols[1] = ColVec.fromSimd(I1 * ood);
            out.cols[2] = ColVec.fromSimd(I2 * ood);
            out.cols[3] = ColVec.fromSimd(I3 * ood);
            return out;
        }
    };
}

/// Outer product `c ⊗ r` → matrix with `r.dim` columns and `c.dim` rows
/// (GLSL `outerProduct`).
pub fn outerProduct(col_vec: anytype, row_vec: anytype) Mat(@TypeOf(row_vec).dim, @TypeOf(col_vec).dim, @TypeOf(col_vec).Element) {
    const T = @TypeOf(col_vec).Element;
    const R = @TypeOf(col_vec).dim;
    const C = @TypeOf(row_vec).dim;
    var m: Mat(C, R, T) = undefined;
    const rv = row_vec.toArray();
    inline for (0..C) |j| m.cols[j] = col_vec.scale(rv[j]);
    return m;
}

// --- aliases (GLM naming: matCxR = C columns, R rows) ----------------------

pub const Mat2 = Mat(2, 2, f32);
pub const Mat3 = Mat(3, 3, f32);
pub const Mat4 = Mat(4, 4, f32);
pub const Mat2x2 = Mat(2, 2, f32);
pub const Mat2x3 = Mat(2, 3, f32);
pub const Mat2x4 = Mat(2, 4, f32);
pub const Mat3x2 = Mat(3, 2, f32);
pub const Mat3x3 = Mat(3, 3, f32);
pub const Mat3x4 = Mat(3, 4, f32);
pub const Mat4x2 = Mat(4, 2, f32);
pub const Mat4x3 = Mat(4, 3, f32);
pub const Mat4x4 = Mat(4, 4, f32);

pub const DMat2 = Mat(2, 2, f64);
pub const DMat3 = Mat(3, 3, f64);
pub const DMat4 = Mat(4, 4, f64);

// ===== folded gtx matrix modules ===========================================
// --- matrix_decompose ---


pub const Decomposed = struct {
    translation: Vec3,
    rotation: Quat,
    scale: Vec3,
    skew: Vec3,
    perspective: Vec4,
};

pub fn decompose(model: Mat4) ?Decomposed {
    const eps = 1e-8;
    var local = model;
    if (@abs(local.at(3, 3)) < eps) return null;

    // Normalize so that local[3][3] == 1.
    const inv33 = 1.0 / local.at(3, 3);
    inline for (0..4) |i| local.cols[i] = local.cols[i].scale(inv33);

    // Perspective matrix: clear the projective column, also tests singularity.
    var persp = local;
    inline for (0..3) |i| persp.cols[i] = persp.cols[i].set(3, 0);
    persp.cols[3] = persp.cols[3].set(3, 1);
    if (@abs(persp.determinant()) < eps) return null;

    // Isolate perspective.
    var perspective = Vec4.init(0, 0, 0, 1);
    if (local.at(0, 3) != 0 or local.at(1, 3) != 0 or local.at(2, 3) != 0) {
        const rhs = Vec4.init(local.at(0, 3), local.at(1, 3), local.at(2, 3), local.at(3, 3));
        perspective = persp.inverse().transpose().mulVec(rhs);
        inline for (0..3) |i| local.cols[i] = local.cols[i].set(3, 0);
        local.cols[3] = local.cols[3].set(3, 1);
    }

    const translation = Vec3.init(local.at(3, 0), local.at(3, 1), local.at(3, 2));

    // Columns of the upper-left 3x3.
    var c0 = Vec3.init(local.at(0, 0), local.at(0, 1), local.at(0, 2));
    var c1 = Vec3.init(local.at(1, 0), local.at(1, 1), local.at(1, 2));
    var c2 = Vec3.init(local.at(2, 0), local.at(2, 1), local.at(2, 2));

    var scale: Vec3 = undefined;
    var skew: Vec3 = undefined;
    scale.x = c0.length();
    c0 = c0.normalize();
    skew.z = c0.dot(c1);
    c1 = c1.sub(c0.scale(skew.z));
    scale.y = c1.length();
    c1 = c1.normalize();
    skew.z /= scale.y;
    skew.y = c0.dot(c2);
    c2 = c2.sub(c0.scale(skew.y));
    skew.x = c1.dot(c2);
    c2 = c2.sub(c1.scale(skew.x));
    scale.z = c2.length();
    c2 = c2.normalize();
    skew.y /= scale.z;
    skew.x /= scale.z;

    if (c0.dot(c1.cross(c2)) < 0) {
        scale = scale.neg();
        c0 = c0.neg();
        c1 = c1.neg();
        c2 = c2.neg();
    }

    return .{
        .translation = translation,
        .rotation = Quat.fromMat3(Mat3.fromColumns(.{ c0, c1, c2 })),
        .scale = scale,
        .skew = skew,
        .perspective = perspective,
    };
}

/// Rebuild a 4x4 matrix from decomposed components (inverse of `decompose`;
/// exact for affine transforms, approximate for the perspective row).
pub fn recompose(d: Decomposed) Mat4 {
    const rm = d.rotation.toMat3();
    const c0n = rm.cols[0];
    const c1n = rm.cols[1];
    const c2n = rm.cols[2];
    const c0 = c0n.scale(d.scale.x);
    const c1 = c1n.scale(d.scale.y).add(c0n.scale(d.skew.z * d.scale.y));
    const c2 = c2n.scale(d.scale.z).add(c1n.scale(d.skew.x * d.scale.z)).add(c0n.scale(d.skew.y * d.scale.z));
    return Mat4.fromColumns(.{
        Vec4.fromVec3(c0, d.perspective.x),
        Vec4.fromVec3(c1, d.perspective.y),
        Vec4.fromVec3(c2, d.perspective.z),
        Vec4.init(d.translation.x, d.translation.y, d.translation.z, d.perspective.w),
    });
}


// --- matrix_interpolation ---


pub fn axisAngleMatrix(axis: Vec3, angle: f32) Mat4 {
    return transform.rotation(angle, axis);
}

/// Decompose both transforms, interpolate (slerp rotation, lerp T/S), recompose.
pub fn interpolate(m1: Mat4, m2: Mat4, delta: f32) Mat4 {
    const d1 = decompose(m1) orelse return m1;
    const d2 = decompose(m2) orelse return m2;
    const t = d1.translation.lerp(d2.translation, delta);
    const s = d1.scale.lerp(d2.scale, delta);
    const r = d1.rotation.slerp(d2.rotation, delta);
    return transform.translation(t).mul(r.toMat4()).mul(transform.scaling(s));
}

/// Return the rotation-only part of an affine transform (translation zeroed).
pub fn extractMatrixRotation(m: Mat4) Mat4 {
    return Mat4.fromColumns(.{
        vec.Vec4.init(m.at(0, 0), m.at(0, 1), m.at(0, 2), 0),
        vec.Vec4.init(m.at(1, 0), m.at(1, 1), m.at(1, 2), 0),
        vec.Vec4.init(m.at(2, 0), m.at(2, 1), m.at(2, 2), 0),
        vec.Vec4.init(0, 0, 0, 1),
    });
}


// --- matrix_factorisation ---


fn flipRows(m: anytype) @TypeOf(m) {
    const M = @TypeOf(m);
    const n = M.cols_n;
    var out: M = undefined;
    inline for (0..n) |c| {
        var arr = m.cols[c].toArray();
        inline for (0..n / 2) |k| {
            const tmp = arr[k];
            arr[k] = arr[n - 1 - k];
            arr[n - 1 - k] = tmp;
        }
        out.cols[c] = M.ColVec.fromArray(arr);
    }
    return out;
}
fn flipCols(m: anytype) @TypeOf(m) {
    const M = @TypeOf(m);
    const n = M.cols_n;
    var out: M = undefined;
    inline for (0..n) |c| out.cols[c] = m.cols[n - 1 - c];
    return out;
}

// === PCA (covariance + symmetric eigensolve) ===


/// Covariance matrix of points about the origin.
pub fn covariance(points: []const Vec3) Mat3 {
    return covarianceCentered(points, Vec3.splat(0));
}
/// Covariance matrix of points about `center`.
pub fn covarianceCentered(points: []const Vec3, center: Vec3) Mat3 {
    var m = Mat3.zero();
    for (points) |p| {
        const a = p.sub(center).toArray();
        inline for (0..3) |x| {
            inline for (0..3) |y| m.cols[x] = m.cols[x].set(y, m.at(x, y) + a[x] * a[y]);
        }
    }
    if (points.len > 0) {
        const inv = 1.0 / @as(f32, @floatFromInt(points.len));
        inline for (0..3) |x| m.cols[x] = m.cols[x].scale(inv);
    }
    return m;
}

pub const Eigen = struct {
    /// Eigenvalues.
    values: Vec3,
    /// Eigenvectors, one per column (column `i` ↔ `values[i]`).
    vectors: Mat3,
};

/// Eigenvalues and eigenvectors of a symmetric real 3x3 matrix (cyclic Jacobi).
pub fn eigenSymmetric(m: Mat3) Eigen {
    var a: [3][3]f32 = undefined;
    inline for (0..3) |i| inline for (0..3) |j| {
        a[i][j] = m.at(j, i);
    };
    var v: [3][3]f32 = .{ .{ 1, 0, 0 }, .{ 0, 1, 0 }, .{ 0, 0, 1 } };

    var sweep: usize = 0;
    while (sweep < 50) : (sweep += 1) {
        const off = @abs(a[0][1]) + @abs(a[0][2]) + @abs(a[1][2]);
        if (off < 1e-12) break;
        inline for (.{ [2]usize{ 0, 1 }, [2]usize{ 0, 2 }, [2]usize{ 1, 2 } }) |pq| {
            const p = pq[0];
            const q = pq[1];
            if (@abs(a[p][q]) > 1e-15) {
                const phi = 0.5 * std.math.atan2(2 * a[p][q], a[q][q] - a[p][p]);
                const c = @cos(phi);
                const s = @sin(phi);
                const app = a[p][p];
                const aqq = a[q][q];
                const apq = a[p][q];
                a[p][p] = c * c * app - 2 * s * c * apq + s * s * aqq;
                a[q][q] = s * s * app + 2 * s * c * apq + c * c * aqq;
                a[p][q] = 0;
                a[q][p] = 0;
                inline for (0..3) |r| {
                    if (r != p and r != q) {
                        const arp = a[r][p];
                        const arq = a[r][q];
                        a[r][p] = c * arp - s * arq;
                        a[p][r] = a[r][p];
                        a[r][q] = s * arp + c * arq;
                        a[q][r] = a[r][q];
                    }
                }
                inline for (0..3) |k| {
                    const vkp = v[k][p];
                    const vkq = v[k][q];
                    v[k][p] = c * vkp - s * vkq;
                    v[k][q] = s * vkp + c * vkq;
                }
            }
        }
    }
    return .{
        .values = Vec3.init(a[0][0], a[1][1], a[2][2]),
        .vectors = Mat3.fromColumns(.{
            Vec3.init(v[0][0], v[1][0], v[2][0]),
            Vec3.init(v[0][1], v[1][1], v[2][1]),
            Vec3.init(v[0][2], v[1][2], v[2][2]),
        }),
    };
}

const testing = std.testing;
const Vec2 = vec.Vec2;
const Vec3 = vec.Vec3;
const Vec4 = vec.Vec4;

test "identity and mulVec" {
    const m = Mat4.identity();
    const v = Vec4.init(1, 2, 3, 4);
    try testing.expect(m.mulVec(v).eql(v));
    try testing.expectEqual(@as(f32, 1), m.at(0, 0));
    try testing.expectEqual(@as(f32, 0), m.at(1, 0));
}

test "matrix product = identity when A * inverse(A)" {
    const a = Mat3.fromColumns(.{
        Vec3.init(2, 0, 1),
        Vec3.init(1, 3, 2),
        Vec3.init(1, 0, 4),
    });
    const ai = a.inverse();
    try testing.expect(a.mul(ai).approxEql(Mat3.identity(), 1e-5));
    try testing.expect(ai.mul(a).approxEql(Mat3.identity(), 1e-5));
}

test "mat4 inverse round trip + determinant" {
    const a = Mat4.fromColumns(.{
        Vec4.init(1, 0, 0, 0),
        Vec4.init(0, 2, 0, 0),
        Vec4.init(0, 0, 3, 0),
        Vec4.init(4, 5, 6, 1),
    });
    try testing.expectApproxEqAbs(@as(f32, 6), a.determinant(), 1e-4);
    try testing.expect(a.mul(a.inverse()).approxEql(Mat4.identity(), 1e-5));
}

test "transpose (non-square)" {
    const a = Mat2x3.fromColumns(.{ Vec3.init(1, 2, 3), Vec3.init(4, 5, 6) });
    const t = a.transpose(); // Mat3x2
    try testing.expectEqual(@as(f32, 1), t.at(0, 0));
    try testing.expectEqual(@as(f32, 2), t.at(1, 0));
    try testing.expectEqual(@as(f32, 4), t.at(0, 1));
}

test "outerProduct" {
    const m = outerProduct(Vec3.init(1, 2, 3), Vec2.init(10, 20)); // Mat2x3
    try testing.expectEqual(@as(f32, 10), m.at(0, 0));
    try testing.expectEqual(@as(f32, 20), m.at(1, 0));
    try testing.expectEqual(@as(f32, 60), m.at(1, 2));
}

test "affineInverse matches full inverse" {
    var a = Mat4.identity();
    a.cols[0] = Vec4.init(0, 1, 0, 0); // 90° rotation about Z ...
    a.cols[1] = Vec4.init(-1, 0, 0, 0);
    a.cols[3] = Vec4.init(5, 6, 7, 1); // ... plus a translation
    try testing.expect(a.mul(a.affineInverse()).approxEql(Mat4.identity(), 1e-5));
    try testing.expect(a.affineInverse().approxEql(a.inverse(), 1e-5));
}

test "rigidInverse / mulAffine / mulChain" {
    var a = Mat4.identity();
    a.cols[0] = Vec4.init(0, 1, 0, 0);
    a.cols[1] = Vec4.init(-1, 0, 0, 0);
    a.cols[3] = Vec4.init(5, 6, 7, 1);
    try testing.expect(a.mul(a.rigidInverse()).approxEql(Mat4.identity(), 1e-5));
    try testing.expect(a.rigidInverse().approxEql(a.affineInverse(), 1e-5));
    var b = Mat4.identity();
    b.cols[3] = Vec4.init(1, 2, 3, 1);
    try testing.expect(a.mulAffine(b).approxEql(a.mul(b), 1e-5));
    try testing.expect(Mat4.mulChain(&.{ a, b }).approxEql(a.mul(b), 1e-5));
}

test "Mat.diagonal (vector) + non-square" {
    const m = Mat3.diagonal(vec.Vec3.init(2, 3, 4));
    try testing.expectEqual(@as(f32, 2), m.at(0, 0));
    try testing.expectEqual(@as(f32, 4), m.at(2, 2));
    try testing.expectEqual(@as(f32, 0), m.at(0, 1));
    try testing.expectEqual(@as(f32, 3), Mat3.diagonal(@as(f32, 3)).at(1, 1)); // scalar form
    try testing.expectEqual(@as(f32, 5), Mat(2, 3, f32).diagonal(Vec2.init(5, 6)).at(0, 0)); // non-square
}

test "Mat.crossMatrix" {
    const a = Vec3.init(1, 2, 3);
    const b = Vec3.init(4, 5, 6);
    try testing.expect(Mat3.crossMatrix(a).mulVec(b).approxEql(a.cross(b), 1e-6));
    try testing.expect(Mat4.crossMatrix(a).mulVec(vec.Vec4.fromVec3(b, 0)).approxEql(vec.Vec4.fromVec3(a.cross(b), 0), 1e-6));
}

test "decompose TRS (+ trivial perspective)" {
    const S = Mat4.fromColumns(.{
        Vec4.init(2, 0, 0, 0), Vec4.init(0, 3, 0, 0), Vec4.init(0, 0, 4, 0), Vec4.init(0, 0, 0, 1),
    });
    const R = Mat4.fromColumns(.{
        Vec4.init(0, 1, 0, 0), Vec4.init(-1, 0, 0, 0), Vec4.init(0, 0, 1, 0), Vec4.init(0, 0, 0, 1),
    });
    var T = Mat4.identity();
    T.cols[3] = Vec4.init(5, 6, 7, 1);
    const d = decompose(T.mul(R).mul(S)).?;
    try testing.expect(d.translation.approxEql(Vec3.init(5, 6, 7), 1e-4));
    try testing.expect(d.scale.approxEql(Vec3.init(2, 3, 4), 1e-4));
    try testing.expect(d.rotation.rotateVec(Vec3.init(1, 0, 0)).approxEql(Vec3.init(0, 1, 0), 1e-4));
    try testing.expect(d.perspective.approxEql(Vec4.init(0, 0, 0, 1), 1e-5));
}

test "decompose recovers a perspective row" {
    var m = Mat4.identity();
    m.cols[0] = m.cols[0].set(3, 0.1); // local[0][3]
    m.cols[2] = m.cols[2].set(3, -1); // typical projective term
    const d = decompose(m).?;
    // perspective should be non-trivial now
    try testing.expect(!d.perspective.approxEql(Vec4.init(0, 0, 0, 1), 1e-5));
}

test "matrix_interpolation endpoints" {
    const a = Mat4.identity();
    const b = transform.translation(Vec3.init(10, 0, 0));
    try testing.expect(interpolate(a, b, 0).approxEql(a, 1e-4));
    const mid = interpolate(a, b, 0.5).mulVec(Vec4.init(0, 0, 0, 1));
    try testing.expect(mid.approxEql(Vec4.init(5, 0, 0, 1), 1e-4));
}

test "qr / rq decomposition" {
    const a = Mat3.fromColumns(.{
        Vec3.init(2, 1, 0),
        Vec3.init(1, 3, 1),
        Vec3.init(0, 1, 4),
    });
    const d = a.qr();
    try testing.expect(d.q.mul(d.r).approxEql(a, 1e-4)); // A = QR
    try testing.expect(d.q.transpose().mul(d.q).approxEql(Mat3.identity(), 1e-4)); // Q orthonormal
    // R upper-triangular
    try testing.expectApproxEqAbs(@as(f32, 0), d.r.at(0, 1), 1e-5);

    const e = a.rq();
    try testing.expect(e.r.mul(e.q).approxEql(a, 1e-4)); // A = RQ
    try testing.expect(e.q.transpose().mul(e.q).approxEql(Mat3.identity(), 1e-4));
}

test "Mat.fromRows (row-major) + fromColumns" {
    const m = Mat3.fromRows(.{ Vec3.init(1, 2, 3), Vec3.init(4, 5, 6), Vec3.init(7, 8, 9) });
    try testing.expectEqual(@as(f32, 1), m.at(0, 0)); // row 0, col 0
    try testing.expectEqual(@as(f32, 2), m.at(1, 0)); // row 0, col 1
    try testing.expectEqual(@as(f32, 4), m.at(0, 1)); // row 1, col 0
    // colMajor == fromColumns
    const c = Mat3.fromColumns(.{ Vec3.init(1, 2, 3), Vec3.init(4, 5, 6), Vec3.init(7, 8, 9) });
    try testing.expectEqual(@as(f32, 1), c.at(0, 0));
    try testing.expectEqual(@as(f32, 2), c.at(0, 1));
}

test "matrix_query" {
    try testing.expect(Mat3.identity().isIdentity(1e-6));
    try testing.expect(Mat3.identity().isNormalized(1e-6));
    try testing.expect(Mat3.identity().isOrthogonal(1e-6));
    try testing.expect(!Mat3.identity().isNull(1e-6));
}

test "PCA covariance + eigen" {
    const pts = [_]Vec3{
        Vec3.init(2, 0, 0),  Vec3.init(-2, 0, 0),
        Vec3.init(0, 0.5, 0), Vec3.init(0, -0.5, 0),
        Vec3.init(0, 0, 0.1), Vec3.init(0, 0, -0.1),
    };
    const cov = covariance(&pts);
    const e = eigenSymmetric(cov);
    inline for (0..3) |i| {
        const ev = e.vectors.cols[i];
        try testing.expect(cov.mulVec(ev).approxEql(ev.scale(e.values.toArray()[i]), 1e-4));
    }
}
