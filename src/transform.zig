//! Affine model transforms (`Transforms(T)`) + camera/projection matrices
//! (`Camera(T)`).
//!
//! Projection variants carry explicit suffixes:
//!   * `Rh`/`Lh` — right/left-handed view space
//!   * `Zo`/`No`  — clip-space depth `[0, 1]` (Vulkan/WebGPU/Metal/D3D) or
//!                  `[-1, 1]` ("negative-one", classic OpenGL)
//!
//! NDC Y direction and reverse-Z are set by the `config.clip` struct and apply
//! to every variant. Defaults target **Vulkan**: right-handed, depth `[0, 1]`,
//! **+Y down**, reverse-Z. Set `math_clip` in your root file to change them.

const std = @import("std");
const vec = @import("vec.zig");
const mat = @import("mat.zig");
const config = @import("config.zig");
const intersect = @import("intersect.zig");
const Vec = vec.Vec;
const Mat = mat.Mat;

/// Transform constructors parameterized by float element type `T`.
pub fn Transforms(comptime T: type) type {
    const Vec2 = Vec(2, T);
    const Vec3 = Vec(3, T);
    const Vec4 = Vec(4, T);
    const Mat3 = Mat(3, 3, T);
    const Mat4 = Mat(4, 4, T);

    return struct {
        // --- model transforms -------------------------------------------

        pub fn translation(v: Vec3) Mat4 {
            var m = Mat4.identity();
            m.cols[3] = Vec4.init(v.x, v.y, v.z, 1);
            return m;
        }
        /// Post-multiply a translation onto `m` (i.e. `m * T(v)`).
        pub fn translate(m: Mat4, v: Vec3) Mat4 {
            var r = m;
            r.cols[3] = m.cols[0].scale(v.x)
                .add(m.cols[1].scale(v.y))
                .add(m.cols[2].scale(v.z))
                .add(m.cols[3]);
            return r;
        }

        pub fn scaling(v: Vec3) Mat4 {
            return Mat4.fromColumns(.{
                Vec4.init(v.x, 0, 0, 0),
                Vec4.init(0, v.y, 0, 0),
                Vec4.init(0, 0, v.z, 0),
                Vec4.init(0, 0, 0, 1),
            });
        }
        pub fn scale(m: Mat4, v: Vec3) Mat4 {
            return Mat4.fromColumns(.{
                m.cols[0].scale(v.x),
                m.cols[1].scale(v.y),
                m.cols[2].scale(v.z),
                m.cols[3],
            });
        }

        /// Rotation matrix of `angle` radians about (unit-normalized) `axis`.
        pub fn rotation(angle: T, axis: Vec3) Mat4 {
            const c = @cos(angle);
            const s = @sin(angle);
            const a = axis.normalize();
            const t = a.scale(1 - c); // temp
            return Mat4.fromColumns(.{
                Vec4.init(c + t.x * a.x, t.x * a.y + s * a.z, t.x * a.z - s * a.y, 0),
                Vec4.init(t.y * a.x - s * a.z, c + t.y * a.y, t.y * a.z + s * a.x, 0),
                Vec4.init(t.z * a.x + s * a.y, t.z * a.y - s * a.x, c + t.z * a.z, 0),
                Vec4.init(0, 0, 0, 1),
            });
        }
        /// Post-multiply a rotation onto `m` (i.e. `m * R`).
        pub fn rotate(m: Mat4, angle: T, axis: Vec3) Mat4 {
            return m.mul(rotation(angle, axis));
        }

        // --- 2D affine (Mat3) -------------------------------------------------

        pub fn translate2d(m: Mat3, v: Vec2) Mat3 {
            var r = m;
            r.cols[2] = m.cols[0].scale(v.x).add(m.cols[1].scale(v.y)).add(m.cols[2]);
            return r;
        }
        pub fn rotate2d(m: Mat3, angle: T) Mat3 {
            const c = @cos(angle);
            const s = @sin(angle);
            return m.mul(Mat3.fromColumns(.{
                Vec3.init(c, s, 0),
                Vec3.init(-s, c, 0),
                Vec3.init(0, 0, 1),
            }));
        }
        pub fn scale2d(m: Mat3, v: Vec2) Mat3 {
            return Mat3.fromColumns(.{ m.cols[0].scale(v.x), m.cols[1].scale(v.y), m.cols[2] });
        }
        pub fn shearX2d(m: Mat3, yk: T) Mat3 {
            return m.mul(Mat3.fromColumns(.{
                Vec3.init(1, 0, 0),
                Vec3.init(yk, 1, 0),
                Vec3.init(0, 0, 1),
            }));
        }
        pub fn shearY2d(m: Mat3, xk: T) Mat3 {
            return m.mul(Mat3.fromColumns(.{
                Vec3.init(1, xk, 0),
                Vec3.init(0, 1, 0),
                Vec3.init(0, 0, 1),
            }));
        }

        // --- 3D shear (Mat4) --------------------------------------------------

        pub fn shearX3d(m: Mat4, s: T, t: T) Mat4 {
            return m.mul(Mat4.fromColumns(.{
                Vec4.init(1, s, t, 0), Vec4.init(0, 1, 0, 0), Vec4.init(0, 0, 1, 0), Vec4.init(0, 0, 0, 1),
            }));
        }
        pub fn shearY3d(m: Mat4, s: T, t: T) Mat4 {
            return m.mul(Mat4.fromColumns(.{
                Vec4.init(1, 0, 0, 0), Vec4.init(s, 1, t, 0), Vec4.init(0, 0, 1, 0), Vec4.init(0, 0, 0, 1),
            }));
        }
        pub fn shearZ3d(m: Mat4, s: T, t: T) Mat4 {
            return m.mul(Mat4.fromColumns(.{
                Vec4.init(1, 0, 0, 0), Vec4.init(0, 1, 0, 0), Vec4.init(s, t, 1, 0), Vec4.init(0, 0, 0, 1),
            }));
        }

        // --- reflection / projection onto lines & planes ----------------------

        /// Reflection about a 2-D line (normal, signed distance from origin).
        pub fn reflect2d(m: Mat3, normal: Vec2, distance: T) Mat3 {
            const n = normal;
            return m.mul(Mat3.fromColumns(.{
                Vec3.init(1 - 2 * n.x * n.x, -2 * n.y * n.x, 0),
                Vec3.init(-2 * n.x * n.y, 1 - 2 * n.y * n.y, 0),
                Vec3.init(-2 * n.x * distance, -2 * n.y * distance, 1),
            }));
        }
        /// Reflection about a 3-D plane (normal, signed distance from origin).
        pub fn reflect3d(m: Mat4, normal: Vec3, distance: T) Mat4 {
            const n = normal;
            return m.mul(Mat4.fromColumns(.{
                Vec4.init(1 - 2 * n.x * n.x, -2 * n.y * n.x, -2 * n.z * n.x, 0),
                Vec4.init(-2 * n.x * n.y, 1 - 2 * n.y * n.y, -2 * n.z * n.y, 0),
                Vec4.init(-2 * n.x * n.z, -2 * n.y * n.z, 1 - 2 * n.z * n.z, 0),
                Vec4.init(-2 * n.x * distance, -2 * n.y * distance, -2 * n.z * distance, 1),
            }));
        }
        /// Orthographic projection onto a 2-D line through the origin.
        pub fn proj2d(m: Mat3, normal: Vec3) Mat3 {
            const n = normal;
            return m.mul(Mat3.fromColumns(.{
                Vec3.init(1 - n.x * n.x, -n.x * n.y, 0),
                Vec3.init(-n.x * n.y, 1 - n.y * n.y, 0),
                Vec3.init(0, 0, 1),
            }));
        }
        /// Orthographic projection onto a 3-D plane through the origin.
        pub fn proj3d(m: Mat4, normal: Vec3) Mat4 {
            const n = normal;
            return m.mul(Mat4.fromColumns(.{
                Vec4.init(1 - n.x * n.x, -n.x * n.y, -n.x * n.z, 0),
                Vec4.init(-n.x * n.y, 1 - n.y * n.y, -n.y * n.z, 0),
                Vec4.init(-n.x * n.z, -n.y * n.z, 1 - n.z * n.z, 0),
                Vec4.init(0, 0, 0, 1),
            }));
        }
        /// Uniform scale + uniform translation (bias).
        pub fn scaleBias(scaleVal: T, bias: T) Mat4 {
            return Mat4.fromColumns(.{
                Vec4.init(scaleVal, 0, 0, 0), Vec4.init(0, scaleVal, 0, 0),
                Vec4.init(0, 0, scaleVal, 0), Vec4.init(bias, bias, bias, 1),
            });
        }

        /// Like `rotate`, but assumes `axis` is already unit length.
        pub fn rotateNormalizedAxis(m: Mat4, angle: T, axis: Vec3) Mat4 {
            const c = @cos(angle);
            const s = @sin(angle);
            const t = axis.scale(1 - c);
            const r = Mat4.fromColumns(.{
                Vec4.init(c + t.x * axis.x, t.x * axis.y + s * axis.z, t.x * axis.z - s * axis.y, 0),
                Vec4.init(t.y * axis.x - s * axis.z, c + t.y * axis.y, t.y * axis.z + s * axis.x, 0),
                Vec4.init(t.z * axis.x + s * axis.y, t.z * axis.y - s * axis.x, c + t.z * axis.z, 0),
                Vec4.init(0, 0, 0, 1),
            });
            return m.mul(r);
        }
    };
}

/// f32 default-precision transforms.
pub const transform = Transforms(f32);

/// Camera/view + projection-matrix builders parameterized by float element `T`.
///
/// Handedness/clip-depth come in explicit variants (`Rh`/`Lh`, `Zo`/`No`); the
/// unsuffixed names dispatch to the configured default (right-handed, depth 0..1).
pub fn Camera(comptime T: type) type {
    const Vec2 = Vec(2, T);
    const Vec3 = Vec(3, T);
    const Vec4 = Vec(4, T);
    const Mat4 = Mat(4, 4, T);
    const M = Transforms(T);

    return struct {
        // --- view (lookAt) ----------------------------------------------

        pub fn lookAtRh(eye: Vec3, center: Vec3, up: Vec3) Mat4 {
            const f = center.sub(eye).normalize();
            const s = f.cross(up).normalize();
            const u = s.cross(f);
            return Mat4.fromColumns(.{
                Vec4.init(s.x, u.x, -f.x, 0),
                Vec4.init(s.y, u.y, -f.y, 0),
                Vec4.init(s.z, u.z, -f.z, 0),
                Vec4.init(-s.dot(eye), -u.dot(eye), f.dot(eye), 1),
            });
        }
        pub fn lookAtLh(eye: Vec3, center: Vec3, up: Vec3) Mat4 {
            const f = center.sub(eye).normalize();
            const s = up.cross(f).normalize();
            const u = f.cross(s);
            return Mat4.fromColumns(.{
                Vec4.init(s.x, u.x, f.x, 0),
                Vec4.init(s.y, u.y, f.y, 0),
                Vec4.init(s.z, u.z, f.z, 0),
                Vec4.init(-s.dot(eye), -u.dot(eye), -f.dot(eye), 1),
            });
        }
        pub const lookAt = switch (config.clip.handedness) {
            .right => lookAtRh,
            .left => lookAtLh,
        };

        // --- variant cores (Rh/Lh × Zo/No, generated wrappers) ----------
        //
        // Each projection family has one comptime-parameterized core; the named
        // `*RhZo`/`*LhNo`/… variants and the config-default forward to it. The
        // depth row (`m22`,`m32`) and the perspective `w` sign are the only
        // things that differ across handedness/clip-depth.

        const Hand = config.Handedness;
        const Depth = config.ClipDepth;

        inline fn wSign(comptime hand: Hand) T {
            return if (hand == .right) -1 else 1;
        }
        /// Perspective depth-row entries: `{ col2.z, col3.z }`.
        inline fn perspectiveDepth(near: T, far: T, comptime hand: Hand, comptime depth: Depth) [2]T {
            return switch (depth) {
                .zero_to_one => .{
                    if (hand == .right) far / (near - far) else far / (far - near),
                    -(far * near) / (far - near),
                },
                .neg_one_to_one => .{
                    if (hand == .right) -(far + near) / (far - near) else (far + near) / (far - near),
                    -(2 * far * near) / (far - near),
                },
            };
        }
        /// Apply the configured clip-space conventions to a finished projection:
        ///   * NDC Y direction — Vulkan (default) has +Y down, so row 1 is negated.
        ///   * reverse-Z — remap clip z to `w - z` (near→1, far→0); `[0,1]` only.
        inline fn clipConv(m: Mat4) Mat4 {
            var r = m;
            if (config.clip.y == .down) {
                inline for (0..4) |i| r.cols[i] = r.cols[i].set(1, -r.cols[i].get(1));
            }
            if (config.clip.reverse_z and config.clip.depth == .zero_to_one) {
                // clip_z' = w - z  ->  row2' = row3 - row2 (per column)
                inline for (0..4) |i| r.cols[i] = r.cols[i].set(2, r.cols[i].get(3) - r.cols[i].get(2));
            }
            return r;
        }

        // --- perspective -------------------------------------------------

        fn perspectiveImpl(fovy: T, aspect: T, near: T, far: T, comptime hand: Hand, comptime depth: Depth) Mat4 {
            const th = @tan(fovy * 0.5);
            const zw = perspectiveDepth(near, far, hand, depth);
            return clipConv(Mat4.fromColumns(.{
                Vec4.init(1.0 / (aspect * th), 0, 0, 0),
                Vec4.init(0, 1.0 / th, 0, 0),
                Vec4.init(0, 0, zw[0], wSign(hand)),
                Vec4.init(0, 0, zw[1], 0),
            }));
        }
        pub fn perspectiveRhZo(fovy: T, aspect: T, near: T, far: T) Mat4 {
            return perspectiveImpl(fovy, aspect, near, far, .right, .zero_to_one);
        }
        pub fn perspectiveRhNo(fovy: T, aspect: T, near: T, far: T) Mat4 {
            return perspectiveImpl(fovy, aspect, near, far, .right, .neg_one_to_one);
        }
        pub fn perspectiveLhZo(fovy: T, aspect: T, near: T, far: T) Mat4 {
            return perspectiveImpl(fovy, aspect, near, far, .left, .zero_to_one);
        }
        pub fn perspectiveLhNo(fovy: T, aspect: T, near: T, far: T) Mat4 {
            return perspectiveImpl(fovy, aspect, near, far, .left, .neg_one_to_one);
        }
        /// Default perspective (uses the configured handedness + clip depth).
        pub fn perspective(fovy: T, aspect: T, near: T, far: T) Mat4 {
            return perspectiveImpl(fovy, aspect, near, far, config.clip.handedness, config.clip.depth);
        }

        /// Right-handed perspective with an infinite far plane, depth [0,1].
        pub fn infinitePerspectiveRhZo(fovy: T, aspect: T, near: T) Mat4 {
            const range = @tan(fovy * 0.5) * near;
            const left = -range * aspect;
            const right = range * aspect;
            const bottom = -range;
            const top = range;
            return clipConv(Mat4.fromColumns(.{
                Vec4.init((2 * near) / (right - left), 0, 0, 0),
                Vec4.init(0, (2 * near) / (top - bottom), 0, 0),
                Vec4.init(0, 0, -1, -1),
                Vec4.init(0, 0, -near, 0),
            }));
        }

        // --- orthographic ------------------------------------------------

        fn orthoImpl(left: T, right: T, bottom: T, top: T, near: T, far: T, comptime hand: Hand, comptime depth: Depth) Mat4 {
            const hsign: T = if (hand == .right) -1 else 1;
            const m22: T = switch (depth) {
                .zero_to_one => hsign * 1.0 / (far - near),
                .neg_one_to_one => hsign * 2.0 / (far - near),
            };
            const tz: T = switch (depth) {
                .zero_to_one => -near / (far - near),
                .neg_one_to_one => -(far + near) / (far - near),
            };
            return clipConv(Mat4.fromColumns(.{
                Vec4.init(2.0 / (right - left), 0, 0, 0),
                Vec4.init(0, 2.0 / (top - bottom), 0, 0),
                Vec4.init(0, 0, m22, 0),
                Vec4.init(-(right + left) / (right - left), -(top + bottom) / (top - bottom), tz, 1),
            }));
        }
        pub fn orthoRhZo(left: T, right: T, bottom: T, top: T, near: T, far: T) Mat4 {
            return orthoImpl(left, right, bottom, top, near, far, .right, .zero_to_one);
        }
        pub fn orthoRhNo(left: T, right: T, bottom: T, top: T, near: T, far: T) Mat4 {
            return orthoImpl(left, right, bottom, top, near, far, .right, .neg_one_to_one);
        }
        pub fn orthoLhZo(left: T, right: T, bottom: T, top: T, near: T, far: T) Mat4 {
            return orthoImpl(left, right, bottom, top, near, far, .left, .zero_to_one);
        }
        pub fn orthoLhNo(left: T, right: T, bottom: T, top: T, near: T, far: T) Mat4 {
            return orthoImpl(left, right, bottom, top, near, far, .left, .neg_one_to_one);
        }
        /// Default orthographic (uses the configured handedness + clip depth).
        pub fn ortho(left: T, right: T, bottom: T, top: T, near: T, far: T) Mat4 {
            return orthoImpl(left, right, bottom, top, near, far, config.clip.handedness, config.clip.depth);
        }

        // --- frustum -----------------------------------------------------

        fn frustumImpl(left: T, right: T, bottom: T, top: T, near: T, far: T, comptime hand: Hand, comptime depth: Depth) Mat4 {
            const zw = perspectiveDepth(near, far, hand, depth);
            const ox = (right + left) / (right - left);
            const oy = (top + bottom) / (top - bottom);
            return clipConv(Mat4.fromColumns(.{
                Vec4.init((2 * near) / (right - left), 0, 0, 0),
                Vec4.init(0, (2 * near) / (top - bottom), 0, 0),
                Vec4.init(if (hand == .right) ox else -ox, if (hand == .right) oy else -oy, zw[0], wSign(hand)),
                Vec4.init(0, 0, zw[1], 0),
            }));
        }
        pub fn frustumRhZo(left: T, right: T, bottom: T, top: T, near: T, far: T) Mat4 {
            return frustumImpl(left, right, bottom, top, near, far, .right, .zero_to_one);
        }
        pub fn frustumRhNo(left: T, right: T, bottom: T, top: T, near: T, far: T) Mat4 {
            return frustumImpl(left, right, bottom, top, near, far, .right, .neg_one_to_one);
        }
        pub fn frustumLhZo(left: T, right: T, bottom: T, top: T, near: T, far: T) Mat4 {
            return frustumImpl(left, right, bottom, top, near, far, .left, .zero_to_one);
        }
        pub fn frustumLhNo(left: T, right: T, bottom: T, top: T, near: T, far: T) Mat4 {
            return frustumImpl(left, right, bottom, top, near, far, .left, .neg_one_to_one);
        }
        /// Default frustum (uses the configured handedness + clip depth).
        pub fn frustum(left: T, right: T, bottom: T, top: T, near: T, far: T) Mat4 {
            return frustumImpl(left, right, bottom, top, near, far, config.clip.handedness, config.clip.depth);
        }

        // --- perspectiveFov (fov + framebuffer width/height) -------------

        fn fovWH(fov: T, width: T, height: T) [2]T {
            const h = @cos(0.5 * fov) / @sin(0.5 * fov);
            return .{ h * height / width, h }; // {w, h}
        }
        fn perspectiveFovImpl(fov: T, width: T, height: T, near: T, far: T, comptime hand: Hand, comptime depth: Depth) Mat4 {
            const wh = fovWH(fov, width, height);
            const zw = perspectiveDepth(near, far, hand, depth);
            return clipConv(Mat4.fromColumns(.{
                Vec4.init(wh[0], 0, 0, 0),
                Vec4.init(0, wh[1], 0, 0),
                Vec4.init(0, 0, zw[0], wSign(hand)),
                Vec4.init(0, 0, zw[1], 0),
            }));
        }
        pub fn perspectiveFovRhZo(fov: T, width: T, height: T, near: T, far: T) Mat4 {
            return perspectiveFovImpl(fov, width, height, near, far, .right, .zero_to_one);
        }
        pub fn perspectiveFovRhNo(fov: T, width: T, height: T, near: T, far: T) Mat4 {
            return perspectiveFovImpl(fov, width, height, near, far, .right, .neg_one_to_one);
        }
        pub fn perspectiveFovLhZo(fov: T, width: T, height: T, near: T, far: T) Mat4 {
            return perspectiveFovImpl(fov, width, height, near, far, .left, .zero_to_one);
        }
        pub fn perspectiveFovLhNo(fov: T, width: T, height: T, near: T, far: T) Mat4 {
            return perspectiveFovImpl(fov, width, height, near, far, .left, .neg_one_to_one);
        }
        /// Default perspectiveFov (uses the configured handedness + clip depth).
        pub fn perspectiveFov(fov: T, width: T, height: T, near: T, far: T) Mat4 {
            return perspectiveFovImpl(fov, width, height, near, far, config.clip.handedness, config.clip.depth);
        }

        // --- infinite perspective (zFar -> infinity) ---------------------

        fn infRowsXY(fovy: T, aspect: T) [2]T {
            const t = @tan(fovy * 0.5);
            return .{ 1.0 / (t * aspect), 1.0 / t };
        }
        pub fn infinitePerspectiveRhNo(fovy: T, aspect: T, near: T) Mat4 {
            const xy = infRowsXY(fovy, aspect);
            return clipConv(Mat4.fromColumns(.{
                Vec4.init(xy[0], 0, 0, 0), Vec4.init(0, xy[1], 0, 0),
                Vec4.init(0, 0, -1, -1),   Vec4.init(0, 0, -2 * near, 0),
            }));
        }
        pub fn infinitePerspectiveLhZo(fovy: T, aspect: T, near: T) Mat4 {
            const xy = infRowsXY(fovy, aspect);
            return clipConv(Mat4.fromColumns(.{
                Vec4.init(xy[0], 0, 0, 0), Vec4.init(0, xy[1], 0, 0),
                Vec4.init(0, 0, 1, 1),     Vec4.init(0, 0, -near, 0),
            }));
        }
        pub fn infinitePerspectiveLhNo(fovy: T, aspect: T, near: T) Mat4 {
            const xy = infRowsXY(fovy, aspect);
            return clipConv(Mat4.fromColumns(.{
                Vec4.init(xy[0], 0, 0, 0), Vec4.init(0, xy[1], 0, 0),
                Vec4.init(0, 0, 1, 1),     Vec4.init(0, 0, -2 * near, 0),
            }));
        }
        pub const infinitePerspective = switch (config.clip.handedness) {
            .right => switch (config.clip.depth) {
                .zero_to_one => infinitePerspectiveRhZo,
                .neg_one_to_one => infinitePerspectiveRhNo,
            },
            .left => switch (config.clip.depth) {
                .zero_to_one => infinitePerspectiveLhZo,
                .neg_one_to_one => infinitePerspectiveLhNo,
            },
        };
        /// Right-handed infinite perspective biased by `ep` to avoid z-fighting at infinity.
        pub fn tweakedInfinitePerspective(fovy: T, aspect: T, near: T, ep: T) Mat4 {
            const xy = infRowsXY(fovy, aspect);
            return clipConv(Mat4.fromColumns(.{
                Vec4.init(xy[0], 0, 0, 0),   Vec4.init(0, xy[1], 0, 0),
                Vec4.init(0, 0, ep - 1, -1), Vec4.init(0, 0, (ep - 2) * near, 0),
            }));
        }

        // --- object <-> window projection --------------------------------

        pub fn projectZo(obj: Vec3, model: Mat4, proj: Mat4, viewport: Vec4) Vec3 {
            var t = proj.mulVec(model.mulVec(Vec4.fromVec3(obj, 1)));
            t = t.divScalar(t.w);
            return Vec3.init(
                (t.x * 0.5 + 0.5) * viewport.z + viewport.x,
                (t.y * 0.5 + 0.5) * viewport.w + viewport.y,
                t.z,
            );
        }
        pub fn projectNo(obj: Vec3, model: Mat4, proj: Mat4, viewport: Vec4) Vec3 {
            var t = proj.mulVec(model.mulVec(Vec4.fromVec3(obj, 1)));
            t = t.divScalar(t.w).scale(0.5).addScalar(0.5);
            return Vec3.init(t.x * viewport.z + viewport.x, t.y * viewport.w + viewport.y, t.z);
        }
        pub const project = switch (config.clip.depth) {
            .zero_to_one => projectZo,
            .neg_one_to_one => projectNo,
        };
        pub fn unProjectZo(win: Vec3, model: Mat4, proj: Mat4, viewport: Vec4) Vec3 {
            const inv = proj.mul(model).inverse();
            const tx = ((win.x - viewport.x) / viewport.z) * 2 - 1;
            const ty = ((win.y - viewport.y) / viewport.w) * 2 - 1;
            const obj = inv.mulVec(Vec4.init(tx, ty, win.z, 1));
            const o = obj.divScalar(obj.w);
            return Vec3.init(o.x, o.y, o.z);
        }
        pub fn unProjectNo(win: Vec3, model: Mat4, proj: Mat4, viewport: Vec4) Vec3 {
            const inv = proj.mul(model).inverse();
            const tx = ((win.x - viewport.x) / viewport.z) * 2 - 1;
            const ty = ((win.y - viewport.y) / viewport.w) * 2 - 1;
            const obj = inv.mulVec(Vec4.init(tx, ty, win.z * 2 - 1, 1));
            const o = obj.divScalar(obj.w);
            return Vec3.init(o.x, o.y, o.z);
        }
        pub const unProject = switch (config.clip.depth) {
            .zero_to_one => unProjectZo,
            .neg_one_to_one => unProjectNo,
        };
        /// Picking matrix that restricts drawing to a small region around `center`.
        pub fn pickMatrix(center: Vec(2, T), delta: Vec(2, T), viewport: Vec4) Mat4 {
            const tmp = Vec3.init(
                (viewport.z - 2 * (center.x - viewport.x)) / delta.x,
                (viewport.w - 2 * (center.y - viewport.y)) / delta.y,
                0,
            );
            const m = M.translate(Mat4.identity(), tmp);
            return M.scale(m, Vec3.init(viewport.z / delta.x, viewport.w / delta.y, 1));
        }

        fn unprojectNdc(inv: Mat4, x: T, y: T, z: T) Vec3 {
            const p = inv.mulVec(Vec4.init(x, y, z, 1));
            return Vec3.init(p.x / p.w, p.y / p.w, p.z / p.w);
        }
        /// A world-space pick ray through a screen pixel. `inv_view_proj` is
        /// `(proj·view).inverse()`. Honors the configured clip conventions.
        pub fn screenToWorldRay(screen_xy: Vec2, viewport: Vec4, inv_view_proj: Mat4) intersect.Ray(T) {
            const nx = ((screen_xy.x - viewport.x) / viewport.z) * 2 - 1;
            var ny = ((screen_xy.y - viewport.y) / viewport.w) * 2 - 1;
            if (config.clip.y == .up) ny = -ny; // screen pixels are top-down
            const near_z: T = switch (config.clip.depth) {
                .zero_to_one => if (config.clip.reverse_z) 1 else 0,
                .neg_one_to_one => -1,
            };
            const far_z: T = switch (config.clip.depth) {
                .zero_to_one => if (config.clip.reverse_z) 0 else 1,
                .neg_one_to_one => 1,
            };
            const np = unprojectNdc(inv_view_proj, nx, ny, near_z);
            const fp = unprojectNdc(inv_view_proj, nx, ny, far_z);
            return intersect.Ray(T).init(np, fp.sub(np).normalize());
        }
    };
}

/// f32 default-precision camera/projection builders.
pub const camera = Camera(f32);

const testing = std.testing;

test "translate then transform a point" {
    const Vec3 = Vec(3, f32);
    const Vec4 = Vec(4, f32);
    const m = transform.translate(Mat(4, 4, f32).identity(), Vec3.init(10, 20, 30));
    const p = m.mulVec(Vec4.init(1, 2, 3, 1));
    try testing.expect(p.approxEql(Vec4.init(11, 22, 33, 1), 1e-5));
}

test "rotate 90deg about Z maps +X to +Y" {
    const Vec3 = Vec(3, f32);
    const Vec4 = Vec(4, f32);
    const m = transform.rotation(std.math.pi / 2.0, Vec3.init(0, 0, 1));
    const r = m.mulVec(Vec4.init(1, 0, 0, 1));
    try testing.expect(r.approxEql(Vec4.init(0, 1, 0, 1), 1e-6));
}

test "lookAt places eye at origin looking down -Z (RH)" {
    const Vec3 = Vec(3, f32);
    const Vec4 = Vec(4, f32);
    const view = camera.lookAt(Vec3.init(0, 0, 5), Vec3.init(0, 0, 0), Vec3.init(0, 1, 0));
    const p = view.mulVec(Vec4.init(0, 0, 0, 1)); // world origin -> view space
    try testing.expect(p.approxEql(Vec4.init(0, 0, -5, 1), 1e-5));
}

test "vulkan default: NDC +Y points down (clip-space Y negated)" {
    const Vec4 = Vec(4, f32);
    const p = camera.perspective(std.math.pi / 3.0, 1.0, 0.1, 100.0);
    // A point above the eye (+Y in view space) must map to negative clip-space Y
    // under the Vulkan default; with `math_clip_y = .up` it would be positive.
    const clip = p.mulVec(Vec4.init(0, 1, -1, 1));
    try testing.expect(clip.y < 0);
}

test "default perspective is reverse-Z (near -> 1, far -> 0)" {
    const Vec4 = Vec(4, f32);
    const near = 0.1;
    const far = 100.0;
    const p = camera.perspective(std.math.pi / 4.0, 16.0 / 9.0, near, far);
    const near_pt = p.mulVec(Vec4.init(0, 0, -near, 1));
    const far_pt = p.mulVec(Vec4.init(0, 0, -far, 1));
    // Vulkan default clip: reverse-Z maps the near plane to 1 and far to 0.
    try testing.expectApproxEqAbs(@as(f32, 1), near_pt.z / near_pt.w, 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 0), far_pt.z / far_pt.w, 1e-4);
}

test "screenToWorldRay points into the scene" {
    const Vec3 = Vec(3, f32);
    const Vec4 = Vec(4, f32);
    const view = camera.lookAt(Vec3.init(0, 0, 5), Vec3.splat(0), Vec3.init(0, 1, 0));
    const proj = camera.perspective(std.math.pi / 3.0, 1.0, 0.1, 100.0);
    const inv = proj.mul(view).inverse();
    const viewport = Vec4.init(0, 0, 800, 600);
    const ray = camera.screenToWorldRay(Vec(2, f32).init(400, 300), viewport, inv); // screen center
    // Centre pixel ray starts near the eye and points toward -Z (into the scene).
    try testing.expect(ray.dir.z < 0);
    try testing.expectApproxEqAbs(@as(f32, 1), ray.dir.length(), 1e-5);
}

test "project / unProject round trip" {
    const Vec3 = Vec(3, f32);
    const Vec4t = Vec(4, f32);
    const model = Mat(4, 4, f32).identity();
    const proj = camera.perspective(std.math.pi / 4.0, 1.0, 0.1, 100.0);
    const viewport = Vec4t.init(0, 0, 800, 600);
    const obj = Vec3.init(0.3, -0.2, -5.0);
    const win = camera.project(obj, model, proj, viewport);
    try testing.expect(camera.unProject(win, model, proj, viewport).approxEql(obj, 1e-3));
}

test "perspectiveFov matches perspective for matching fov" {
    const p1 = camera.perspective(std.math.pi / 3.0, 16.0 / 9.0, 0.1, 100.0);
    const p2 = camera.perspectiveFov(std.math.pi / 3.0, 16.0, 9.0, 0.1, 100.0);
    try testing.expect(p1.approxEql(p2, 1e-5));
}

test "ortho centers the view box" {
    const Vec4 = Vec(4, f32);
    const p = camera.ortho(-1, 1, -1, 1, 0, 1);
    const c = p.mulVec(Vec4.init(0, 0, 0, 1));
    // Center maps to NDC (0,0); Y is unchanged at 0, depth is in [0,1].
    try testing.expectApproxEqAbs(@as(f32, 0), c.x, 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0), c.y, 1e-6);
}

test "default coord config dispatches to RH + 0..1" {
    const CAM = Camera(f32);
    // unsuffixed defaults must equal the RhZo / RH / Zo variants
    try testing.expect(CAM.perspective(1.0, 1.5, 0.1, 100).approxEql(CAM.perspectiveRhZo(1.0, 1.5, 0.1, 100), 1e-6));
    const eye = Vec(3, f32).init(2, 3, 4);
    const ctr = Vec(3, f32).splat(0);
    const up = Vec(3, f32).init(0, 1, 0);
    try testing.expect(CAM.lookAt(eye, ctr, up).approxEql(CAM.lookAtRh(eye, ctr, up), 1e-6));
}

test "transform2d" {
    const Mat3 = mat.Mat3;
    const Vec2 = vec.Vec2;
    const Vec3 = vec.Vec3;
    const m = transform.translate2d(Mat3.identity(), Vec2.init(3, 4));
    const p = m.mulVec(Vec3.init(1, 1, 1));
    try testing.expect(p.approxEql(Vec3.init(4, 5, 1), 1e-6));
}

test "rotate_normalized_axis" {
    const Mat4 = mat.Mat4;
    const Vec3 = vec.Vec3;
    const Vec4 = vec.Vec4;
    const m = transform.rotateNormalizedAxis(Mat4.identity(), std.math.pi / 2.0, Vec3.init(0, 0, 1));
    const p = m.mulVec(Vec4.init(1, 0, 0, 1));
    try testing.expect(p.approxEql(Vec4.init(0, 1, 0, 1), 1e-6));
}
