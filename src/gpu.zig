//! GPU buffer layout — `math.std140` / `math.std430`.
//!
//! Layout-correct wrappers for the vector/matrix types so an `extern struct`
//! of them matches a GLSL/Slang uniform (std140), storage (std430), or push-
//! constant block byte-for-byte. Build with `.from(mathValue)` and read back
//! with `.get()`:
//!
//! ```zig
//! const Uniforms = extern struct {
//!     mvp:       math.std140.Mat4,
//!     color:     math.std140.Vec4,
//!     light_dir: math.std140.Vec3, // 16-aligned, so the next field lands right
//! };
//! var u: Uniforms = .{
//!     .mvp = .from(mvp), .color = .from(rgba), .light_dir = .from(dir),
//! };
//! // @memcpy / map u into the buffer — offsets match the shader.
//! ```
//!
//! ## Or: scalar block layout (zero overhead)
//! If you enable `VK_EXT_scalar_block_layout` (Vulkan 1.2 core) and compile
//! Slang with `-fvk-use-scalar-layout`, `vec3` packs tightly to 12 bytes — which
//! is exactly `math.Vec3`. Then you can skip these wrappers and upload the plain
//! `math.*` types directly. Recommended for new projects.
//!
//! ## std140 vs std430
//! For `Vec2`/`Vec3`/`Vec4`/`Mat3`/`Mat4` the two layouts are identical, so
//! `std430` re-exports them. They differ only for `Mat2` (column stride) and for
//! arrays of scalars/vec2 (std140 rounds the array stride up to 16) — lay arrays
//! out yourself with the matching element type.

const vec = @import("vec.zig");
const mat = @import("mat.zig");

fn Wrap(comptime Math: type, comptime alignment: u29) type {
    return extern struct {
        val: Math align(alignment),
        const Self = @This();
        pub fn from(v: Math) Self {
            return .{ .val = v };
        }
        pub fn get(self: Self) Math {
            return self.val;
        }
    };
}

pub const std140 = struct {
    /// 8-byte aligned `vec2`.
    pub const Vec2 = Wrap(vec.Vec2, 8);
    /// 16-byte aligned `vec3` (12 data bytes + 4 padding → size 16).
    pub const Vec3 = extern struct {
        val: vec.Vec3 align(16),
        _pad: f32 = 0,
        const Self = @This();
        pub fn from(v: vec.Vec3) Self {
            return .{ .val = v };
        }
        pub fn get(self: Self) vec.Vec3 {
            return self.val;
        }
    };
    /// 16-byte aligned `vec4`.
    pub const Vec4 = Wrap(vec.Vec4, 16);
    /// `mat4` as 4 columns, 16-byte aligned (size 64).
    pub const Mat4 = extern struct {
        cols: [4]vec.Vec4 align(16),
        const Self = @This();
        pub fn from(m: mat.Mat4) Self {
            return .{ .cols = m.cols };
        }
        pub fn get(self: Self) mat.Mat4 {
            return mat.Mat4.fromColumns(self.cols);
        }
    };
    /// `mat3` as 3 columns, each padded to 16 bytes (size 48).
    pub const Mat3 = extern struct {
        cols: [3]Vec3,
        const Self = @This();
        pub fn from(m: mat.Mat3) Self {
            return .{ .cols = .{ Vec3.from(m.cols[0]), Vec3.from(m.cols[1]), Vec3.from(m.cols[2]) } };
        }
        pub fn get(self: Self) mat.Mat3 {
            return mat.Mat3.fromColumns(.{ self.cols[0].val, self.cols[1].val, self.cols[2].val });
        }
    };
    /// `mat2` with std140 column stride of 16 bytes (size 32).
    pub const Mat2 = extern struct {
        cols: [2]Col,
        const Self = @This();
        const Col = extern struct { val: vec.Vec2 align(16), _pad: [2]f32 = .{ 0, 0 } };
        pub fn from(m: mat.Mat2) Self {
            return .{ .cols = .{ .{ .val = m.cols[0] }, .{ .val = m.cols[1] } } };
        }
        pub fn get(self: Self) mat.Mat2 {
            return mat.Mat2.fromColumns(.{ self.cols[0].val, self.cols[1].val });
        }
    };
};

pub const std430 = struct {
    // Identical layout to std140 for these:
    pub const Vec2 = std140.Vec2;
    pub const Vec3 = std140.Vec3;
    pub const Vec4 = std140.Vec4;
    pub const Mat3 = std140.Mat3;
    pub const Mat4 = std140.Mat4;
    /// `mat2` with std430 column stride of 8 bytes (tight, size 16).
    pub const Mat2 = extern struct {
        cols: [2]std140.Vec2,
        const Self = @This();
        pub fn from(m: mat.Mat2) Self {
            return .{ .cols = .{ std140.Vec2.from(m.cols[0]), std140.Vec2.from(m.cols[1]) } };
        }
        pub fn get(self: Self) mat.Mat2 {
            return mat.Mat2.fromColumns(.{ self.cols[0].val, self.cols[1].val });
        }
    };
};

const std = @import("std");
const testing = std.testing;

test "std140/std430 sizes & alignments match the GLSL rules" {
    // std140 == std430 for vectors / mat3 / mat4
    inline for (.{ std140, std430 }) |L| {
        try testing.expectEqual(@as(usize, 8), @sizeOf(L.Vec2));
        try testing.expectEqual(@as(usize, 8), @alignOf(L.Vec2));
        try testing.expectEqual(@as(usize, 16), @sizeOf(L.Vec3));
        try testing.expectEqual(@as(usize, 16), @alignOf(L.Vec3));
        try testing.expectEqual(@as(usize, 16), @sizeOf(L.Vec4));
        try testing.expectEqual(@as(usize, 48), @sizeOf(L.Mat3));
        try testing.expectEqual(@as(usize, 16), @alignOf(L.Mat3));
        try testing.expectEqual(@as(usize, 64), @sizeOf(L.Mat4));
        try testing.expectEqual(@as(usize, 16), @alignOf(L.Mat4));
    }
    // mat2 differs: std140 pads columns to 16, std430 keeps them at 8.
    try testing.expectEqual(@as(usize, 32), @sizeOf(std140.Mat2));
    try testing.expectEqual(@as(usize, 16), @sizeOf(std430.Mat2));

    // round-trips preserve the value
    const v = vec.Vec3.init(1, 2, 3);
    try testing.expect(std140.Vec3.from(v).get().eql(v));
    const m = mat.Mat4.identity();
    try testing.expect(std140.Mat4.from(m).get().eql(m));
}
