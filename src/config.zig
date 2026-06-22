//! Comptime clip-space configuration for the projection/view builders
//! (`perspective`, `ortho`, `frustum`, `project`, …).
//!
//! One struct, `Clip`, bundles every clip-space convention. **Defaults target
//! Vulkan/Slang**: right-handed, depth `[0,1]`, NDC **+Y down**, forward-Z.
//! Override the whole thing from your application's root source file:
//!
//! ```zig
//! const math = @import("math");
//! // Vulkan with reverse-Z (near→1, far→0) for better depth precision:
//! pub const math_clip: math.config.Clip = .{ .reverse_z = true };
//! // ...or classic OpenGL:
//! pub const math_clip: math.config.Clip = .{ .depth = .neg_one_to_one, .y = .up };
//! ```
//!
//! The `Rh`/`Lh` and `Zo`/`No` projection-name suffixes select handedness +
//! depth explicitly; `y` and `reverse_z` are orthogonal and apply to every
//! variant.

const root = @import("root");

pub const Handedness = enum { right, left };
pub const ClipDepth = enum { zero_to_one, neg_one_to_one };
pub const ClipY = enum { down, up };

/// Every clip-space convention in one place.
pub const Clip = struct {
    /// View-space handedness.
    handedness: Handedness = .right,
    /// NDC depth range: `[0,1]` (Vulkan/D3D/Metal) or `[-1,1]` (OpenGL).
    depth: ClipDepth = .zero_to_one,
    /// NDC Y direction: `down` (Vulkan) or `up` (OpenGL).
    y: ClipY = .down,
    /// Reverse-Z (near→1, far→0) for better depth-buffer precision. Only
    /// meaningful with `depth = .zero_to_one`; pair it with a GREATER depth
    /// test and a depth clear of 0.
    reverse_z: bool = true,
};

/// Active clip-space convention. Override with `math_clip` in your root file.
pub const clip: Clip = if (@hasDecl(root, "math_clip")) root.math_clip else .{};
