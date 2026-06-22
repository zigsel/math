# math

A generic, comptime-first mathematics library for Zig — vectors, matrices,
quaternions, and the graphics/geometry toolkit around them. Built on `@Vector`
SIMD with a method-chaining type API plus a curated flat layer of genType
(scalar-or-vector) free functions.

```zig
const math = @import("math");

const v = math.Vec3.init(1, 2, 3).normalize();        // method chaining
const d = math.clamp(v.dot(math.Vec3.splat(1)), 0, 1); // genType free function
const mvp = proj.mul(view).mul(model);                 // column-major matrices
```

- **Generic over dimension & element type** — `Vec(N, T)`, `Mat(C, R, T)`,
  `Quat(T)`, `Aabb(T)`, … all parameterized; `f32` aliases (`Vec3`, `Mat4`,
  `Quaternion`) for the common case.
- **SIMD by construction** — every operation lowers to `@Vector`, so you get
  SSE/AVX/NEON for the build target with no hand-written intrinsics.
- **Tight, GPU-uploadable layout** — `Vec3` is 12 bytes, `Mat4` is 64; column-
  major storage matches GLSL/SPIR-V.
- **Vulkan defaults** — right-handed, depth `[0, 1]`, NDC **+Y down**, **reverse-Z**
  out of the box; std140/std430 buffer-layout types for direct GPU upload.
  (All configurable for OpenGL/D3D.)
- **Two APIs, one truth** — types own their behavior as methods; the flat layer
  is genType GLSL-style sugar. No duplicated logic.

Requires **Zig 0.16+**.

## Install

```sh
zig fetch --save "git+https://github.com/zigsel/math"
```

Then in `build.zig`:

```zig
const math_dep = b.dependency("math", .{ .target = target, .optimize = optimize });
exe.root_module.addImport("math", math_dep.module("math"));
```

## API at a glance

The **flat root** holds only genType **scalar/vector** math — the GLSL builtins
(`math.dot`… are methods; `math.clamp`, `math.mix`, `math.sin`, `math.reflect`,
relational ops, …). Everything matrix-, camera-, or domain-specific lives in a
namespace:

| Namespace | What |
|---|---|
| `math.geom`      | vector geometry: norms, angles, vector rotation, projection, queries |
| `math.transform` | model/affine builders (`translate`, `rotate`, `scale`, 2D, shear, reflect/proj onto planes) |
| `math.camera`    | view & projection (`lookAt`, `perspective`, `ortho`, `frustum`, `project`/`unProject`, …) |
| `math.matrix`    | `decompose`/`recompose`, `interpolate`, `qr`/`rq`, `outerProduct`, PCA (`covariance`, `eigenSymmetric`) |
| `math.color`     | sRGB / HSV / YCoCg / XYZ conversions, luminance, saturation |
| `math.noise`     | Perlin, periodic Perlin, and simplex noise (2–4D) |
| `math.rand`      | random distributions (`uniform`, `normal`, `onSphere`, `inBall`, …) |
| `math.ease`      | 30+ Penner easing curves + spring/damping (`expDecay`, `smoothDamp`) |
| `math.curve`     | splines: Bézier (quad/cubic + derivatives + arc length), Catmull-Rom, Hermite, B-spline |
| `math.stats`     | running mean / variance / stddev (Welford) |
| `math.bits`      | integer bit ops, Morton codes, power-of-two |
| `math.pack` / `math.unpack` | normalized floats ⇄ packed integer words |
| `math.intersect` | ray/line × plane/sphere/triangle casts, plus the `Ray(T)` type |
| `math.bounds`    | bounding volumes — `Aabb` / `Sphere` / `Obb` / `Capsule` / `Frustum` + culling |
| `math.euler`     | Euler-angle rotation matrices + extraction |
| `math.fast`      | approximate (fast) `sqrt`/`sin`/`exp`/… |
| `math.util`      | pointer/array interop, hashing, polar coords, gradient paint, mip levels |
| `math.constants` | π, τ, e, √2, golden ratio, … (f32 consts + precision-generic fns) |
| `math.meta`      | comptime type predicates & genType dispatch helpers |
| `math.std140` / `math.std430` | GPU buffer-layout wrappers for direct uniform/SSBO/push-constant upload |
| `math.config`    | clip-space convention (the `Clip` struct) |

**Generic builders** are capitalized and parameterized by element type —
`math.Vec`, `math.Mat`, `math.Quat`, `math.Transforms`, `math.Camera`,
`math.Euler`, `math.Noise`, `math.DualQuat`, and the geometry types
`math.Aabb`, `math.Sphere`, `math.Obb`, `math.Capsule`, `math.Frustum`,
`math.Ray`, `math.Plane`, `math.Line`, `math.Segment`, `math.Rect`,
`math.Triangle`. The **f32 instances** are the lowercase/concrete forms —
`math.transform`, `math.camera`, `math.Vec3`, `math.Mat4`, `math.Aabb3`,
`math.Plane3`, `math.Triangle3`, `math.Rect2`, `math.euler`, `math.noise`.

## Conventions

**Methods vs. free functions.** Behavior that belongs to a value is a *method*:
`a.dot(b)`, `v.length()`, `v.normalize()`, `m.inverse()`, `m.qr()`,
`q.slerp(r, t)`. The flat layer is reserved for genType math that is naturally
polymorphic over scalars *and* vectors (`math.clamp`, `math.mix`, `math.sin`,
`math.pow`) plus the geometric ops with no method form (`math.reflect`,
`math.refract`, `math.faceForward`).

**Column-major.** `Mat(C, R, T)` stores `C` columns; index with `m.at(col, row)`.
`m.mulVec(v)` is `M·v`. Bytes upload directly to GLSL/SPIR-V; in Slang/HLSL use
`mul(v, M)` (or compile column-major).

**genType polymorphism.** GLSL functions are recovered at comptime: the same
`math.clamp` / `math.abs` / `math.mix` accept a scalar *or* a vector and
broadcast scalar arguments component-wise.

**Clip space (Vulkan by default).** One comptime struct holds every convention;
override the whole thing from your root source file:

```zig
// Defaults: { .handedness = .right, .depth = .zero_to_one, .y = .down, .reverse_z = true }
pub const math_clip: math.config.Clip = .{ .depth = .neg_one_to_one, .y = .up }; // classic OpenGL
```

`y` (NDC direction) and `reverse_z` (near→1, far→0, for depth precision — pair it
with a `GREATER` depth test and a depth clear of 0) apply to every projection.
The `Rh`/`Lh` and `Zo`/`No` name suffixes (`perspectiveRhZo`, `orthoLhNo`, …)
select handedness and depth range explicitly.

**GPU upload.** Two options for matching shader memory layout:

1. **Scalar block layout (recommended).** Enable the `scalarBlockLayout` feature
   (core since Vulkan 1.2, so just a feature toggle on 1.3) and compile Slang with
   `-fvk-use-scalar-layout`; then `vec3` packs tightly to 12 bytes — exactly
   `math.Vec3` — and you upload the plain `math.*` types directly, no padding.
2. **std140 / std430 wrappers.** Otherwise declare buffer structs with the
   layout-correct types so the bytes match by construction:

   ```zig
   const Uniforms = extern struct {
       view_proj:  math.std140.Mat4,
       camera_pos: math.std140.Vec3, // 16-aligned; next field lands correctly
       time:       f32,
   };
   const u: Uniforms = .{ .view_proj = .from(vp), .camera_pos = .from(pos), .time = t };
   // memcpy / map `u` straight into the buffer.
   ```

   `.from(mathValue)` packs, `.get()` reads back. `std430` shares the vector/
   matrix layout (they differ only for `mat2` and scalar/vec2 array stride).

## Examples

Self-contained, runnable programs in [`examples/`](examples). Run one with
`zig build example-<name>`, or all with `zig build examples`:

| Example | Covers |
|---|---|
| [`vectors`](examples/vectors.zig)       | `Vec(N, T)`, arithmetic, swizzle, casts, bool masks + relational |
| [`matrices`](examples/matrices.zig)     | `Mat(C, R, T)`, products, inverse, decompose, QR, PCA |
| [`rotations`](examples/rotations.zig)   | quaternions, dual quaternions, Euler angles |
| [`camera`](examples/camera.zig)         | model transforms, MVP, project/unProject, frustum culling (Vulkan) |
| [`scalar_math`](examples/scalar_math.zig) | flat genType math, constants, fast approximations, `meta` |
| [`geometry`](examples/geometry.zig)     | `geom` ops, ray casts, shapes (plane/triangle/…), bounding volumes |
| [`graphics`](examples/graphics.zig)     | color spaces, noise (incl. fbm/turbulence/ridged), random |
| [`packing`](examples/packing.zig)       | bit ops, Morton codes, normalized packing |
| [`animation`](examples/animation.zig)   | easing, Bézier/B-spline, spring smoothing, running stats |

## Develop

```sh
zig build test        # run the test suite
zig build examples    # build & run every example
zig build docs        # generate HTML docs into zig-out/docs
```
