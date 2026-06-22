//! math — a generic, comptime-first mathematics library for Zig.
//!
//! Built on `@Vector` SIMD with a method-chaining type API plus a curated flat
//! layer of genType (scalar-or-vector) free functions in the GLSL tradition.
//!
//! ```zig
//! const math = @import("math");
//! const v = math.Vec3.init(1, 2, 3).normalize();
//! const d = math.dot(v, math.Vec3.splat(1));         // flat genType free function
//! const m = math.transform.translate(math.Mat4.identity(), v);  // namespaced builder
//! const p = math.camera.perspective(1.0, 1.5, 0.1, 100);
//! ```
//!
//! ## Layout
//! - **Flat root**: genType scalar/vector math (`sin`, `clamp`, `mix`, `dot`, …)
//!   and the concrete/​generic type builders.
//! - **Namespaces**: everything domain- or matrix-specific —
//!   `geom` (vector geometry), `transform` (model/affine), `camera`
//!   (view/projection), `matrix` (factorisation/decompose/query), `color`,
//!   `noise`, `rand`, `ease`, `curve`, `bits`, `pack`/`unpack`, `intersect`,
//!   `bounds`, `euler`, `fast`, `util`, `constants`, `meta`, `config`.

// --- private implementation modules ----------------------------------------
const vec = @import("vec.zig");
const mat = @import("mat.zig");
const quat = @import("quat.zig");
const relational = @import("relational.zig");
const xform = @import("transform.zig");
const num = @import("num.zig");
const packmod = @import("pack.zig");
const noisemod = @import("noise.zig");
const eulermod = @import("euler.zig");
const dualquat = @import("dualquat.zig");

// --- public namespaces ------------------------------------------------------
pub const geom = @import("geom.zig"); // vector geometry (norms, angles, projection, queries)
pub const bits = @import("bits.zig"); // integer bit ops, Morton codes
pub const pack = packmod.pack; // pack normalized floats → integer words
pub const unpack = packmod.unpack;
pub const rand = @import("random.zig"); // random distributions
pub const color = @import("color.zig"); // color-space conversions
pub const ease = @import("easing.zig"); // easing curves
pub const fast = @import("fast.zig"); // approximate (fast) math
pub const curve = @import("curve.zig"); // interpolating splines
pub const intersect = @import("intersect.zig"); // ray/line intersection
pub const bounds = @import("bounds.zig"); // bounding volumes & frustum culling
pub const util = @import("util.zig"); // misc helpers (ptr interop, hash, PCA, …)
pub const meta = @import("meta.zig"); // comptime type predicates + genType dispatch helpers
pub const constants = @import("constants.zig"); // math constants
pub const config = @import("config.zig"); // comptime clip-space defaults

const gpu = @import("gpu.zig"); // std140/std430 buffer-layout wrappers
pub const std140 = gpu.std140;
pub const std430 = gpu.std430;

pub const stats = @import("stats.zig"); // running statistics

pub const noise = noisemod.noise; // f32 default; `math.Noise(T)` for other precisions
pub const euler = eulermod.euler; // f32 default; `math.Euler(T)` for other precisions
pub const transform = xform.transform; // f32 model/affine builders; `math.Transforms(T)` generic
pub const camera = xform.camera; // f32 view/projection builders; `math.Camera(T)` generic

/// Matrix factorisation, decomposition, interpolation, and queries.
pub const matrix = struct {
    pub const decompose = mat.decompose;
    pub const recompose = mat.recompose;
    pub const Decomposed = mat.Decomposed;
    pub const interpolate = mat.interpolate;
    pub const axisAngleMatrix = mat.axisAngleMatrix;
    pub const extractRotation = mat.extractMatrixRotation;
    pub const outerProduct = mat.outerProduct;
    // PCA
    pub const covariance = mat.covariance;
    pub const covarianceCentered = mat.covarianceCentered;
    pub const Eigen = mat.Eigen;
    pub const eigenSymmetric = mat.eigenSymmetric;
};

// --- generic type builders --------------------------------------------------
pub const Vec = vec.Vec;
pub const Mat = mat.Mat;
pub const Quat = quat.Quat;
pub const Transforms = xform.Transforms; // generic model/affine builders
pub const Camera = xform.Camera; // generic view/projection builders
pub const Euler = eulermod.Euler; // generic Euler-angle builders
pub const Noise = noisemod.Noise; // generic noise
pub const DualQuat = dualquat.DualQuat; // generic dual quaternion
pub const Aabb = bounds.Aabb; // generic AABB builder
pub const Sphere = bounds.Sphere; // generic bounding-sphere builder
pub const Obb = bounds.Obb; // generic oriented-bounding-box builder
pub const Capsule = bounds.Capsule; // generic capsule builder
pub const Frustum = bounds.Frustum; // generic frustum builder
pub const Ray = intersect.Ray; // generic ray builder
const shapes = @import("shapes.zig");
pub const Plane = shapes.Plane; // generic plane builder
pub const Line = shapes.Line; // generic (infinite) line builder
pub const Segment = shapes.Segment; // generic line-segment builder
pub const Rect = shapes.Rect; // generic 2-D rectangle builder
pub const Triangle = shapes.Triangle; // generic triangle builder

// --- concrete type aliases --------------------------------------------------
pub const Vec1 = vec.Vec1;
pub const Vec2 = vec.Vec2;
pub const Vec3 = vec.Vec3;
pub const Vec4 = vec.Vec4;
pub const DVec1 = vec.DVec1;
pub const IVec1 = vec.IVec1;
pub const UVec1 = vec.UVec1;
pub const BVec1 = vec.BVec1;
pub const DVec2 = vec.DVec2;
pub const DVec3 = vec.DVec3;
pub const DVec4 = vec.DVec4;
pub const IVec2 = vec.IVec2;
pub const IVec3 = vec.IVec3;
pub const IVec4 = vec.IVec4;
pub const UVec2 = vec.UVec2;
pub const UVec3 = vec.UVec3;
pub const UVec4 = vec.UVec4;
pub const BVec2 = vec.BVec2;
pub const BVec3 = vec.BVec3;
pub const BVec4 = vec.BVec4;

pub const Mat2 = mat.Mat2;
pub const Mat3 = mat.Mat3;
pub const Mat4 = mat.Mat4;
pub const Mat2x2 = mat.Mat2x2;
pub const Mat2x3 = mat.Mat2x3;
pub const Mat2x4 = mat.Mat2x4;
pub const Mat3x2 = mat.Mat3x2;
pub const Mat3x3 = mat.Mat3x3;
pub const Mat3x4 = mat.Mat3x4;
pub const Mat4x2 = mat.Mat4x2;
pub const Mat4x3 = mat.Mat4x3;
pub const Mat4x4 = mat.Mat4x4;
pub const DMat2 = mat.DMat2;
pub const DMat3 = mat.DMat3;
pub const DMat4 = mat.DMat4;

pub const Quaternion = quat.Quaternion;
pub const DQuaternion = quat.DQuaternion;

/// f32 concrete geometry types (generic builders are `math.Aabb(T)`, etc.).
pub const Aabb3 = bounds.Aabb(f32);
pub const Sphere3 = bounds.Sphere(f32);
pub const Obb3 = bounds.Obb(f32);
pub const Capsule3 = bounds.Capsule(f32);
pub const Frustum3 = bounds.Frustum(f32);
pub const Ray3 = intersect.Ray(f32);
pub const Plane3 = shapes.Plane(f32);
pub const Line3 = shapes.Line(f32);
pub const Segment3 = shapes.Segment(f32);
pub const Triangle3 = shapes.Triangle(f32);
pub const Rect2 = shapes.Rect(f32);

// ===========================================================================
// Flat genType layer — scalar-or-vector free functions.
// Domain-specific and matrix-builder functions live in the namespaces above.
// ===========================================================================

// --- geometric free functions (no method form; the rest live under math.geom) -
// length/distance/dot/cross/normalize are Vec methods: `a.dot(b)`, `v.length()`.
pub const faceForward = geom.faceForward;
pub const reflect = geom.reflect;
pub const refract = geom.refract;

// --- common -----------------------------------------------------------------
pub const abs = num.abs;
pub const sign = num.sign;
pub const floor = num.floor;
pub const ceil = num.ceil;
pub const trunc = num.trunc;
pub const round = num.round;
pub const roundEven = num.roundEven;
pub const fract = num.fract;
pub const mod = num.mod;
pub const fmod = num.fmod;
pub const modf = num.modf;
pub const min = num.min;
pub const max = num.max;
pub const clamp = num.clamp;
pub const saturate = num.saturate;
pub const mix = num.mix;
pub const step = num.step;
pub const smoothstep = num.smoothstep;
pub const smootherstep = num.smootherstep;
pub const fma = num.fma;
pub const isnan = num.isnan;
pub const isinf = num.isinf;
pub const isfinite = num.isfinite;
pub const isdenormal = num.isdenormal;
pub const frexp = num.frexp;
pub const ldexp = num.ldexp;
pub const floatBitsToInt = num.floatBitsToInt;
pub const floatBitsToUint = num.floatBitsToUint;
pub const intBitsToFloat = num.intBitsToFloat;
pub const uintBitsToFloat = num.uintBitsToFloat;

// --- trigonometric ----------------------------------------------------------
pub const radians = num.radians;
pub const degrees = num.degrees;
pub const sin = num.sin;
pub const cos = num.cos;
pub const tan = num.tan;
pub const asin = num.asin;
pub const acos = num.acos;
pub const atan = num.atan;
pub const atan2 = num.atan2;
pub const sinh = num.sinh;
pub const cosh = num.cosh;
pub const tanh = num.tanh;
pub const asinh = num.asinh;
pub const acosh = num.acosh;
pub const atanh = num.atanh;
// reciprocal / inverse-reciprocal trig
pub const sec = num.sec;
pub const csc = num.csc;
pub const cot = num.cot;
pub const asec = num.asec;
pub const acsc = num.acsc;
pub const acot = num.acot;
pub const sech = num.sech;
pub const csch = num.csch;
pub const coth = num.coth;
pub const asech = num.asech;
pub const acsch = num.acsch;
pub const acoth = num.acoth;

// --- exponential ------------------------------------------------------------
pub const pow = num.pow;
pub const exp = num.exp;
pub const log = num.log;
pub const exp2 = num.exp2;
pub const log2 = num.log2;
pub const sqrt = num.sqrt;
pub const inverseSqrt = num.inverseSqrt;
pub const pow2 = num.pow2;
pub const pow3 = num.pow3;
pub const pow4 = num.pow4;
pub const logBase = num.logBase;

// --- relational (return boolean vectors) ------------------------------------
pub const lessThan = relational.lessThan;
pub const lessThanEqual = relational.lessThanEqual;
pub const greaterThan = relational.greaterThan;
pub const greaterThanEqual = relational.greaterThanEqual;
pub const equal = relational.equal;
pub const notEqual = relational.notEqual;
pub const epsilonEqual = num.epsilonEqual;
pub const epsilonNotEqual = num.epsilonNotEqual;

// --- extended min/max, NaN-aware --------------------------------------------
pub const fmin = num.fmin;
pub const fmax = num.fmax;
pub const fclamp = num.fclamp;
pub const min3 = num.min3;
pub const min4 = num.min4;
pub const max3 = num.max3;
pub const max4 = num.max4;
pub const fcompMin = num.fcompMin;
pub const fcompMax = num.fcompMax;

// --- angles (radians) -------------------------------------------------------
pub const wrapAngle = num.wrapAngle;
pub const normalizeAngle = num.normalizeAngle;
pub const deltaAngle = num.deltaAngle;
pub const lerpAngle = num.lerpAngle;

// --- wrapping / texture addressing ------------------------------------------
pub const repeat = num.repeat;
pub const mirrorRepeat = num.mirrorRepeat;
pub const mirrorClamp = num.mirrorClamp;
pub const wrapClamp = num.wrapClamp;

// --- integer rounding / multiples / ULP -------------------------------------
pub const iround = num.iround;
pub const uround = num.uround;
pub const isMultiple = num.isMultiple;
pub const ceilMultiple = num.ceilMultiple;
pub const floorMultiple = num.floorMultiple;
pub const roundMultiple = num.roundMultiple;
pub const nextFloat = num.nextFloat;
pub const prevFloat = num.prevFloat;
pub const floatDistance = num.floatDistance;

// --- misc genType -----------------------------------------------------------
pub const gaussian = num.gauss; // Gaussian falloff exp(-(x-µ)²/2σ²); see math.rand.normal for sampling
pub const compNormalize = num.compNormalize;
pub const compScale = num.compScale;

test {
    _ = vec;
    _ = mat;
    _ = quat;
    _ = meta;
    _ = constants;
    _ = relational;
    _ = xform;
    _ = bits;
    _ = num;
    _ = geom;
    _ = packmod;
    _ = noisemod;
    _ = rand;
    _ = color;
    _ = ease;
    _ = fast;
    _ = eulermod;
    _ = dualquat;
    _ = curve;
    _ = intersect;
    _ = bounds;
    _ = util;
    _ = gpu;
    _ = stats;
    _ = shapes;
}
