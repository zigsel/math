//! Generic vectors `Vec(N, T)` for any dimension `N` and scalar element `T`.
//! `N ≤ 4` get named `.x/.y/.z/.w` fields; `N > 4` are array-backed (`.data`).
//!
//! Design:
//!   * Storage is an `extern struct` of **named fields** (`x`, `y`, `z`, `w`) so
//!     you get GLM-style member access and a tight, GPU-uploadable layout
//!     (a `Vec3` is 12 bytes, not the 16 a raw `@Vector(3, f32)` would take).
//!   * All math converts the fields into a builtin `@Vector(N, T)` in-register
//!     via `simd()`, computes with hardware SIMD, and lowers back with
//!     `fromSimd()`. For `vec4` this is a free `@bitCast`; for `vec2`/`vec3` the
//!     compiler folds the field loads.
//!   * Shared operations live once in `VecImpl` and are mixed into each size
//!     variant via decl aliases (Zig removed `usingnamespace`, so this is the
//!     idiomatic replacement).
//!
//! Components can be addressed as position `xyzw`, color `rgba`, or texcoord
//! `stpq` through `swizzle` (they all map to the same indices).

const std = @import("std");
const sc = @import("meta.zig");

/// Construct the vector type for dimension `dim` and element type `T`.
pub fn Vec(comptime n: comptime_int, comptime T: type) type {
    return switch (n) {
        1 => extern struct {
            x: T,

            pub const dim: comptime_int = 1;
            pub const Element = T;
            pub const is_math_vector = true;
            const Self = @This();
            const Impl = VecImpl(Self, 1, T);

            pub inline fn init(xs: T) Self {
                return .{ .x = xs };
            }
            pub inline fn simd(self: Self) @Vector(1, T) {
                return .{self.x};
            }
            pub inline fn fromSimd(v: @Vector(1, T)) Self {
                return .{ .x = v[0] };
            }

            // --- mixin ---
            pub const splat = Impl.splat;
            pub const fromArray = Impl.fromArray;
            pub const toArray = Impl.toArray;
            pub const get = Impl.get;
            pub const set = Impl.set;
            pub const add = Impl.add;
            pub const sub = Impl.sub;
            pub const mul = Impl.mul;
            pub const div = Impl.div;
            pub const scale = Impl.scale;
            pub const divScalar = Impl.divScalar;
            pub const addScalar = Impl.addScalar;
            pub const subScalar = Impl.subScalar;
            pub const neg = Impl.neg;
            pub const dot = Impl.dot;
            pub const lengthSq = Impl.lengthSq;
            pub const length = Impl.length;
            pub const distance = Impl.distance;
            pub const distanceSq = Impl.distanceSq;
            pub const normalize = Impl.normalize;
            pub const normalizeOrZero = Impl.normalizeOrZero;
            pub const min = Impl.min;
            pub const max = Impl.max;
            pub const clamp = Impl.clamp;
            pub const clampScalar = Impl.clampScalar;
            pub const abs = Impl.abs;
            pub const lerp = Impl.lerp;
            pub const sum = Impl.sum;
            pub const product = Impl.product;
            pub const minComponent = Impl.minComponent;
            pub const maxComponent = Impl.maxComponent;
            pub const eql = Impl.eql;
            pub const approxEql = Impl.approxEql;
            pub const cast = Impl.cast;
            pub const swizzle = Impl.swizzle;
            pub const withSwizzle = Impl.withSwizzle;
            pub const format = Impl.format;
            pub const any = Impl.any;
            pub const all = Impl.all;
            pub const not = Impl.not;
        },
        2 => extern struct {
            x: T,
            y: T,

            pub const dim: comptime_int = 2;
            pub const Element = T;
            pub const is_math_vector = true;
            const Self = @This();
            const Impl = VecImpl(Self, 2, T);

            pub inline fn init(xs: T, ys: T) Self {
                return .{ .x = xs, .y = ys };
            }
            pub inline fn simd(self: Self) @Vector(2, T) {
                return .{ self.x, self.y };
            }
            pub inline fn fromSimd(v: @Vector(2, T)) Self {
                return .{ .x = v[0], .y = v[1] };
            }

            // --- mixin: shared operations (see VecImpl) ---
            pub const splat = Impl.splat;
            pub const fromArray = Impl.fromArray;
            pub const toArray = Impl.toArray;
            pub const get = Impl.get;
            pub const set = Impl.set;
            pub const add = Impl.add;
            pub const sub = Impl.sub;
            pub const mul = Impl.mul;
            pub const div = Impl.div;
            pub const scale = Impl.scale;
            pub const divScalar = Impl.divScalar;
            pub const addScalar = Impl.addScalar;
            pub const subScalar = Impl.subScalar;
            pub const neg = Impl.neg;
            pub const dot = Impl.dot;
            pub const lengthSq = Impl.lengthSq;
            pub const length = Impl.length;
            pub const distance = Impl.distance;
            pub const distanceSq = Impl.distanceSq;
            pub const normalize = Impl.normalize;
            pub const normalizeOrZero = Impl.normalizeOrZero;
            pub const min = Impl.min;
            pub const max = Impl.max;
            pub const clamp = Impl.clamp;
            pub const clampScalar = Impl.clampScalar;
            pub const abs = Impl.abs;
            pub const lerp = Impl.lerp;
            pub const sum = Impl.sum;
            pub const product = Impl.product;
            pub const minComponent = Impl.minComponent;
            pub const maxComponent = Impl.maxComponent;
            pub const eql = Impl.eql;
            pub const approxEql = Impl.approxEql;
            pub const cast = Impl.cast;
            pub const swizzle = Impl.swizzle;
            pub const withSwizzle = Impl.withSwizzle;
            pub const format = Impl.format;
            pub const any = Impl.any;
            pub const all = Impl.all;
            pub const not = Impl.not;
        },
        3 => extern struct {
            x: T,
            y: T,
            z: T,

            pub const dim: comptime_int = 3;
            pub const Element = T;
            pub const is_math_vector = true;
            const Self = @This();
            const Impl = VecImpl(Self, 3, T);

            pub inline fn init(xs: T, ys: T, zs: T) Self {
                return .{ .x = xs, .y = ys, .z = zs };
            }
            /// Build a `Vec3` from a `Vec2` and a scalar `z`.
            pub inline fn fromVec2(v: Vec(2, T), zs: T) Self {
                return .{ .x = v.x, .y = v.y, .z = zs };
            }
            pub inline fn simd(self: Self) @Vector(3, T) {
                return .{ self.x, self.y, self.z };
            }
            pub inline fn fromSimd(v: @Vector(3, T)) Self {
                return .{ .x = v[0], .y = v[1], .z = v[2] };
            }
            /// Cross product (3-D only).
            pub inline fn cross(a: Self, b: Self) Self {
                const av = a.simd();
                const bv = b.simd();
                const a_yzx = @shuffle(T, av, av, @Vector(3, i32){ 1, 2, 0 });
                const a_zxy = @shuffle(T, av, av, @Vector(3, i32){ 2, 0, 1 });
                const b_yzx = @shuffle(T, bv, bv, @Vector(3, i32){ 1, 2, 0 });
                const b_zxy = @shuffle(T, bv, bv, @Vector(3, i32){ 2, 0, 1 });
                return Self.fromSimd(a_yzx * b_zxy - a_zxy * b_yzx);
            }

            // --- mixin ---
            pub const splat = Impl.splat;
            pub const fromArray = Impl.fromArray;
            pub const toArray = Impl.toArray;
            pub const get = Impl.get;
            pub const set = Impl.set;
            pub const add = Impl.add;
            pub const sub = Impl.sub;
            pub const mul = Impl.mul;
            pub const div = Impl.div;
            pub const scale = Impl.scale;
            pub const divScalar = Impl.divScalar;
            pub const addScalar = Impl.addScalar;
            pub const subScalar = Impl.subScalar;
            pub const neg = Impl.neg;
            pub const dot = Impl.dot;
            pub const lengthSq = Impl.lengthSq;
            pub const length = Impl.length;
            pub const distance = Impl.distance;
            pub const distanceSq = Impl.distanceSq;
            pub const normalize = Impl.normalize;
            pub const normalizeOrZero = Impl.normalizeOrZero;
            pub const min = Impl.min;
            pub const max = Impl.max;
            pub const clamp = Impl.clamp;
            pub const clampScalar = Impl.clampScalar;
            pub const abs = Impl.abs;
            pub const lerp = Impl.lerp;
            pub const sum = Impl.sum;
            pub const product = Impl.product;
            pub const minComponent = Impl.minComponent;
            pub const maxComponent = Impl.maxComponent;
            pub const eql = Impl.eql;
            pub const approxEql = Impl.approxEql;
            pub const cast = Impl.cast;
            pub const swizzle = Impl.swizzle;
            pub const withSwizzle = Impl.withSwizzle;
            pub const format = Impl.format;
            pub const any = Impl.any;
            pub const all = Impl.all;
            pub const not = Impl.not;
        },
        4 => extern struct {
            x: T,
            y: T,
            z: T,
            w: T,

            pub const dim: comptime_int = 4;
            pub const Element = T;
            pub const is_math_vector = true;
            const Self = @This();
            const Impl = VecImpl(Self, 4, T);

            pub inline fn init(xs: T, ys: T, zs: T, ws: T) Self {
                return .{ .x = xs, .y = ys, .z = zs, .w = ws };
            }
            pub inline fn fromVec3(v: Vec(3, T), ws: T) Self {
                return .{ .x = v.x, .y = v.y, .z = v.z, .w = ws };
            }
            pub inline fn simd(self: Self) @Vector(4, T) {
                return @bitCast(self); // vec4 layout == @Vector(4, T) layout
            }
            pub inline fn fromSimd(v: @Vector(4, T)) Self {
                return @bitCast(v);
            }

            // --- mixin ---
            pub const splat = Impl.splat;
            pub const fromArray = Impl.fromArray;
            pub const toArray = Impl.toArray;
            pub const get = Impl.get;
            pub const set = Impl.set;
            pub const add = Impl.add;
            pub const sub = Impl.sub;
            pub const mul = Impl.mul;
            pub const div = Impl.div;
            pub const scale = Impl.scale;
            pub const divScalar = Impl.divScalar;
            pub const addScalar = Impl.addScalar;
            pub const subScalar = Impl.subScalar;
            pub const neg = Impl.neg;
            pub const dot = Impl.dot;
            pub const lengthSq = Impl.lengthSq;
            pub const length = Impl.length;
            pub const distance = Impl.distance;
            pub const distanceSq = Impl.distanceSq;
            pub const normalize = Impl.normalize;
            pub const normalizeOrZero = Impl.normalizeOrZero;
            pub const min = Impl.min;
            pub const max = Impl.max;
            pub const clamp = Impl.clamp;
            pub const clampScalar = Impl.clampScalar;
            pub const abs = Impl.abs;
            pub const lerp = Impl.lerp;
            pub const sum = Impl.sum;
            pub const product = Impl.product;
            pub const minComponent = Impl.minComponent;
            pub const maxComponent = Impl.maxComponent;
            pub const eql = Impl.eql;
            pub const approxEql = Impl.approxEql;
            pub const cast = Impl.cast;
            pub const swizzle = Impl.swizzle;
            pub const withSwizzle = Impl.withSwizzle;
            pub const format = Impl.format;
            pub const any = Impl.any;
            pub const all = Impl.all;
            pub const not = Impl.not;
        },
        // Any dimension > 4: array-backed storage (no named x/y/z/w fields).
        // All operations still work through the dimension-agnostic VecImpl.
        else => extern struct {
            data: [n]T,

            pub const dim: comptime_int = n;
            pub const Element = T;
            pub const is_math_vector = true;
            const Self = @This();
            const Impl = VecImpl(Self, n, T);

            /// Build from an `n`-element array (no variadic init beyond size 4).
            pub inline fn init(arr: [n]T) Self {
                return .{ .data = arr };
            }
            pub inline fn simd(self: Self) @Vector(n, T) {
                return self.data;
            }
            pub inline fn fromSimd(v: @Vector(n, T)) Self {
                return .{ .data = v };
            }

            // --- mixin ---
            pub const splat = Impl.splat;
            pub const fromArray = Impl.fromArray;
            pub const toArray = Impl.toArray;
            pub const get = Impl.get;
            pub const set = Impl.set;
            pub const add = Impl.add;
            pub const sub = Impl.sub;
            pub const mul = Impl.mul;
            pub const div = Impl.div;
            pub const scale = Impl.scale;
            pub const divScalar = Impl.divScalar;
            pub const addScalar = Impl.addScalar;
            pub const subScalar = Impl.subScalar;
            pub const neg = Impl.neg;
            pub const dot = Impl.dot;
            pub const lengthSq = Impl.lengthSq;
            pub const length = Impl.length;
            pub const distance = Impl.distance;
            pub const distanceSq = Impl.distanceSq;
            pub const normalize = Impl.normalize;
            pub const normalizeOrZero = Impl.normalizeOrZero;
            pub const min = Impl.min;
            pub const max = Impl.max;
            pub const clamp = Impl.clamp;
            pub const clampScalar = Impl.clampScalar;
            pub const abs = Impl.abs;
            pub const lerp = Impl.lerp;
            pub const sum = Impl.sum;
            pub const product = Impl.product;
            pub const minComponent = Impl.minComponent;
            pub const maxComponent = Impl.maxComponent;
            pub const eql = Impl.eql;
            pub const approxEql = Impl.approxEql;
            pub const cast = Impl.cast;
            pub const swizzle = Impl.swizzle;
            pub const withSwizzle = Impl.withSwizzle;
            pub const format = Impl.format;
            pub const any = Impl.any;
            pub const all = Impl.all;
            pub const not = Impl.not;
        },
    };
}

/// Shared, dimension-agnostic vector operations. Each method works entirely
/// through the `simd()`/`fromSimd()` bridge, so adding an op here makes it
/// available on every vector size.
fn VecImpl(comptime Self: type, comptime dim: comptime_int, comptime T: type) type {
    const VT = @Vector(dim, T);
    return struct {
        pub inline fn splat(s: T) Self {
            const v: VT = @splat(s);
            return Self.fromSimd(v);
        }
        pub inline fn fromArray(a: [dim]T) Self {
            return Self.fromSimd(a);
        }
        pub inline fn toArray(self: Self) [dim]T {
            return self.simd();
        }
        pub inline fn get(self: Self, i: usize) T {
            return self.toArray()[i];
        }
        pub inline fn set(self: Self, i: usize, val: T) Self {
            var a = self.toArray();
            a[i] = val;
            return Self.fromArray(a);
        }

        pub inline fn add(a: Self, b: Self) Self {
            return Self.fromSimd(a.simd() + b.simd());
        }
        pub inline fn sub(a: Self, b: Self) Self {
            return Self.fromSimd(a.simd() - b.simd());
        }
        pub inline fn mul(a: Self, b: Self) Self {
            return Self.fromSimd(a.simd() * b.simd());
        }
        pub inline fn div(a: Self, b: Self) Self {
            return Self.fromSimd(a.simd() / b.simd());
        }
        pub inline fn scale(a: Self, s: T) Self {
            const v: VT = @splat(s);
            return Self.fromSimd(a.simd() * v);
        }
        pub inline fn divScalar(a: Self, s: T) Self {
            const v: VT = @splat(s);
            return Self.fromSimd(a.simd() / v);
        }
        pub inline fn addScalar(a: Self, s: T) Self {
            const v: VT = @splat(s);
            return Self.fromSimd(a.simd() + v);
        }
        pub inline fn subScalar(a: Self, s: T) Self {
            const v: VT = @splat(s);
            return Self.fromSimd(a.simd() - v);
        }
        pub inline fn neg(a: Self) Self {
            return Self.fromSimd(-a.simd());
        }

        pub inline fn dot(a: Self, b: Self) T {
            return @reduce(.Add, a.simd() * b.simd());
        }
        pub inline fn lengthSq(a: Self) T {
            return a.dot(a);
        }
        pub inline fn length(a: Self) T {
            comptime sc.requireFloat(Self);
            return @sqrt(a.dot(a));
        }
        pub inline fn distance(a: Self, b: Self) T {
            return b.sub(a).length();
        }
        pub inline fn distanceSq(a: Self, b: Self) T {
            return b.sub(a).lengthSq();
        }
        pub inline fn normalize(a: Self) Self {
            comptime sc.requireFloat(Self);
            return a.scale(1.0 / a.length());
        }
        pub inline fn normalizeOrZero(a: Self) Self {
            comptime sc.requireFloat(Self);
            const l = a.length();
            if (l == 0) return splat(0);
            return a.scale(1.0 / l);
        }

        pub inline fn min(a: Self, b: Self) Self {
            return Self.fromSimd(@min(a.simd(), b.simd()));
        }
        pub inline fn max(a: Self, b: Self) Self {
            return Self.fromSimd(@max(a.simd(), b.simd()));
        }
        pub inline fn clamp(a: Self, lo: Self, hi: Self) Self {
            return a.max(lo).min(hi);
        }
        pub inline fn clampScalar(a: Self, lo: T, hi: T) Self {
            return a.max(splat(lo)).min(splat(hi));
        }
        pub inline fn abs(a: Self) Self {
            return Self.fromSimd(@abs(a.simd()));
        }
        pub inline fn lerp(a: Self, b: Self, t: T) Self {
            const vt: VT = @splat(t);
            return Self.fromSimd(a.simd() + (b.simd() - a.simd()) * vt);
        }

        pub inline fn sum(a: Self) T {
            return @reduce(.Add, a.simd());
        }
        pub inline fn product(a: Self) T {
            return @reduce(.Mul, a.simd());
        }
        pub inline fn minComponent(a: Self) T {
            return @reduce(.Min, a.simd());
        }
        pub inline fn maxComponent(a: Self) T {
            return @reduce(.Max, a.simd());
        }

        // --- boolean-mask reductions (for Vec(N, bool)) ---------------------
        /// True if any component is true.
        pub inline fn any(b: Self) bool {
            return @reduce(.Or, b.simd());
        }
        /// True if all components are true.
        pub inline fn all(b: Self) bool {
            return @reduce(.And, b.simd());
        }
        /// Component-wise logical NOT.
        pub inline fn not(b: Self) Self {
            const f: VT = @splat(false);
            return Self.fromSimd(b.simd() == f);
        }

        pub inline fn eql(a: Self, b: Self) bool {
            return @reduce(.And, a.simd() == b.simd());
        }
        pub inline fn approxEql(a: Self, b: Self, eps: T) bool {
            const d = @abs(a.simd() - b.simd());
            const e: VT = @splat(eps);
            return @reduce(.And, d <= e);
        }

        /// Convert element type (e.g. `IVec3 -> Vec3`).
        pub inline fn cast(self: Self, comptime U: type) Vec(dim, U) {
            var r: @Vector(dim, U) = undefined;
            const s = self.simd();
            inline for (0..dim) |i| r[i] = scalarCast(U, s[i]);
            return Vec(dim, U).fromSimd(r);
        }

        /// Comptime swizzle. Accepts `xyzw`, `rgba`, or `stpq` letters
        /// interchangeably; returns a scalar for length 1, else `Vec(len, T)`.
        ///   `v.swizzle("xy")`, `v.swizzle("zyx")`, `c.swizzle("rgb")`
        pub inline fn swizzle(self: Self, comptime spec: []const u8) (if (spec.len == 1) T else Vec(spec.len, T)) {
            const idx = comptime blk: {
                var arr: [spec.len]i32 = undefined;
                for (spec, 0..) |c, i| {
                    const li = letterIndex(c);
                    if (li >= dim) @compileError("swizzle component out of range for this vector size");
                    arr[i] = li;
                }
                break :blk arr;
            };
            const v = self.simd();
            if (spec.len == 1) return v[@intCast(idx[0])];
            const mask: @Vector(spec.len, i32) = idx;
            return Vec(spec.len, T).fromSimd(@shuffle(T, v, v, mask));
        }

        /// Functional write-swizzle: a copy with the `spec` components replaced
        /// by those of `values` (e.g. `v.withSwizzle("xz", Vec2.init(a, b))`).
        /// `spec.len` must be 2..4 (set a single component with `v.x = ...`).
        pub inline fn withSwizzle(self: Self, comptime spec: []const u8, values: Vec(spec.len, T)) Self {
            const idx = comptime blk: {
                var arr: [spec.len]usize = undefined;
                for (spec, 0..) |ch, i| {
                    const li = letterIndex(ch);
                    if (li >= dim) @compileError("swizzle component out of range for this vector size");
                    arr[i] = @intCast(li);
                }
                break :blk arr;
            };
            var out = self;
            const va = values.toArray();
            inline for (0..spec.len) |i| out = out.set(idx[i], va[i]);
            return out;
        }

        /// `std.fmt` integration (use `{f}`): prints `vec3(x, y, z)`.
        pub fn format(self: Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
            try writer.print("vec{d}(", .{dim});
            const a = self.toArray();
            inline for (0..dim) |i| {
                if (i > 0) try writer.writeAll(", ");
                if (comptime T == bool) {
                    try writer.print("{}", .{a[i]});
                } else {
                    try writer.print("{d}", .{a[i]});
                }
            }
            try writer.writeAll(")");
        }
    };
}

fn letterIndex(comptime c: u8) i32 {
    return switch (c) {
        'x', 'r', 's' => 0,
        'y', 'g', 't' => 1,
        'z', 'b', 'p' => 2,
        'w', 'a', 'q' => 3,
        else => @compileError("invalid swizzle component letter"),
    };
}

fn scalarCast(comptime U: type, x: anytype) U {
    const X = @TypeOf(x);
    if (comptime sc.isFloat(U)) {
        if (comptime sc.isFloat(X)) return @floatCast(x);
        if (comptime sc.isInt(X)) return @floatFromInt(x);
    } else if (comptime sc.isInt(U)) {
        if (comptime sc.isFloat(X)) return @intFromFloat(x);
        if (comptime sc.isInt(X)) return @intCast(x);
    }
    if (comptime U == X) return x;
    @compileError("unsupported scalar cast " ++ @typeName(X) ++ " -> " ++ @typeName(U));
}

// ---------------------------------------------------------------------------
// Concrete type aliases (GLM naming)
// ---------------------------------------------------------------------------

pub const Vec1 = Vec(1, f32);
pub const Vec2 = Vec(2, f32);
pub const Vec3 = Vec(3, f32);
pub const Vec4 = Vec(4, f32);
pub const DVec1 = Vec(1, f64);
pub const IVec1 = Vec(1, i32);
pub const UVec1 = Vec(1, u32);
pub const BVec1 = Vec(1, bool);

pub const DVec2 = Vec(2, f64);
pub const DVec3 = Vec(3, f64);
pub const DVec4 = Vec(4, f64);

pub const IVec2 = Vec(2, i32);
pub const IVec3 = Vec(3, i32);
pub const IVec4 = Vec(4, i32);

pub const UVec2 = Vec(2, u32);
pub const UVec3 = Vec(3, u32);
pub const UVec4 = Vec(4, u32);

pub const BVec2 = Vec(2, bool);
pub const BVec3 = Vec(3, bool);
pub const BVec4 = Vec(4, bool);

test "vec basics: ctor, layout, arithmetic" {
    try std.testing.expectEqual(@as(usize, 8), @sizeOf(Vec2));
    try std.testing.expectEqual(@as(usize, 12), @sizeOf(Vec3));
    try std.testing.expectEqual(@as(usize, 16), @sizeOf(Vec4));

    const a = Vec3.init(1, 2, 3);
    const b = Vec3.init(4, 5, 6);
    try std.testing.expect(a.add(b).eql(Vec3.init(5, 7, 9)));
    try std.testing.expect(a.scale(2).eql(Vec3.init(2, 4, 6)));
    try std.testing.expectEqual(@as(f32, 32), a.dot(b));
    try std.testing.expect(a.cross(b).eql(Vec3.init(-3, 6, -3)));
}

test "vec chaining + normalize" {
    const v = Vec3.init(0, 3, 4).normalize();
    try std.testing.expectApproxEqAbs(@as(f32, 1), v.length(), 1e-6);
    const w = Vec3.init(1, 1, 1).add(Vec3.splat(1)).scale(0.5);
    try std.testing.expect(w.eql(Vec3.splat(1)));
}

test "swizzle (xyzw / rgba / stpq)" {
    const v = Vec4.init(1, 2, 3, 4);
    try std.testing.expect(v.swizzle("xy").eql(Vec2.init(1, 2)));
    try std.testing.expect(v.swizzle("zyx").eql(Vec3.init(3, 2, 1)));
    try std.testing.expect(v.swizzle("wzyx").eql(Vec4.init(4, 3, 2, 1)));
    try std.testing.expect(v.swizzle("rgb").eql(Vec3.init(1, 2, 3)));
    try std.testing.expectEqual(@as(f32, 3), v.swizzle("z"));
}

test "withSwizzle + format" {
    const v = Vec3.init(1, 2, 3);
    try std.testing.expect(v.withSwizzle("xz", Vec2.init(9, 8)).eql(Vec3.init(9, 2, 8)));
    var buf: [64]u8 = undefined;
    try std.testing.expectEqualStrings("vec3(1, 2, 3)", try std.fmt.bufPrint(&buf, "{f}", .{v}));
}

test "vec1" {
    const a = Vec1.init(3);
    try std.testing.expectEqual(@as(usize, 4), @sizeOf(Vec1));
    try std.testing.expectEqual(@as(f32, 3), a.x);
    try std.testing.expect(a.add(a).eql(Vec1.init(6)));
    try std.testing.expectEqual(@as(f32, 3), a.length());
}

test "vec cast" {
    const i = IVec3.init(1, 2, 3);
    try std.testing.expect(i.cast(f32).eql(Vec3.init(1, 2, 3)));
    const f = Vec3.init(1.7, 2.2, 3.9);
    try std.testing.expect(f.cast(i32).eql(IVec3.init(1, 2, 3)));
}

test "any-N vector (n > 4, array-backed)" {
    const V8 = Vec(8, f32);
    const a = V8.init(.{ 1, 2, 3, 4, 5, 6, 7, 8 });
    const b = V8.splat(2);
    try std.testing.expectEqual(@as(f32, 36), a.sum());
    try std.testing.expectEqual(@as(f32, 10), a.add(b).get(7)); // 8 + 2
    try std.testing.expectEqual(@as(f32, 8), a.maxComponent());
    try std.testing.expect(a.scale(0.5).eql(V8.init(.{ 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4 })));
    try std.testing.expectApproxEqAbs(@as(f32, 1), a.normalize().length(), 1e-6);
    try std.testing.expectEqual(@as(usize, 8 * @sizeOf(f32)), @sizeOf(V8)); // tight layout
}
