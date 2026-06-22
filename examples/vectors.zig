//! Vectors — `math.Vec(N, T)` and the concrete aliases (`Vec3`, `IVec2`, …).
//! Run: `zig build example-vectors`

const std = @import("std");
const math = @import("math");
const print = std.debug.print;

pub fn main() void {
    vecConstruct();
    vecArithmetic();
    vecGeometry();
    vecSwizzle();
    vecGeneric();
    vecBoolMask();
}

fn vecConstruct() void {
    const a = math.Vec3.init(1, 2, 3); // component constructor
    const b = math.Vec3.splat(0.5); // broadcast one scalar
    const c = math.Vec3.fromArray(.{ 4, 5, 6 }); // from an array
    const d = math.Vec4.fromVec3(a, 1); // extend Vec3 -> Vec4 (w = 1)
    print("a={f} b={f} c={f} d={f}\n", .{ a, b, c, d });
    print("a.x={d} a.get(2)={d}\n", .{ a.x, a.get(2) });
}

fn vecArithmetic() void {
    const a = math.Vec3.init(1, 2, 3);
    const b = math.Vec3.init(4, 5, 6);
    // Method-chaining API; every op returns a new vector.
    print("a+b      = {f}\n", .{a.add(b)});
    print("a*2      = {f}\n", .{a.scale(2)});
    print("a*b      = {f}\n", .{a.mul(b)}); // component-wise
    print("(a+b)/2  = {f}\n", .{a.add(b).scale(0.5)});
    print("lerp     = {f}\n", .{a.lerp(b, 0.25)});
    print("sum/prod = {d} {d}\n", .{ a.sum(), a.product() });
}

fn vecGeometry() void {
    const a = math.Vec3.init(1, 0, 0);
    const b = math.Vec3.init(0, 1, 0);
    // dot/cross/length/distance/normalize are METHODS (not free functions).
    print("dot       = {d}\n", .{a.dot(b)});
    print("cross     = {f}\n", .{a.cross(b)});
    print("length    = {d}\n", .{math.Vec3.init(3, 4, 0).length()});
    print("lengthSq  = {d}\n", .{math.Vec3.init(3, 4, 0).lengthSq()});
    print("normalize = {f}\n", .{math.Vec3.init(0, 3, 4).normalize()});
    print("distance  = {d}\n", .{a.distance(b)});
}

fn vecSwizzle() void {
    const v = math.Vec4.init(1, 2, 3, 4);
    // Letters from xyzw / rgba / stpq are interchangeable.
    print("v.zyx  = {f}\n", .{v.swizzle("zyx")});
    print("v.xy   = {f}\n", .{v.swizzle("xy")});
    print("v.rgb  = {f}\n", .{v.swizzle("rgb")});
    print("v.z    = {d}\n", .{v.swizzle("z")}); // single letter -> scalar
    // Functional write-swizzle.
    print("set xz = {f}\n", .{v.withSwizzle("xz", math.Vec2.init(9, 8))});
}

fn vecGeneric() void {
    // Any dimension and element type via the generic builder.
    const V7 = math.Vec(7, f64);
    const u = V7.init(.{ 1, 2, 3, 4, 5, 6, 7 }); // N>4 takes an array
    print("V7 sum={d} max={d}\n", .{ u.sum(), u.maxComponent() });

    // Integer vectors + element-type casts.
    const iv = math.IVec3.init(1, 2, 3);
    print("IVec3->Vec3 = {f}\n", .{iv.cast(f32)});
    print("Vec3->IVec3 = {f}\n", .{math.Vec3.init(1.7, 2.2, 3.9).cast(i32)});
}

fn vecBoolMask() void {
    const a = math.Vec3.init(1, 5, 3);
    const b = math.Vec3.init(2, 4, 3);
    // Relational free functions return a boolean vector you reduce with
    // .any()/.all()/.not().
    const mask = math.lessThan(a, b); // (true, false, false)
    print("a<b any={} all={}\n", .{ mask.any(), mask.all() });
    print("equal(a,b) = {f}\n", .{math.equal(a, b)});
    // GLSL `mix` with a bool mask selects per component.
    print("select     = {f}\n", .{math.mix(a, b, mask)});
}
