//! GLSL vector relational functions. Comparisons return a boolean vector
//! (`Vec(N, bool)`); reduce or invert it with `.any()`/`.all()`/`.not()` methods.

const std = @import("std");
const sc = @import("meta.zig");
const vec = @import("vec.zig");
const Vec = vec.Vec;

fn BVec(comptime T: type) type {
    return Vec(T.dim, bool);
}

pub fn lessThan(a: anytype, b: @TypeOf(a)) BVec(@TypeOf(a)) {
    return BVec(@TypeOf(a)).fromSimd(a.simd() < b.simd());
}
pub fn lessThanEqual(a: anytype, b: @TypeOf(a)) BVec(@TypeOf(a)) {
    return BVec(@TypeOf(a)).fromSimd(a.simd() <= b.simd());
}
pub fn greaterThan(a: anytype, b: @TypeOf(a)) BVec(@TypeOf(a)) {
    return BVec(@TypeOf(a)).fromSimd(a.simd() > b.simd());
}
pub fn greaterThanEqual(a: anytype, b: @TypeOf(a)) BVec(@TypeOf(a)) {
    return BVec(@TypeOf(a)).fromSimd(a.simd() >= b.simd());
}
pub fn equal(a: anytype, b: @TypeOf(a)) BVec(@TypeOf(a)) {
    return BVec(@TypeOf(a)).fromSimd(a.simd() == b.simd());
}
pub fn notEqual(a: anytype, b: @TypeOf(a)) BVec(@TypeOf(a)) {
    return BVec(@TypeOf(a)).fromSimd(a.simd() != b.simd());
}

// any / all / not are `Vec(N, bool)` methods: `mask.any()`, `mask.all()`,
// `mask.not()` (see vec.zig).

const testing = std.testing;
const Vec3 = vec.Vec3;

test "relational" {
    const a = Vec3.init(1, 2, 3);
    const b = Vec3.init(3, 2, 1);
    try testing.expect(lessThan(a, b).eql(vec.BVec3.init(true, false, false)));
    try testing.expect(equal(a, b).eql(vec.BVec3.init(false, true, false)));
    try testing.expect(greaterThan(a, b).eql(vec.BVec3.init(false, false, true)));
    try testing.expect(greaterThan(a, b).not().eql(vec.BVec3.init(true, true, false)));
    try testing.expect(lessThan(a, b).any());
    try testing.expect(!lessThan(a, b).all());
    try testing.expect(lessThanEqual(a, a).all());
}
