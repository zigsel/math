//! Bit twiddling & packing — `math.bits` (integer ops, Morton codes) and
//! `math.pack`/`math.unpack` (normalized floats <-> integer words).
//! Run: `zig build example-packing`

const std = @import("std");
const math = @import("math");
const print = std.debug.print;

pub fn main() void {
    bitsOps();
    bitsMorton();
    packNormalized();
    packGeneric();
}

fn bitsOps() void {
    print("popcount(0b1011) = {d}\n", .{math.bits.count(@as(u32, 0b1011))});
    print("isPow2(16)       = {}\n", .{math.bits.isPow2(@as(u32, 16))});
    print("ceilPow2(9)      = {d}\n", .{math.bits.ceilPow2(@as(u32, 9))});
    print("msb(0b10010000)  = {d}\n", .{math.bits.msb(@as(u32, 0b1001_0000))});
}

fn bitsMorton() void {
    // Interleave coordinates into a Morton (Z-order) code and back.
    const code = math.bits.interleave2(0xABCD, 0x1234);
    const d = math.bits.deinterleave2(code);
    print("morton code = 0x{X}  ->  x=0x{X} y=0x{X}\n", .{ code, d.x, d.y });
}

fn packNormalized() void {
    // Pack a color into a single 32-bit RGBA8 word, then unpack it.
    const color = math.Vec4.init(1, 0.5, 0.25, 1);
    const word = math.pack.unorm4x8(color);
    print("packed RGBA8 = 0x{X:0>8}\n", .{word});
    print("unpacked     = {f}\n", .{math.unpack.unorm4x8(word)});
}

fn packGeneric() void {
    // Comptime-width normalized packing to any integer type.
    const v = math.Vec2.init(0.1, 0.9);
    const packed_u16 = math.pack.unorm(u16, v);
    print("unorm(u16)   = {f}\n", .{packed_u16});
    print("round-trip   = {f}\n", .{math.unpack.unorm(f32, packed_u16)});
}
