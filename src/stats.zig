//! Running statistics — `math.stats`. Welford's online algorithm: feed values
//! one at a time and read mean / variance / stddev / min / max in O(1) memory,
//! numerically stable.

const std = @import("std");

/// Online accumulator over a float type `T`.
pub fn Accumulator(comptime T: type) type {
    return struct {
        count: u64 = 0,
        mean: T = 0,
        m2: T = 0,
        min: T = std.math.inf(T),
        max: T = -std.math.inf(T),

        const Self = @This();

        pub fn add(self: *Self, x: T) void {
            self.count += 1;
            const delta = x - self.mean;
            self.mean += delta / @as(T, @floatFromInt(self.count));
            self.m2 += delta * (x - self.mean);
            self.min = @min(self.min, x);
            self.max = @max(self.max, x);
        }
        pub fn addSlice(self: *Self, xs: []const T) void {
            for (xs) |x| self.add(x);
        }
        /// Population variance (÷ n).
        pub fn variance(self: Self) T {
            return if (self.count > 0) self.m2 / @as(T, @floatFromInt(self.count)) else 0;
        }
        /// Sample variance (÷ n−1, Bessel-corrected).
        pub fn sampleVariance(self: Self) T {
            return if (self.count > 1) self.m2 / @as(T, @floatFromInt(self.count - 1)) else 0;
        }
        pub fn stddev(self: Self) T {
            return @sqrt(self.variance());
        }
        pub fn sampleStddev(self: Self) T {
            return @sqrt(self.sampleVariance());
        }
        pub fn range(self: Self) T {
            return self.max - self.min;
        }
    };
}

/// f32 default.
pub const Stats = Accumulator(f32);

const testing = std.testing;

test "running mean / variance / range" {
    var s = Stats{};
    s.addSlice(&.{ 2, 4, 4, 4, 5, 5, 7, 9 });
    try testing.expectApproxEqAbs(@as(f32, 5), s.mean, 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 4), s.variance(), 1e-5); // population
    try testing.expectApproxEqAbs(@as(f32, 2), s.stddev(), 1e-5);
    try testing.expectEqual(@as(f32, 7), s.range());
    try testing.expectEqual(@as(u64, 8), s.count);
}
