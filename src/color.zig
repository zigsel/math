//! Color space conversions — `math.color`.
//!
//! Conversions are generic over the float element type: pass any `Vec(3, T)` /
//! `Vec(4, T)` and the element type is inferred. The sRGB transfer functions are
//! genType (scalar or vector). f32 is the common case.

const std = @import("std");
const sc = @import("meta.zig");
const vec = @import("vec.zig");
const mat = @import("mat.zig");

// --- sRGB transfer (genType) ------------------------------------------------

pub fn linearToSrgb(c: anytype) @TypeOf(c) {
    return sc.map1(c, struct {
        fn f(x: anytype) @TypeOf(x) {
            const T = @TypeOf(x);
            return if (x <= 0.0031308) x * 12.92 else 1.055 * std.math.pow(T, x, 1.0 / 2.4) - 0.055;
        }
    }.f);
}
pub fn srgbToLinear(c: anytype) @TypeOf(c) {
    return sc.map1(c, struct {
        fn f(x: anytype) @TypeOf(x) {
            const T = @TypeOf(x);
            return if (x <= 0.04045) x / 12.92 else std.math.pow(T, (x + 0.055) / 1.055, 2.4);
        }
    }.f);
}

// --- HSV (hue in degrees [0,360)) -------------------------------------------

pub fn hsvToRgb(hsv: anytype) @TypeOf(hsv) {
    const V = @TypeOf(hsv);
    const h = hsv.x / 60.0;
    const s = hsv.y;
    const v = hsv.z;
    const i: i32 = @intFromFloat(@floor(h));
    const f = h - @floor(h);
    const p = v * (1.0 - s);
    const q = v * (1.0 - s * f);
    const t = v * (1.0 - s * (1.0 - f));
    return switch (@mod(i, 6)) {
        0 => V.init(v, t, p),
        1 => V.init(q, v, p),
        2 => V.init(p, v, t),
        3 => V.init(p, q, v),
        4 => V.init(t, p, v),
        else => V.init(v, p, q),
    };
}
pub fn rgbToHsv(rgb: anytype) @TypeOf(rgb) {
    const V = @TypeOf(rgb);
    const T = V.Element;
    const mx = @max(rgb.x, @max(rgb.y, rgb.z));
    const mn = @min(rgb.x, @min(rgb.y, rgb.z));
    const d = mx - mn;
    const s = if (mx == 0) 0 else d / mx;
    if (d == 0) return V.init(0, s, mx);
    var h: T = if (mx == rgb.x)
        (rgb.y - rgb.z) / d
    else if (mx == rgb.y)
        2.0 + (rgb.z - rgb.x) / d
    else
        4.0 + (rgb.x - rgb.y) / d;
    h *= 60.0;
    if (h < 0) h += 360.0;
    return V.init(h, s, mx);
}

// --- HSL (hue in degrees [0,360)) -------------------------------------------

pub fn rgbToHsl(rgb: anytype) @TypeOf(rgb) {
    const V = @TypeOf(rgb);
    const T = V.Element;
    const mx = @max(rgb.x, @max(rgb.y, rgb.z));
    const mn = @min(rgb.x, @min(rgb.y, rgb.z));
    const d = mx - mn;
    const l = (mx + mn) * 0.5;
    if (d == 0) return V.init(0, 0, l);
    const s = d / (1.0 - @abs(2.0 * l - 1.0));
    var h: T = if (mx == rgb.x)
        @mod((rgb.y - rgb.z) / d, 6.0)
    else if (mx == rgb.y)
        (rgb.z - rgb.x) / d + 2.0
    else
        (rgb.x - rgb.y) / d + 4.0;
    h *= 60.0;
    if (h < 0) h += 360.0;
    return V.init(h, s, l);
}
pub fn hslToRgb(hsl: anytype) @TypeOf(hsl) {
    const V = @TypeOf(hsl);
    const h = hsl.x / 60.0;
    const s = hsl.y;
    const l = hsl.z;
    const c = (1.0 - @abs(2.0 * l - 1.0)) * s;
    const x = c * (1.0 - @abs(@mod(h, 2.0) - 1.0));
    const m = l - c * 0.5;
    const i: i32 = @intFromFloat(@floor(h));
    const base = switch (@mod(i, 6)) {
        0 => V.init(c, x, 0),
        1 => V.init(x, c, 0),
        2 => V.init(0, c, x),
        3 => V.init(0, x, c),
        4 => V.init(x, 0, c),
        else => V.init(c, 0, x),
    };
    return base.addScalar(m);
}

// --- YCoCg ------------------------------------------------------------------

pub fn rgbToYcocg(rgb: anytype) @TypeOf(rgb) {
    const V = @TypeOf(rgb);
    return V.init(
        rgb.x / 4.0 + rgb.y / 2.0 + rgb.z / 4.0,
        rgb.x / 2.0 - rgb.z / 2.0,
        -rgb.x / 4.0 + rgb.y / 2.0 - rgb.z / 4.0,
    );
}
pub fn ycocgToRgb(c: anytype) @TypeOf(c) {
    const V = @TypeOf(c);
    const tmp = c.x - c.z;
    return V.init(tmp + c.y, c.x + c.z, tmp - c.y);
}
/// Reversible (lossless) YCoCg.
pub fn rgbToYcocgR(rgb: anytype) @TypeOf(rgb) {
    const V = @TypeOf(rgb);
    return V.init(rgb.y * 0.5 + (rgb.x + rgb.z) * 0.25, rgb.x - rgb.z, rgb.y - (rgb.x + rgb.z) * 0.5);
}
pub fn ycocgRToRgb(c: anytype) @TypeOf(c) {
    const V = @TypeOf(c);
    const tmp = c.x - c.z * 0.5;
    const g = c.z + tmp;
    const b = tmp - c.y * 0.5;
    return V.init(b + c.y, g, b);
}

// --- luminance / saturation -------------------------------------------------

pub fn luminance(rgb: anytype) @TypeOf(rgb).Element {
    const V = @TypeOf(rgb);
    return rgb.dot(V.init(0.33, 0.59, 0.11));
}
/// Saturation-adjustment matrix (`s=0` → greyscale, `s=1` → identity).
pub fn saturation(s: anytype) mat.Mat(4, 4, @TypeOf(s)) {
    const T = @TypeOf(s);
    const V3 = vec.Vec(3, T);
    const V4 = vec.Vec(4, T);
    const col = V3.init(0.2126, 0.7152, 0.0722).scale(1 - s);
    return mat.Mat(4, 4, T).fromColumns(.{
        V4.init(col.x + s, col.x, col.x, 0),
        V4.init(col.y, col.y + s, col.y, 0),
        V4.init(col.z, col.z, col.z + s, 0),
        V4.init(0, 0, 0, 1),
    });
}

// --- XYZ chromatic conversions (ported verbatim from GLM) -------------------

pub fn linearSrgbToXyzD65(c: anytype) @TypeOf(c) {
    const V = @TypeOf(c);
    const m = V.init(0.490, 0.17697, 0.2);
    const n = V.init(0.31, 0.8124, 0.01063);
    const o = V.init(0.490, 0.01, 0.99);
    return m.mul(c).add(n.mul(c)).add(o.mul(c)).scale(5.650675255693055);
}
pub fn linearSrgbToXyzD50(c: anytype) @TypeOf(c) {
    const V = @TypeOf(c);
    const m = V.init(0.436030342570117, 0.222438466210245, 0.013897440074263);
    const n = V.init(0.385101860087134, 0.716942745571917, 0.097076381494207);
    const o = V.init(0.143067806654203, 0.060618777416563, 0.713926257896652);
    return m.mul(c).add(n.mul(c)).add(o.mul(c));
}
pub fn xyzD65ToLinearSrgb(c: anytype) @TypeOf(c) {
    const V = @TypeOf(c);
    const m = V.init(0.41847, -0.091169, 0.0009209);
    const n = V.init(-0.15866, 0.25243, 0.015708);
    const o = V.init(0.0009209, -0.0025498, 0.1786);
    return m.mul(c).add(n.mul(c)).add(o.mul(c));
}
pub fn xyzD65ToXyzD50(c: anytype) @TypeOf(c) {
    const V = @TypeOf(c);
    const m = V.init(1.047844353856414, 0.029549007606644, -0.009250984365223);
    const n = V.init(0.022898981050086, 0.990508028941971, 0.015072338237051);
    const o = V.init(-0.050206647741605, -0.017074711360960, 0.751717835079977);
    return m.mul(c).add(n.mul(c)).add(o.mul(c));
}

const testing = std.testing;
const Vec3 = vec.Vec3;
const Mat4 = mat.Mat4;
test "color round trips" {
    const c = Vec3.init(0.2, 0.6, 0.4);
    try testing.expect(srgbToLinear(linearToSrgb(c)).approxEql(c, 1e-5));
    try testing.expect(hsvToRgb(rgbToHsv(c)).approxEql(c, 1e-4));
    try testing.expect(hslToRgb(rgbToHsl(c)).approxEql(c, 1e-4));
    try testing.expect(ycocgToRgb(rgbToYcocg(c)).approxEql(c, 1e-5));
    try testing.expect(ycocgRToRgb(rgbToYcocgR(c)).approxEql(c, 1e-5));
    try testing.expect(saturation(@as(f32, 1)).approxEql(Mat4.identity(), 1e-6));
}
test "color generic over f64" {
    const V = vec.Vec(3, f64);
    const c = V.init(0.2, 0.6, 0.4);
    try testing.expect(hsvToRgb(rgbToHsv(c)).approxEql(c, 1e-12));
    try testing.expect(ycocgToRgb(rgbToYcocg(c)).approxEql(c, 1e-12));
}
