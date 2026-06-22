//! Perlin (classic), periodic Perlin, and simplex noise — `math.noise`.
//! Dimensions 2-4, generic over the float element type via `Noise(T)`; root
//! exposes the f32 instance as `math.noise` and the builder as `math.Noise`.
//!
//! Ported from Stefan Gustavson's permutation-free GLSL implementations (the
//! same ones GLM uses). Internally uses raw `@Vector` math so the GLSL operator
//! expressions translate directly; inputs/outputs use our `Vec` types.

const std = @import("std");
const vec = @import("vec.zig");

/// Noise functions parameterized by float element type `T`.
pub fn Noise(comptime T: type) type {
    return struct {
        const Vec2 = vec.Vec(2, T);
        const Vec3 = vec.Vec(3, T);
        const Vec4 = vec.Vec(4, T);

        const V2 = @Vector(2, T);
        const V3 = @Vector(3, T);
        const V4 = @Vector(4, T);

        inline fn s2(x: T) V2 {
            return @splat(x);
        }
        inline fn s3(x: T) V3 {
            return @splat(x);
        }
        inline fn s4(x: T) V4 {
            return @splat(x);
        }
        inline fn dotv(a: anytype, b: @TypeOf(a)) T {
            return @reduce(.Add, a * b);
        }
        inline fn sw4(v: V4, comptime a: i32, comptime b: i32, comptime c: i32, comptime d: i32) V4 {
            return @shuffle(T, v, v, @Vector(4, i32){ a, b, c, d });
        }
        inline fn fr3(x: V3) V3 {
            return x - @floor(x);
        }
        inline fn fr4(x: V4) V4 {
            return x - @floor(x);
        }

        inline fn mod289_2(x: V2) V2 {
            return x - @floor(x * s2(1.0 / 289.0)) * s2(289.0);
        }
        inline fn mod289_3(x: V3) V3 {
            return x - @floor(x * s3(1.0 / 289.0)) * s3(289.0);
        }
        inline fn mod289_4(x: V4) V4 {
            return x - @floor(x * s4(1.0 / 289.0)) * s4(289.0);
        }
        inline fn modv3(x: V3, y: V3) V3 {
            return x - y * @floor(x / y);
        }
        inline fn modv4(x: V4, y: V4) V4 {
            return x - y * @floor(x / y);
        }
        inline fn permute3(x: V3) V3 {
            return mod289_3((x * s3(34.0) + s3(1.0)) * x);
        }
        inline fn permute4(x: V4) V4 {
            return mod289_4((x * s4(34.0) + s4(1.0)) * x);
        }
        inline fn permuteScalar(x: T) T {
            return @mod(((x * 34.0) + 1.0) * x, 289.0);
        }
        inline fn taylorInvSqrt4(r: V4) V4 {
            return s4(1.79284291400159) - s4(0.85373472095314) * r;
        }
        inline fn taylorInvSqrtScalar(r: T) T {
            return 1.79284291400159 - 0.85373472095314 * r;
        }
        inline fn fade2(t: V2) V2 {
            return t * t * t * (t * (t * s2(6.0) - s2(15.0)) + s2(10.0));
        }
        inline fn fade3(t: V3) V3 {
            return t * t * t * (t * (t * s3(6.0) - s3(15.0)) + s3(10.0));
        }
        inline fn fade4(t: V4) V4 {
            return t * t * t * (t * (t * s4(6.0) - s4(15.0)) + s4(10.0));
        }
        inline fn step3(edge: V3, x: V3) V3 {
            return @select(T, x < edge, s3(0.0), s3(1.0));
        }
        inline fn step4(edge: V4, x: V4) V4 {
            return @select(T, x < edge, s4(0.0), s4(1.0));
        }

        // ===================================================================
        // Simplex noise
        // ===================================================================

        /// Simplex noise, 2-D, range ≈ [-1, 1].
        pub fn simplex2(p: Vec2) T {
            const v = p.simd();
            const C = V4{ 0.211324865405187, 0.366025403784439, -0.577350269189626, 0.024390243902439 };
            var i = @floor(v + s2(dotv(v, V2{ C[1], C[1] })));
            const x0 = v - i + s2(dotv(i, V2{ C[0], C[0] }));
            const i1v = if (x0[0] > x0[1]) V2{ 1, 0 } else V2{ 0, 1 };
            var x12 = V4{ x0[0], x0[1], x0[0], x0[1] } + V4{ C[0], C[0], C[2], C[2] };
            x12[0] -= i1v[0];
            x12[1] -= i1v[1];
            i = mod289_2(i);
            const p3 = permute3(permute3(V3{ i[1], i[1] + i1v[1], i[1] + 1 }) + V3{ i[0], i[0] + i1v[0], i[0] + 1 });
            const x12xy = V2{ x12[0], x12[1] };
            const x12zw = V2{ x12[2], x12[3] };
            var m = @max(s3(0.5) - V3{ dotv(x0, x0), dotv(x12xy, x12xy), dotv(x12zw, x12zw) }, s3(0.0));
            m = m * m;
            m = m * m;
            const x = s3(2.0) * fr3(p3 * s3(C[3])) - s3(1.0);
            const h = @abs(x) - s3(0.5);
            const ox = @floor(x + s3(0.5));
            const a0 = x - ox;
            m = m * (s3(1.79284291400159) - s3(0.85373472095314) * (a0 * a0 + h * h));
            const g = V3{
                a0[0] * x0[0] + h[0] * x0[1],
                a0[1] * x12[0] + h[1] * x12[1],
                a0[2] * x12[2] + h[2] * x12[3],
            };
            return 130.0 * dotv(m, g);
        }

        /// Simplex noise, 3-D, range ≈ [-1, 1].
        pub fn simplex3(p: Vec3) T {
            const v = p.simd();
            const C = V2{ 1.0 / 6.0, 1.0 / 3.0 };
            var i = @floor(v + s3(dotv(v, s3(C[1]))));
            const x0 = v - i + s3(dotv(i, s3(C[0])));
            const g = step3(V3{ x0[1], x0[2], x0[0] }, x0);
            const l = s3(1.0) - g;
            const i1v = @min(g, V3{ l[2], l[0], l[1] });
            const i2v = @max(g, V3{ l[2], l[0], l[1] });
            const x1 = x0 - i1v + s3(C[0]);
            const x2 = x0 - i2v + s3(C[1]);
            const x3 = x0 - s3(0.5);
            i = mod289_3(i);
            const pp = permute4(permute4(permute4(s4(i[2]) + V4{ 0, i1v[2], i2v[2], 1 }) +
                s4(i[1]) + V4{ 0, i1v[1], i2v[1], 1 }) +
                s4(i[0]) + V4{ 0, i1v[0], i2v[0], 1 });
            const n_ = 0.142857142857;
            const ns = s3(n_) * V3{ 0.0, 0.5, 1.0 } - V3{ 0.0, 1.0, 0.0 };
            const j = pp - s4(49.0) * @floor(pp * s4(ns[2] * ns[2]));
            const x_ = @floor(j * s4(ns[2]));
            const y_ = @floor(j - s4(7.0) * x_);
            const gx = x_ * s4(ns[0]) + s4(ns[1]);
            const gy = y_ * s4(ns[0]) + s4(ns[1]);
            const gh = s4(1.0) - @abs(gx) - @abs(gy);
            const b0 = V4{ gx[0], gx[1], gy[0], gy[1] };
            const b1 = V4{ gx[2], gx[3], gy[2], gy[3] };
            const s0 = @floor(b0) * s4(2.0) + s4(1.0);
            const s1 = @floor(b1) * s4(2.0) + s4(1.0);
            const sh = -step4(gh, s4(0.0));
            const a0 = sw4(b0, 0, 2, 1, 3) + sw4(s0, 0, 2, 1, 3) * V4{ sh[0], sh[0], sh[1], sh[1] };
            const a1 = sw4(b1, 0, 2, 1, 3) + sw4(s1, 0, 2, 1, 3) * V4{ sh[2], sh[2], sh[3], sh[3] };
            var p0 = V3{ a0[0], a0[1], gh[0] };
            var p1 = V3{ a0[2], a0[3], gh[1] };
            var p2 = V3{ a1[0], a1[1], gh[2] };
            var p3v = V3{ a1[2], a1[3], gh[3] };
            const nrm = taylorInvSqrt4(V4{ dotv(p0, p0), dotv(p1, p1), dotv(p2, p2), dotv(p3v, p3v) });
            p0 = p0 * s3(nrm[0]);
            p1 = p1 * s3(nrm[1]);
            p2 = p2 * s3(nrm[2]);
            p3v = p3v * s3(nrm[3]);
            var m = @max(s4(0.6) - V4{ dotv(x0, x0), dotv(x1, x1), dotv(x2, x2), dotv(x3, x3) }, s4(0.0));
            m = m * m;
            return 42.0 * dotv(m * m, V4{ dotv(p0, x0), dotv(p1, x1), dotv(p2, x2), dotv(p3v, x3) });
        }

        fn grad4(j: T, ip: V4) V4 {
            const pf = @floor(fr3(s3(j) * V3{ ip[0], ip[1], ip[2] }) * s3(7.0)) * s3(ip[2]) - s3(1.0);
            const pw = 1.5 - dotv(@abs(pf), V3{ 1, 1, 1 });
            var p = V4{ pf[0], pf[1], pf[2], pw };
            const s = @select(T, p < s4(0.0), s4(1.0), s4(0.0));
            p[0] += (s[0] * 2.0 - 1.0) * s[3];
            p[1] += (s[1] * 2.0 - 1.0) * s[3];
            p[2] += (s[2] * 2.0 - 1.0) * s[3];
            return p;
        }

        /// Simplex noise, 4-D, range ≈ [-1, 1].
        pub fn simplex4(p: Vec4) T {
            const F4: T = 0.309016994374947451;
            const C = V4{ 0.138196601125011, 0.276393202250021, 0.414589803375032, -0.447213595499958 };
            const v = p.simd();
            var i = @floor(v + s4(dotv(v, s4(F4))));
            const x0 = v - i + s4(dotv(i, s4(C[0])));

            const isX = step3(V3{ x0[1], x0[2], x0[3] }, V3{ x0[0], x0[0], x0[0] });
            const isYZ = step3(V3{ x0[2], x0[3], x0[3] }, V3{ x0[1], x0[1], x0[2] });
            var idx0 = V4{ isX[0] + isX[1] + isX[2], 1.0 - isX[0], 1.0 - isX[1], 1.0 - isX[2] };
            idx0[1] += isYZ[0] + isYZ[1];
            idx0[2] += 1.0 - isYZ[0];
            idx0[3] += 1.0 - isYZ[1];
            idx0[2] += isYZ[2];
            idx0[3] += 1.0 - isYZ[2];

            const i3v = @min(@max(idx0, s4(0.0)), s4(1.0));
            const i2v = @min(@max(idx0 - s4(1.0), s4(0.0)), s4(1.0));
            const i1v = @min(@max(idx0 - s4(2.0), s4(0.0)), s4(1.0));
            const x1 = x0 - i1v + s4(C[0]);
            const x2 = x0 - i2v + s4(C[1]);
            const x3 = x0 - i3v + s4(C[2]);
            const x4 = x0 + s4(C[3]);

            i = mod289_4(i);
            const j0 = permuteScalar(permuteScalar(permuteScalar(permuteScalar(i[3]) + i[2]) + i[1]) + i[0]);
            const j1 = permute4(permute4(permute4(permute4(s4(i[3]) + V4{ i1v[3], i2v[3], i3v[3], 1.0 }) +
                s4(i[2]) + V4{ i1v[2], i2v[2], i3v[2], 1.0 }) +
                s4(i[1]) + V4{ i1v[1], i2v[1], i3v[1], 1.0 }) +
                s4(i[0]) + V4{ i1v[0], i2v[0], i3v[0], 1.0 });

            const ip = V4{ 1.0 / 294.0, 1.0 / 49.0, 1.0 / 7.0, 0.0 };
            var p0 = grad4(j0, ip);
            var p1 = grad4(j1[0], ip);
            var p2 = grad4(j1[1], ip);
            var p3v = grad4(j1[2], ip);
            var p4 = grad4(j1[3], ip);

            const nrm = taylorInvSqrt4(V4{ dotv(p0, p0), dotv(p1, p1), dotv(p2, p2), dotv(p3v, p3v) });
            p0 = p0 * s4(nrm[0]);
            p1 = p1 * s4(nrm[1]);
            p2 = p2 * s4(nrm[2]);
            p3v = p3v * s4(nrm[3]);
            p4 = p4 * s4(taylorInvSqrtScalar(dotv(p4, p4)));

            var m0 = @max(s3(0.6) - V3{ dotv(x0, x0), dotv(x1, x1), dotv(x2, x2) }, s3(0.0));
            var m1 = @max(s2(0.6) - V2{ dotv(x3, x3), dotv(x4, x4) }, s2(0.0));
            m0 = m0 * m0;
            m1 = m1 * m1;
            return 49.0 * (dotv(m0 * m0, V3{ dotv(p0, x0), dotv(p1, x1), dotv(p2, x2) }) +
                dotv(m1 * m1, V2{ dotv(p3v, x3), dotv(p4, x4) }));
        }

        // ===================================================================
        // Classic Perlin noise (+ periodic variants)
        // ===================================================================

        fn perlin2Impl(pt: Vec2, comptime periodic: bool, rep: V2) T {
            const P = pt.simd();
            const Pxyxy = V4{ P[0], P[1], P[0], P[1] };
            var Pi = @floor(Pxyxy) + V4{ 0, 0, 1, 1 };
            const Pf = fr4(Pxyxy) - V4{ 0, 0, 1, 1 };
            if (periodic) Pi = modv4(Pi, V4{ rep[0], rep[1], rep[0], rep[1] });
            Pi = mod289_4(Pi);
            const ix = sw4(Pi, 0, 2, 0, 2);
            const iy = sw4(Pi, 1, 1, 3, 3);
            const fx = sw4(Pf, 0, 2, 0, 2);
            const fy = sw4(Pf, 1, 1, 3, 3);
            const ii = permute4(permute4(ix) + iy);
            var gx = s4(2.0) * fr4(ii * s4(0.0243902439)) - s4(1.0);
            const gy = @abs(gx) - s4(0.5);
            gx = gx - @floor(gx + s4(0.5));
            var g00 = V2{ gx[0], gy[0] };
            var g10 = V2{ gx[1], gy[1] };
            var g01 = V2{ gx[2], gy[2] };
            var g11 = V2{ gx[3], gy[3] };
            const nrm = taylorInvSqrt4(V4{ dotv(g00, g00), dotv(g01, g01), dotv(g10, g10), dotv(g11, g11) });
            g00 = g00 * s2(nrm[0]);
            g01 = g01 * s2(nrm[1]);
            g10 = g10 * s2(nrm[2]);
            g11 = g11 * s2(nrm[3]);
            const n00 = dotv(g00, V2{ fx[0], fy[0] });
            const n10 = dotv(g10, V2{ fx[1], fy[1] });
            const n01 = dotv(g01, V2{ fx[2], fy[2] });
            const n11 = dotv(g11, V2{ fx[3], fy[3] });
            const fade_xy = fade2(V2{ Pf[0], Pf[1] });
            const nx0 = n00 + fade_xy[0] * (n10 - n00);
            const nx1 = n01 + fade_xy[0] * (n11 - n01);
            return 2.3 * (nx0 + fade_xy[1] * (nx1 - nx0));
        }

        fn perlin3Impl(pt: Vec3, comptime periodic: bool, rep: V3) T {
            const P = pt.simd();
            var Pi0 = @floor(P);
            var Pi1 = Pi0 + s3(1.0);
            if (periodic) {
                Pi0 = modv3(Pi0, rep);
                Pi1 = modv3(Pi1, rep);
            }
            Pi0 = mod289_3(Pi0);
            Pi1 = mod289_3(Pi1);
            const Pf0 = fr3(P);
            const Pf1 = Pf0 - s3(1.0);
            const ix = V4{ Pi0[0], Pi1[0], Pi0[0], Pi1[0] };
            const iy = V4{ Pi0[1], Pi0[1], Pi1[1], Pi1[1] };
            const iz0 = s4(Pi0[2]);
            const iz1 = s4(Pi1[2]);
            const ixy = permute4(permute4(ix) + iy);
            const ixy0 = permute4(ixy + iz0);
            const ixy1 = permute4(ixy + iz1);

            var gx0 = ixy0 * s4(1.0 / 7.0);
            var gy0 = fr4(@floor(gx0) * s4(1.0 / 7.0)) - s4(0.5);
            gx0 = fr4(gx0);
            const gz0 = s4(0.5) - @abs(gx0) - @abs(gy0);
            const sz0 = step4(gz0, s4(0.0));
            gx0 = gx0 - sz0 * (step4(s4(0.0), gx0) - s4(0.5));
            gy0 = gy0 - sz0 * (step4(s4(0.0), gy0) - s4(0.5));

            var gx1 = ixy1 * s4(1.0 / 7.0);
            var gy1 = fr4(@floor(gx1) * s4(1.0 / 7.0)) - s4(0.5);
            gx1 = fr4(gx1);
            const gz1 = s4(0.5) - @abs(gx1) - @abs(gy1);
            const sz1 = step4(gz1, s4(0.0));
            gx1 = gx1 - sz1 * (step4(s4(0.0), gx1) - s4(0.5));
            gy1 = gy1 - sz1 * (step4(s4(0.0), gy1) - s4(0.5));

            var g000 = V3{ gx0[0], gy0[0], gz0[0] };
            var g100 = V3{ gx0[1], gy0[1], gz0[1] };
            var g010 = V3{ gx0[2], gy0[2], gz0[2] };
            var g110 = V3{ gx0[3], gy0[3], gz0[3] };
            var g001 = V3{ gx1[0], gy1[0], gz1[0] };
            var g101 = V3{ gx1[1], gy1[1], gz1[1] };
            var g011 = V3{ gx1[2], gy1[2], gz1[2] };
            var g111 = V3{ gx1[3], gy1[3], gz1[3] };

            const norm0 = taylorInvSqrt4(V4{ dotv(g000, g000), dotv(g010, g010), dotv(g100, g100), dotv(g110, g110) });
            g000 = g000 * s3(norm0[0]);
            g010 = g010 * s3(norm0[1]);
            g100 = g100 * s3(norm0[2]);
            g110 = g110 * s3(norm0[3]);
            const norm1 = taylorInvSqrt4(V4{ dotv(g001, g001), dotv(g011, g011), dotv(g101, g101), dotv(g111, g111) });
            g001 = g001 * s3(norm1[0]);
            g011 = g011 * s3(norm1[1]);
            g101 = g101 * s3(norm1[2]);
            g111 = g111 * s3(norm1[3]);

            const n000 = dotv(g000, Pf0);
            const n100 = dotv(g100, V3{ Pf1[0], Pf0[1], Pf0[2] });
            const n010 = dotv(g010, V3{ Pf0[0], Pf1[1], Pf0[2] });
            const n110 = dotv(g110, V3{ Pf1[0], Pf1[1], Pf0[2] });
            const n001 = dotv(g001, V3{ Pf0[0], Pf0[1], Pf1[2] });
            const n101 = dotv(g101, V3{ Pf1[0], Pf0[1], Pf1[2] });
            const n011 = dotv(g011, V3{ Pf0[0], Pf1[1], Pf1[2] });
            const n111 = dotv(g111, Pf1);

            const fade_xyz = fade3(Pf0);
            const nz0 = V4{ n000, n100, n010, n110 };
            const nz1 = V4{ n001, n101, n011, n111 };
            const n_z = nz0 + s4(fade_xyz[2]) * (nz1 - nz0);
            const nyz0 = V2{ n_z[0], n_z[1] };
            const nyz1 = V2{ n_z[2], n_z[3] };
            const n_yz = nyz0 + s2(fade_xyz[1]) * (nyz1 - nyz0);
            return 2.2 * (n_yz[0] + fade_xyz[0] * (n_yz[1] - n_yz[0]));
        }

        fn perlin4Impl(pt: Vec4, comptime periodic: bool, rep: V4) T {
            const P = pt.simd();
            var Pi0 = @floor(P);
            var Pi1 = Pi0 + s4(1.0);
            if (periodic) {
                Pi0 = modv4(Pi0, rep);
                Pi1 = modv4(Pi1, rep);
            }
            Pi0 = mod289_4(Pi0);
            Pi1 = mod289_4(Pi1);
            const Pf0 = fr4(P);
            const Pf1 = Pf0 - s4(1.0);
            const ix = V4{ Pi0[0], Pi1[0], Pi0[0], Pi1[0] };
            const iy = V4{ Pi0[1], Pi0[1], Pi1[1], Pi1[1] };
            const iz0 = s4(Pi0[2]);
            const iz1 = s4(Pi1[2]);
            const iw0 = s4(Pi0[3]);
            const iw1 = s4(Pi1[3]);
            const ixy = permute4(permute4(ix) + iy);
            const ixy0 = permute4(ixy + iz0);
            const ixy1 = permute4(ixy + iz1);
            const ixy00 = permute4(ixy0 + iw0);
            const ixy01 = permute4(ixy0 + iw1);
            const ixy10 = permute4(ixy1 + iw0);
            const ixy11 = permute4(ixy1 + iw1);

            const G = struct {
                fn group(ixyXX: V4) [4]V4 {
                    var gx = ixyXX * s4(1.0 / 7.0);
                    var gy = @floor(gx) * s4(1.0 / 7.0);
                    var gz = @floor(gy) * s4(1.0 / 6.0);
                    gx = fr4(gx) - s4(0.5);
                    gy = fr4(gy) - s4(0.5);
                    gz = fr4(gz) - s4(0.5);
                    const gw = s4(0.75) - @abs(gx) - @abs(gy) - @abs(gz);
                    const sw = step4(gw, s4(0.0));
                    gx = gx - sw * (step4(s4(0.0), gx) - s4(0.5));
                    gy = gy - sw * (step4(s4(0.0), gy) - s4(0.5));
                    return .{ gx, gy, gz, gw };
                }
            };
            const a00 = G.group(ixy00);
            const a01 = G.group(ixy01);
            const a10 = G.group(ixy10);
            const a11 = G.group(ixy11);

            const corner = struct {
                fn pick(g: [4]V4, comptime k: usize) V4 {
                    return V4{ g[0][k], g[1][k], g[2][k], g[3][k] };
                }
            };
            var g0000 = corner.pick(a00, 0);
            var g1000 = corner.pick(a00, 1);
            var g0100 = corner.pick(a00, 2);
            var g1100 = corner.pick(a00, 3);
            var g0010 = corner.pick(a10, 0);
            var g1010 = corner.pick(a10, 1);
            var g0110 = corner.pick(a10, 2);
            var g1110 = corner.pick(a10, 3);
            var g0001 = corner.pick(a01, 0);
            var g1001 = corner.pick(a01, 1);
            var g0101 = corner.pick(a01, 2);
            var g1101 = corner.pick(a01, 3);
            var g0011 = corner.pick(a11, 0);
            var g1011 = corner.pick(a11, 1);
            var g0111 = corner.pick(a11, 2);
            var g1111 = corner.pick(a11, 3);

            const n00 = taylorInvSqrt4(V4{ dotv(g0000, g0000), dotv(g0100, g0100), dotv(g1000, g1000), dotv(g1100, g1100) });
            g0000 = g0000 * s4(n00[0]);
            g0100 = g0100 * s4(n00[1]);
            g1000 = g1000 * s4(n00[2]);
            g1100 = g1100 * s4(n00[3]);
            const n01 = taylorInvSqrt4(V4{ dotv(g0001, g0001), dotv(g0101, g0101), dotv(g1001, g1001), dotv(g1101, g1101) });
            g0001 = g0001 * s4(n01[0]);
            g0101 = g0101 * s4(n01[1]);
            g1001 = g1001 * s4(n01[2]);
            g1101 = g1101 * s4(n01[3]);
            const n10 = taylorInvSqrt4(V4{ dotv(g0010, g0010), dotv(g0110, g0110), dotv(g1010, g1010), dotv(g1110, g1110) });
            g0010 = g0010 * s4(n10[0]);
            g0110 = g0110 * s4(n10[1]);
            g1010 = g1010 * s4(n10[2]);
            g1110 = g1110 * s4(n10[3]);
            const n11 = taylorInvSqrt4(V4{ dotv(g0011, g0011), dotv(g0111, g0111), dotv(g1011, g1011), dotv(g1111, g1111) });
            g0011 = g0011 * s4(n11[0]);
            g0111 = g0111 * s4(n11[1]);
            g1011 = g1011 * s4(n11[2]);
            g1111 = g1111 * s4(n11[3]);

            const f0 = Pf0;
            const f1 = Pf1;
            const n0000 = dotv(g0000, f0);
            const n1000 = dotv(g1000, V4{ f1[0], f0[1], f0[2], f0[3] });
            const n0100 = dotv(g0100, V4{ f0[0], f1[1], f0[2], f0[3] });
            const n1100 = dotv(g1100, V4{ f1[0], f1[1], f0[2], f0[3] });
            const n0010 = dotv(g0010, V4{ f0[0], f0[1], f1[2], f0[3] });
            const n1010 = dotv(g1010, V4{ f1[0], f0[1], f1[2], f0[3] });
            const n0110 = dotv(g0110, V4{ f0[0], f1[1], f1[2], f0[3] });
            const n1110 = dotv(g1110, V4{ f1[0], f1[1], f1[2], f0[3] });
            const n0001 = dotv(g0001, V4{ f0[0], f0[1], f0[2], f1[3] });
            const n1001 = dotv(g1001, V4{ f1[0], f0[1], f0[2], f1[3] });
            const n0101 = dotv(g0101, V4{ f0[0], f1[1], f0[2], f1[3] });
            const n1101 = dotv(g1101, V4{ f1[0], f1[1], f0[2], f1[3] });
            const n0011 = dotv(g0011, V4{ f0[0], f0[1], f1[2], f1[3] });
            const n1011 = dotv(g1011, V4{ f1[0], f0[1], f1[2], f1[3] });
            const n0111 = dotv(g0111, V4{ f0[0], f1[1], f1[2], f1[3] });
            const n1111 = dotv(g1111, f1);

            const fade_xyzw = fade4(Pf0);
            const a = V4{ n0000, n1000, n0100, n1100 };
            const b = V4{ n0001, n1001, n0101, n1101 };
            const n_0w = a + s4(fade_xyzw[3]) * (b - a);
            const c = V4{ n0010, n1010, n0110, n1110 };
            const d = V4{ n0011, n1011, n0111, n1111 };
            const n_1w = c + s4(fade_xyzw[3]) * (d - c);
            const n_zw = n_0w + s4(fade_xyzw[2]) * (n_1w - n_0w);
            const nyzw0 = V2{ n_zw[0], n_zw[1] };
            const nyzw1 = V2{ n_zw[2], n_zw[3] };
            const n_yzw = nyzw0 + s2(fade_xyzw[1]) * (nyzw1 - nyzw0);
            return 2.2 * (n_yzw[0] + fade_xyzw[0] * (n_yzw[1] - n_yzw[0]));
        }

        /// Classic Perlin noise, range ≈ [-1, 1].
        pub fn perlin2(p: Vec2) T {
            return perlin2Impl(p, false, undefined);
        }
        pub fn perlin3(p: Vec3) T {
            return perlin3Impl(p, false, undefined);
        }
        pub fn perlin4(p: Vec4) T {
            return perlin4Impl(p, false, undefined);
        }

        /// Periodic classic Perlin noise (tiles with period `rep`).
        pub fn pnoise2(p: Vec2, rep: Vec2) T {
            return perlin2Impl(p, true, rep.simd());
        }
        pub fn pnoise3(p: Vec3, rep: Vec3) T {
            return perlin3Impl(p, true, rep.simd());
        }
        pub fn pnoise4(p: Vec4, rep: Vec4) T {
            return perlin4Impl(p, true, rep.simd());
        }

        // ===================================================================
        // Fractal (multi-octave) noise
        // ===================================================================

        /// Base Perlin noise for any of the supported dimensions.
        fn perlin(p: anytype) T {
            return switch (@TypeOf(p).dim) {
                2 => perlin2(p),
                3 => perlin3(p),
                4 => perlin4(p),
                else => @compileError("noise supports 2-D, 3-D, 4-D points only"),
            };
        }

        /// Parameters for the fractal variants. `lacunarity` scales frequency
        /// per octave (≈2), `gain` scales amplitude per octave (≈0.5).
        pub const Fractal = struct { octaves: u32 = 5, lacunarity: T = 2.0, gain: T = 0.5 };

        /// Fractal Brownian motion — summed Perlin octaves, range ≈ [-1, 1].
        pub fn fbm(p: anytype, opts: Fractal) T {
            var sum: T = 0;
            var amp: T = 1;
            var freq: T = 1;
            var norm: T = 0;
            var i: u32 = 0;
            while (i < opts.octaves) : (i += 1) {
                sum += amp * perlin(p.scale(freq));
                norm += amp;
                amp *= opts.gain;
                freq *= opts.lacunarity;
            }
            return sum / norm;
        }
        /// Turbulence — summed |Perlin| octaves (billowy; smoke/clouds), range ≈ [0, 1].
        pub fn turbulence(p: anytype, opts: Fractal) T {
            var sum: T = 0;
            var amp: T = 1;
            var freq: T = 1;
            var norm: T = 0;
            var i: u32 = 0;
            while (i < opts.octaves) : (i += 1) {
                sum += amp * @abs(perlin(p.scale(freq)));
                norm += amp;
                amp *= opts.gain;
                freq *= opts.lacunarity;
            }
            return sum / norm;
        }
        /// Ridged multifractal — sharp ridges (mountains/fire), range ≈ [0, 1].
        pub fn ridged(p: anytype, opts: Fractal) T {
            var sum: T = 0;
            var amp: T = 1;
            var freq: T = 1;
            var norm: T = 0;
            var i: u32 = 0;
            while (i < opts.octaves) : (i += 1) {
                const r = 1.0 - @abs(perlin(p.scale(freq)));
                sum += amp * r * r;
                norm += amp;
                amp *= opts.gain;
                freq *= opts.lacunarity;
            }
            return sum / norm;
        }
    };
}

/// f32 default-precision noise.
pub const noise = Noise(f32);

const testing = std.testing;
const NV2 = vec.Vec2;
const NV3 = vec.Vec3;
const NV4 = vec.Vec4;

test "noise lattice zeros, range, determinism" {
    const n = noise;
    // 2-D simplex & all classic perlin vanish at integer lattice points.
    try testing.expectApproxEqAbs(@as(f32, 0), n.simplex2(NV2.init(0, 0)), 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 0), n.perlin2(NV2.init(0, 0)), 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 0), n.perlin3(NV3.init(0, 0, 0)), 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 0), n.perlin4(NV4.init(0, 0, 0, 0)), 1e-5);

    var x: f32 = 0;
    while (x < 8) : (x += 0.41) {
        const ns = [_]f32{
            n.simplex2(NV2.init(x, x * 1.3)),
            n.simplex3(NV3.init(x, x * 0.7, -x)),
            n.simplex4(NV4.init(x, -x, x * 0.5, 1.0)),
            n.perlin2(NV2.init(x, -x)),
            n.perlin3(NV3.init(x, x, -x)),
            n.perlin4(NV4.init(x, -x, x, -x)),
            n.pnoise2(NV2.init(x, -x), NV2.init(4, 4)),
            n.pnoise3(NV3.init(x, x, x), NV3.init(4, 4, 4)),
            n.pnoise4(NV4.init(x, x, x, x), NV4.init(4, 4, 4, 4)),
        };
        for (ns) |val| try testing.expect(val >= -1.6 and val <= 1.6);
        try testing.expectEqual(ns[2], n.simplex4(NV4.init(x, -x, x * 0.5, 1.0)));
    }
}

test "perlin3/4 are continuous (no cell-boundary jumps)" {
    const n = noise;
    var prev4 = n.perlin4(NV4.init(0, 0, 0, 0));
    var prev3 = n.perlin3(NV3.init(0, 0, 0));
    var t: f32 = 0;
    while (t < 4.0) : (t += 0.025) {
        const c4 = n.perlin4(NV4.init(t, t * 0.5, -t * 0.3, t * 0.7));
        const c3 = n.perlin3(NV3.init(t, -t * 0.4, t * 0.6));
        try testing.expect(@abs(c4 - prev4) < 0.3);
        try testing.expect(@abs(c3 - prev3) < 0.3);
        prev4 = c4;
        prev3 = c3;
    }
}

test "pnoise tiles with its period" {
    const n = noise;
    const rep = NV2.init(5, 5);
    try testing.expectApproxEqAbs(n.pnoise2(NV2.init(1.3, 2.1), rep), n.pnoise2(NV2.init(1.3 + 5, 2.1), rep), 1e-4);
    try testing.expectApproxEqAbs(n.pnoise2(NV2.init(1.3, 2.1), rep), n.pnoise2(NV2.init(1.3, 2.1 + 5), rep), 1e-4);
}

test "noise generic over f64" {
    const N = Noise(f64);
    const V = vec.Vec(2, f64);
    try testing.expectApproxEqAbs(@as(f64, 0), N.perlin2(V.init(0, 0)), 1e-12);
    try testing.expect(@abs(N.simplex2(V.init(1.5, -2.3))) <= 1.6);
}
