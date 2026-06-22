//! Geometry — `math.geom` (vector geometry), `math.intersect` (ray/line casts),
//! and `math.bounds` (AABB / sphere).
//! Run: `zig build example-geometry`

const std = @import("std");
const math = @import("math");
const print = std.debug.print;

pub fn main() void {
    geomReflect();
    geomAngles();
    geomProject();
    castRay();
    boundsVolumes();
    shapePrimitives();
    boundingVolumes();
}

fn geomReflect() void {
    // reflect/refract/faceForward are GLSL free functions (no method form).
    const i = math.Vec3.init(1, -1, 0).normalize();
    const n = math.Vec3.init(0, 1, 0);
    print("reflect = {f}\n", .{math.reflect(i, n)});
    print("refract = {f}\n", .{math.refract(i, n, 1.0)});
}

fn geomAngles() void {
    const a = math.Vec3.init(1, 0, 0);
    const b = math.Vec3.init(0, 1, 0);
    print("angle          = {d} rad\n", .{math.geom.angle(a, b)});
    print("signed (2D)    = {d}\n", .{math.geom.orientedAngle2(math.Vec2.init(0, 1), math.Vec2.init(1, 0))});
    print("areOrthogonal  = {}\n", .{math.geom.areOrthogonal(a, b, 1e-6)});
}

fn geomProject() void {
    const a = math.Vec3.init(2, 3, 0);
    const b = math.Vec3.init(1, 0, 0);
    print("proj a onto b  = {f}\n", .{math.geom.proj(a, b)});
    print("perp component = {f}\n", .{math.geom.perp(a, b)});
    print("closest on seg = {f}\n", .{math.geom.closestPointOnLine(math.Vec3.init(0.5, 5, 0), math.Vec3.splat(0), math.Vec3.init(1, 0, 0))});
}

fn castRay() void {
    // The Ray type bundles origin + direction and offers method-form tests.
    const ray = math.Ray3.init(math.Vec3.init(0, 0, 0), math.Vec3.init(1, 0, 0));
    if (ray.sphere(math.Vec3.init(5, 0, 0), 1)) |t| {
        print("ray hit sphere at t={d}, point={f}\n", .{ t, ray.at(t) });
    }
    // Free-function form (Möller–Trumbore triangle test) is also available.
    const hit = math.intersect.rayTriangle(
        math.Vec3.init(0.25, 0.25, 1),
        math.Vec3.init(0, 0, -1),
        math.Vec3.init(0, 0, 0),
        math.Vec3.init(1, 0, 0),
        math.Vec3.init(0, 1, 0),
    );
    if (hit) |h| print("ray hit triangle: t={d} bary=({d},{d})\n", .{ h.t, h.u, h.v });
}

fn boundsVolumes() void {
    const pts = [_]math.Vec3{ math.Vec3.init(1, 2, 3), math.Vec3.init(-1, 0, 5), math.Vec3.init(2, -2, 1) };
    const box = math.Aabb3.fromPoints(&pts);
    print("aabb min/max = {f} / {f}\n", .{ box.min, box.max });
    print("contains 0   = {}\n", .{box.contains(math.Vec3.splat(0))});
    const sphere = box.boundingSphere();
    print("sphere c/r   = {f} / {d}\n", .{ sphere.center, sphere.radius });
}

fn shapePrimitives() void {
    // Plane / Triangle / Segment / Rect.
    const plane = math.Plane3.fromPoints(math.Vec3.splat(0), math.Vec3.init(1, 0, 0), math.Vec3.init(0, 1, 0));
    print("plane dist (0,0,3) = {d}\n", .{plane.distance(math.Vec3.init(0, 0, 3))});

    const tri = math.Triangle3.init(math.Vec3.splat(0), math.Vec3.init(1, 0, 0), math.Vec3.init(0, 1, 0));
    print("triangle area      = {d}, centroid = {f}\n", .{ tri.area(), tri.centroid() });
    print("closest to (-1,-1) = {f}\n", .{tri.closestPoint(math.Vec3.init(-1, -1, 0))});

    const seg = math.Segment3.init(math.Vec3.splat(0), math.Vec3.init(10, 0, 0));
    print("segment closest    = {f}\n", .{seg.closestPoint(math.Vec3.init(3, 5, 0))});

    const r = math.Rect2.fromPosSize(math.Vec2.init(0, 0), math.Vec2.init(4, 2));
    print("rect contains(2,1) = {}\n", .{r.contains(math.Vec2.init(2, 1))});
}

fn boundingVolumes() void {
    // Oriented box, capsule, frustum.
    const obb = math.Obb3.fromAabb(math.Aabb3.init(math.Vec3.splat(-1), math.Vec3.splat(1)));
    print("obb vs obb         = {}\n", .{obb.intersects(math.Obb3.fromAabb(math.Aabb3.init(math.Vec3.splat(0.5), math.Vec3.splat(2))))});

    const cap = math.Capsule3.init(math.Vec3.init(0, 0, 0), math.Vec3.init(0, 4, 0), 1);
    print("capsule contains   = {}\n", .{cap.contains(math.Vec3.init(0.5, 2, 0))});

    const view = math.camera.lookAt(math.Vec3.init(0, 0, 5), math.Vec3.splat(0), math.Vec3.init(0, 1, 0));
    const proj = math.camera.perspective(math.radians(@as(f32, 60)), 1.0, 0.1, 100.0);
    const frustum = math.Frustum3.fromViewProj(proj.mul(view));
    print("frustum sees origin= {}\n", .{frustum.containsPoint(math.Vec3.splat(0))});
}
