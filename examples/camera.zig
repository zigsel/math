//! Transforms & camera — `math.transform` (model/affine) and `math.camera`
//! (view/projection). Defaults target Vulkan: right-handed, depth [0,1], +Y down.
//! Run: `zig build example-camera`

const std = @import("std");
const math = @import("math");
const print = std.debug.print;

pub fn main() void {
    xformModel();
    xform2d();
    cameraMvp();
    cameraProject();
    cameraCull();
    gpuUpload();
}

fn xformModel() void {
    // Builders that take a matrix POST-multiply onto it, so you can chain them
    // in T * R * S order.
    const m = math.transform.translate(
        math.transform.rotate(
            math.transform.scaling(math.Vec3.splat(2)),
            math.radians(@as(f32, 90)),
            math.Vec3.init(0, 0, 1),
        ),
        math.Vec3.init(10, 0, 0),
    );
    print("TRS * origin = {f}\n", .{m.mulVec(math.Vec4.init(0, 0, 0, 1))});
    print("TRS * +X     = {f}\n", .{m.mulVec(math.Vec4.init(1, 0, 0, 0))}); // scaled+rotated
}

fn xform2d() void {
    // 2-D affine transforms operate on Mat3.
    const m = math.transform.rotate2d(
        math.transform.translate2d(math.Mat3.identity(), math.Vec2.init(3, 4)),
        math.radians(@as(f32, 90)),
    );
    print("2D move+rot * (1,0) = {f}\n", .{m.mulVec(math.Vec3.init(1, 0, 1))});
}

fn cameraMvp() void {
    const model = math.Mat4.identity();
    const view = math.camera.lookAt(math.Vec3.init(0, 0, 5), math.Vec3.splat(0), math.Vec3.init(0, 1, 0));
    const proj = math.camera.perspective(math.radians(@as(f32, 60)), 16.0 / 9.0, 0.1, 100.0);
    const mvp = proj.mul(view).mul(model);

    const clip = mvp.mulVec(math.Vec4.init(0, 1, 0, 1)); // a point above the origin
    print("clip = {f}\n", .{clip});
    print("ndc  = {f}  (Vulkan: +Y is DOWN, so y<0)\n", .{clip.swizzle("xyz").scale(1.0 / clip.w)});
    // All clip-space conventions live in one struct. Defaults are Vulkan
    // (right-handed, [0,1] depth, +Y down, reverse-Z). For classic OpenGL set
    // this in your root source file:
    //   pub const math_clip: math.config.Clip = .{ .depth = .neg_one_to_one, .y = .up };
}

fn gpuUpload() void {
    // Lay out a uniform block with std140-correct alignment so it memcpys 1:1
    // to the shader. (Or enable scalar block layout and upload math.* directly.)
    const Uniforms = extern struct {
        view_proj: math.std140.Mat4,
        camera_pos: math.std140.Vec3, // 16-aligned: the next field still lands right
        time: f32,
    };
    const proj = math.camera.perspective(math.radians(@as(f32, 60)), 16.0 / 9.0, 0.1, 100.0);
    const view = math.camera.lookAt(math.Vec3.init(0, 2, 5), math.Vec3.splat(0), math.Vec3.init(0, 1, 0));
    const u: Uniforms = .{
        .view_proj = .from(proj.mul(view)),
        .camera_pos = .from(math.Vec3.init(0, 2, 5)),
        .time = 1.5,
    };
    print("Uniforms size={d}B align={d}B (std140)\n", .{ @sizeOf(Uniforms), @alignOf(Uniforms) });
    print("camera_pos read back = {f}\n", .{u.camera_pos.get()});
}

fn cameraProject() void {
    const model = math.Mat4.identity();
    const proj = math.camera.perspective(math.radians(@as(f32, 45)), 1.0, 0.1, 100.0);
    const viewport = math.Vec4.init(0, 0, 1920, 1080);
    const world = math.Vec3.init(0.3, -0.2, -5);
    const win = math.camera.project(world, model, proj, viewport);
    const back = math.camera.unProject(win, model, proj, viewport);
    print("world->window = {f}\n", .{win});
    print("round-trip ok = {}\n", .{back.approxEql(world, 1e-3)});
}

fn cameraCull() void {
    // Frustum culling against the bounding-volume types.
    const view = math.camera.lookAt(math.Vec3.init(0, 0, 5), math.Vec3.splat(0), math.Vec3.init(0, 1, 0));
    const proj = math.camera.perspective(math.radians(@as(f32, 60)), 1.0, 0.1, 100.0);
    const planes = math.bounds.frustumPlanes(proj.mul(view));
    const near_box = math.Aabb3.init(math.Vec3.splat(-1), math.Vec3.splat(1));
    const far_box = math.Aabb3.init(math.Vec3.splat(50), math.Vec3.splat(52));
    print("near box visible = {}\n", .{math.bounds.aabbInFrustum(planes, near_box)});
    print("far  box visible = {}\n", .{math.bounds.aabbInFrustum(planes, far_box)});
}
