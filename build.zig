const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // The library module, importable by consumers as `math`.
    const mod = b.addModule("math", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });

    // `zig build test`
    const lib_tests = b.addTest(.{ .root_module = mod });
    const run_lib_tests = b.addRunArtifact(lib_tests);
    const test_step = b.step("test", "Run all library tests");
    test_step.dependOn(&run_lib_tests.step);

    // `zig build docs` — emit generated API documentation to zig-out/docs
    const docs = b.addInstallDirectory(.{
        .source_dir = lib_tests.getEmittedDocs(),
        .install_dir = .prefix,
        .install_subdir = "docs",
    });
    const docs_step = b.step("docs", "Generate API documentation");
    docs_step.dependOn(&docs.step);

    // `zig build examples` runs them all; `zig build example-<name>` runs one.
    const examples = [_][]const u8{
        "vectors",   "matrices", "rotations", "camera",  "scalar",
        "geometry",  "graphics", "packing",   "animation",
    };
    const examples_step = b.step("examples", "Build & run every example");
    for (examples) |name| {
        const exe = b.addExecutable(.{
            .name = b.fmt("example-{s}", .{name}),
            .root_module = b.createModule(.{
                .root_source_file = b.path(b.fmt("examples/{s}.zig", .{name})),
                .target = target,
                .optimize = optimize,
                .imports = &.{.{ .name = "math", .module = mod }},
            }),
        });
        const run = b.addRunArtifact(exe);
        const one = b.step(b.fmt("example-{s}", .{name}), b.fmt("Run the {s} example", .{name}));
        one.dependOn(&run.step);
        examples_step.dependOn(&run.step);
    }
}
