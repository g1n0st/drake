def _nvcc_object(
        name,
        src = None,
        hdrs = [],
        includes = [],
        extra_srcs = [],
        extra_includes = [],
        nvcc_opts = []):
    """Compile a specific `src` using nvcc.

    src: The src (.cu) file to compile.
    hdrs: List of headers.
    includes: List of include paths.
    extra_srcs: List of extra sources, e.g. Eigen, needed to compile `src`.
    extra_includes: List of include paths for the extra sources.
    nvcc_opts: List of additional nvcc options, see:
        https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#nvcc-command-options  # noqa
    """

    # Process the include paths.
    include_args = []
    if hdrs and not includes:
        # Find genrule path to the first header.
        hdr_as_makevar = "$(rootpath {})".format(hdrs[0])

        # Find genrule path to the workspace holding the first header.
        # (Use `dirname` to pare off the header basename and then one more
        # dirname for each component of the package_name.)
        i_as_makevar = hdr_as_makevar
        for _ in range(native.package_name().count("/") + 2):
            i_as_makevar = "$$(dirname {})".format(i_as_makevar)
        include_args.append("-I {}".format(i_as_makevar))
    for i in includes:
        # Find the first hdr that uses this include path.
        if i == ".":
            matching_hdr = hdrs[0]
            relative_to_include = hdrs[0]
        else:
            matching_hdr = [h for h in hdrs if h.startswith(i + "/")][0]
            relative_to_include = matching_hdr[len(i) + 1:]

        # Find genrule path to this matching_hdr.
        hdr_as_makevar = "$(rootpath {})".format(matching_hdr)

        # Find genrule path to this i.
        i_as_makevar = hdr_as_makevar
        for _ in range(relative_to_include.count("/") + 1):
            i_as_makevar = "$$(dirname {})".format(i_as_makevar)
        include_args.append("-I {}".format(i_as_makevar))
    for path in extra_includes:
        include_args.append("-I {}".format(path))

    # Run nvcc.
    native.genrule(
        name = name + "_genrule",
        outs = [name],
        srcs = [src] + hdrs + extra_srcs + [
            "@cuda//:nvcc",
            "@cuda//:usrlocal",
        ],
        cmd = " ".join([
            "$(location @cuda//:nvcc)",
            # For non-cuda code we have -std=c++20 in tools/bazel.rc, but nvcc
            # does not support 20 yet so we'll have to stick with 14 for now.
            # TODO(jeremy.nimmer) Possibly we could use 17 here now on Jammy?
            "-std=c++14",
            # This align with tools/bazel.rc --force_pic.
            "--compiler-options=-fPIC",
            # Don't bother supporting -c dbg mode here.
            "-O2 -DNDEBUG",
            "-use_fast_math",
            # Eigen requires some experimental extensions.
            "--expt-relaxed-constexpr",
            # This is the default nvcc code-gen options.
            "-gencode arch=compute_89,code=[compute_89,sm_89]",
            # The double-precision version of atomicAdd requires CUDA compute
            # capability 6.x but the Tesla M60 on Jenkins only supports 5.2.
            # This will generate and include code compatible with 6.0, which
            # the graphic card will use if it is capable of.
            "-gencode arch=compute_75,code=sm_75",
            # Input => output compile only (don't link).
            "$(location {}) -c -o $@".format(src),
        ] + include_args + nvcc_opts),
        tags = ["manual"],
        visibility = ["//visibility:private"],
    )

def nvcc_library(
        name,
        srcs = [],
        hdrs = [],
        includes = [],
        nvcc_opts = [],
        deps = []):
    """Declares a cc_library-like target.

    Only a few kwargs are supported, compared to the typical cc_library.  If we
    need more then we can add them, but with careful consideration to how they
    map into something nvcc understands.
    """

    # When this library requires Eigen, we'll compile its CUDA files using the
    # unstable development branch of Eigen but its C++ code using our default
    # stable Eigen release.  C++ code should never use the unstable branch.
    if "@eigen" in deps:
        nvcc_extra_srcs = [
            "@eigen_unstable//:Eigen/Core",
            "@eigen_unstable//:hdrs",
        ]
        nvcc_extra_includes = [
            "$$(dirname $$(dirname $(rootpath @eigen_unstable//:Eigen/Core)))",
        ]
    else:
        nvcc_extra_srcs = []
        nvcc_extra_includes = []

    # Compile srcs using nvcc.
    nvcc_objs = []
    for src in srcs:
        if not src.endswith(".cu"):
            fail("Not a CUDA file: " + src)
        obj_name = src + ".o"
        _nvcc_object(
            name = obj_name,
            src = src,
            hdrs = hdrs,
            includes = includes,
            extra_srcs = nvcc_extra_srcs,
            extra_includes = nvcc_extra_includes,
            nvcc_opts = nvcc_opts,
        )
        nvcc_objs.append(obj_name)
    native.cc_library(
        name = name,
        srcs = nvcc_objs,
        hdrs = hdrs,
        deps = deps + ["@cuda//:cudart"],
        linkstatic = 1,
    )
