load("//tools/workspace:github.bzl", "github_archive")

def poisson_disk_sampling_internal_repository(
        name,
        mirrors = None):
    github_archive(
        name = name,
        repository = "thinks/poisson-disk-sampling",
        commit = "b5d11d6325878c5e120364e673eadcd3df1cb473",
        sha256 = "7ca1ba1d3454d64a884477b723ecb4d0ea5d4138a469b1bdb57e542b4f340499",  # noqa
        build_file = ":package.BUILD.bazel",
        mirrors = mirrors,
    )
