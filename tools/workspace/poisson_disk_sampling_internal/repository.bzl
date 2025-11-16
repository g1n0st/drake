load("//tools/workspace:github.bzl", "github_archive")

def poisson_disk_sampling_internal_repository(
        name,
        mirrors = None):
    github_archive(
        name = name,
        repository = "thinks/tph_poisson",
        commit = "b5d11d6325878c5e120364e673eadcd3df1cb473",
        sha256 = "1feb36cb9a10a115dc6bd5b1f0ed5e83574f18321f36f953db09d2ba8c4c1fcf",  # noqa
        build_file = ":package.BUILD.bazel",
        mirrors = mirrors,
    )
