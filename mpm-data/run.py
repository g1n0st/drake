import subprocess

for exact in [False, True]:
    for _small_, small_dt in enumerate([1e-3]):
        for _big_ in range(10):
            big_dt = (_big_ + 1) * 1e-3
            small_name = ["1e-3", "1e-4"][_small_]
            big_name = str(_big_ + 1) + "e-3"

            if exact:
                subprocess.run(
                    f"bazel run mpm_cloth --config omp -- --friction 1.0 --testcase 2 --simulation_time 0.5 --write-files --substep {small_name} --time_step {big_name} --exact_line_search",
                    cwd="/home/changyu/drake/examples/multibody/deformable",
                    shell=True
                )
                subprocess.run(f"rm -rf exact_small_{small_name}_big_{big_name}", cwd="/home/changyu/drake/mpm-data", shell=True)
                subprocess.run(f"mkdir exact_small_{small_name}_big_{big_name}", cwd="/home/changyu/drake/mpm-data", shell=True)
                subprocess.run(f"mv *.json exact_small_{small_name}_big_{big_name}/", cwd="/home/changyu/drake/mpm-data", shell=True)
            else:
                subprocess.run(
                    f"bazel run mpm_cloth --config omp -- --friction 1.0 --testcase 2 --simulation_time 0.5 --write-files --substep {small_name} --time_step {big_name}",
                    cwd="/home/changyu/drake/examples/multibody/deformable",
                    shell=True
                )
                subprocess.run(f"rm -rf small_{small_name}_big_{big_name}", cwd="/home/changyu/drake/mpm-data", shell=True)
                subprocess.run(f"mkdir small_{small_name}_big_{big_name}", cwd="/home/changyu/drake/mpm-data", shell=True)
                subprocess.run(f"mv *.json small_{small_name}_big_{big_name}/", cwd="/home/changyu/drake/mpm-data", shell=True)
