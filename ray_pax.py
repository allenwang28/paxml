from typing import Any, Callable, Iterable, List
import tpu_manager
import argparse
import logging
import ray


def pax_setup_commands(address: str) -> List[str]:
    return [
        "gsutil cp gs://pax-on-cloud-tpu-project/wheels/20221103/paxml*.whl .",
        "gsutil cp gs://pax-on-cloud-tpu-project/wheels/20221103/praxis*.whl .",
        "pip3 install ~/paxml*.whl",
        "pip3 install ~/praxis*.whl",
        "pip3 install ray[default]",
        "sudo mkdir /dev/shm",
        "sudo mount -t tmpfs -o size=100g tmpfs /dev/shm",
        f"ray start --address={address} --load-code-from-local --resources='" + '{"tpu": 1}\'',
    ]


class PaxGcpManager(tpu_manager.GcpTpuManager):

    def initialize_vms(self):
        self.run_commands_on_workers(pax_setup_commands(self._ray_head_address))

    def deploy_changes(self):
        self.copy_files_to_workers("paxml/")

    def run(self):
        logging.info("Initializing VMs.")
        self.initialize_vms()
        self.deploy_changes()
        self.nodes = []
        for _ in range(self.get_num_nodes()):
            self.run_pax.remote()

    @ray.remote(resources={"tpu": 1})
    def run_pax(self):
        from absl import app
        from absl import flags
        from paxml import main
        from fiddle import absl_flags
        import jax

        FLAGS = flags.FLAGS
        FLAGS.exp = "tasks.lm.params.lm_cloud.LmCloudSpmd2B"
        FLAGS.job_log_dir = "/tmp/pax_dir"
        # Provide access to --jax_backend_target and --jax_xla_backend flags.
        jax.config.config_with_absl()

        flags.mark_flag_as_required('job_log_dir')
        app.run(main, flags_parser=absl_flags.flags_parser)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tpu_name", type=str, help="The name of the TPU.")
    parser.add_argument("project", type=str, help="The name of the project.")
    parser.add_argument("zone", type=str, help="The name of the zone.")
    args, _ = parser.parse_known_args()

    head_info = ray.init(address="auto")
    address = head_info.address_info["address"]

    manager = PaxGcpManager(
        tpu_name=args.tpu_name,
        zone=args.zone,
        ray_head_address=address)

    manager.run()

    ray.shutdown()


if __name__ == "__main__":
    main()
