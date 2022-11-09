# need to run `gcloud compute config-ssh` on the VM first
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
        "sudo pip3 install ray[default]",
        "pip3 install ray[default]",
        "pip3 install protobuf==3.15",
        "sudo mkdir -p /dev/shm",
        "sudo mount -t tmpfs -o size=100g tmpfs /dev/shm",
        f"ray start --address={address} --resources='" + '{"tpu": 1}\'',
    ]


@ray.remote(resources={"tpu": 1})
class PaxWorker:
    def __init__(self, tpu_name: str ,worker_index: int):
        self._worker_index = worker_index
        self._tpu_name = tpu_name

    def run(self):
        from absl import app
        from absl import flags
        from absl import logging
        from paxml import main
        from fiddle import absl_flags
        import jax
        import ray
        import os
        os.environ["TPU_MIN_LOG_LEVEL"] = "0"

        if self._worker_index == 0:
            logging.set_verbosity(logging.INFO)
        else:
            logging.set_verbosity(logging.ERROR)

        assert ray.is_initialized()

        flags.DEFINE_string("node-ip-address", None, "hack")
        flags.DEFINE_string("node-manager-port", None, "hack")
        flags.DEFINE_string("object-store-name", None, "hack")
        flags.DEFINE_string("raylet-name", None, "hack")
        flags.DEFINE_string("redis-address", None, "hack")
        flags.DEFINE_string("storage", None, "hack")
        flags.DEFINE_string("temp-dir", None, "hack")
        flags.DEFINE_string("metrics-agent-port", None, "hack")
        flags.DEFINE_string("logging-rotate-bytes", None, "hack")
        flags.DEFINE_string("logging-rotate-backup-count", None, "hack")
        flags.DEFINE_string("gcs-address", None, "hack")
        flags.DEFINE_string("redis-password", None, "hack")
        flags.DEFINE_string("startup-token", None, "hack")
        FLAGS = flags.FLAGS
        print("Flags retrieved.")
        FLAGS.exp = "paxml.tasks.lm.params.lm_cloud.LmCloudSpmd2B"
        FLAGS.job_log_dir = "/tmp/pax_dir"
        # Provide access to --jax_backend_target and --jax_xla_backend flags.
        jax.config.config_with_absl()

        app.run(main.main, flags_parser=absl_flags.flags_parser)

    def __repr__(self):
        return f"{self._tpu_name}-w-{self._worker_index}"


class PaxGcpManager(tpu_manager.GcpTpuManager):

    def initialize_vms(self):
        self.run_commands_on_workers(pax_setup_commands(self._ray_head_address))

    def deploy_changes(self):
        self.copy_files_to_workers("paxml/")

    def run(self):
        logging.info("Initializing VMs.")
        self.deploy_changes()
        self.initialize_vms()
        self._nodes = []

        run_futures = []
        for i in range(self.get_num_nodes()):
            worker = PaxWorker(tpu_name=self._tpu_name, worker_index=i).remote()
            run_futures.append(worker.run.remote())
            self._nodes.append(worker)
        for run_future in run_futures:
            ray.get(run_future)


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("tpu_name", type=str, help="The name of the TPU.")
    parser.add_argument("zone", type=str, help="The name of the zone.")
    args, _ = parser.parse_known_args()

    head_info = ray.init()
    address = head_info.address_info["address"]

    manager = PaxGcpManager(
        tpu_name=args.tpu_name,
        zone=args.zone,
        ray_head_address=address)

    manager.run()

    ray.shutdown()


if __name__ == "__main__":
    main()
