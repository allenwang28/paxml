from distutils.file_util import copy_file
import functools
import os
import requests
import subprocess

from typing import Any, Callable, List, Mapping

import multiprocessing
from fabric import Connection
import logging


@functools.lru_cache()
def get_bearer() -> str:
    return subprocess.check_output("gcloud auth print-access-token", shell=True).decode("utf-8").strip()


@functools.lru_cache()
def get_project() -> str:
    return subprocess.check_output("gcloud config list --format 'value(core.project)'", shell=True).decode(
        "utf-8").strip()


def get_tpu_metadata(name: str, zone: str) -> Mapping[str, str]:

    headers = {
        'Authorization': f'Bearer {get_bearer()}',
    }

    response = requests.get(
        f'https://tpu.googleapis.com/v2alpha1/projects/{get_project()}/locations/{zone}/nodes/{name}',
        headers=headers)

    return response.json()


def get_ip_addresses(name: str, zone: str) -> List[str]:
    info = get_tpu_metadata(name, zone)
    outputs = []
    for i in info["networkEndpoints"]:
        outputs.append(i["ipAddress"])
    return outputs


def connect(ip_address: str) -> Connection:
    return Connection(
        ip_address,
        connect_kwargs={
            "key_filename": os.path.expanduser('~/.ssh/google_compute_engine'),
        },
    )


class GcpTpuManager:

    def __init__(self, tpu_name: str, zone: str, ray_head_address: str):
        self._tpu_name = tpu_name
        self._zone = zone
        self._ip_addresses = get_ip_addresses(name=self._tpu_name, zone=self._zone)
        self._ray_head_address = ray_head_address
        self._connections = {}
        for ip_address in self._ip_addresses:
            self._connections[ip_address] = connect(ip_address)

    def _run_on_worker(self, ip_address: str, commands: Iterable[str], verbose: bool = True):
        for command in commands:
            logging.info("Running {} on {}".format(command, ip_address))
            if command.startswith('sudo'):
                # Strip 'sudo' from command
                command = command[5:]
                output = self._connections[ip_address].sudo(command)
                if verbose:
                    logging.info(f"{ip_address}: " + output.stdout)
            else:
                output = self._connections[ip_address].run(command)
                if verbose:
                    logging.info(f"{ip_address}: " + output.stdout)

    def _run_per_worker(self, fn: Callable[..., Any]):
        """Runs a callable function for all workers."""
        with multiprocessing.Pool(processes=len(self._ip_addresses)) as p:
            p.map(fn, self._ip_addresses)

    def run_commands_on_workers(self, commands: List[str]):
        """Runs a list of commands for all workers."""
        self._run_per_worker(functools.partial(self._run_on_worker, commands=commands))

    def copy_files_to_workers(self, files: List[str]):
        def copy_file_to_worker(ip_address: str):
            for file in files:
                self._connections[ip_address].put(file)

        self._run_per_worker(copy_file_to_worker)

    def get_num_nodes(self):
        return len(self._ip_addresses)