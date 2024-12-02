"""makes a virtual environment venv in the root directory of the repo
and installs all dependencies in the requirements.txt file at the root
directory of the repo"""

import subprocess
import sys
from pathlib import Path
from platform import system
from time import sleep


PYTHON_VERSION = "3.10.12"


def make_venv() -> None:
    """builds a virtual environment in the root directory of the repo named
    venv"""
    subprocess.run(
        [
            "python",
            "-m",
            "venv",
            Path(__file__).parent.parent.joinpath("venv").absolute(),
        ]
    )


def process_reqs(stdout: bytes) -> list[str]:
    """takes the byte dump from running pip freeze in windows command prompt
    and returns a list of all installed packages"""
    raw_reqs: list[str] = stdout.decode().split("\r\n")
    reqs: list[str] = []
    for raw_req in raw_reqs:
        if raw_req:
            reqs.append(raw_req.split("==")[0])
    return reqs


def clean_venv() -> None:
    """cleans the virtual environment in the root directory of the repo named
    venv"""
    # get all installed packages
    process = subprocess.Popen(
        [
            Path(__file__)
            .parent.parent.joinpath("venv/Scripts/python")
            .absolute(),
            "-m",
            "pip",
            "freeze",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout, stderr = process.communicate()
    # remove installed packages if any are present
    if stdout == b"":
        return
    subprocess.run(
        [
            Path(__file__)
            .parent.parent.joinpath("venv/Scripts/python")
            .absolute(),
            "-m",
            "pip",
            "uninstall",
            "-y",
        ]
        + process_reqs(stdout)
    )


def install_packages() -> None:
    """uses pip to install all of the dependencies specified in the
    requirements.txt of the root directory in the virtual environment venv"""
    subprocess.run(
        [
            Path(__file__)
            .parent.parent.joinpath("venv/Scripts/python")
            .absolute(),
            "-m",
            "pip",
            "--timeout",
            "1000",
            "install",
            "-r",
            Path(__file__)
            .parent.parent.joinpath("requirements.txt")
            .absolute(),
        ]
    )


def update_pip() -> None:
    """updates pip in the virtual environment in the root directory of the
    repo named venv"""
    subprocess.run(
        [
            Path(__file__)
            .parent.parent.joinpath("venv/Scripts/python")
            .absolute(),
            "-m",
            "pip",
            "install",
            "--upgrade",
            "pip",
        ]
    )


def correct_version() -> bool:
    """checks installed version of python vs PYTHON_VERSION"""
    version = PYTHON_VERSION.split(".")
    wrong_version = False
    for i in range(len(version)):
        wrong_version = wrong_version or sys.version_info[i] != int(version[i])
    return not wrong_version


if __name__ == "__main__":
    if system() != "Windows":
        print("[ERROR] this script is only meant for Windows\nHalting")
        exit()
    if not correct_version():
        print("[WARNING] python 3.10.12 is recommended\nWill continue in:")
        for i in range(3, 0, -1):
            print(i)
            sleep(1)

    if (
        Path(__file__)
        .parent.parent.joinpath("venv/Scripts/python.exe")
        .exists()
    ):
        clean_venv()
    else:
        make_venv()
    update_pip()
    install_packages()
