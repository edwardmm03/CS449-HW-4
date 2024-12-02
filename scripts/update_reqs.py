from pathlib import Path
from typing import Union
from argparse import ArgumentParser


def print_reqs(reqs: list[tuple[str, str]]) -> str:
    result = ""
    for req in reqs:
        result += f"{req[0]}=={req[1]}"
    return result


def find_package(
    package: str, reqs: list[tuple[str, str]]
) -> Union[tuple[str, str], None]:
    for req in reqs:
        if package == req[0]:
            return req
    return None


def matching_reqs(
    old_reqs: list[tuple[str, str]], new_reqs: list[tuple[str, str]]
) -> list[tuple[str, str]]:
    matches: list[tuple[str, str]] = []
    for package in old_reqs:
        new_package = find_package(package[0], new_reqs)
        # if new_package is None:
        #    new_package = package
        if new_package is not None:
            matches.append(new_package)
    return matches


def to_reqs(lines: list[str]) -> list[tuple[str, str]]:
    reqs: list[tuple[str, str]] = []
    for line in lines:
        package_version: list[str] = line.split("==")
        if len(package_version) == 2:
            reqs.append((package_version[0], package_version[1]))
    return reqs


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--src", "-s", default="")
    arg_parser.add_argument("--dest", "-d", default="../requirements.txt")
    arg_parser.add_argument("--out", "-o", default="../requirements.txt")
    args = arg_parser.parse_args()

    if args.src == "" or args.dest == "" or args.out == "":
        exit()

    with Path(__file__).parent.joinpath(args.src).open("r") as file:
        new_reqs = to_reqs(file.readlines())
    with Path(__file__).parent.joinpath(args.dest).open("r") as file:
        old_reqs = to_reqs(file.readlines())
    with Path(__file__).parent.joinpath(args.out).open("w") as file:
        file.write(print_reqs(matching_reqs(old_reqs, new_reqs)))
