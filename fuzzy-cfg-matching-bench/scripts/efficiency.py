import functools

from pydriller import Repository, Git, Commit
import utils
from utils import eprint, group, with_file
import psutil
import multiprocessing as mp
import os
import subprocess
import itertools as it
import shutil
import shlex
import json
from datetime import datetime
import sys
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Iterable, Any
import argparse
from argparseutils import SplittingArgumentParser, wrap_type, absolute_path, existing_file, json_from_file
from argparseutils import ExtendAction as Extend
from urllib.parse import urlparse, urlunparse
from dateutil.parser import parse as parse_date

configurations = {}
variations: List[Dict] = []
build_compdb = None
goblint_path = None
diff_exclude = []



################################################################################
# Usage: efficiency.py --help
#
# By default:
# - Executing the script will overwrite the directory 'result_efficiency' in the cwd.
# - The script for building the compilation database is assumed to be at 'build_compdb.sh' in the current directory
#
# 'configurations' should be a mapping of identifier to a (partial) goblint configuration,
# multiple arguments are joined left-to-right
#

# The single test runs are mapped to processors according to the coremapping. The one specified in the section below
# should work for Intel machines, otherwise you might need to adapt it according to the description.


# result_dir    = str((Path.cwd() / 'result_efficiency').absolute())
# maxCLOC       = parsed_args.max_cloc
# url           = urlunparse(parsed_args.url)
# repo_name     = Path(parsed_args.url.path).stem
# build_compdb  = str(parsed_args.build_compdb)
# # conf_base     = "zstd-race-baseline" # very minimal: "zstd-minimal"
# # conf_incrpost = "zstd-race-incrpostsolver"
# begin         = parsed_args.begin
# to            = parsed_args.end
# diff_exclude  = parsed_args.diff_exclude
# analyzer_dir  = str(parsed_args.analyzer_dir.absolute())
# only_collect_results = parsed_args.only_collect_results # can be turned on to collect results, if data collection was aborted before the creation of result tables
# numcores = len(parsed_args.cores)
# coremapping = parsed_args.cores


################################################################################
# Usage: python3 incremental_smallcommits.py <full_path_analyzer_dir> <number_of_cores>
# Executing the script will overwrite the directory 'result_efficiency' in the cwd.
# The script for building the compilation database is assumed to be found in the analyzers script directory and the
# config file is assumed to be found in the conf directory of the analyzers' repository.
# The single test runs are mapped to processors according to the coremapping. The one specified in the section below
# should work for Intel machines, otherwise you might need to adapt it according to the description.
# if len(sys.argv) != 3:
#     print(
#         "Wrong number of parameters.\n"
#         "Use script like this: python3 parallel_benchmarking.py "
#         "<path to goblint directory> <number of processes>"
#     )
#     exit()
# result_dir = Path.cwd() / "result_efficiency"
# maxCLOC = None  # can be deactivated with None
# url           = "https://github.com/cmatsuoka/figlet.git"
# repo_name     = "figlet"
# build_compdb  = Path.cwd() / "build_compdb_figlet.sh"
# # # conf_base     = "figlet-baseline" # very minimal: "zstd-minimal"
# # # conf_incrpost = "figlet-incrpostsolver"
# begin         = datetime(2011,1,12)
# to            = datetime(2011,1,13)
#
# with open("configurations.json", "r") as conf_file:
#     configurations = json.load(conf_file)
#
# with open("variations.json", "r") as var_file:
#     variations = json.load(var_file)


# url = "https://github.com/mlichvar/chrony"
# repo_name = "chrony"
# build_compdb = Path("build_compdb_chrony.sh")
# begin = datetime(2022, 1, 1)
# to = datetime(2022, 3, 31)
# diff_exclude = []  # ["build", "doc", "examples", "tests", "zlibWrapper", "contrib"]
# analyzer_dir = sys.argv[1]
# goblint_path = Path(analyzer_dir) / "goblint"
# only_collect_results = True  # can be turned on to collect results, if data collection was aborted before the creation of result tables
# ################################################################################
# try:
#     num_cores = int(sys.argv[2])
# except ValueError:
#     print(
#         "Parameter should be a number.\nUse script like this: python3 parallel_benchmarking.py <absolute path to goblint directory> <number of processes>"
#     )
#     exit()
# avail_phys_cores = psutil.cpu_count(logical=False)
# allowedcores = avail_phys_cores - 1
# if not only_collect_results and num_cores > allowedcores:
#     print(
#         "Not enough physical cores on this machine (exist: ",
#         avail_phys_cores,
#         " allowed: ",
#         allowedcores,
#         ")",
#     )
#     exit()
# For equal load distribution, choose a processes to core mapping,
# use only physical cores and have an equal number of processes per cache.
# The layout of physical/logical cores and sharing of caches is machine dependent. To find out use: 'lscpu --all --extended'.
# For our test server:
# coremapping1 = [i for i in range(num_cores - num_cores//2)]
# coremapping2 = [i for i in range(avail_phys_cores//2, avail_phys_cores//2 + num_cores//2)]
# coremapping = [coremapping1[i//2] if i%2==0 else coremapping2[i//2] for i in range(len(coremapping1) + len(coremapping2))]
# coremapping = list(range(1, num_cores + 1))
################################################################################


# from Python 3.8
def _shlex_join(split_command):
    """Return a shell-escaped string from *split_command*."""
    return ' '.join(shlex.quote(arg) for arg in split_command)


def filter_commits_false_pred(repo_path, diff_exclude, max_cloc):
    def pred(c: Commit):
        rel_cloc = utils.calculateRelCLOC(str(repo_path), c, diff_exclude)
        return rel_cloc == 0 or (max_cloc is not None and rel_cloc > maxCLOC) or not c.parents

    return pred


def check_variations(variations: Iterable):
    saved = set()
    for i, variation in enumerate(variations):
        if "name" not in variation:
            yield f"missing required property 'name' for variation at index {i}"
            continue

        for prop in (p for p in ("configurations",) if p not in variation):
            yield f"missing required property {prop} for configuration {variation['name']}"
            break

        else:
            if "incremental-load" in variation and variation["incremental-load"] not in saved:
                yield (
                    f"variation {variation['name']}: tries to load "
                    + f"{variation['load']} which has not been saved"
                )

            if variation.get("incremental-save", False):
                saved.add(variation["name"])


def analyze_small_commits_in_repo(cwd: Path, repo_source_dir: Path, commits: List[Commit], process_id: Any):
    task_marker_process = f"[{process_id}]"
    os.chdir(cwd)
    out_dir = cwd / "out"
    count_analyzed = 0
    count_failed = 0
    repo_path = cwd / "repository"
    if not repo_path.is_dir():
        shutil.copytree(repo_source_dir, cwd / "repository")
    repository = Git(repo_path)

    eprint(f"{task_marker_process} process will handle {len(commits)} commit(s)")

    for commit_n, commit in enumerate(commits, start=1):
        failed = []

        out_try = out_dir / commit.hash
        parent = repository.get_commit(commit.parents[0])

        task_marker_commit = f"{process_id}:{commit.hash[:6]}"

        for variation_n, variation in enumerate(variations, start=1):
            task_marker_var = f"{task_marker_commit}:{variation['name']}"
            eprint(f"[{task_marker_var}] variation = {json.dumps(variation)}")

            out_var = out_try / "variation" / variation["name"]

            prev_failed, prev_done = ((out_var / marker).exists() for marker in (".failed", ".done"))
            if prev_failed or prev_done:
                if prev_failed:
                    failed.append(variation)
                eprint(f"[{task_marker_var}] skipped already-analyzed variation, "
                        f"which {'failed' if prev_failed else 'completed successfully'} during previous run")
                continue

            # remove existing results (could happen if something went really wrong)
            try:
                shutil.rmtree(out_var)
            except FileNotFoundError:
                pass
            out_var.mkdir(parents=True)

            commit_var = dict(parent=parent, child=commit)[variation.get("commit", "child")]

            eprint(
                f"[{task_marker_var}] "
                f"on commit {commit_var.hash[:6]} ({commit_n}/{len(commits)}), "
                f"variation {variation['name']} ({variation_n}/{len(variations)})"
            )

            save: Optional[bool] = variation.get("incremental-save", False)
            load: Optional[str] = variation.get("incremental-load", None)

            config_var = [configurations[k] for k in variation.get("configurations", ())]

            extra_config = dict(
                printstats=True,
                result="json",
                incremental=dict(load=bool(load), save=bool(save)),
                dbg={"stats-json-out": str(out_var / "stats.json")}
            )
            if load:
                extra_config["incremental"]["load-dir"] = str(out_try / "variation" / load / "incremental_data")
            if save:
                extra_config["incremental"]["save-dir"] = str(out_var / "incremental_data")

            def cleanup_var():
                try:
                    (out_var / utils.analyzer_log).unlink()
                except FileNotFoundError:
                    pass

            try:
                utils.analyze_commit(
                    repository,
                    goblint_path,
                    build_compdb,
                    commit_var,
                    out_var,
                    config_var + [extra_config]
                )
            except subprocess.CalledProcessError as e:
                eprint(
                    f"[{task_marker_var}] aborted because command '{_shlex_join(e.cmd)}' failed"
                )
                (out_var / ".failed").touch()
                count_failed += 1
                failed.append(variation)

                cleanup_var()
                continue

            stats = with_file(out_var / "stats.json", "r", json.load, default={})
            timing = dict(utils.flatten_timing(stats["timing"], parents=("timing",), drop_prefix=1)) if stats else {}
            solver_stats = stats["solver"] if stats else {}

            with (out_var / "data.json").open("w") as data_file:
                json.dump(
                    dict(
                        hash=commit.hash,
                        variation=variation["name"],
                        **utils.extract_from_analyzer_log(
                            out_var / utils.analyzer_log
                        ),
                        **solver_stats,
                        **timing,
                    ),
                    data_file
                )

            (out_var / ".done").touch()
            count_analyzed += 1
            eprint(f"[{task_marker_var}] completed successfully")

            cleanup_var()


        out_try.mkdir(exist_ok=True)
        with (out_try / "commit.json").open("w") as file:
            json.dump(
                dict(
                    hash=commit.hash,
                    parent_hash=parent.hash,
                    changed_lines_of_code=commit.lines,
                    relevant_changed_lines_of_code=utils.calculateRelCLOC(repository.path, commit, diff_exclude),
                    failed=failed,
                ),
                file
            )

    num_analyses = count_analyzed + count_failed
    eprint(f"{task_marker_process} process done, total={num_analyses} analyzed={count_analyzed} failed={count_failed}")


def run_per_process(core, cwd, repository_source_dir, commits, process_id=None):
    psutil.Process().cpu_affinity([core])
    analyze_small_commits_in_repo(cwd, repository_source_dir, commits, process_id=process_id)


def analyze_make_chunks(url, begin, end, result_dir, make_commit_rejecter, core_mapping):
    repo_name = Path(url.path).stem
    repo_path = result_dir / repo_name

    repo_exists = repo_path.is_dir()

    # calculate actual interesting commits up-front to allow for similar load distribution
    repo = Repository(
        urlunparse(url) if not repo_exists else str(repo_path),
        since=begin,
        to=end,
        only_no_merge=True,
        only_modifications_with_file_types=[".c", ".h"],
        **dict(clone_repo_to=result_dir) if not repo_exists else {}
    )
    reject = make_commit_rejecter(result_dir / repo_name)
    commits = [c for c in repo.traverse_commits() if not reject(c)]

    eprint(f"number of interesting commits: {len(commits)}")
    chunks = list(zip(core_mapping, group(commits, len(core_mapping))))
    tally = dict()
    for _, chunk in chunks:
        k = len(chunk)
        tally[k] = tally.get(k, 0) + 1
    eprint(", ".join(f"{n_cores} core(s) will each handle {n_commits} commit(s)" for n_commits, n_cores in tally.items()))

    return repo_path, chunks


def analyze_chunks_of_commits_in_parallel(repo_path, result_dir, chunks):
    processes = []

    for i, (core, commits_process) in enumerate(chunks):
        process_dir = result_dir / "process" / str(i)
        process_dir.mkdir(exist_ok=True, parents=True)

        args = core, process_dir, repo_path, commits_process
        kwargs = dict(process_id=i)
        p = mp.Process(target=run_per_process, args=args, kwargs=kwargs)
        p.start()
        processes.append(p)
        # run_per_process(*args, **kwargs)

    for p in processes:
        p.join()


def collect_results(result_dir):
    def iterdir_existing(d):
        try:
            yield from d.iterdir()
        except FileNotFoundError:
            yield from ()

    data_paths = (
        variation_dir / "data.json"
        for proc_dir in iterdir_existing(result_dir / "process")
        for commit_dir in iterdir_existing(proc_dir / "out")
        for variation_dir in iterdir_existing(commit_dir / "variation") if variation_dir.is_dir()
    )
    results = []
    for data_path in data_paths:
        if not data_path.exists():
            continue
        with data_path.open("r") as data_file:
            results.append(json.load(data_file))

    pd.json_normalize(results).to_csv(result_dir / "results.csv")


def main():
    parser = SplittingArgumentParser(fromfile_prefix_chars="@")
    parser.add_argument("url", type=urlparse)
    parser.add_argument("begin", type=wrap_type(parse_date))
    parser.add_argument("end", type=wrap_type(parse_date))
    parser.add_argument("--build-compdb", type=existing_file, default="build_compdb.sh",
                        help="path to executable to build compdb, default '%(default)s' in current directory")
    parser.add_argument("--analyzer", type=existing_file, default=shutil.which("goblint"),
                        help="path to goblint, default 'goblint' on path")

    parser.add_argument("--configurations", type=json_from_file, nargs="+", default=[], action=Extend,
                        help="a mapping {config id -> goblint config}; multiple mappings are joined")
    parser.add_argument("--variations", type=json_from_file, nargs="+", default=[], action=Extend,
                        help="list of variations; each variation is run for each commit, "
                             "configurations reference the config IDs defined in the configuration argument; "
                             "see 'check_variations' in source of efficiency.py; "
                             "multiple variation lists are joined")
    parser.add_argument("--restart", action="store_true",
                        help="Remove the existing (partial) results before starting.")

    parser.add_argument("--max-cloc", type=int)
    parser.add_argument("-c", "--cores", type=int, nargs="+", default=[], action=Extend)
    parser.add_argument("--only-collect-results", action="store_true")
    parser.add_argument("--only-show-distribution", action="store_true")
    parser.add_argument("-e", "--diff-exclude", nargs="+", default=[], action=Extend)
    parser.add_argument("-o", "--result-directory", type=absolute_path, default="result_efficiency")

    parsed_args = parser.parse_args()

    if parsed_args.analyzer is None:
        eprint("goblint not found on PATH, use '--analyzer' option to set path")
        return

    global configurations, variations
    configurations = dict(it.chain.from_iterable(c.items() for c in parsed_args.configurations))
    variations = list(it.chain.from_iterable(parsed_args.variations))

    global goblint_path, build_compdb
    goblint_path = parsed_args.analyzer
    build_compdb = parsed_args.build_compdb

    global diff_exclude
    diff_exclude = parsed_args.diff_exclude

    problems = list(check_variations(variations))
    if problems:
        eprint(*problems, sep="\n")
        return

    result_dir = parsed_args.result_directory
    if not parsed_args.only_collect_results:
        if parsed_args.restart and result_dir.exists():
            shutil.rmtree(result_dir)
        result_dir.mkdir(exist_ok=True)
        repo_path, chunks = analyze_make_chunks(
            parsed_args.url,
            parsed_args.begin,
            parsed_args.end,
            result_dir,
            lambda repo_dir: filter_commits_false_pred(repo_dir, parsed_args.diff_exclude, parsed_args.max_cloc),
            parsed_args.cores if parsed_args.cores else [0]
        )
        if parsed_args.only_show_distribution:
            return
        analyze_chunks_of_commits_in_parallel(repo_path, result_dir, chunks)
    collect_results(result_dir)


if __name__ == "__main__":
    main()


# find result_efficiency/ -regextype egrep -regex '.*.json(lines)?' \
# -exec bash -c 'printf "%s\n" "$1"; jq . < "$1" | sponge "$1"' -- {} \;
