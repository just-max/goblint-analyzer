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
from typing import Optional, List

################################################################################
# Usage: python3 incremental_smallcommits.py <full_path_analyzer_dir> <number_of_cores>
# Executing the script will overwrite the directory 'result_efficiency' in the cwd.
# The script for building the compilation database is assumed to be found in the analyzers script directory and the
# config file is assumed to be found in the conf directory of the analyzers' repository.
# The single test runs are mapped to processors according to the coremapping. The one specified in the section below
# should work for Intel machines, otherwise you might need to adapt it according to the description.
if len(sys.argv) != 3:
    print(
        "Wrong number of parameters.\n"
        "Use script like this: python3 parallel_benchmarking.py "
        "<path to goblint directory> <number of processes>"
    )
    exit()
result_dir = Path.cwd() / "result_efficiency"
maxCLOC = None  # can be deactivated with None
url           = "https://github.com/cmatsuoka/figlet.git"
repo_name     = "figlet"
build_compdb  = Path.cwd() / "build_compdb_figlet.sh"
# # conf_base     = "figlet-baseline" # very minimal: "zstd-minimal"
# # conf_incrpost = "figlet-incrpostsolver"
begin         = datetime(2011,1,12)
to            = datetime(2011,1,13)

with open("configurations.json", "r") as conf_file:
    configurations = json.load(conf_file)

with open("variations.json", "r") as var_file:
    variations = json.load(var_file)


# url = "https://github.com/mlichvar/chrony"
# repo_name = "chrony"
# build_compdb = Path("build_compdb_chrony.sh")
# begin = datetime(2022, 1, 1)
# to = datetime(2022, 3, 31)
diff_exclude = []  # ["build", "doc", "examples", "tests", "zlibWrapper", "contrib"]
analyzer_dir = sys.argv[1]
goblint_path = Path(analyzer_dir) / "goblint"
only_collect_results = True  # can be turned on to collect results, if data collection was aborted before the creation of result tables
################################################################################
try:
    num_cores = int(sys.argv[2])
except ValueError:
    print(
        "Parameter should be a number.\nUse script like this: python3 parallel_benchmarking.py <absolute path to goblint directory> <number of processes>"
    )
    exit()
avail_phys_cores = psutil.cpu_count(logical=False)
allowedcores = avail_phys_cores - 1
if not only_collect_results and num_cores > allowedcores:
    print(
        "Not enough physical cores on this machine (exist: ",
        avail_phys_cores,
        " allowed: ",
        allowedcores,
        ")",
    )
    exit()
# For equal load distribution, choose a processes to core mapping,
# use only physical cores and have an equal number of processes per cache.
# The layout of physical/logical cores and sharing of caches is machine dependent. To find out use: 'lscpu --all --extended'.
# For our test server:
# coremapping1 = [i for i in range(num_cores - num_cores//2)]
# coremapping2 = [i for i in range(avail_phys_cores//2, avail_phys_cores//2 + num_cores//2)]
# coremapping = [coremapping1[i//2] if i%2==0 else coremapping2[i//2] for i in range(len(coremapping1) + len(coremapping2))]
coremapping = list(range(1, num_cores + 1))
################################################################################


def filter_commits_false_pred(repo_path):
    def pred(c: Commit):
        relCLOC = utils.calculateRelCLOC(str(repo_path), c, diff_exclude)
        return relCLOC == 0 or (maxCLOC is not None and relCLOC > maxCLOC) or not c.parents

    return pred


def check_variations(variations):
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


def analyze_small_commits_in_repo(cwd: Path, repo_source_dir: Path, commits: List[Commit], configurations, variations, process_id):
    task_marker_process = f"[{process_id}]"
    os.chdir(cwd)
    out_dir = cwd / "out"
    count_analyzed = 0
    count_failed = 0
    repo_path = shutil.copytree(repo_source_dir, cwd / "repository")
    repository = Git(repo_path)

    eprint(f"{task_marker_process} process will handle {len(commits)} commit(s)")

    for commit in commits:
        out_try = out_dir / commit.hash
        parent = repository.get_commit(commit.parents[0])

        task_marker_commit = f"[{process_id}:{commit.hash[:6]}]"

        failed = True
        try:
            for variation in variations:
                task_marker_var = f"[{process_id}:{commit.hash[:6]}:{variation['name']}]"
                eprint(f"{task_marker_var} variation = {json.dumps(variation)}")

                commit_var = dict(parent=parent, child=commit)[variation.get("commit", "child")]
                out_var = out_try / "variation" / variation["name"]
                out_var.mkdir(parents=True)

                eprint(f"{task_marker_var} on commit {commit_var.hash[:6]}")

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

                utils.analyze_commit(
                    repository,
                    goblint_path,
                    build_compdb,
                    commit_var,
                    out_var,
                    config_var + [extra_config]
                )

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

            count_analyzed += 1
            failed = False
        except subprocess.CalledProcessError as e:
            eprint(
                f"{task_marker_commit} aborted because command '{shlex.join(e.cmd)}' failed"
            )
            count_failed += 1
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

    num_commits = count_analyzed + count_failed
    eprint(f"{task_marker_process} process done, total={num_commits} analyzed={count_analyzed} failed={count_failed}")


def run_per_process(core, cwd, repository_source_dir, commits, process_id=None):
    psutil.Process().cpu_affinity([core])
    analyze_small_commits_in_repo(cwd, repository_source_dir, commits, configurations, variations, process_id=process_id)


def analyze_chunks_of_commits_in_parallel(result_dir):
    processes = []

    # calculate actual interesting commits up-front to allow for similar load distribution
    repo = Repository(
        url,
        since=begin,
        to=to,
        only_no_merge=True,
        only_modifications_with_file_types=[".c", ".h"],
        clone_repo_to=str(Path.cwd())
    )
    repo_path = Path.cwd() / repo_name
    reject = filter_commits_false_pred(result_dir / repo_name)
    commits = [c for c in repo.traverse_commits() if not reject(c)]

    eprint(f"number of interesting commits: {len(commits)}")

    for i, commits_process in enumerate(group(commits, num_cores)):
        process_dir = result_dir / "process" / str(i)
        process_dir.mkdir(parents=True)

        args = coremapping[i], process_dir, repo_path, commits_process
        kwargs = dict(process_id=i)
        p = mp.Process(target=run_per_process, args=args, kwargs=kwargs)
        p.start()
        processes.append(p)
        # run_per_process(*args, **kwargs)

    for p in processes:
        p.join()


def collect_results(result_dir):
    data_paths = (
        variation_dir / "data.json"
        for proc_dir in (result_dir / "process").iterdir()
        for commit_dir in (proc_dir / "out").iterdir()
        for variation_dir in (commit_dir / "variation").iterdir() if variation_dir.is_dir()
    )
    results = []
    for data_path in data_paths:
        if not data_path.exists():
            continue
        with data_path.open("r") as data_file:
            results.append(json.load(data_file))

    pd.json_normalize(results).to_csv(result_dir / "results.csv")


problems = list(check_variations(variations))
if problems:
    eprint(*problems, sep="\n")
    exit()

if not only_collect_results:
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    os.mkdir(result_dir)
    analyze_chunks_of_commits_in_parallel(result_dir)
collect_results(result_dir)
