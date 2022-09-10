from pydriller import Repository, Git
import utils
from utils import eprint
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
from typing import Optional

################################################################################
# Usage: python3 incremental_smallcommits.py <full_path_analyzer_dir> <number_of_cores>
# Executing the script will overwrite the directory 'result_efficiency' in the cwd.
# The script for building the compilation database is assumed to be found in the analyzers script directory and the
# config file is assumed to be found in the conf directory of the analyzers repository.
# The single test runs are mapped to processors according to the coremapping. The one specified in the section below
# should work for Intel machines, otherwise you might need to adapt it according to the description.
if len(sys.argv) != 3:
      print("Wrong number of parameters.\nUse script like this: python3 parallel_benchmarking.py <path to goblint directory> <number of processes>")
      exit()
result_dir    = os.path.join(os.getcwd(), 'result_efficiency')
maxCLOC       = None # can be deactivated with None
# url           = "https://github.com/cmatsuoka/figlet.git"
# repo_name     = "figlet"
# build_compdb  = "build_compdb_figlet.sh"
# # conf_base     = "figlet-baseline" # very minimal: "zstd-minimal"
# # conf_incrpost = "figlet-incrpostsolver"
# begin         = datetime(2011,1,12)
# to            = datetime(2011,1,13)
url           = "https://github.com/mlichvar/chrony"
repo_name     = "chrony"
build_compdb  = "build_compdb_chrony.sh"
begin         = datetime(2022, 1, 1)
to            = datetime(2022, 3, 31)
diff_exclude  = [] # ["build", "doc", "examples", "tests", "zlibWrapper", "contrib"]
analyzer_dir  = sys.argv[1]
only_collect_results = False # can be turned on to collect results, if data collection was aborted before the creation of result tables
################################################################################
try:
    numcores = int(sys.argv[2])
except ValueError:
    print("Parameter should be a number.\nUse script like this: python3 parallel_benchmarking.py <absolute path to goblint directory> <number of processes>")
    exit()
avail_phys_cores = psutil.cpu_count(logical=False)
allowedcores = avail_phys_cores - 1
if not only_collect_results and numcores > allowedcores:
    print("Not enough physical cores on this machine (exist: ", avail_phys_cores, " allowed: ", allowedcores, ")")
    exit()
# For equal load distribution, choose a processes to core mapping,
# use only physical cores and have an equal number of processes per cache.
# The layout of physical/logical cores and sharing of caches is machine dependent. To find out use: 'lscpu --all --extended'.
# For our test server:
# coremapping1 = [i for i in range(numcores - numcores//2)]
# coremapping2 = [i for i in range(avail_phys_cores//2, avail_phys_cores//2 + numcores//2)]
# coremapping = [coremapping1[i//2] if i%2==0 else coremapping2[i//2] for i in range(len(coremapping1) + len(coremapping2))]
coremapping = list(range(1, numcores + 1))
################################################################################

configurations = [
    {
        "name": "parent",
        "conf": "chrony",
        "commit": "parent",
        "save": True
    },
    {
        "name": "incremental",
        "conf": "chrony",
        "commit": "child",
        "load": "parent"
    },
    {
        "name": "incremental-cfg",
        "conf": "chrony-cfg",
        "commit": "child",
        "load": "parent"
    }
]

def filter_commits_false_pred(repo_path):
    def pred(c):
        relCLOC = utils.calculateRelCLOC(repo_path, c, diff_exclude)
        return relCLOC == 0 or (maxCLOC is not None and relCLOC > maxCLOC)
    return pred

def check_configurations(configurations):
    saved = set()
    for i, configuration in enumerate(configurations):
        for prop in ("name", "commit", "conf"):
            if prop not in configuration:
                return f"missing required property {prop} for configuration at index {i}"
        if configuration["commit"] not in ("parent", "child"):
            return f"configuration {configuration['name']}: commit must be one of 'parent' or 'child'"
        if "load" in configuration and configuration["load"] not in saved:
            return (
                f"configuration {configuration['name']}: tries to load "
                + f"{configuration['load']} which has not been saved")
        if configuration.get("save", False):
            saved.add(configuration["name"])
    return None

def analyze_small_commits_in_repo(cwd: Path, out_dir: Path, from_c, to_c, process_id):
    out_dir = Path(out_dir)
    count_analyzed = 0
    count_skipped = 0
    count_failed = 0
    analyzed_commits = {}
    repo_path = Path.cwd() / repo_name

    for commit in it.islice(
        it.filterfalse(
            filter_commits_false_pred(repo_path),
            Repository(
                url, since=begin, to=to, only_no_merge=True, clone_repo_to=cwd
            ).traverse_commits(),
        ),
        from_c,
        to_c,
    ):
        gr = Git(repo_path)

        #print("\n" + commit.hash)
        #print('changed LOC: ', commit.lines)
        #print('merge commit: ', commit.merge)

        # skip merge commits and commits that have no or less than maxCLOC of relevant code changes
        relCLOC = utils.calculateRelCLOC(repo_path, commit, diff_exclude) # use this to filter commits by actually relevant changes
        #print("relCLOC: ", relCLOC)
        if relCLOC == 0 or (maxCLOC is not None and relCLOC > maxCLOC) or not commit.parents:
            #print('Skip this commit: merge commit or too many relevant changed LOC')
            count_skipped+=1
            continue

        # analyze
        try_num = from_c + count_analyzed + count_failed + 1
        task_marker = f'[{process_id}:{try_num}]'
        out_try = out_dir / str(try_num)
        parent = gr.get_commit(commit.parents[0])

        incremental_data_path = cwd / "incremental_data"
        failed = True
        try:
            completed = set()
            for configuration in configurations:
                task_marker = f"[{process_id}:{try_num}:{configuration['name']}]"
                eprint(f"{task_marker} configuration = {json.dumps(configuration)}")
                utils.reset_incremental_data(incremental_data_path)
                options = ["--enable", "printstats"]

                commit_hash = (
                    dict(
                        parent=parent,
                        child=commit
                    )[configuration["commit"]]).hash

                eprint(f"{task_marker} on commit {commit_hash[:6]}")
                out_cfg = out_try / configuration["name"]
                out_cfg.mkdir(parents=True)
                save: Optional[bool] = configuration.get("save", False)
                load: Optional[str] = configuration.get("load", None)

                if load:
                    load_path = out_try / load / "incremental_data"
                    eprint(f"{task_marker} load incremental data from {load_path}")
                    shutil.copytree(load_path, incremental_data_path)

                options += ["--enable" if load else "--disable", "incremental.load"]
                options += ["--enable" if save else "--disable", "incremental.save"]

                utils.analyze_commit(
                    analyzer_dir,
                    gr,
                    str(repo_path),
                    build_compdb,
                    commit_hash,
                    out_cfg,
                    configuration["conf"],
                    options
                )

                if save:
                    save_path = out_cfg / "incremental_data"
                    eprint(f"{task_marker} save incremental data to {save_path}")
                    shutil.move(incremental_data_path, save_path)

                with (out_cfg / "data.json").open("w") as data_file:
                    json.dump(
                        dict(
                            hash=commit.hash,
                            parent_hash=parent.hash,
                            configuration=configuration["name"],
                            CLOC=commit.lines,
                            relCLOC=relCLOC,
                            **utils.extract_from_analyzer_log(out_cfg / utils.analyzerlog)),
                        data_file)

            count_analyzed += 1
            failed = False
        except subprocess.CalledProcessError as e:
            eprint(f"{task_marker} aborted because command '{shlex.join(e.cmd)}' failed")
            count_failed += 1
        out_try.mkdir(exist_ok=True)
        with (out_try / "commit_properties.log").open("w+") as file:
            json.dump({"hash": commit.hash, "parent_hash": parent.hash, "CLOC": commit.lines, "relCLOC": relCLOC, "failed": failed}, file)
        analyzed_commits[try_num] = (str(commit.hash)[:6], relCLOC)

    num_commits = count_analyzed + count_skipped + count_failed
    print("\nCommits traversed in total: ", num_commits)
    print("Analyzed: ", count_analyzed)
    print("Failed: ", count_failed)
    print("Skipped: ", count_skipped)

def collect_data(outdir):
    data = {"Commit": [], "Failed?": [], "Changed LOC": [], "Relevant changed LOC": [], "Changed/Added/Removed functions": [],
      utils.header_runtime_parent: [], utils.header_runtime_incr_child: [],
      utils.header_runtime_incr_posts_child: [], utils.header_runtime_incr_posts_rel_child: [],
      "Change in number of race warnings": []}
    for t in os.listdir(outdir):
        parentlog = os.path.join(outdir, t, 'parent', utils.analyzerlog)
        childlog = os.path.join(outdir, t, 'child', utils.analyzerlog)
        childpostslog = os.path.join(outdir, t, 'child-incr-post', utils.analyzerlog)
        childpostsrellog = os.path.join(outdir, t, 'child-rel', utils.analyzerlog)
        commit_prop_log = os.path.join(outdir, t, 'commit_properties.log')
        t = int(t)
        commit_prop = json.load(open(commit_prop_log, "r"))
        data["Changed LOC"].append(commit_prop["CLOC"])
        data["Relevant changed LOC"].append(commit_prop["relCLOC"])
        data["Failed?"].append(commit_prop["failed"])
        data["Commit"].append(commit_prop["hash"][:7])
        if commit_prop["failed"] == True:
            data[utils.header_runtime_parent].append(0)
            data[utils.header_runtime_incr_child].append(0)
            data[utils.header_runtime_incr_posts_child].append(0)
            data[utils.header_runtime_incr_posts_rel_child].append(0)
            data["Changed/Added/Removed functions"].append(0)
            data["Change in number of race warnings"].append(0)
            continue
        parent_info = utils.extract_from_analyzer_log(parentlog)
        child_info = utils.extract_from_analyzer_log(childlog)
        child_posts_info = utils.extract_from_analyzer_log(childpostslog)
        child_posts_rel_info = utils.extract_from_analyzer_log(childpostsrellog)
        data["Changed/Added/Removed functions"].append(int(child_info["changed"]) + int(child_info["added"]) + int(child_info["removed"]))
        data[utils.header_runtime_parent].append(float(parent_info["runtime"]))
        data[utils.header_runtime_incr_child].append(float(child_info["runtime"]))
        data[utils.header_runtime_incr_posts_child].append(float(child_posts_info["runtime"]))
        data[utils.header_runtime_incr_posts_rel_child].append(float(child_posts_rel_info["runtime"]))
        data["Change in number of race warnings"].append(int(child_info["race_warnings"] - int(parent_info["race_warnings"])))
    return data

def collect_data(out_dir, from_c, to_c):
    yield from ()
    for i in range(from_c, to_c + 1):
        try:
            with (out_dir / str(i) / "commit_properties.log").open("r") as cpf:
                commit_props = json.load(cpf)
            for configuration in configurations:
                    from_log = utils.extract_from_analyzer_log(out_dir / str(i) / configuration["name"] / utils.analyzerlog)
                    from_log.update(commit_props)
                    yield from_log
        except FileNotFoundError as e:
            eprint(f"could not extract log: {e}")

def runperprocess(core, from_c, to_c, process_id=None):
    if not only_collect_results:
        psutil.Process().cpu_affinity([core])
    cwd  = Path.cwd()
    out_dir = cwd / "out"
    if not only_collect_results:
        if out_dir.exists() and out_dir.is_dir():
            shutil.rmtree(outdir)
        analyze_small_commits_in_repo(cwd, out_dir, from_c, to_c, process_id=process_id)
    # return
    data = collect_data(out_dir, from_c, to_c)
    df = pd.DataFrame.from_records(data)
    # df = pd.DataFrame(data_set)
    #df.sort_index(inplace=True, key=lambda idx: idx.map(lambda x: int(x.split(":")[0])))
    print(df)
    df.to_csv('results.csv', sep =';')

def analyze_chunks_of_commits_in_parallel():
    processes = []

    # calculate actual number of interesting commits up-front to allow for similar load distribution
    iter = it.filterfalse(filter_commits_false_pred(os.path.join(os.getcwd(), repo_name)), Repository(url, since=begin, to=to, only_no_merge=True, clone_repo_to=os.getcwd()).traverse_commits())
    num_commits = sum(1 for _ in iter)
    print("Number of potentially interesting commits:", num_commits)
    perprocess = num_commits // numcores if num_commits % numcores == 0 else num_commits // numcores + 1
    print("Per process: " + str(perprocess))

    for i in range(numcores):
        dir = "process" + str(i)
        if not only_collect_results:
            os.mkdir(dir)
        os.chdir(dir)
        # run script
        start = perprocess * i
        end = perprocess * (i + 1) if i < numcores - 1 else num_commits
        if not only_collect_results:
            p = mp.Process(target=runperprocess, args=[coremapping[i], start, end], kwargs=dict(process_id=i))
            # runperprocess(coremapping[i], start, end, process_id=i)
            p.start()
            processes.append(p)
            # time.sleep(random.randint(5,60)) # add random delay between process creation to try to reduce interference
        else:
            runperprocess(coremapping[i], start, end)
        os.chdir(result_dir)

    for p in processes:
        p.join()

def merge_results():
    filename = "results.csv"
    frames = []
    for process_dir in os.listdir("."):
        path = os.path.join(process_dir, filename)
        if os.path.exists(path):
            t = pd.read_csv(path, index_col=0, sep=";")
            frames.append(t)
    if len(frames) > 0:
        df = pd.concat(frames)
        #df.sort_index(inplace=True, key=lambda idx: idx.map(lambda x: int(x.split(":")[0])))
        df.to_csv('total_results.csv', sep=";")


if not only_collect_results:
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    os.mkdir(result_dir)

os.chdir(result_dir)

problem = check_configurations(configurations)
if problem is not None:
    eprint(problem)
    exit()

analyze_chunks_of_commits_in_parallel()
merge_results()
