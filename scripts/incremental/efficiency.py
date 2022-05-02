from pydriller import Repository, Git
import utils
import psutil
import multiprocessing as mp
import os
import subprocess
import itertools
import shutil
import re
import json
from datetime import datetime
import sys
import pandas as pd
import matplotlib
matplotlib.use("pgf")
matplotlib.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "font.family": "serif",
        "font.size": 8,
        "text.usetex": True,
        "pgf.rcfonts": False,
        "axes.unicode_minus": False,
    }
)
import matplotlib.pyplot as plt

################################################################################
# Usage: python3 incremental_smallcommits.py <full_path_analyzer_dir> <repo_url> <repo_name> <name_of_build_script>
#     <name_of_config> <begin> <from_commit_index> <to_commit_index>
# Executing the script will overwrite the directory 'result_efficiency' in the cwd.
# The script for building the compilation database is assumed to be found in the analyzers script directory and the
# config file is assumed to be found in the conf directory of the analyzers repository.
if len(sys.argv) != 3:
      print("Wrong number of parameters.\nUse script like this: python3 parallel_benchmarking.py <path to goblint directory> <number of processes>")
      exit()
result_dir = os.path.join(os.getcwd(), 'result_efficiency')
maxCLOC       = 50
url           = "https://github.com/facebook/zstd"
repo_name     = "zstd"
build_compdb  = "build_compdb_zstd.sh"
conf          = "big-benchmarks1"
begin         = datetime(2021,8,1)
to            = datetime(2022,2,1)
diff_exclude  = ["build", "doc", "examples", "tests", "zlibWrapper", "contrib"]
analyzer_dir  = sys.argv[1]
try:
    numcores = int(sys.argv[2])
except ValueError:
    print("Parameter should be a number.\nUse script like this: python3 parallel_benchmarking.py <path to goblint directory> <number of processes>")
    exit()
################################################################################


def reset_incremental_data(incr_data_dir):
    if os.path.exists(incr_data_dir) and os.path.isdir(incr_data_dir):
        shutil.rmtree(incr_data_dir)

def analyze_small_commits_in_repo(cwd, outdir, from_c, to_c):
    count_analyzed = 0
    count_skipped = 0
    count_failed = 0
    analyzed_commits = {}
    repo_path = os.path.join(cwd, repo_name)

    for commit in itertools.islice(Repository(url, since=begin, to=to, only_no_merge=True, clone_repo_to=cwd).traverse_commits(), from_c, to_c):
        gr = Git(repo_path)

        #print("\n" + commit.hash)
        #print('changed LOC: ', commit.lines)
        #print('merge commit: ', commit.merge)

        # skip merge commits and commits that have less than maxCLOC of relevant code changes
        relCLOC = utils.calculateRelCLOC(repo_path, commit, diff_exclude) # use this to filter commits by actually relevant changes
        #print("relCLOC: ", relCLOC)
        if maxCLOC is not None and relCLOC > maxCLOC:
            #print('Skip this commit: merge commit or too many relevant changed LOC')
            count_skipped+=1
            continue

        # analyze
        try_num = from_c + count_analyzed + count_failed + 1
        outtry = os.path.join(outdir, str(try_num))
        parent = gr.get_commit(commit.parents[0])
        #print('Analyze this commit incrementally. #', try_num)

        reset_incremental_data(os.path.join(cwd, 'incremental_data'))
        failed = True
        try:
            #print('Starting from parent', str(parent.hash), ".")
            outparent = os.path.join(outtry, 'parent')
            os.makedirs(outparent)
            add_options = ['--disable', 'incremental.load', '--enable', 'incremental.save']
            utils.analyze_commit(analyzer_dir, gr, repo_path, build_compdb, parent.hash, outparent, conf, add_options)

            #print('And now analyze', str(commit.hash), 'incrementally.')
            outchild = os.path.join(outtry, 'child')
            os.makedirs(outchild)
            add_options = ['--enable', 'incremental.load', '--disable', 'incremental.save']
            utils.analyze_commit(analyzer_dir, gr, repo_path, build_compdb, commit.hash, outchild, conf, add_options)

            #print('And again incremental, this time reluctantly')
            outchildrel = os.path.join(outtry, 'child-rel')
            os.makedirs(outchildrel)
            add_options = ['--enable', 'incremental.load', '--disable', 'incremental.save', '--enable', 'incremental.reluctant.on']
            utils.analyze_commit(analyzer_dir, gr, repo_path, build_compdb, commit.hash, outchildrel, conf, add_options)

            #print('And again incremental, this time with exhaustive restarting')
            outchildexrest = os.path.join(outtry, 'child-ex-rest')
            os.makedirs(outchildexrest)
            add_options = ['--enable', 'incremental.load', '--disable', 'incremental.save', '--enable', 'incremental.reluctant.on', '--enable', 'incremental.restart.sided.enabled']
            utils.analyze_commit(analyzer_dir, gr, repo_path, build_compdb, parent.hash, outchildexrest, conf, add_options)

            '''
            #print('And again incremental, this time with cfg comparison')
            outchildcfg = os.path.join(outtry, 'child-cfg-comp')
            os.makedirs(outchildcfg)
            add_options = ['--enable', 'incremental.load', '--disable', 'incremental.save', '--set', 'incremental.compare', 'cfg']
            analyze_commit(gr, commit.hash, outchildcfg, add_options)
            '''
            count_analyzed+=1
            failed = False
        except subprocess.CalledProcessError as e:
            print('Aborted because command ', e.cmd, 'failed.')
            count_failed+=1
        os.makedirs(outtry, exist_ok=True)
        with open(os.path.join(outtry,'commit_properties.log'), "w+") as file:
            json.dump({"hash": commit.hash, "parent_hash": parent.hash, "CLOC": commit.lines, "relCLOC": relCLOC, "failed": failed}, file)
        analyzed_commits[try_num]=(str(commit.hash)[:6], relCLOC)

    num_commits = count_analyzed + count_skipped + count_failed
    print("\nCommits traversed in total: ", num_commits)
    print("Analyzed: ", count_analyzed)
    print("Failed: ", count_failed)
    print("Skipped: ", count_skipped)


def extract_from_analyzer_log(log):
    def find_line(pattern):
        file = open(log, "r")
        for line in file.readlines():
            m = re.search(pattern, line)
            if m:
                file.close()
                return m.groupdict()
    runtime_pattern = 'TOTAL[ ]+(?P<runtime>[0-9\.]+) s'
    change_info_pattern = 'change_info = { unchanged = (?P<unchanged>[0-9]*); changed = (?P<changed>[0-9]*); added = (?P<added>[0-9]*); removed = (?P<removed>[0-9]*) }'
    r = find_line(runtime_pattern)
    ch = find_line(change_info_pattern) or {"unchanged": 0, "changed": 0, "added": 0, "removed": 0}
    d = dict(list(r.items()) + list(ch.items()))
    file = open(log, "r")
    num_racewarnings = file.read().count('[Warning][Race]')
    d["race_warnings"] = num_racewarnings
    file.close()
    return d


def collect_data(outdir):
    index = []
    data = {"Failed?": [], "Changed LOC": [], "Relevant changed LOC": [], "Changed/Added/Removed functions": [],
      "Runtime for parent commit (non-incremental)": [], "Runtime for commit (incremental)": [],
      "Runtime for commit (incremental, reluctant)": [], "Runtime for commit (incremental, extensive restarting)": [],
      "Runtime for commit (incremental, cfg comparison)": [], "Change in number of race warnings": []}
    for t in os.listdir(outdir):
        parentlog = os.path.join(outdir, t, 'parent', 'analyzer.log')
        childlog = os.path.join(outdir, t, 'child', 'analyzer.log')
        childrellog = os.path.join(outdir, t, 'child-rel', 'analyzer.log')
        childexreslog = os.path.join(outdir, t, 'child-ex-rest', 'analyzer.log')
        #childcfglog = os.path.join(outdir, t, 'child-cfg-comp', 'analyzer.log')
        commit_prop_log = os.path.join(outdir, t, 'commit_properties.log')
        t = int(t)
        commit_prop = json.load(open(commit_prop_log, "r"))
        data["Changed LOC"].append(commit_prop["CLOC"])
        data["Relevant changed LOC"].append(commit_prop["relCLOC"])
        data["Failed?"].append(commit_prop["failed"])
        index.append(str(t) + ": " + commit_prop["hash"][:7])
        if commit_prop["failed"] == True:
            data["Runtime for parent commit (non-incremental)"].append(0)
            data["Runtime for commit (incremental)"].append(0)
            data["Runtime for commit (incremental, reluctant)"].append(0)
            data["Runtime for commit (incremental, extensive restarting)"].append(0)
            data["Runtime for commit (incremental, cfg comparison)"].append(0)
            data["Changed/Added/Removed functions"].append(0)
            data["Change in number of race warnings"].append(0)
            continue
        parent_info = extract_from_analyzer_log(parentlog)
        child_info = extract_from_analyzer_log(childlog)
        child_rel_info = extract_from_analyzer_log(childrellog)
        child_exres_info = extract_from_analyzer_log(childexreslog)
        #child_cfg_info = extract_from_analyzer_log(childcfglog)
        data["Changed/Added/Removed functions"].append(int(child_info["changed"]) + int(child_info["added"]) + int(child_info["removed"]))
        data["Runtime for parent commit (non-incremental)"].append(float(parent_info["runtime"]))
        data["Runtime for commit (incremental)"].append(float(child_info["runtime"]))
        data["Runtime for commit (incremental, reluctant)"].append(float(child_rel_info["runtime"]))
        data["Runtime for commit (incremental, extensive restarting)"].append(float(child_exres_info["runtime"]))
        data["Runtime for commit (incremental, cfg comparison)"].append(0) #float(child_cfg_info["runtime"]))
        data["Change in number of race warnings"].append(int(parent_info["race_warnings"]) - int(child_info["race_warnings"]))
    return {"index": index, "data": data}

def runperprocess(core, from_c, to_c):
    psutil.Process().cpu_affinity([core])
    cwd  = os.getcwd()
    outdir = os.path.join(cwd, 'out')
    if os.path.exists(outdir) and os.path.isdir(outdir):
      shutil.rmtree(outdir)
    analyze_small_commits_in_repo(cwd, outdir, from_c, to_c)
    data_set = collect_data(outdir)
    df = pd.DataFrame(data_set["data"], index=data_set["index"])
    df.sort_index(inplace=True, key=lambda idx: idx.map(lambda x: int(x.split(":")[0])))
    print(df)
    df.to_csv('results.csv')

def analyze_chunks_of_commits_in_parallel():
    avail_phys_cores = psutil.cpu_count(logical=False)
    allowedcores = avail_phys_cores - 2
    if numcores > allowedcores:
        print("Not enough physical cores on this maching (exist: ", avail_phys_cores, " allowed: ", allowedcores, ")")
        exit()
    # For equal load distribution, choose a processes to core mapping,
    # use only physical cores and have an equal number of processes per cache.
    # The layout of physical/logical cores and sharing of caches is machine dependent. To find out use: 'lscpu --all --extended'.
    # For our test server:
    coremapping1 = [i for i in range(numcores - numcores//2)]
    coremapping2 = [i for i in range(avail_phys_cores//2, avail_phys_cores//2 + numcores//2)]
    coremapping = [coremapping1[i//2] if i%2==0 else coremapping2[i//2] for i in range(len(coremapping1) + len(coremapping2))]
    processes = []

    # calculate number of interesting commits
    num_commits = sum(1 for _ in Repository(url, since=begin, to=to, only_no_merge=True).traverse_commits())
    print("Number of potentially interesting commits:", num_commits)
    perprocess = num_commits // numcores

    for i in range(numcores):
        dir = "process" + str(i)
        os.mkdir(dir)
        os.chdir(dir)
        # run script
        start = perprocess * i
        end = perprocess * (i + 1) if i < numcores - 1 else num_commits
        p = mp.Process(target=runperprocess, args=[coremapping[i], start, end])
        p.start()
        processes.append(p)
        # time.sleep(random.randint(5,60)) # add random delay between process creation to try to reduce interference
        os.chdir(result_dir)

    for p in processes:
        p.join()

def merge_results():
    filename = "results.csv"
    frames = []
    for process_dir in os.listdir("."):
        path = os.path.join(process_dir, filename)
        if os.path.exists(path):
            t = pd.read_csv(path, index_col=0)
            frames.append(t)
    if len(frames) > 0:
        df = pd.concat(frames)
        df.sort_index(inplace=True, key=lambda idx: idx.map(lambda x: int(x.split(":")[0])))
        df.to_csv('total_results.csv')


if os.path.exists(result_dir):
    shutil.rmtree(result_dir)
os.mkdir(result_dir)
os.chdir(result_dir)

analyze_chunks_of_commits_in_parallel()
merge_results()
