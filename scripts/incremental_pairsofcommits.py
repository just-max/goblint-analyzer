from pydriller import Repository, Git
import os
from pathlib import Path
import subprocess
import itertools
import shutil
import re
import json
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


def reset_incremental_data():
    incr_data_dir = os.path.join(cwd, 'incremental_data')
    if os.path.exists(incr_data_dir) and os.path.isdir(incr_data_dir):
        shutil.rmtree(incr_data_dir)

def analyze_commit(gr, commit_hash, outdir, extra_options):
    gr.checkout(commit_hash)

    prepare_command = ['sh', os.path.join(analyzer_dir, 'scripts', build_compdb)]
    with open(outdir+'/prepare.log', "w+") as outfile:
        subprocess.run(prepare_command, cwd = repo_path, check=True, stdout=outfile, stderr=subprocess.STDOUT)
        outfile.close()

    analyze_command = [os.path.join(analyzer_dir, 'goblint'), '--conf', os.path.join(analyzer_dir, 'conf', conf + '.json'), *extra_options, repo_path]
    with open(outdir+'/analyzer.log', "w+") as outfile:
        subprocess.run(analyze_command, check=True, stdout=outfile, stderr=subprocess.STDOUT)
        outfile.close()

def calculateRelCLOC(commit):
    relcloc = 0
    for f in commit.modified_files:
        _, extension = os.path.splitext(f.filename)
        if not (extension == ".h" or extension == ".c"):
            continue
        filepath = f.new_path
        if filepath is None:
            filepath = f.old_path
        parents = Path(filepath).parents
        parents = list(map(lambda x: os.path.join(repo_path, x), parents))
        if any(dir in parents for dir in paths_to_exclude):
            continue
        relcloc = relcloc + f.added_lines + f.deleted_lines
    return relcloc

def analyze_small_commits_in_repo():
    global count_analyzed
    global count_skipped
    global count_failed
    global analyzed_commits
    for commit in itertools.islice(Repository(url, since=begin, only_no_merge=True, clone_repo_to=cwd).traverse_commits(), from_c, to_c):
        gr = Git(repo_path)

        #print("\n" + commit.hash)
        #print('changed LOC: ', commit.lines)
        #print('merge commit: ', commit.merge)

        # skip merge commits and commits that have less than maxCLOC of relevant code changes
        relCLOC = calculateRelCLOC(commit) # use this to filter commits by actually relevant changes
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

        reset_incremental_data()
        failed = True
        try:
            #print('Starting from parent', str(parent.hash), ".")
            outparent = os.path.join(outtry, 'parent')
            os.makedirs(outparent)
            add_options = ['--disable', 'incremental.load', '--enable', 'incremental.save']
            analyze_commit(gr, parent.hash, outparent, add_options)

            #print('And now analyze', str(commit.hash), 'incrementally.')
            outchild = os.path.join(outtry, 'child')
            os.makedirs(outchild)
            add_options = ['--enable', 'incremental.load', '--disable', 'incremental.save']
            analyze_commit(gr, commit.hash, outchild, add_options)

            #print('And again incremental, this time reluctantly')
            outchildrel = os.path.join(outtry, 'child-rel')
            os.makedirs(outchildrel)
            add_options = ['--enable', 'incremental.load', '--disable', 'incremental.save', '--enable', 'incremental.reluctant.on']
            analyze_commit(gr, commit.hash, outchildrel, add_options)

            count_analyzed+=1
            failed = False
        except subprocess.CalledProcessError as e:
            print('Aborted because command ', e.cmd, 'failed.')
            count_failed+=1
        os.makedirs(outtry, exist_ok=True)
        with open(os.path.join(outtry,'commit_properties.log'), "w+") as file:
            json.dump({"hash": commit.hash, "parent_hash": parent.hash, "CLOC": commit.lines, "relCLOC": relCLOC, "failed": failed}, file)
        analyzed_commits[try_num]=(str(commit.hash)[:6], relCLOC)


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


def collect_data():
    index = []
    data = {"Failed?": [], "Changed LOC": [], "Relevant changed LOC": [], "Changed/Added/Removed functions": [],
      "Runtime for parent commit (non-incremental)": [], "Runtime for commit (incremental)": [],
      "Runtime for commit (incremental, reluctant)": [], "Change in number of race warnings": []}
    for t in os.listdir(outdir):
        parentlog = os.path.join(outdir, t, 'parent', 'analyzer.log')
        childlog = os.path.join(outdir, t, 'child', 'analyzer.log')
        childrellog = os.path.join(outdir, t, 'child-rel', 'analyzer.log')
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
            data["Changed/Added/Removed functions"].append(0)
            data["Change in number of race warnings"].append(0)
            continue
        parent_info = extract_from_analyzer_log(parentlog)
        child_info = extract_from_analyzer_log(childlog)
        child_rel_info = extract_from_analyzer_log(childrellog)
        data["Changed/Added/Removed functions"].append(int(child_info["changed"]) + int(child_info["added"]) + int(child_info["removed"]))
        data["Runtime for parent commit (non-incremental)"].append(float(parent_info["runtime"]))
        data["Runtime for commit (incremental)"].append(float(child_info["runtime"]))
        data["Runtime for commit (incremental, reluctant)"].append(float(child_rel_info["runtime"]))
        data["Change in number of race warnings"].append(int(parent_info["race_warnings"]) - int(child_info["race_warnings"]))
    return {"index": index, "data": data}

def plot(data_set):
    df = pd.DataFrame(data_set["data"], index=data_set["index"]) # TODO: index=analyzed_commits
    df.sort_index(inplace=True, key=lambda idx: idx.map(lambda x: int(x.split(":")[0])))
    print(df)
    df.to_csv('results.csv')

    df.plot.bar(rot=0, width=0.7, figsize=(25,10))
    plt.xticks(rotation=45, ha='right', rotation_mode='anchor')
    plt.xlabel('Commit')
    plt.tight_layout()
    plt.savefig("figure.pdf")


def main(full_path_analyzer, url_arg, repo_name_arg, build_script, conf_arg, begin_arg, start, end):
  global analyzer_dir
  global url
  global repo_name
  global build_compdb
  global conf
  global begin
  global from_c, to_c
  global analyzed_commits, count_analyzed, count_skipped, count_failed
  global cwd, outdir, repo_path, paths_to_exclude, maxCLOC

  analyzer_dir = full_path_analyzer
  url           = url_arg
  repo_name     = repo_name_arg
  build_compdb  = build_script
  conf          = conf_arg
  begin         = begin_arg
  from_c        = start
  to_c          = end
  maxCLOC       = 50
  dirs_to_exclude  = ["build", "doc", "examples", "tests", "zlibWrapper", "contrib"]

  cwd  = os.getcwd()
  outdir = os.path.join(cwd, 'out')
  repo_path = os.path.normpath(os.path.join(cwd, repo_name))
  paths_to_exclude = list(map(lambda x: os.path.join(repo_path, x), dirs_to_exclude))

  analyzed_commits = {}
  count_analyzed = 0
  count_skipped = 0
  count_failed = 0

  if os.path.exists(outdir) and os.path.isdir(outdir):
    shutil.rmtree(outdir)
  analyze_small_commits_in_repo()
  num_commits = count_analyzed + count_skipped + count_failed
  print("\nCommits traversed in total: ", num_commits)
  print("Analyzed: ", count_analyzed)
  print("Failed: ", count_failed)
  print("Skipped: ", count_skipped)

  data = collect_data()
  plot(data)
