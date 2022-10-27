
0. Clone this repository and check out `fuzzy-cfg-matching-bench`

    ```sh
    git clone https://github.com/just-max/goblint-analyzer analyzer
    cd analyzer
    git switch fuzzy-cfg-matching-bench
    ```

0. Compile Goblint

    ```sh
    make setup
    make
    ```

0. Change directories
    ```sh
    cd fuzzy-matching-cfg-bench
    ```

0. (Optional) Set the local python version with `pyenv`

    ```sh
    apt install libbz2-dev  # or equivalent, needed for pandas
    pyenv install 3.8.10  # or later version
    pyenv local 3.8.10
    ```

0. (Optional) Create a virtual environment with `virtualenv`

    ```sh
    pip install virtualenv
    virtualenv venv
    . venv/bin/activate
    ```

0. Install dependencies

    ```sh
    pip install -r requirements.txt
    ```

0. Run the benchmarking script for a specific project, e.g. `figlet`

    ```sh
    cd projects/figlet
    python ../../scripts/efficiency.py @efficiency.args \
      --analyzer ../../../goblint \
      --batch-size 4 --batch-index 0 \
      --cores 0 1 2 3
    ```

    Adjust the list of CPU cores (`0 1 2 3`) as appropriate (see `efficiency.py --help` and source code).
    Batch size controls the number of analyzed commits per core, batch index controls which batch to run.

    For many cores, try using `seq`, e.g.

    ```sh
    seq 0 63 | xargs python ... --cores
    ```

    to run on cores 0 through 63.

0. The script can be stopped with `SIGINT` (Ctrl+C/`KeyboardInterrupt`). Progress in running analyses is lost, but starting the script **with the same arguments** again will not redo completed or failed runs. Using different arguments will break in unexpected ways; pass `--restart` or delete the output directory to start from the beginning.

0. (Optional) With `jq` and `sponge` installed (e.g. `apt install jq moreutils`), clean up intermediate `json` files for viewing.

    ```sh
    find result_efficiency/ \
      -regextype egrep -regex '.*.json(lines)?' \
      -exec bash -c 'printf "%s\n" "$1"; jq . < "$1" | sponge "$1"' -- {} \;
    ```
