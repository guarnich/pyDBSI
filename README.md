Here are the two things you asked for.

-----

## 1\. Updated `examples/README.md`

Here is the complete text for the `examples/README.md` file, now including the section for the Jupyter Notebook.

I also made minor structural improvements, like numbering the methods.

````markdown
# DBSI Fitting Examples

This folder contains methods for running the DBSI model fitting using the `dbsi_toolbox`.

## 1. Installation (Required)

Before running any examples, you must install the toolbox in "editable mode." This links the package to your Python environment, allowing you to import it.

From the **root directory** (the folder containing `setup.py`), run:

```bash
pip install -e .
````

(The `.` refers to the current directory).

-----

## 2\. Method 1: Command Line Interface (CLI)

This is the most robust method for integrating into automated pipelines (e.g., Bash scripts, SLURM).

**Script:** `run_dbsi_cli.py`

### How to Run

Pass the file paths as arguments directly in your terminal.

#### Command Template

```bash
python examples/run_dbsi_cli.py \
    --nii  "<path_to_your_file.nii.gz>" \
    --bval "<path_to_your_file.bval>" \
    --bvec "<path_to_your_file.bvec>" \
    --mask "<path_to_your_mask.nii.gz>" \
    --out  "<directory_for_results>" \
    --prefix "my_output_prefix"
```

-----

## 3\. Method 2: Interactive Jupyter Notebook

This method is ideal for interactive use, debugging, and visualizing the results slice-by-slice.

**Notebook:** `tutorial.ipynb`

### How to Run

1.  Ensure you have Jupyter installed (`pip install jupyterlab`).
2.  Start the Jupyter server from your terminal:
    ```bash
    jupyter-lab
    ```
3.  Open `tutorial.ipynb` in the Jupyter interface.
4.  Edit the file paths in the **second code cell** to point to your data.
5.  Run the cells sequentially from top to bottom.

<!-- end list -->

```

---