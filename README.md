
# pyDBSI: Python DBSI Fitting Toolbox

A Python toolbox for fitting the Diffusion Basis Spectrum Imaging (DBSI) model to diffusion-weighted MRI data.

## 🧠 What is DBSI?

**Diffusion Basis Spectrum Imaging (DBSI)** is an advanced diffusion MRI model designed to overcome the limitations of standard Diffusion Tensor Imaging (DTI).

While DTI struggles in areas with complex pathologies (like inflammation, edema, and axonal injury co-existing) [Wang et al., Quantification of increased cellularity during inflammatory demyelination, 2011; Kéri, Diffusion Basis Restricted Fraction as a Putative..., 2025], DBSI was developed to resolve and quantify these individual components. It achieves this by modeling the diffusion signal as a combination of:

1.  **Anisotropic Tensors:** Representing water diffusion along organized structures like axonal fibers.
2.  **A Spectrum of Isotropic Tensors:** Representing water diffusing freely in different environments [Shirani et al., Diffusion basis spectrum imaging for identifying pathologies in MS subtypes, 2019; Cross and Song, A new imaging modality to non-invasively assess..., 2017].

### Key Metrics

This toolbox provides maps for all DBSI parameters, allowing you to quantify distinct tissue properties:

  * **Fiber Fraction (f\_fiber):** Reflects the apparent density of axonal fibers [Shirani et al., Diffusion basis spectrum imaging for identifying pathologies in MS subtypes, 2019; Vavasour et al., Characterisation of multiple sclerosis neuroinflammation..., 2022].
  * **Axial Diffusivity (D\_axial / AD):** A marker for axonal integrity; a decrease often suggests axonal injury [Lavadi et al., Diffusion basis spectrum imaging detects axonal injury..., 2025; Tu et al., Diffusion Basis Spectrum Imaging detects evolving..., 2012].
  * **Radial Diffusivity (D\_radial / RD):** A marker for myelin integrity; an increase often suggests demyelination [Lavadi et al., Diffusion basis spectrum imaging detects axonal injury..., 2025; Shirani et al., Diffusion basis spectrum imaging for identifying pathologies in MS subtypes, 2019].
  * **Restricted Fraction (f\_restricted):** The key isotropic component, modeling water in highly restricted environments. This metric serves as a putative marker for **cellularity** (e.g., inflammation, gliosis, or tumor cells) [Wang et al., Quantification of increased cellularity during inflammatory demyelination, 2011; Kéri, Diffusion Basis Restricted Fraction as a Putative..., 2025].
  * **Hindered & Water Fractions (f\_hindered, f\_water):** Isotropic components representing water in less dense environments, such as vasogenic **edema** or tissue loss [Tu et al., Diffusion Basis Spectrum Imaging detects evolving..., 2012; Shirani et al., Diffusion basis spectrum imaging for identifying pathologies in MS subtypes, 2019].

This project provides a simple, open-source Python implementation to fit this powerful model to your data.

-----

## 🚀 Installation

This package is designed to be installed in "editable" mode, which allows you to use it as a toolbox while continuing to develop it.

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/guarnich/dbsi-fitting-toolbox.git
    cd dbsi-fitting-toolbox
    ```

2.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Install the Toolbox:**

    ```bash
    pip install -e .
    ```

    (The `.` refers to the current directory). This makes the `dbsi_toolbox` importable from anywhere in your Python environment.

-----

## ⚡ Quickstart: How to Run

This toolbox is run from the command line. You provide the input NIfTI files and the output directory.

**See the `examples/` folder for more detailed guides and a Jupyter Notebook tutorial.**

### Command Line Interface (CLI)

The easiest way to run the model is using the `run_dbsi.py` script.

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

#### Arguments

  * `--nii`: (Required) Path to your 4D diffusion-weighted NIfTI file.
  * `--bval`: (Required) Path to your `.bval` file.
  * `--bvec`: (Required) Path to your `.bvec` file.
  * `--mask`: (Required) Path to your 3D binary brain mask NIfTI file.
  * `--out`: (Required) Path to the folder where all output maps will be saved.
  * `--prefix`: (Optional) A prefix for all output files (e.g., `sub-01_dbsi`). Default is `dbsi_cli`.
  * `--method`: (Optional) The optimization algorithm. Default is `least_squares`.

### Example

```bash
python examples/run_dbsi_cli.py \
    --nii  "subject/dwi/dwi_preproc.nii.gz" \
    --bval "subject/dwi/dwi.bval" \
    --bvec "subject/dwi/dwi.bvec" \
    --mask "subject/dwi/brain_mask.nii.gz" \
    --out  "subject/dbsi_results" \
    --prefix "sub-01_ses-01"
```

This command will fit the DBSI model to every voxel inside the mask and save \~25+ parameter maps (e.g., `sub-01_ses-01_f_fiber.nii.gz`, `sub-01_ses-01_f_restricted.nii.gz`, etc.) in the `subject/dbsi_results` folder.

-----