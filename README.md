
# pyDBSI: Python DBSI Fitting Toolbox

A Python toolbox for fitting the Diffusion Basis Spectrum Imaging (DBSI) model to diffusion-weighted MRI data.

## 🧠 What is DBSI?

**Diffusion Basis Spectrum Imaging (DBSI)** is an advanced diffusion MRI model designed to overcome the limitations of standard Diffusion Tensor Imaging (DTI).

While DTI struggles in areas with complex pathologies (like inflammation, edema, and axonal injury co-existing) [Wang, Y. et al., Quantification of increased cellularity during inflammatory demyelination, 2011; Kéri, S., Diffusion Basis Restricted Fraction as a Putative Magnetic Resonance Imaging Marker of Neuroinflammation: Histological Evidence, Diagnostic Accuracy, and Translational Potential, 2025], DBSI was developed to resolve and quantify these individual components. It achieves this by modeling the diffusion signal as a combination of:

1.  **Anisotropic Tensors:** Representing water diffusion along organized structures like axonal fibers.
2.  **A Spectrum of Isotropic Tensors:** Representing water diffusing freely in different environments [Shirani, A. et al., Diffusion basis spectrum imaging for identifying pathologies in MS subtypes, 2019; Cross, A.H. and Song, S.K., A new imaging modality to non-invasively assess multiple sclerosis pathology, 2017].

### Key Metrics

This toolbox provides maps for all DBSI parameters, allowing you to quantify distinct tissue properties:

  * **Fiber Fraction (f\_fiber):** Reflects the apparent density of axonal fibers [Shirani, A. et al., Diffusion basis spectrum imaging for identifying pathologies in MS subtypes, 2019; Vavasour, I.M. et al., Characterisation of multiple sclerosis neuroinflammation and neurodegeneration with relaxation and diffusion basis spectrum imaging, 2022].
  * **Axial Diffusivity (D\_axial / AD):** A marker for axonal integrity; a decrease often suggests axonal injury [Lavadi, R.S. et al., Diffusion basis spectrum imaging detects axonal injury in the optic nerve following traumatic brain injury, 2025; Tu, T.W. et al., Diffusion Basis Spectrum Imaging detects evolving axonal injury, demyelination and inflammation in the course of EAE, 2012].
  * **Radial Diffusivity (D\_radial / RD):** A marker for myelin integrity; an increase often suggests demyelination [Lavadi, R.S. et al., Diffusion basis spectrum imaging detects axonal injury in the optic nerve following traumatic brain injury, 2025; Shirani, A. et al., Diffusion basis spectrum imaging for identifying pathologies in MS subtypes, 2019].
  * **Restricted Fraction (f\_restricted):** The key isotropic component, modeling water in highly restricted environments. This metric serves as a putative marker for **cellularity** (e.g., inflammation, gliosis, or tumor cells) [Wang, Y. et al., Quantification of increased cellularity during inflammatory demyelination, 2011; Kéri, S., Diffusion Basis Restricted Fraction as a Putative Magnetic Resonance Imaging Marker of Neuroinflammation, 2025].
  * **Hindered & Water Fractions (f\_hindered, f\_water):** Isotropic components representing water in less dense environments, such as vasogenic **edema** or tissue loss [Tu, T.W. et al., Diffusion Basis Spectrum Imaging detects evolving axonal injury, demyelination and inflammation in the course of EAE, 2012; Shirani, A. et al., Diffusion basis spectrum imaging for identifying pathologies in MS subtypes, 2019].

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
python examples/run_dbsi.py \
    --nii  "<path_to_your_file.nii.gz>" \
    --bval "<path_to_your_file.bval>" \
    --bvec "<path_to_your_file.bvec>" \
    --mask "<path_to_your_mask.nii.gz>" \
    --out  "<directory_for_results>" \
    --prefix "my_output_prefix"
```

### Example

```bash
python examples/run_dbsi.py \
    --nii  "subject/dwi/dwi_preproc.nii.gz" \
    --bval "subject/dwi/dwi.bval" \
    --bvec "subject/dwi/dwi.bvec" \
    --mask "subject/dwi/brain_mask.nii.gz" \
    --out  "subject/dbsi_results" \
    --prefix "sub-01_ses-01"
```

This command will fit the DBSI model to every voxel inside the mask and save \~25+ parameter maps (e.g., `sub-01_ses-01_f_fiber.nii.gz`, `sub-01_ses-01_f_restricted.nii.gz`, etc.) in the `subject/dbsi_results` folder.

-----

## 🛠️ Command-Line Options

You can view all available command-line options by running the script with the `--help` flag.

```bash
python examples/run_dbsi_cli.py --help
```

This will display the following message:

```
usage: run_dbsi_cli.py [-h] --nii NII --bval BVAL --bvec BVEC --mask MASK --out OUT
                       [--prefix PREFIX] [--method {least_squares,differential_evolution}]

Fits the complete DBSI model to DWI data.

Required Arguments:
  --nii NII             Path to the 4D NIfTI file (.nii.gz)
  --bval BVAL           Path to the .bval file
  --bvec BVEC           Path to the .bvec file
  --mask MASK           Path to the 3D brain mask NIfTI file.
  --out OUT             Output directory for the NIfTI maps

Optional Arguments:
  -h, --help            show this help message and exit
  --prefix PREFIX       Prefix for output files (default: dbsi_cli)
  --method {least_squares,differential_evolution}
                        Optimization algorithm (default: least_squares)
```

-----
