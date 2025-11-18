from setuptools import setup, find_packages

setup(
    name="dbsi_toolbox",
    version="0.2.0", 
    author="Francesco Guarnaccia",
    description="A comprehensive toolbox for Diffusion Basis Spectrum Imaging (Standard & Deep Learning)",
    url="https://github.com/guarnich/pyDBSI",
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'dbsi-fit=scripts.dbsi_cli:main',
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires='>=3.8',
    install_requires=[
        "numpy",
        "scipy",
        "nibabel",
        "dipy",
        "tqdm",
        "pandas",
        "matplotlib",
        "torch",  
    ],
)