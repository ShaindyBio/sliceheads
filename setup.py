from setuptools import setup, find_packages

setup(
    name="sliceheads",
    version="0.1.0",
    author="Moshe",
    description="A deep learning pipeline for medical 3D volume slice embedding and head classification.",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "h5py",
        "nibabel",
        "pillow",
        "torch",
        "transformers",
        "tqdm",
        "scikit-learn",
        "aeon",
        "pyyaml"
    ],
    python_requires=">=3.8",
)
