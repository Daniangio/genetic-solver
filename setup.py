from setuptools import setup, find_packages
from pathlib import Path

# see https://packaging.python.org/guides/single-sourcing-package-version/
version_dict = {}
with open(Path(__file__).parents[0] / "gensol/_version.py") as fp:
    exec(fp.read(), version_dict)
version = version_dict["__version__"]
del version_dict

setup(
    name="gensol",
    version=version,
    author="Daniele Angioletti",
    description="Genetic Solver",
    python_requires=">=3.8",
    packages=find_packages(include=["gensol", "gensol.*"]),
    install_requires=[
        "ipykernel",
        "numpy",
        "tqdm",
        "pandas"
    ],
    zip_safe=True,
)