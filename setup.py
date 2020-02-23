from setuptools import setup, find_packages

# read the contents of README file
from os import path
from io import open

# get __version__ from _version.py
ver_file = path.join("cen", "version.py")
with open(ver_file) as fp:
    exec(fp.read())

this_directory = path.abspath(path.dirname(__file__))


# read the contents of README.rst
def readme():
    with open(path.join(this_directory, "README.rst"), encoding="utf-8") as fp:
        return fp.read()


# read the contents of requirements.txt
def requirements():
    with open(
        path.join(this_directory, "requirements.txt"), encoding="utf-8"
    ) as fp:
        return fp.read().splitlines()


setup(
    name="cen",
    version=__version__,
    description="Contextual Explanation Networks.",
    long_description=readme(),
    long_description_content_type="text/x-rst",
    author="Maruan Al-Shedivat",
    author_email="maruan@alshedivat.com",
    url="https://github.com/alshedivat/cen",
    keywords=[
        "deep learning",
        "machine learning",
        "explainability",
        "interpretability",
        "tensorflow",
        "keras",
        "python",
    ],
    packages=find_packages(exclude=["tests"]),
    package_data={
        "cen": [
            "configs/*.yaml",
            "configs/**/*.yaml",
            "configs/**/**/*.yaml",
        ],
    },
    install_requires=requirements(),
    setup_requires=["setuptools>=38.6.0"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
)
