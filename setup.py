import setuptools

exec(open("mintypython/version.py").read())

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="MintyPython",
    version=__version__,
    author="Kmilo Aparicio",
    author_email="kmilo.aparicio@gmail.com",
    description="Model INTerpretation with pYthon",
    long_description=long_description,
    long_description_content_type="text/markdown",
    # url="https://git.forge.lmig.com/projects/PS/repos/mintypython",
    packages=setuptools.find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "xgboost",
        "shap",
        "bokeh>=3.0.3",
        "pyarrow",
        # "lightgbm>=3.3.2,<4",
        "matplotlib",
        "seaborn",
        "deepdiff",
        # "scorepyon @ git+ssh://git@github.com/lmigtech/scorepyon_bra.git@1.3.2",
        "scorepyon @ git+https://github.com/Trajano81/scoring_pricing.git",
        "ipython",
    ],
    python_requires=">=3.8",
)