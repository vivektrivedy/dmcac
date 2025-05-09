from setuptools import setup, find_packages
setup(
    name="dmcac",
    version="0.1.0",
    description="Divergence-Minimisation Cross-Attention Retrieval",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=[ln.strip() for ln in open("requirements.txt")],
    python_requires=">=3.9",
)
