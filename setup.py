from pathlib import Path

import pkg_resources as pkg
from setuptools import find_packages, setup

# Settings
PARENT = Path(__file__).resolve().parent  # root directory
README = (PARENT / "README.md").read_text(encoding="utf-8")

REQUIREMENTS_FILENAME = "requirements.txt"
REQUIREMENTS = [
    f"{x.name}{x.specifier}"
    for x in pkg.parse_requirements((PARENT / REQUIREMENTS_FILENAME).read_text())
]


setup(
    name="ask_paper",
    version="1.1",
    python_requires=">=3.8",
    license="MIT License",
    description="Library to store text resources (especially ArXiv papers) and using llm to find informations within them",
    long_description=README,
    long_description_content_type="text/markdown",
    author="MK",
    packages=find_packages(),  # required
    include_package_data=True,
    install_requires=REQUIREMENTS,
    extras_require={
        "dev": ["pytest", "black"],
    },
)
