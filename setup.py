from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="tokdu",
    version="0.1.0",
    packages=find_packages(),
    py_modules=["tokdu"],
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "tokdu=tokdu:main",
        ],
    },
    description="A token counting TUI tool that respects .gitignore and skips binary files",
    author="Halil Tezcan KARABULUT",
    author_email="unitythemaker@gmail.com",
    url="https://github.com/unitythemaker/tokdu",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
