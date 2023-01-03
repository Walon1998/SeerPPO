from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="SeerPPO",
    version="0.0.43",
    url="https://github.com/Walon1998/SeerPPO",
    author="Neville Walo",
    author_email="neville.walo@gmail.com",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    keywords="rl, seer, ppo",
    packages=find_packages(),
    python_requires=">=3.7, <4",
    install_requires=["numpy",
                      "numba",
                      "rlgym",
                      "gym",
                      "torch",
                      "rlbot",
                      "aiohttp",
                      "tqdm",
                      "setuptools", ],
)
