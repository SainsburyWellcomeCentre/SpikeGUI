# module imports
import re
import os
import os.path as op
from setuptools import setup, find_packages

# sets the readme file object
curdir = op.dirname(op.realpath(__file__))
readme = open(op.join(curdir, 'README.md')).read()

requirements = [
    "mpldatacursor >= 0.6.2",
    "numpy >= 1.18.4",
    "pandas >= 0.25.3",
    "PeakUtils >= 1.3.2",
    "PyQt5 >= 5.11.3",
    "PyQt5-sip >= 4.19.13",
    "pyqt5-tools >= 5.11.2.1.3",
    "rpy2 >= 3.3.3",
    "scikit-image >= 0.16.2",
    "scikit-learn >= 0.21.2",
    "scikit-posthocs >= 0.6.2",
    "scipy >= 1.5.2",
    "seaborn >= 0.9.0",
    "setuptools >= 39.0.1",
    "shapely <= 1.7.1",
    "tqdm >= 4.28.1",
    "xlsxwriter >= 1.1.2",
    "venn >= 0.1.3",
    "matplotlib-venn >= 0.1.15",
    "python-Levenshtein >= 0.12.0",
    "cached-property >= 1.5.1",
    "configobj >= 5.0.6",
]

setup(
    name='spikeGUI',
    version='1.0',
    description='GUI-based program that analyses the spiking activity from Neuropixel recording experiment',
    long_description=readme,
    install_requires=[
        # ''               # margrie_libs
        # 'https://github.com/stephenlenzi/probez.git',      # probez repository
        # ''               # pyphys
        # ''               # rotation_analysis
        # ''               # vest_phys
        requirements,
    ],
    python_requires=">=3.6",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "spikeGUI = analysis_guis.spike_gui:main",
        ]
    },
    include_package_data=True,
    author="Richard Faville",
    author_email="richard.faville@gmail.com",
    url="https://github.com/RichardFav/spikeGUI",
)
