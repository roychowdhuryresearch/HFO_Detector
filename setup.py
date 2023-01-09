from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='HFODetector',
    version='0.0.13',
    description='Package for detecting HFOs',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/roychowdhuryresearch/HFO_Detector',
    author='Xin Chen, Hoyoung Chung',
    author_email='xinchen98@g.ucla.edu, taylorchung@ucla.edu',
    license='MIT License',
    python_requires='>=3.6',
    packages=find_packages(include=['HFODetector']),
    install_requires=['mne',
                      'numpy',
                      'scipy',
                      'matplotlib',
                      'pandas',
                      'scikit-image'
                      ],

    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
    ],
)
