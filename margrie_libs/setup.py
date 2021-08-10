from setuptools import setup, find_packages

requirements = ['numpy', 'scipy', 'rpy2', 'matplotlib', 'tqdm', 'pandas']

setup(
    name='margrie_libs',
    version='0.1',
    install_requires=requirements,
    packages=find_packages(exclude=['config', 'docs', 'tests*']),
    url='',
    license='MIT',
    author='Charly Rousseau, Stephen C. Lenzi',
    author_email='',
    description='Shared libraries of the Margrie lab'
)
