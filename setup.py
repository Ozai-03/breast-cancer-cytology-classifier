from setuptools import setup, find_packages

setup(
    name='breast-cancer-cytology-classifier',
    version='0.1.0',
    description='Utilities for UCI Breast Cancer Wisconsin (Diagnostic) FNA cytology classification',
    packages_dir={'': 'src'},
    packages=['src'],
    install_requires=[
        'pandas',
        'scikit-learn',
        'matplotlib',
        'seaborn',
    ],
    author='Mathew Peguero',
    author_email='mnpeguerobusiness@gmail.com'
)