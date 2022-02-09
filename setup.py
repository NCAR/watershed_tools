from setuptools import setup, find_packages
import pathlib
from pip.req import parse_requirements

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
readme = (here / 'README.md').read_text(encoding='utf-8')

# Arguments marked as "Required" below must be included for upload to PyPI.
# Fields marked as "Optional" may be commented out.

setup(
    name='watershed_tools',  # Required
    version='0.0.0',  # Required
    license='GPLv3', # Optional
    description='SUMMA workflow repository',  # Optional
    long_description=long_description,  # Optional
    long_description_content_type='text/markdown',  # Optional
    url='https://github.com/NCAR/watershed_tools/',  # Optional
    author=('Andy Wood', 'Hongli Liu') , # Optional
    author_email=('andywood@ucar.edu','hongli.liu@usask.ca'),  # Optional
    keywords='hydrology, SUMMA',  # Optional
    install_reqs=parse_requirements('requirements.txt', session='hack'),
    # When your source code is in a subdirectory under the project root, e.g.
    # `src/`, it is necessary to specify the `package_dir` argument.
    package_dir={'': 'utils'},  # Optional
    packages=find_packages(where='utils'),  # Required
    python_requires='>=3.6, <4', # Optional
)
