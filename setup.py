import sys
import subprocess
from package_settings import NAME, VERSION, PACKAGES, DESCRIPTION
from setuptools import setup

# Calling only at the egg_info step gives us the wanted depth first behavior
if 'egg_info' in sys.argv:
    subprocess.check_call(['pip3', 'install', '--no-warn-conflicts', '--upgrade', '-r', 'requirements.txt'])

setup(
    name=NAME,
    version=VERSION,
    long_description=DESCRIPTION,
    author='Max Bartolo',
    author_email='max@bartolo.ai',
    packages=PACKAGES,
    include_package_data=True,
    package_data={
        '': ['*.*'],
    },
)
