from setuptools import setup, find_packages
from typing import List

def get_requirements(file_path):
    HYPHEN_E_DOT = '-e .'
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace('/n','') for req in requirements]
    
    if HYPHEN_E_DOT in requirements:
        requirements.remove(HYPHEN_E_DOT)
    
    return requirements

setup(
    name = 'BankMarketing',
    version = '0.0.1',
    author = 'Vipina Manjunatha',
    author_email = 'vipina1394@gmail.com',
    packages= find_packages(),
    install_requires = get_requirements('requirements.txt')
)