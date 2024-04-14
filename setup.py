from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = '-e .'
def get_requirement(file_path:str)->List[str]:
    '''
    This function will return a list of requirements
    '''
    requirements = []
    with open(file_path) as file_obj:
        reads = file_obj.readlines()
        # below line is used bcz in readlines() it will add '\n' in the list, so we repace it
        requirements = [req.replace('\n', '') for req in reads]
        if HYPEN_E_DOT in requirements:  # removing the HYPE_E_DOT from the list
            requirements.remove(HYPEN_E_DOT)
    
    return requirements


setup(
name = 'mlprojects',
version = '0.0.1',
author = '',
auther_emails = '',
packages=find_packages(), 
# this find_package will check all the folder that are created as package (__init__.py) and install their packages
install_requires=get_requirement('requirements.txt')
)