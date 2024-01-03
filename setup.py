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
        requirements = [req.replace('\n', '') for req in reads]
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    
    return requirements


setup(
name = 'mlprojects',
version = '0.0.1',
author = '',
auther_emails = '',
packages=find_packages(),
install_require=['pandas', 'numpy', 'seaborn']
)