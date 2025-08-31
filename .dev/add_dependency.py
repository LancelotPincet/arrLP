#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2025-08-28
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet

"""
Adds a dependency to arrLP
"""



# %% Libraries
from devlp import path
import toml
import subprocess



# %% Main function
def main() :
    print('\nRunning add_dependency :')

    todefine = True
    while todefine :
        name = input('     New dependency to add to arrLP ? >>> ')
        ok = input(f'     "{name}" will be added to arrLP ? [y]/n >>> ')
        if ok.lower() in ["", "y", "yes", "true"] :
            todefine = False
    
    proj_path = path.parent / 'libsLP/arrLP/pyproject.toml'
    with open(proj_path, "r") as file :
        data = toml.load(file)
    dependencies = data['project']['dependencies'] + [name.lower()]
    data['project']['dependencies'] = sorted(dependencies)
    with open(proj_path, "w") as file :
        toml.dump(data, file)
    print('     Pyproject file filled')

    subprocess.run(["uv", "venv", "--clear"], cwd=path.parent / 'libsLP/arrLP', check=True, stdout=subprocess.PIPE)
    print('     venv cleared')
    subprocess.run(["uv", "sync", "--all-packages"], cwd=path.parent / 'libsLP/arrLP', check=True, stdout=subprocess.PIPE)
    print('     venv synched')

    # End
    print('add_dependency finished!\n')



# %% Main function run
if __name__ == "__main__":
    main()