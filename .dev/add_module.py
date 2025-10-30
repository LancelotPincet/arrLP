#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 

# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet

"""
Create new module for arrLP
"""



# %% Libraries
from devlp import path, copy_file
from datetime import datetime
import os
import json



# %% Main function
def main() :
    print('\nRunning add_module :')

    # Ask for script name
    already_exists = True
    while already_exists :
        name = input('     New module name ? >>> ')
        module_path = path.parent / f'libsLP/arrLP/src/arrlp/modules/{name}_LP'
        script_path = path.parent / f'libsLP/arrLP/src/arrlp/scripts/{name}_LP'
        if module_path.exists() :
            print('     This module already exists :/')
            crush = input("     Do you want to crush existing module? y/[no] >>> ")
            if str(crush).lower() in ["y", "yes", "true", "1"] :
                crush = input("     Sure? y/[no] >>> ")
                if str(crush).lower() in ["y", "yes", "true", "1"] :
                    already_exists = False
        elif script_path.exists() :
            print('     This module already exists as a script :/')
        else :
            already_exists = False
    description = input('     what will the module do ? >>> ')
    date = datetime.now().strftime("%Y-%m-%d")

    # Print informations
    print(f'     name : {name}')
    print(f'     description : {description}')
    print(f'     date : {date}')
    if not module_path.exists() :
        os.mkdir(module_path)

    # Get infos
    infos = {}
    infos['modulename'] = name
    infos['moduledate'] = date
    infos['moduledescription'] = description
    infos['modulelib'] = arrLP
    infos['modulelowerlib'] = arrlp
    infos['moduleequals'] = "="*len(name)
    infos['modlibeq'] = "="*len("arrLP")
        
    # Write python module
    python_path = path / '_templates/libraries/main_files/module.py'
    newpython_path = module_path / f'{name}.py'
    copy_file(python_path, newpython_path, infos)
    with open(module_path / "__init__.py", "w") as file :
        file.write("")
    print('     Module file written [to complete]')

    # Add test
    python_path = path / '_templates/libraries/main_files/test.py'
    newpython_path = module_path / f'test_{name}.py'
    copy_file(python_path, newpython_path, infos)
    print('     Test file written [to complete]')

    # Add module to json file
    modjson = {}
    modjson['module'] = f"modules/{name}_LP/{name}"
    modjson['object'] = name
    modjson['description'] = description
    modjson['date'] = date
    json_path = path.parent / f'libsLP/arrLP/src/arrlp/modules.json'
    with open(json_path, "r") as file :
        data = json.load(file)
    data[name] = modjson
    with open(json_path, "w") as file :
        json.dump(data, file, indent=4, sort_keys=True)
    print("     Module json updated")

    # Add module documentation
    if name[0] == (name.lower())[0] : # if function
        rst_path = path / '_templates/libraries/documentation/add_function.rst'
    else : # Class
        rst_path = path / '_templates/libraries/documentation/add_class.rst'
    newrst_path = path.parent / f'libsLP/arrLP/docs/source/{name}.rst'
    copy_file(rst_path, newrst_path, infos)
    print('     Documentation rst file added')

    # Add module to documentation list
    allmodules_path = path.parent / f'libsLP/arrLP/docs/source/modules.rst'
    string = allmodules_path.read_text()
    string = string.replace(f"   {name}\n", "")
    with open(allmodules_path, "w") as file :
        file.write(string)
    with open(allmodules_path, "a") as file :
        file.write(f"   {name}\n")
    print('     Module added to documentation list')

    # End
    print('add_module finished!\n')



# %% Main function run
if __name__ == "__main__":
    main()