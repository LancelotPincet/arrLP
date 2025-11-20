Installation
============

In this section, we will see how **arrLP** can be installed in your Python projects.
We will first do a quick :ref:`reminder on CLI basics <_cli_reminder>`_ necessary to install comfortably the library.
Then, we will explain how to :ref:`create a fresh project <_create_project>`_ compatible with it using *uv*.
Finally, we will show how to :ref:`contribute to the source code development <_get_source_code>`_ by cloning the repository with *git*.

.. _cli_reminder:

1. CLI reminder
---------------

For installation, you will need to use a **C**ommand **L**ine **I**nterface (CLI) like your system terminal.
If you are using Windows or macOS, you might not be familiar with these, so we will first do a quick reminder for these two OS.
If you are using Linux, I suppose you will not need these and you can skip to the :ref:`project creation part <_create_project>`_.

.. _opening_the_terminal:

Opening the terminal
~~~~~~~~~~~~~~~~~~~~

.. tabs::

   .. tab:: Windows
      First, you will need to open a terminal window: search for **"Windows PowerShell"** in the Start Menu (or press **Win + R**, type **PowerShell**, then press **Enter**)
      A terminal window will appear with something like: ``PS C:\Users\myusername> _``

   .. tab:: macOS
      First, you will need to open a terminal window: open **Applications → Utilities → Terminal** (or press **Cmd + Space**, type **Terminal**, and press **Enter**)
      A terminal window will appear with something like: ``MacBook:~ myusername$ _``

When the cursor (``_``) is blinking, it means you can type a command.  
The path you see on the left of the line is your **current directory**—the folder from which commands will be executed.

Commands generally follow this structure: ``command-name argument --option1 value --option2 value``
Options usually start with ``--option-name``, but one-letter shortcuts like `-y` for `--yes` may exist.
When a command is known by your system, you can type the beginning and press **Tab** to auto-complete it.
To learn more about available commands and their options, refer to online documentation (or ask your favorite AI assistant).

.. warning::

   For real noobs like me who come from Python courses, you might be used to Python editors ("IDE" = **I**ntegrated **D**evelopment **E**nvironment) with a shell. Therefore, you might think the PowerShell command lines accept Python commands, **which is wrong**.
   
   The Python editors shell launch under the hood a Python interpreter. You can identify if you are in a Python interpreter as the Python CLI starts with `>>> _`. To launch a Python interpreter in the terminal and get the same behavior as your Python editors, you need to have Python installed on your machine and write `python` to open the interpreter.

.. _navigation_commands:

Navigation commands
~~~~~~~~~~~~~~~~~~~

In the OS terminal, `cd` (**C**hange **D**irectory) lets you move between folders.

.. tabs::

   .. tab:: Windows
      In Windows, folders are separated by ``\``
      ``cd`` command can be used various ways:
         - change directory using an absolute path: ``cd full\path\to\new\directory``
         - go to the parent folder: ``cd ..``
         - navigate to a folder inside your current directory: ``cd relative\path\inside\current\directory``
         - combine movements, for example to reach a folder inside the grandparent directory:  ``cd ..\..\relative\path\to\folder``
         - go to your user home directory: ``cd ~``

   .. tab:: macOS
      In macOS, folders are separated by ``/`` (like in Linux)
      ``cd`` command can be used various ways:
         - change directory using an absolute path: ``cd full/path/to/new/directory``
         - go to the parent folder: ``cd ..``
         - navigate to a folder inside your current directory: ``cd relative/path/inside/current/directory``
         - combine movements, for example to reach a folder inside the grandparent directory:  ``cd ../../relative/path/to/folder``
         - go to your user home directory: ``cd ~``
         - go back to previous directory (macOS only): ``cd -``

.. _folder_management_commands:

Folder management commands
~~~~~~~~~~~~~~~~~~~~~~~~~~
 
.. tabs::

   .. tab:: Windows
      - list folders and files in current **dir**ectory: ``dir``
      - **m**a**k**e a new **dir**ectory (=folder): ``mkdir MyFolderName``
      - **r**e**m**ove a **dir**ectory: ``rmdir MyFolderName``
      - **del**ete a file: ``del MyFile.txt``


   .. tab:: macOS
      - **l**i**s**t folders and files in current directory: ``ls``
      - **m**a**k**e a new **dir**ectory (=folder): ``mkdir MyFolderName``
      - **r**e**m**ove a directory (**r**ecursive delete): ``rm -r MyFolderName``
      - **r**e**m**ove a file: ``rm MyFile.txt``

.. _create_project:

2. Create project
-----------------

Here, we want to create a new Python project with a dependency on **arrLP**.
To make this kind of project functional, there are several steps to follow :
- :ref:`initialize the project folder <_initialize_project_with_uv>`_:
   1. define a *pyproject.toml* file to write dependencies and other metadata
   2. define a *README.md* file
   3. add dependencies
- :ref:`create a virtual environment <_create_virtual_environment_with_uv>`_:
   1. Solve the dependencies versions
   2. install Python on your machine
   3. create a virtual environment
   4. activate the virtual environment
   5. install dependencies
- :ref:`run the script from a fresh terminal <_run_script_with_uv>`_:
   1. activate the virtual environment
   2. run the script

As these are tedious to achieve manually, I highly recommend using `*uv* <https://docs.astral.sh/uv/getting-started/installation/>`_ from *Astral* for installation and environment management as it is a very fast and all-in-one solution.
However, if you are used to using *Anaconda*, we will also show how to initialize with this distribution at the :ref:`end of this subsection <_create_project_with_anaconda>`_.

.. _install_uv:

Install *uv*
~~~~~~~~~~~~

To check if *uv* is installed, you can try the following command. If it gives you a version, than *uv* is already installed!

.. code-block:: shell

   uv --version

If it fails, you need to install *uv*.

.. tabs::

   .. tab:: Linux
      you can install with *curl*:
      .. code-block:: shell

         curl -LsSf https://astral.sh/uv/install.sh | sh

      If *curl* did not seem to be installed, install it before following the command corresponding to your distro:
      - **Debian** and **Ubuntu**: ``sudo apt update && sudo apt install curl -y``
      - **Fedora**: ``sudo dnf install curl -y``
      - **Arch Linux** and **Manjaro**: ``sudo pacman -S curl``

   .. tab:: Windows
      I recommand using a package manager for the installation. You can use one the following:
      - `**WinGet <https://winstall.app/>`_: ``winget install --id=astral-sh.uv  -e``
      - `**Scoop <https://scoop.sh/>`_: ``scoop install main/uv``
      - `**Chocolatey <https://chocolatey.org/install>`_: ``choco install uv``

      If you want to install *uv* using standalone installers or other methods, please refer to their `installation website <https://docs.astral.sh/uv/getting-started/installation/>`_

   .. tab:: macOS
      I recommand using a package manager for the installation. You can use one the following:
      - `**Homebrew <https://brew.sh/>`_: ``brew install uv``
      - **curl**: Follow the same steps as for the Linux protocole.
      - `**MacPorts <https://www.macports.org/install.php>`_: ``sudo port selfupdate && sudo port install uv``

      If you want to install *uv* using standalone installers or other methods, please refer to their `installation website <https://docs.astral.sh/uv/getting-started/installation/>`_

Now *uv* should be installed on your macOS machine!

.. _initialize_project_with_uv:

Initialize a project with *uv*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let's assume we changed directory with `cd` to the parent folder of our future project.
To initialize a project called `my_project` with *uv*, use command:

.. code-block:: shell

   uv init my_project
   cd my_project

This will create a folder with a *pyproject.toml* file, a *README.md* file and a *main.py* file.

Then, we just need to add the dependencies to uor project, typically dependency to **arrLP**, with the command:

.. code-block:: shell

   uv add arrLP

You can redo this command for all the dependencies you might need by replacing the "arrLP" by the other dependency name (the name used in the command *pip*).
For more informations, go to the [*uv*](https://docs.astral.sh/uv) website.
The project is now initialized!

.. _create_virtual_environment_with_uv:

Create virtual environment with *uv*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With *uv*, you can do all five steps defined for the Python virtual environment definition above at once. First go to the project directory with `cd path_to_my_project`(replace `path_to_my_project`).
*uv* can then:
1. look at the *pyproject.toml* file and solve the Python & dependencies versions that are necessary by storing them in a *uv.lock* file
2. look if the necessary Python distribution is in your machine, if not will download it online
3. look if you have already a virtual environment folder (*.venv*), if not will create it
4. activate the *.venv*
5. look if all dependencies versions are installed, if not will install them from *pip*

*uv* gives you commands if you want to do these individually:
- step 1 only: `uv lock`
- step 2 only: `uv python install`
- step 3 to 4: `uv venv`
- step 1 to 5: `uv sync`
For more informations, go to the [*uv*](https://docs.astral.sh/uv) website.

At the end, to create the virtual environment, you only need to use:

.. code-block:: shell

   uv sync

The project virtual environment *.venv* should now be created!

.. _run_script_with_uv:

Run script with *uv*
~~~~~~~~~~~~~~~~~~~~







3. Run your scripts with uv
---------------------------

Once your project is set, you can run any scripts from system terminal inside the project folder with **arrLP** dependency using a simple command:

.. code-block:: shell

   uv run my_script.py

For more information on script dependencies with uv, please refer to their `script guide <https://docs.astral.sh/uv/guides/scripts/>`_.

Some scripts may be provided with the library :doc:`(see Reference Guide) <scripts>`.
To use them, you can just download them from the `GitHub repository <https://github.com/LancelotPincet/arrLP/tree/main/src/arrlp/scripts>`_.

4. Run your script from an editor
---------------------------------

In practice, python scripts are often launched from an editor. Here are the following protocoles for most commonly used editors.

Visual Studio Code
~~~~~~~~~~~~~~~~~~

The console in **Visual Studio Code** (or any wrapper editor like **Cursor**) is a system terminal, so if you installed *uv* as suggested, you can just use :

.. code-block:: shell

   uv run my_script.py

**Visual Studio Code** also proposes a *Python extension* to give the ability to run scripts via a button (IPython under the hood).
If you use this button, do not forget to activate the *.venv* environment created (it may activate automatically if you run from project path, but this is not very consistent).
To manually activate : ``Ctrl+Shift+P → Python: Select Interpreter`` and select ``./.venv/Scripts/python.exe``

Anaconda tools (Spyder and Jupyter)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you are using *Anaconda*, you probably are coding via *Spyder* and *Jupyter*. By default, these use the *root environment* of Anaconda with lots of packages pre-included (but not *arrLP*!).

The **first option** is to manually launch a kernel with the *.venv* we created before on your *Anaconda Spyder/Jupyter*.

- Spyder: ``Preferences → Python Interpreter`` and select ``./.venv/Scripts/python.exe``
- Jupyter: ``ipykernel`` must be installed in *.venv* (``uv add jupyter ipykernel``), then ``Kernel → Change Kernel``
This first option can be tedious every day.

The **second option** is to install Spyder and Jupyter in the *.venv*.

.. code-block:: shell

   uv add jupyter ipykernel
   uv add spyder

Then you can call these in a terminal to open directly the softwares with the *.venv*
However installing *Spyder* with *uv* (= *pip* installation) can sometimes be difficult with dependencies.

Other editors
~~~~~~~~~~~~~

As other editors have not been tested, please refer to dedicated support to run scripts from virtual environments with these.

5. Get source code
------------------

If you want to use the source code locally to modify the library, you can `git clone` the `GitHub source code <https://github.com/LancelotPincet/arrLP>`_.

First you need to have `git <https://git-scm.com/downloads>`_ installed on your computer. 
Go to the local directory where you want to save the repository (change ``PATH_TO_REPO_PARENT``):

.. code-block:: shell

   cd PATH_TO_REPO_PARENT

Then clone the repository:

.. code-block:: shell

   git clone https://github.com/LancelotPincet/arrLP.git

Now the library source code should be present

If you want to contribute, you can do a pull-request in the GitHub repository.