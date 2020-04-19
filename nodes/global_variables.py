import git
import os

"""
Stores variables used by multiple classes and scripts. 
Specifically the absolute path in the system of the nodes folder.

"""

path = os.path.dirname(os.path.realpath(__file__))
git_repo = git.Repo(path, search_parent_directories=True)
git_root = git_repo.git.rev_parse("--show-toplevel")
