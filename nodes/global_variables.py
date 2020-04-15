import git
import os

path = os.path.dirname(os.path.realpath(__file__))
git_repo = git.Repo(path, search_parent_directories=True)
git_root = git_repo.git.rev_parse("--show-toplevel")
