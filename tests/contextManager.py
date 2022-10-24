import os
from contextlib import contextmanager

@contextmanager
def set_dir(dir):
    curr_dir = os.getcwd()
    os.chdir(dir)
    print("Directory changed to given")
    yield
    os.chdir(curr_dir)
    print("Directory reverted back to original")