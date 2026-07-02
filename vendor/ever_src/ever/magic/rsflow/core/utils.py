import os
import shutil


def _rm_temp(dirs):
    for d in dirs:
        shutil.rmtree(d)


def _is_exists(fps):
    return all([os.path.exists(fp) for fp in fps])


def _error_handler(logging, message, temp_dirs):
    logging.error(message)
    _rm_temp(temp_dirs)
