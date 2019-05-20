import os


def folders_starting_with(src_dir, prefix):
    files = [f for f in os.listdir(src_dir) if f.startswith(prefix)]
    files = [os.path.join(src_dir, f) for f in files]
    folders = filter_dirs(files)
    return folders


def filter_dirs(files_list):
    return [f for f in files_list if os.path.isdir(f)]
