import os


def store_meta_data(path_to_meta_data, key, value):
    if not os.path.isfile(path_to_meta_data):
        config = ConfigObj()
        config.filename = path_to_meta_data

    config = ConfigObj(path_to_meta_data)
    config[key] = value
    config.write()


def get_meta_data(file_name, key):
    if not os.path.isfile(file_name):
        raise KeyError('requested meta file does not exist')
    config = ConfigObj(file_name)

    if key not in config.keys():
        raise KeyError('the requested key is not present in the meta file')

    return config[key]