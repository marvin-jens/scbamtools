import os


def load(config_file="", load_defaults=True, args={}):
    """
    Tries to load YAML configuration from
        0) builtin default from scbamtools package (if load_defaults=True)
        1) update with yaml loaded from config_file (if provided)
        2) update with kwargs from args (for commandline overrides)
    """
    from yaml import load, Loader
    from argparse import Namespace

    # def recursive_to_namespace(d):
    #     for k, v in d.items():
    #         if type(v) is dict:
    # d[k] = Namespace()
    def recursive_update(dst, src):
        for k, v in src.items():
            if type(v) is dict:
                dst[k] = recursive_update(dst.get(k, {}), v)
            # elif type(v) is list:
            #     dst[k] = dst.get(k, []).extend(v)
            else:
                dst[k] = v
        return dst

    default_path = os.path.join(os.path.dirname(__file__), "default.yaml")
    default = load(open(default_path), Loader=Loader)
    config = default.copy()

    if config_file:
        try:
            user_config = load(open(config_file), Loader=Loader)
            config = recursive_update(config, user_config)
        except FileNotFoundError:
            pass

    if args:
        config = recursive_update(config, args)

    return Namespace(**config)


def save(config, filename):
    from yaml import dump, Dumper

    open(filename, "wt").write(dump(config))
