"""
Configuration classes.
Configuration is supposed to be accessed as class attributes and managed by
Config class.

Licensed under The MIT License
Written by Jean Da Rolt
"""

import yaml


class MetaConfig(type):
    """Metaclass used to block modifications to Config attributes if the
    configuration is frozen."""
    def __setattr__(cls, name, value):
        if Config._FROZEN and name != '_FROZEN':
            raise Exception('Configuration is frozen.')
        return super(MetaConfig, cls).__setattr__(name, value)


class Config(metaclass=MetaConfig):
    """Stores execution configuration.
    Usage:
        Initialization:
            Config.load_default(config_fn)
            Config.merge(other_fn)
            Config.freeze()
        Get:
            Config.ATTRIBUTE1.ATTRIBUTE1_1.ATTRIBUTE1_1_1
        """
    DEFAULT_CONFIG_FN = './mrcnn/config/base_config.yml'
    _FROZEN = False

    _DEFAULT_LOADED = False

    class ConfigNode():  # !pylint: disable=R0903
        """This class is used as an empty object where each instance
        will store different attributes according to the configuration
        tree.
        """
        pass

    def __new__(cls, *args, **kwargs):  # !pylint: disable=W0613
        """This class should not be instantiated."""
        raise Exception('Config class is not meant to be instantiated.')

    @classmethod
    def load_default(cls, config_fn=DEFAULT_CONFIG_FN):
        """Load configurations in config_fn as class attributes of
        this class.

        Args:
            config_fn: PATH to YAML file containing configuration.
        """
        cls._load(config_fn)
        cls._DEFAULT_LOADED = True

    @classmethod
    def merge(cls, config_fn):
        """Merge configuration present at config_fn into this class.

        Args:
            config_fn: YAML file containing configuration.
        """
        if not Config._DEFAULT_LOADED:
            raise Exception('Default configuration should be loaded '
                            'before loading actual configurations')
        cls._load(config_fn)

    @classmethod
    def to_string(cls):
        """Recursively convert to string."""
        return str(_to_dict())

    @staticmethod
    def display():
        """Displays configurations."""
        print(yaml.dump(_to_dict()))

    @classmethod
    def freeze(cls):
        """Blocks the configuration so it cannot be modified. Used to prevent
        changes in the configuration during execution."""
        cls._FROZEN = True

    @classmethod
    def unfreeze(cls):
        """Unblocks the configuration to be changed."""
        cls._FROZEN = False

    @classmethod
    def _load(cls, config_fn):
        """Load configurations in config_fn as class attributes of
        this class.

        Args:
            config_fn: YAML file containing configuration.
        """

        with open(config_fn) as stream:
            config_dict = yaml.safe_load(stream)

        cls._build_config_tree(cls, config_dict)

    @classmethod
    def _build_config_tree(cls, parent, value):
        """Convert a dictionary to class attributes in a tree-fashion. The root
        object is Config class.
        Note: Recursive function.
        Args:
            parent: object or class where attributes are created.
            value: value to be inserted into parent.
        """
        if isinstance(value, dict):  # parent has children
            for child_name, child_value in value.items():
                if isinstance(child_value, dict):
                    child_node = Config.ConfigNode()
                    if not hasattr(parent, child_name):
                        if not cls._DEFAULT_LOADED:
                            setattr(parent, child_name, child_node)
                        else:
                            raise Exception(f"Attribute {child_name} not "
                                            f"defined in base_config.yml")
                    else:
                        child_node = getattr(parent, child_name)
                    Config._build_config_tree(child_node, child_value)
                else:
                    if not cls._DEFAULT_LOADED or (cls._DEFAULT_LOADED) and\
                       hasattr(parent, child_name):
                        setattr(parent, child_name, child_value)
                    else:
                        raise Exception(f"Attribute {child_name} not "
                                        f"defined in base_config.yml")

    @staticmethod
    def dump(filename):
        with open(filename, 'w') as output_file:
            yaml.dump(_to_dict(), output_file)


def _to_dict(dict_node={}, node=Config):
    if type(node).__name__ not in ['MetaConfig', 'ConfigNode']:
        if isinstance(node, (dict, list)):
            return node
        return str(node).rstrip()

    for child_name, child in node.__dict__.items():
        if child_name == 'ConfigNode':
            continue
        if child_name.startswith('_'):
            continue
        if isinstance(node.__dict__[child_name], classmethod):
            continue
        if isinstance(node.__dict__[child_name], staticmethod):
            continue
        dict_node[child_name] = _to_dict({}, child)

    return dict_node
