"""
Configuration classes.
Configuration is supposed to be accessed as class attributes and managed by
Config class.
"""

import yaml


class MetaConfig(type):
    """Metaclass used to block modifications to Config attributes if the
    configuration is frozen."""
    def __setattr__(self, name, value):
        if Config._FROZEN and name != '_FROZEN':
            raise Exception('Configuration is frozen.')
        return super(MetaConfig, self).__setattr__(name, value)


class Config(metaclass=MetaConfig):
    """Stores execution configuration."""
    DEFAULT_CONFIG_FN = 'base_config.yml'

    _DEFAULT_LOADED = False
    _FROZEN = False
    _CURRENT_CFG = None

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
        to_str = ""
        if cls._DEFAULT_LOADED:
            to_str += 'Default configuration: \n'
        else:
            to_str += 'Modified configuration: \n'
        to_str += str(cls._CURRENT_CFG)
        return to_str

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
            cls._CURRENT_CFG = config_dict

        cls._build_config_tree(cls, config_dict)

    @staticmethod
    def _build_config_tree(parent, value):
        """Convert a dictionary to class attributes in a tree-fashion. The root
        object is Config class.
        Note: Recursive function.
        Args:
            parent: object or class where attributes are created.
            value: value to be inserted into parent.
        """
        if isinstance(value, dict):  # parent has children
            for child_name, child_value in value.items():
                print(child_name)
                print(child_value)
                print(type(child_value))
                if isinstance(child_value, dict):
                    child_node = Config.ConfigNode()
                    setattr(parent, child_name, child_node)
                    Config._build_config_tree(child_node, child_value)
                else:
                    setattr(parent, child_name, child_value)
