import inspect
from collections import defaultdict


class naming_system:
    used_name = set()
    name_dict = defaultdict(lambda: 1)

    @classmethod
    def autoname(cls, identifier):
        class_name = type(identifier).__name__.lower()
        name_count = cls.name_dict[class_name]
        while True:
            new_name = f"{class_name}_{name_count}"
            if new_name in cls.used_name:
                cls.name_dict[class_name] += 1
                name_count = cls.name_dict[class_name]
            else:
                break
        return new_name

    @classmethod
    def add_name(cls, name):
        cls.used_name.add(name)


def has_arg(fn, name, accept_all=False):
    signature = inspect.signature(fn)
    parameter = signature.parameters.get(name)
    if parameter is None:
        if accept_all:
            for param in signature.parameters.values():
                if param.kind == inspect.Parameter.VAR_KEYWORD:
                    return True
        return False
    return (parameter.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                               inspect.Parameter.KEYWORD_ONLY))
