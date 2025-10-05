import inspect


def inspect_class(cls):
    """
    Inspect a class and return its type, name, and file.

    Args:
        cls: An instance of a class to inspect.
    Returns:
        A dictionary with keys 'type', 'name', and 'file'.
    """

    type_ = type(cls)
    module = inspect.getmodule(cls.__class__)
    assert module is not None, "Could not find module for class"

    file = module.__file__
    name = module.__name__

    assert isinstance(name, str), "name must be a string"
    assert isinstance(file, str), "file must be a string"

    extracted_name = name.split(".")[-1]
    assert file.endswith(".py"), "file must be a .py"
    extracted_file = file.split("/")[-1][:-3]

    return {"type": type_, "name": extracted_name, "file": extracted_file}
