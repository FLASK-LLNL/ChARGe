import inspect


def inspect_class(cls):

    type_ = type(cls)
    module = inspect.getmodule(cls.__class__)
    file = module.__file__ if module else "Unknown"
    name = module.__name__ if module else "Unknown"

    print(f"\nType of the class: {type_}")
    print(f"Name of the class: {name}")
    print(f"File path of the class: {file}")
    extracted_name = name.split(".")[-1]
    print(f"Extracted class name: {extracted_name}")
    extracted_file = file.split("/")[-1][:-3] if file != "Unknown" else "Unknown"
    print(f"Extracted file name: {extracted_file}")

    return {"type": type_, "name": extracted_name, "file": extracted_file}
