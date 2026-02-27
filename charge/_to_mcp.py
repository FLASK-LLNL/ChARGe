from charge.tasks.task import Task
from typing import Type
import inspect
import textwrap


def task_to_mcp(class_info, methods_list) -> str:
    """
    Convert an Task class to an MCP server definition string.
    """
    return_str = ""
    return_str += "from mcp.server.fastmcp import FastMCP\n"
    return_str += f"from {class_info['file']} import {class_info['name']}\n\n"

    return_str += 'mcp = FastMCP("Hypothesis MCP Server")\n\n'
    return_str += f"# Instance of the class\n"
    return_str += f"obj = {class_info['name']}()\n\n"

    for method in methods_list:
        sig = inspect.signature(method)
        params = []
        call_args = []
        for name, param in sig.parameters.items():
            if name == "self":
                continue
            annot = ""
            if param.annotation is not inspect._empty:
                annot = f": {inspect.formatannotation(param.annotation)}"
            default = ""
            if param.default is not inspect._empty:
                default = f"={repr(param.default)}"
            params.append(f"{name}{annot}{default}")
            call_args.append(f"{name}={name}")
        return_annot = ""
        if sig.return_annotation is not inspect._empty:
            return_annot = f" -> {inspect.formatannotation(sig.return_annotation)}"
        return_str += "@mcp.tool()\n"
        return_str += f"def {method.__name__}({', '.join(params)}){return_annot}:\n"
        if method.__doc__:
            docstring = textwrap.indent(
                f'"""{textwrap.dedent(method.__doc__)}"""', "    "
            )
            return_str += f"{docstring}\n"
        return_str += f"    return obj.{method.__name__}({', '.join(call_args)})\n\n"

    return_str += "\n"
    return_str += 'if __name__ == "__main__":\n'
    return_str += '    mcp.run(transport="stdio")\n'
    return return_str
