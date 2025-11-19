from pathlib import Path
import mkdocs_gen_files

# --- Configuration ---
# 1. The path to the directory *containing* the package folder 'charge'. (Project Root)
SRC_ROOT = Path(".") 
# 2. The name of your top-level package.
PACKAGE_NAME = "charge" 
# 3. The target directory inside 'docs' for the generated reference.
REF_DIR = Path("reference")

nav = mkdocs_gen_files.Nav()

# Walk through all Python files within the 'charge' package directory
# path will be, e.g., 'charge/_tags.py', 'charge/__init__.py', 'charge/clients.py'
paths = list((SRC_ROOT / PACKAGE_NAME).rglob("*.py"))

def get_sort_key(path):
    """
    Sort key function to ensure directories appear before files.
    Returns a list of tuples (priority, name), where priority is 0 for directories and 1 for files.
    """
    parts = path.relative_to(SRC_ROOT).parts
    return [(0, part) if i < len(parts) - 1 else (1, part) for i, part in enumerate(parts)]

for path in sorted(paths, key=get_sort_key):
    
    # 1. Get the path relative to the root, without the extension
    # e.g., 'charge/_tags' or 'charge/__init__'
    module_path = path.relative_to(SRC_ROOT).with_suffix("")
    
    # Split into parts: e.g., ['charge', '_tags']
    parts = list(module_path.parts)
    
    # --- Logic to handle __init__.py files ---
    if parts[-1] == "__init__":
        # Drop the "__init__" part. Remaining parts: ['charge']
        parts = parts[:-1]
        
        # Set the documentation filename to index.md for the package root.
        doc_path = REF_DIR / Path(*parts) / "index.md"
    
    # --- Logic to handle regular modules (e.g., _tags.py) ---
    else:
        # Keep all parts: e.g., ['charge', '_tags']
        
        # Set the documentation filename: e.g., reference/charge/_tags.md
        doc_path = REF_DIR / module_path.with_suffix(".md")

    # The full Python identifier for mkdocstrings (e.g., 'charge' or 'charge._tags')
    ident = ".".join(parts)

    # --- FINAL CHECKS ---
    if not ident:
        # Skip if somehow the identifier is still empty (shouldn't happen now)
        continue

    # 2. Add entry to the Nav object. 
    # This automatically uses the fully qualified name (ident) as the title 
    # and doc_path as the link, structured by the 'parts' list.
    nav[parts] = doc_path.relative_to(REF_DIR).as_posix()

    # 3. Create the Markdown file with the mkdocstrings injection
    with mkdocs_gen_files.open(doc_path, "w") as fd:
        fd.write(f"# `{ident}`\n\n")
        fd.write(f"::: {ident}\n")

    # Set edit link path to the source file
    mkdocs_gen_files.set_edit_path(doc_path, path)


# 4. Write the final SUMMARY.md navigation file
with mkdocs_gen_files.open(REF_DIR / "SUMMARY.md", "w") as nav_file:
    # This will generate the correct nested structure, like:
    # * [charge](charge/index.md)
    #   * [charge._tags](charge/_tags.md)
    #   ...
    nav_file.writelines(nav.build_literate_nav())