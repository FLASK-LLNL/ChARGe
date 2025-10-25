"""Installation helper for packages with conflicting dependencies."""
import subprocess
import sys
import click


PACKAGE_GROUPS = {
    "chemprice": {
        "packages": ["chemprice"],
        "description": "Chemical pricing analysis tools"
    },
    "aizynthfinder": {
        "packages": ["aizynthfinder", "reaction-utils"],
        "description": "AiZynthfinder planning and reaction utilities"
    },
}

def install_packages_from_group(group: str):
    commands = []
    for pkg in PACKAGE_GROUPS[group]["packages"]:
        commands.append({
            "cmd": [sys.executable, '-m', 'pip', 'install', '--no-deps', pkg],
            "desc": f"Installing {pkg} (--no-deps)"
        })
    return commands


def run_pip_command(cmd, description):
    """Run a pip command and handle errors gracefully."""
    click.echo(f"\n→ {description}...")
    click.echo(f"  Command: {' '.join(cmd)}")
    
    try:
        subprocess.check_call(cmd)
        click.secho("✓ Success!", fg="green")
        return True
    except subprocess.CalledProcessError as e:
        click.secho(f"✗ Failed: {e}", fg="red", err=True)
        return False


@click.command()
@click.option(
    '--no-main',
    is_flag=True,
    help='Skip installation of main package (only install optional packages)'
)
@click.option(
    '--no-extras',
    is_flag=True,
    help='Installation of main package only (no optional packages)'
)
@click.option(
    '--editable/--no-editable',
    default=True,
    help='Install main package in editable mode (default: editable)'
)
@click.option(
    '--git-tag',
    required=False,
    help='Install from github using the tag'
)
@click.option(
    '--extras',
    type=click.Choice(['all', 'autogen', 'aizynthfinder', 'ollama', 'gemini', 'rdkit', 'flask', 'chemprop','chemprice'], case_sensitive=False),
    default=['all'],
    multiple=True,
    help='Extras to install for main package (default: all)'
)
@click.option(
    '--dry-run',
    is_flag=True,
    help='Show what would be installed without actually installing'
)
def main(no_extras, no_main, editable, git_tag, extras, dry_run):
    """
    Install ChARGe and its key package dependencies without sub-dependencies.
    
    This installer handles packages with conflicting dependency requirements
    by installing them with --no-deps after the main package is installed.
    
    Examples:
    
        # Full installation (default)
        $ charge-install

        # Minimal installation
        $ charge-install --no-extras
        
        # Only install chemprice
        $ charge-install --extras chemprice
        
        # Only install aizynthfinder tools
        $ charge-install --extras aizynthfinder
        
        # Install only optional packages (assumes main package already installed)
        $ charge-install --no-main
        
        # See what would be installed
        $ charge-install --dry-run
    """
    click.secho("="*60, fg="cyan")
    click.secho("Installation Helper for ChARGe", fg="cyan", bold=True)
    click.secho("="*60, fg="cyan")
    
    if dry_run:
        click.secho("\n[DRY RUN MODE - No packages will be installed]\n", fg="yellow", bold=True)
    
    commands = []
    failed = []

    if no_extras:
        click.secho(f"\n[WARNING - No optional packages will be installed flag --no-extras overriding --extras={extras}]\n", fg="yellow", bold=True)
        extras = []

    package_location = '.'
    if git_tag:
        package_location = f"git+https://github.com/FLASK-LLNL/ChARGe.git@{git_tag}"
        editable = False

    # Determine which packages to install
    # Main package installation
    if not no_main:
        install_cmd = [sys.executable, '-m', 'pip', 'install']
        if editable:
            install_cmd.append('-e')

        if not git_tag:
            if extras:
                install_cmd.append(f'{package_location}[{",".join(extras)}]')
            else:
                install_cmd.append(f'{package_location}')
        else:
            if extras:
                install_cmd.append(f'charge[{",".join(extras)}]@{package_location}')
            else:
                install_cmd.append(f'{package_location}')
        
        commands.append({
            "cmd": install_cmd,
            "desc": f"Installing main package from {package_location}{' (editable)' if editable else ''}" +
                   (f" with [{extras}] extras" if extras else "")
        })

    # Optional package groups
    if 'chemprice' in extras  or 'all' in extras:
        commands.extend(install_packages_from_group("chemprice"))
    else:
        click.echo(f"\n⊘ Skipping chemprice")

    if 'aizynthfinder' in extras or 'all' in extras:
        commands.extend(install_packages_from_group("aizynthfinder"))
    else:
        click.echo(f"\n⊘ Skipping aizynthfinder packages")

    # Show plan
    if commands:
        click.echo(f"\n{len(commands)} installation step(s) planned:")
        for i, step in enumerate(commands, 1):
            click.echo(f"  {i}. {step['desc']}")
    else:
        click.secho("\n⚠ No packages selected for installation", fg="yellow")
        return

    if dry_run:
        click.secho("\n[Dry run complete - no changes made]", fg="yellow")
        return
    
    # Execute installations
    click.echo()
    for step in commands:
        success = run_pip_command(step['cmd'], step['desc'])
        if not success:
            failed.append(step['desc'])
    
    # Summary
    click.echo("\n" + "="*60)
    if failed:
        click.secho("⚠ Installation completed with errors", fg="yellow", bold=True)
        click.echo("\nFailed steps:")
        for item in failed:
            click.secho(f"  ✗ {item}", fg="red")
        click.echo("\nYou may need to install these packages manually.")
        sys.exit(1)
    else:
        click.secho("✓ Installation complete!", fg="green", bold=True)
    click.secho("="*60, fg="cyan")


if __name__ == "__main__":
    main()
