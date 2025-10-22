"""Installation helper for packages with conflicting dependencies."""
import subprocess
import sys
import click


PACKAGE_GROUPS = {
    "chemprice": {
        "packages": ["chemprice"],
        "description": "Chemical pricing analysis tools"
    },
    "synthesis": {
        "packages": ["aizynthfinder", "reaction-utils"],
        "description": "AI synthesis planning and reaction utilities"
    },
}


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
    '--skip-chemprice',
    is_flag=True,
    help='Skip installation of chemprice package'
)
@click.option(
    '--skip-synthesis',
    is_flag=True,
    help='Skip installation of synthesis packages (aizynthfinder, reaction-utils)'
)
@click.option(
    '--only',
    type=click.Choice(['chemprice', 'synthesis'], case_sensitive=False),
    help='Only install specified package group'
)
@click.option(
    '--no-main',
    is_flag=True,
    help='Skip installation of main package (only install optional packages)'
)
@click.option(
    '--editable/--no-editable',
    default=True,
    help='Install main package in editable mode (default: editable)'
)
@click.option(
    '--extras',
    default='all',
    help='Extras to install for main package (default: all)'
)
@click.option(
    '--dry-run',
    is_flag=True,
    help='Show what would be installed without actually installing'
)
def main(skip_chemprice, skip_synthesis, only, no_main, editable, extras, dry_run):
    """
    Install ChARGe and its key package dependencies without sub-dependencies.
    
    This installer handles packages with conflicting dependency requirements
    by installing them with --no-deps after the main package is installed.
    
    Examples:
    
        # Full installation (default)
        $ charge-install
        
        # Skip chemprice
        $ charge-install --skip-chemprice
        
        # Only install synthesis tools
        $ charge-install --only synthesis
        
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
    
    # Determine which packages to install
    if only:
        skip_chemprice = (only != 'chemprice')
        skip_synthesis = (only != 'synthesis')
        click.echo(f"\n→ Installing only: {only}")
    
    # Main package installation
    if not no_main:
        install_cmd = [sys.executable, '-m', 'pip', 'install']
        if editable:
            install_cmd.append('-e')
        
        if extras:
            install_cmd.append(f'.[{extras}]')
        else:
            install_cmd.append('.')
        
        commands.append({
            "cmd": install_cmd,
            "desc": f"Installing main package{' (editable)' if editable else ''}" + 
                   (f" with [{extras}] extras" if extras else "")
        })
    
    # Optional package groups
    if not skip_chemprice:
        for pkg in PACKAGE_GROUPS["chemprice"]["packages"]:
            commands.append({
                "cmd": [sys.executable, '-m', 'pip', 'install', '--no-deps', pkg],
                "desc": f"Installing {pkg} (--no-deps)"
            })
    else:
        click.echo(f"\n⊘ Skipping chemprice")
    
    if not skip_synthesis:
        packages = PACKAGE_GROUPS["synthesis"]["packages"]
        commands.append({
            "cmd": [sys.executable, '-m', 'pip', 'install', '--no-deps'] + packages,
            "desc": f"Installing synthesis packages: {', '.join(packages)} (--no-deps)"
        })
    else:
        click.echo(f"\n⊘ Skipping synthesis packages")
    
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
