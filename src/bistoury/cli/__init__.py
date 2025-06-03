"""
Command-line interface for Bistoury.
"""

from .commands import cli

def main():
    """Main entry point for the CLI."""
    cli()

__all__ = ['cli', 'main'] 