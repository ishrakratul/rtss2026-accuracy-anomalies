"""
terminal.py

Small helper for printing table/figure titles so they stand out in a
terminal: bold + colored text with a bordered banner when stdout is a
real terminal, and a plain ASCII banner (still clearly bordered) when
output is redirected to a file or pipe, where ANSI codes would just show
up as garbage characters.
"""
import sys

_RESET = "\033[0m"
_BOLD = "\033[1m"
_CYAN = "\033[96m"


def _supports_color() -> bool:
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def print_title(title: str, width: int = 78):
    """Print a bold, colored, bordered banner for a table/figure title.
    Falls back to a plain ASCII banner when stdout isn't a terminal
    (e.g. output redirected to a file)."""
    bar = "=" * width
    if _supports_color():
        print(f"{_BOLD}{_CYAN}{bar}{_RESET}")
        print(f"{_BOLD}{_CYAN}{title}{_RESET}")
        print(f"{_BOLD}{_CYAN}{bar}{_RESET}")
    else:
        print(bar)
        print(f"  {title}")
        print(bar)


def print_subtitle(text: str):
    """A lighter-weight bold highlight for sub-sections within a table's
    output (e.g. a specific cascade or condition), without a full banner."""
    if _supports_color():
        print(f"{_BOLD}{_CYAN}{text}{_RESET}")
    else:
        print(f">>> {text}")
