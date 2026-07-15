# -*- coding: utf-8 -*-
"""Shared rich-based console formatting helpers (boxed tables).

These are used by the CLI commands (preprocess, run) to present parameters,
results and timing in consistent rounded boxes.  A fresh Console is created at
print time so it picks up the current sys.stdout -- including the Tee that
run_dvc installs to mirror output into run_log.txt.  When stdout is not a real
terminal (e.g. through Tee) rich renders the boxes as plain unicode without
ANSI colour codes, keeping the log file clean.
"""


def kv_box(title, rows, border_style="cyan", value_justify="left"):
    """Print a key/value table inside a rounded box.

    Parameters
    ----------
    title : str
        Panel title.
    rows : iterable of (label, value)
        Rows are rendered in order; both fields are coerced to str.  Values may
        contain rich markup (labels are shown verbatim as bold cyan).
    border_style : str
        Rich style for the panel border.
    value_justify : str
        'left' or 'right' justification for the value column.
    """
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich import box

    tbl = Table(box=None, show_header=False, pad_edge=False)
    tbl.add_column("", style="bold cyan", justify="left")
    tbl.add_column("", justify=value_justify)
    for label, value in rows:
        # Labels via Text() so bracketed units are not parsed as rich markup.
        tbl.add_row(Text(str(label)), str(value))

    Console().print(Panel(tbl, title=f"[bold]{title}[/bold]",
                          border_style=border_style, box=box.ROUNDED,
                          padding=(1, 3), expand=False))


def fmt_duration(seconds):
    """Human-readable duration string, e.g. '3.42 s' or '1m 05.3s'."""
    if seconds >= 60.0:
        m, s = divmod(seconds, 60.0)
        return f"{int(m)}m {s:04.1f}s"
    return f"{seconds:.2f} s"
