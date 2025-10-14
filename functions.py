import pandas as pd

def humanize_number(x):
    """Convert large numbers to human-readable format."""
    try:
        x = float(x)
    except Exception:
        return "N/A"
    if pd.isna(x):
        return "N/A"
    abs_x = abs(x)
    if abs_x >= 1e12:
        return f"{x/1e12:.2f}T"
    if abs_x >= 1e9:
        return f"{x/1e9:.2f}B"
    if abs_x >= 1e6:
        return f"{x/1e6:.2f}M"
    if abs_x >= 1e3:
        return f"{x/1e3:.2f}K"
    return f"{x:.0f}"