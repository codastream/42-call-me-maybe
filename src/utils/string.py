def is_quoted_string(s: str) -> bool:
    """Return true if the string is surrounded by unescaped double quotes"""
    s = s.strip()
    if len(s) < 2:
        return False
    return s.startswith('"') and s.endswith('"') and s[-2] != '\\'
