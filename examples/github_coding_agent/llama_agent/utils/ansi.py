# ANSI color codes for terminal output
ANSI_RED = "\033[91m"
ANSI_GREEN = "\033[92m" 
ANSI_YELLOW = "\033[93m"
ANSI_BLUE = "\033[94m"
ANSI_MAGENTA = "\033[95m"
ANSI_CYAN = "\033[96m"
ANSI_RESET = "\033[0m"
ANSI_BOLD = "\033[1m"

def red(text: any) -> str:
    """Wrap text in red ANSI color"""
    return f"{ANSI_RED}{text}{ANSI_RESET}"

def green(text: any) -> str:
    """Wrap text in green ANSI color"""
    return f"{ANSI_GREEN}{text}{ANSI_RESET}"

def yellow(text: any) -> str:
    """Wrap text in yellow ANSI color"""
    return f"{ANSI_YELLOW}{text}{ANSI_RESET}"

def blue(text: any) -> str:
    """Wrap text in blue ANSI color"""
    return f"{ANSI_BLUE}{text}{ANSI_RESET}"

def magenta(text: any) -> str:
    """Wrap text in magenta ANSI color"""
    return f"{ANSI_MAGENTA}{text}{ANSI_RESET}"

def cyan(text: any) -> str:
    """Wrap text in cyan ANSI color"""
    return f"{ANSI_CYAN}{text}{ANSI_RESET}"

def bold(text: any) -> str:
    """Wrap text in bold ANSI style"""
    return f"{ANSI_BOLD}{text}{ANSI_RESET}"