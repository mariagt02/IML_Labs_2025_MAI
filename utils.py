import platform
import re
import os


class GlobalConfig:
    """
    A utility class for global configuration settings.
    It provides default paths for results and output files used across the project.
    """
    DEFAULT_RESULTS_PATH = os.path.join("res", "k_ibl_hyperparameters") # Path where the k_ibl_hyperparameters results are stored
    DEFAULT_STATS_OUTPUT_PATH = os.path.join("res", "stat_tests") # Path where the Friedman-Nemenyi test results will be saved




class TerminalColor:
    """
    A utility class for terminal text coloring and formatting.
    It supports ANSI escape codes for colorizing text in terminals that support it.
    The class automatically detects if the terminal supports ANSI colors and provides methods to colorize text,
    strip ANSI escape codes, and initialize color settings based on the platform.
    """
    __ENABLED = None
    __CODES = {
        "blue": "\033[34m",
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "purple": "\033[35m",
        "grey": "\033[90m",
        "orange": "\033[38;5;214m"
    }
    __RESET = "\033[0m"
    __ANSI_ESCAPE_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    

    @staticmethod
    def _init():
        """
        Initializes the terminal color settings based on the platform.
        """
        if platform.system() in ["Linux", "Darwin"]:
            TerminalColor.__ENABLED = True
        elif platform.system() == "Windows":
            try:
                import ctypes
                kernel32 = ctypes.windll.kernel32
                kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
                TerminalColor.__ENABLED = True
            except Exception:
                TerminalColor.__ENABLED = False

    @staticmethod
    def colorize(text: str, color: str, bold: bool = False) -> str:
        """
        Colorizes the given text with the specified color and optional bold formatting.
        
        Args:
            text (str): The text to colorize.
            color (str): The color to apply. Must be one of the keys in TerminalColor.CODES.
            bold (bool): If True, applies bold formatting to the text.
        
        Returns:
            str: The colorized text with ANSI escape codes, or the original text if coloring is not enabled or the color is invalid.
        """
        if TerminalColor.__ENABLED is None:
            TerminalColor._init()

        if TerminalColor.__ENABLED and color in TerminalColor.__CODES:
            prefix = "\033[1m" if bold else ""
            return f"{prefix}{TerminalColor.__CODES[color]}{text}{TerminalColor.__RESET}"
        return text

    @staticmethod
    def strip(text: str) -> str:
        """
        Strips ANSI escape codes from the given text.
        
        Args:
            text (str): The text from which to strip ANSI escape codes.
        
        Returns:
            str: The text without ANSI escape codes.
        """
        return TerminalColor.__ANSI_ESCAPE_RE.sub("", text)

    @property
    def ERROR(self):
        return TerminalColor.colorize("ERROR:", color="red", bold=True)
    
    @property
    def WARNING(self):
        return TerminalColor.colorize("WARNING:", color="yellow", bold=True)
    
    @property
    def SUCCESS(self):
        return TerminalColor.colorize("SUCCESS:", color="green", bold=True)