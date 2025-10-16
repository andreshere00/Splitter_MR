# ---------------------------------- #
# ------------ Splitter ------------ #
# ---------------------------------- #


class SplitterInputWarning(UserWarning):
    """
    Warning raised when the splitter input is suspicious (e.g., empty text or
    text expected to be JSON but not parseable as JSON).
    """

    pass
