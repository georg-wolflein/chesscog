"""Core exceptions.
"""


class RecognitionException(Exception):
    """Exception representing an error in the chess recognition pipeline.
    """

    def __init__(self, message: str = "unknown error"):
        super().__init__("chess recognition error: " + message)


class ChessboardNotLocatedException(RecognitionException):
    """Exception if the chessboard could not be located.
    """

    def __init__(self, reason: str = None):
        message = "chessboard could not be located"
        if reason:
            message += ": " + reason
        super().__init__(message)
