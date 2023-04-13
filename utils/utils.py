def relpath(file: str) -> str:
    """Always locate to the correct relative path"""
    from sys import _getframe
    from pathlib import Path
    frame = _getframe(1)
    curr_file = Path(frame.f_code.co_filename)
    return str(curr_file.parent.joinpath(file).resolve())
