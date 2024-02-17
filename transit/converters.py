import time

def parse_time_to_seconds(time_str: str) -> int:
    """Converts a time string to the number of seconds since midnight.
    """
    try:
        if not isinstance(time_str, str):
            raise ValueError("Input must be a string.")

        parts = time_str.split(':')
        if len(parts) != 3:
            raise ValueError("Time string must be in 'HH:MM:SS' format.")
        
        # Allowed hours up to 48 to account for trips after midnight
        h, m, s = map(int, parts)
        if not (0 <= h < 48 and 0 <= m < 60 and 0 <= s < 60):
            raise ValueError("Hours must be in 0-25, minutes and seconds must be in 0-59.")

        return h * 3600 + m * 60 + s

    except ValueError as e:
        raise ValueError(f"Invalid time format or value: {e}")

    except Exception as e:
        raise ValueError(f"An error occurred while parsing the time: {e}")

def parse_seconds_to_time(seconds: int) -> str:
    """Converts the number of seconds since midnight to a time string."""
    return time.strftime('%H:%M:%S', time.gmtime(seconds))