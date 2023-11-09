import time

# Уперто откуда-то с гитхаба
def parse_time_to_seconds(time_str: str) -> int:
    """Converts a time string to the number of seconds since midnight.
    """
    try:
        # Ensure the input is a string
        if not isinstance(time_str, str):
            raise ValueError("Input must be a string.")

        # Split the time string into components
        parts = time_str.split(':')
        if len(parts) != 3:
            raise ValueError("Time string must be in 'HH:MM:SS' format.")

        # Допустимы значения часов до 48, чтобы учесть автобусы после полуночи
        h, m, s = map(int, parts)
        if not (0 <= h < 48 and 0 <= m < 60 and 0 <= s < 60):
            raise ValueError("Hours must be in 0-25, minutes and seconds must be in 0-59.")

        # Calculate and return total seconds
        return h * 3600 + m * 60 + s

    except ValueError as e:
        # Re-raise the ValueError with a more specific message
        raise ValueError(f"Invalid time format or value: {e}")

    except Exception as e:
        # Catch any other exceptions that may occur and raise as ValueError
        raise ValueError(f"An error occurred while parsing the time: {e}")

def parse_seconds_to_time(seconds: int) -> str:
    """Converts the number of seconds since midnight to a time string."""
    return time.strftime('%H:%M:%S', time.gmtime(seconds))