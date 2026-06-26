# src/logger.py

from datetime import datetime

class StreamlitLogger:

    def __init__(self):
        self.logs = []

    def _add(self, level, msg):

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        entry = {
            "time": timestamp,
            "level": level,
            "message": msg
        }

        self.logs.append(entry)

    def info(self, msg):
        self._add("INFO", msg)

    def warning(self, msg):
        self._add("WARNING", msg)

    def error(self, msg):
        self._add("ERROR", msg)

    def success(self, msg):
        self._add("SUCCESS", msg)

    def get_logs(self):
        return self.logs