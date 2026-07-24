class CustomUTF8Decoder:

    def __init__(self) -> None:
        """Initialize buffer"""
        self.buffer = b""

    def decode(self, chunk: bytes) -> str:
        """Decode while maintaining state in a buffer"""
        self.buffer += chunk
        try:
            text = self.buffer.decode("utf-8")
            self.buffer = b""
            return text
        except UnicodeDecodeError:
            return ""

    def flush(self) -> str:
        """To be called after generation to get potential orphan bytes"""
        res = self.buffer.decode("utf-8", errors="replace")
        self.buffer = b""
        return res
