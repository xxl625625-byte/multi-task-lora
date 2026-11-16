"""字级分词。选项代号必须是一个字符，因此也是一个 token。"""

from __future__ import annotations


class CharTokenizer:
    def __init__(self, texts: list[str]):
        chars = set("ABC")
        for text in texts:
            chars.update(text)
        ordered = sorted(chars)
        self.stoi = {char: index + 1 for index, char in enumerate(ordered)}
        self.itos = {index: char for char, index in self.stoi.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.stoi) + 1

    def encode(self, text: str) -> list[int]:
        try:
            return [self.stoi[char] for char in text]
        except KeyError as exc:
            raise KeyError(f"词表缺少字符 {exc}") from exc

    def token_id(self, code: str) -> int:
        if len(code) != 1:
            raise ValueError(f"选项代号必须是单字符，收到 {code}")
        return self.stoi[code]

    def check_codes(self, codes: list[str]) -> None:
        for code in codes:
            if len(self.encode(code)) != 1:
                raise ValueError(f"代号 {code} 不是单个 token")
