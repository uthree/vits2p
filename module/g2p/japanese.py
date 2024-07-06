from typing import List
from .language_module import LanguageModule
import pyopenjtalk


class JapaneseModule(LanguageModule):
    def __init__(self):
        pass

    def g2p(self, text: str) -> List[str]:
        return pyopenjtalk.g2p(text).split(" ")

    def vocabulary(self) -> List[str]:
        return ['I', 'N', 'U', 'a', 'b', 'by', 'ch', 'cl', 'd',
                'dy', 'e', 'f', 'g', 'gy', 'h', 'hy', 'i', 'j', 'k', 'ky',
                'm', 'my', 'n', 'ny', 'o', 'p', 'py', 'r', 'ry', 's', 'sh',
                't', 'ts', 'ty', 'u', 'v', 'w', 'y', 'z', 'pau']