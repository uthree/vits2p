from typing import List

# 言語を追加するときはこのクラスを継承して実装する。
class LanguageModule():
    def __init__():
        pass
    
    # テキストを音素列に変換
    def g2p(self, text: str) -> List[str]:
        pass

    # この言語が使用する全ての音素を返す。(重複なし)
    def vocabulary(self) -> List[str]:
        pass