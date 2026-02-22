"""
encoder.py – Shared linguistic encoder for Chinese text representation.

Used by both main.py (demo) and finetune_experiment.py (supercomputer run).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import jieba
from pypinyin import pinyin as _pinyin, Style

try:
    import opencc as _opencc
except ImportError:
    _opencc = None

try:
    from dragonmapper import hanzi as _dragonmapper_hanzi
except ImportError:
    _dragonmapper_hanzi = None


class LinguisticEncoder:
    """
    Converts Chinese text into various linguistic representations.

    Parameters
    ----------
    ids_path
        Path to a CHISE IDS file (tab-delimited: U+XXXX <TAB> char <TAB> decomp).
        Falls back to a small built-in sample when omitted.
    wubi_path
        Path to a Wubi code dictionary (whitespace-delimited key-value pairs).
    stroke_path
        Path to a stroke-sequence dictionary (whitespace-delimited key-value pairs).
    """

    _PINYIN_TO_ZHUYIN: Dict[str, str] = {
        'a': 'ㄚ', 'o': 'ㄛ', 'e': 'ㄜ', 'ai': 'ㄞ', 'ei': 'ㄟ',
        'ao': 'ㄠ', 'ou': 'ㄡ', 'an': 'ㄢ', 'en': 'ㄣ', 'ang': 'ㄤ',
        'eng': 'ㄥ', 'er': 'ㄦ', 'b': 'ㄅ', 'p': 'ㄆ', 'm': 'ㄇ',
        'f': 'ㄈ', 'd': 'ㄉ', 't': 'ㄊ', 'n': 'ㄋ', 'l': 'ㄌ',
        'g': 'ㄍ', 'k': 'ㄎ', 'h': 'ㄏ', 'j': 'ㄐ', 'q': 'ㄑ',
        'x': 'ㄒ', 'zh': 'ㄓ', 'ch': 'ㄔ', 'sh': 'ㄕ', 'r': 'ㄖ',
        'z': 'ㄗ', 'c': 'ㄘ', 's': 'ㄙ', 'y': 'ㄧ', 'w': 'ㄨ',
    }

    _IDS_LAYOUT_MARKERS = set('⿰⿱⿲⿳⿴⿵⿶⿷⿸⿹⿺⿻')

    def __init__(
        self,
        ids_path: Optional[str] = None,
        wubi_path: Optional[str] = None,
        stroke_path: Optional[str] = None,
        cangjie_path: Optional[str] = None,
    ) -> None:
        self._cc_s2t = _opencc.OpenCC('s2t') if _opencc else None
        self.ids_map = self._load_ids_file(ids_path) if ids_path else self._mock_ids_data()
        self.wubi_map = self._load_kv_file(wubi_path) if wubi_path else self._mock_wubi_data()
        self.stroke_map = self._load_kv_file(stroke_path) if stroke_path else self._mock_stroke_data()
        self.cangjie_map = self._load_kv_file(cangjie_path) if cangjie_path else self._mock_cangjie_data()

    # ------------------------------------------------------------------
    # Individual representation methods
    # ------------------------------------------------------------------

    def to_pinyin(self, text: str, with_tone: bool = True) -> str:
        """Return space-joined pinyin romanisation."""
        style = Style.TONE if with_tone else Style.NORMAL
        return ' '.join(item[0] for item in _pinyin(text, style=style, heteronym=False))

    def to_zhuyin(self, text: str) -> str:
        """Return space-joined Zhuyin (Bopomofo) phonetic notation."""
        if _dragonmapper_hanzi:
            return _dragonmapper_hanzi.to_zhuyin(text)
        # Fallback: derive Zhuyin from pinyin via lookup table
        syllables = _pinyin(text, style=Style.TONE3, heteronym=False)
        result: List[str] = []
        for item in syllables:
            syl = item[0]
            zhuyin: List[str] = []
            for key in sorted(self._PINYIN_TO_ZHUYIN, key=len, reverse=True):
                if key in syl:
                    zhuyin.append(self._PINYIN_TO_ZHUYIN[key])
                    syl = syl.replace(key, '', 1)
            result.append(''.join(zhuyin))
        return ' '.join(result)

    def to_radicals(self, text: str) -> str:
        """Return space-joined flat IDS components (layout markers stripped)."""
        parts: List[str] = []
        for char in text:
            decomp = self.ids_map.get(char, char)
            parts.extend(c for c in decomp if c not in self._IDS_LAYOUT_MARKERS)
        return ' '.join(parts)

    def segment_morphemes(self, text: str, mode: str = 'accurate') -> str:
        """Return space-joined word segmentation using jieba."""
        if mode == 'accurate':
            words = jieba.cut(text, cut_all=False)
        elif mode == 'full':
            words = jieba.cut(text, cut_all=True)
        else:
            words = jieba.cut_for_search(text)
        return ' '.join(words)

    def to_traditional(self, text: str) -> str:
        """Convert Simplified Chinese to Traditional Chinese."""
        if self._cc_s2t is None:
            raise RuntimeError("opencc is not installed; cannot perform script conversion.")
        return self._cc_s2t.convert(text)

    def to_wubi(self, text: str) -> List[str]:
        """Return list of Wubi key codes, one per character."""
        return [self.wubi_map.get(ch, '<UNK>') for ch in text]

    def to_cangjie(self, text: str) -> List[str]:
        """Return list of Cangjie key codes, one per character."""
        return [self.cangjie_map.get(ch, '<UNK>') for ch in text]

    def to_strokes(self, text: str) -> List[str]:
        """Return flat list of stroke atoms across all characters."""
        strokes: List[str] = []
        for ch in text:
            strokes.extend(self.stroke_map.get(ch, ''))
        return strokes

    # ------------------------------------------------------------------
    # Strategy dispatch (used by main.py demo)
    # ------------------------------------------------------------------

    def encode(self, text: str, strategy: str) -> List[str]:
        """Encode *text* with the named *strategy* and return a token list."""
        if strategy == 'baseline':
            return list(text)
        if strategy == 'pinyin_no_tone':
            return self.to_pinyin(text, with_tone=False).split()
        if strategy == 'pinyin_tone':
            return self.to_pinyin(text, with_tone=True).split()
        if strategy == 'zhuyin':
            return self.to_zhuyin(text).split()
        if strategy == 'morphological':
            return self.segment_morphemes(text).split()
        if strategy == 'simplified_traditional':
            return list(text) + ['<SEP>'] + list(self.to_traditional(text))
        if strategy == 'radical_flat':
            return [t for t in self.to_radicals(text).split() if t]
        if strategy == 'radical_structural':
            result: List[str] = []
            for ch in text:
                result.extend(self.ids_map.get(ch, ch))
            return result
        if strategy == 'wubi':
            return self.to_wubi(text)
        if strategy == 'cangjie':
            return self.to_cangjie(text)
        if strategy == 'stroke_sequence':
            return self.to_strokes(text)
        raise ValueError(f"Unknown strategy: {strategy!r}")

    # ------------------------------------------------------------------
    # File loaders
    # ------------------------------------------------------------------

    def _load_ids_file(self, path: str) -> Dict[str, str]:
        """Parse a CHISE IDS file (U+XXXX <TAB> char <TAB> decomposition ...)."""
        mapping: Dict[str, str] = {}
        with open(path, encoding='utf-8') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.strip().split('\t')
                if len(parts) < 3:
                    continue
                char = parts[1]
                # Strip bracketed source tags, e.g. [GTV], [J], etc.
                clean, skip = '', False
                for ch in parts[2]:
                    if ch == '[':
                        skip = True
                    elif ch == ']':
                        skip = False
                    elif not skip:
                        clean += ch
                mapping[char] = clean.strip()
        return mapping

    def _load_kv_file(self, path: str) -> Dict[str, str]:
        """Parse a whitespace-delimited key-value file."""
        mapping: Dict[str, str] = {}
        with open(path, encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    mapping[parts[0]] = parts[1]
        return mapping

    # ------------------------------------------------------------------
    # Built-in sample data (fallback when no files provided)
    # ------------------------------------------------------------------

    def _mock_ids_data(self) -> Dict[str, str]:
        return {
            '机': '⿰木几', '器': '⿱口𠙻', '翻': '⿰番羽',
            '译': '⿰讠义', '学': '⿱𦥑子', '中': '中', '文': '文',
        }

    def _mock_wubi_data(self) -> Dict[str, str]:
        return {
            '机': 'sm', '器': 'kkk', '翻': 'tol', '译': 'ycf',
            '学': 'ip', '中': 'k', '文': 'yy',
        }

    def _mock_cangjie_data(self) -> Dict[str, str]:
        # Cangjie codes from UNIHAN kCangjie field
        return {
            '机': 'DHN', '器': 'RRIKR', '翻': 'HWSMM', '译': 'IVEQ',
            '学': 'FBND', '中': 'L', '文': 'YK',
        }

    def _mock_stroke_data(self) -> Dict[str, str]:
        # h=heng(一) s=shu(丨) p=pie(丿) n=na(乀) z=zhe(乛) d=dian(丶)
        return {
            '机': 'hsphzn', '器': 'hzhhzh', '翻': 'phsz',
            '译': 'zh', '学': 'zzh', '中': 'shsz', '文': 'zhpn',
        }
