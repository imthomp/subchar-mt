"""
encoder.py – Shared linguistic encoder for Chinese text representation.

Used by both main.py (demo) and finetune_experiment.py (supercomputer run).
"""

from __future__ import annotations

import hashlib
import struct
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

try:
    import sentencepiece as _spm
except ImportError:
    _spm = None


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
        sp_model_path: Optional[str] = None,
        random_index_seed: int = 42,
    ) -> None:
        self._cc_s2t = _opencc.OpenCC('s2t') if _opencc else None
        self.ids_map = self._load_ids_file(ids_path) if ids_path else self._mock_ids_data()
        self.wubi_map = self._load_kv_file(wubi_path) if wubi_path else self._mock_wubi_data()
        self.stroke_map = self._load_kv_file(stroke_path) if stroke_path else self._mock_stroke_data()
        self.cangjie_map = self._load_kv_file(cangjie_path) if cangjie_path else self._mock_cangjie_data()

        # SentencePiece model (optional — needed for to_sentencepiece())
        self._sp_model: Optional[object] = None
        if sp_model_path and _spm is not None:
            self._sp_model = _spm.SentencePieceProcessor()
            self._sp_model.Load(sp_model_path)

        # Random-index mapping: deterministic per-character index derived from a seeded hash.
        # Uses a deterministic hash so the mapping is stable across processes without storing
        # a full vocab table — any character maps to a consistent 4-digit token.
        self._ri_seed = random_index_seed

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

    def to_radicals(self, text: str, depth: int = 1) -> str:
        """
        Return space-joined IDS components at the given decomposition depth.

        depth=1 (default): direct components only — ideograph-level (Han et al. rxd1).
          e.g. 語 → 言 吾
        depth=2: intermediate — decompose each component one more level (Han et al. rxd2).
          e.g. 語 → 言 五 口   (吾 → 五 口; Han et al. found this COLLAPSES performance)
        depth≥6: near-primitive — recursive until no further decomposition possible.
          e.g. 語 → 言 一 𫝀 口

        Han, Jones & Smeaton (arXiv 2512.15556) found rxd1 ≈ rxd3 ≈ word baseline
        but rxd2 collapses to ~12.86 BLEU. This depth parameter enables that ablation.
        """
        def _decomp_char(ch: str, remaining_depth: int) -> List[str]:
            if remaining_depth == 0:
                return [ch]
            raw = self.ids_map.get(ch)
            if raw is None:
                return [ch]
            components = [c for c in raw if c not in self._IDS_LAYOUT_MARKERS]
            if not components:
                return [ch]
            result: List[str] = []
            for comp in components:
                result.extend(_decomp_char(comp, remaining_depth - 1))
            return result

        parts: List[str] = []
        for char in text:
            parts.extend(_decomp_char(char, depth))
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

    def to_bytes(self, text: str) -> str:
        """
        Encode each character as its UTF-8 bytes in lowercase hex, space-joined.
        Non-linguistic control: captures character identity via encoding, not semantics.
        E.g. 中 → 'e4 b8 ad'
        """
        tokens: List[str] = []
        for ch in text:
            tokens.extend(f"{b:02x}" for b in ch.encode("utf-8"))
        return " ".join(tokens)

    def to_random_index(self, text: str) -> str:
        """
        Replace each character with a deterministic pseudo-random 4-digit token.
        Non-linguistic control from Si et al. (2023): if this matches linguistic encodings,
        the benefit is structural (sequence length / BPE alignment), not semantic.

        The index is derived via a seeded hash of (seed, char) so it is:
          - consistent across calls / processes (same char always maps to same token)
          - different from linguistic indices (not ord(), not stroke order)
          - stable when the encoder is reconstructed with the same seed
        """
        tokens: List[str] = []
        for ch in text:
            # Mix seed with the codepoint to get a 16-bit index in 0000–9999
            raw = struct.pack(">II", self._ri_seed, ord(ch))
            h = int(hashlib.sha256(raw).hexdigest()[:4], 16)  # 0–65535
            tokens.append(f"{h % 10000:04d}")
        return " ".join(tokens)

    def to_sentencepiece(self, text: str) -> str:
        """
        Segment Chinese text with a trained SentencePiece unigram model.
        Data-driven subword baseline: contrasts jieba's dictionary-based morphemes
        with a purely statistical segmentation.

        Requires a pre-trained model loaded via sp_model_path at construction time.
        Falls back to baseline (raw characters) if no model is loaded.
        """
        if self._sp_model is None:
            return text
        pieces = self._sp_model.EncodeAsPieces(text)
        # Strip the leading ▁ (U+2581) word-boundary marker that SentencePiece inserts
        return " ".join(p.lstrip("▁") or p for p in pieces)

    def to_selective_decomp(
        self,
        text: str,
        tokenizer_vocab: set,
        decomp_method: str = 'radicals',
    ) -> str:
        """
        Inference-only selective decomposition (Saunders, Feely & Byrne, WAT 2020).

        Only decomposes characters that are OUT-OF-VOCABULARY for the model's tokenizer
        (i.e., would be split into byte-fallback or <unk> tokens). Known characters are
        kept as-is; unknown ones are expanded to their IDS radical components.

        This is the "principled fix" the roadmap identifies: decompose when it helps
        (unknown chars), leave alone when it doesn't (known chars).

        Parameters
        ----------
        text : str
            Raw Chinese source text.
        tokenizer_vocab : set
            Set of token strings known to the model (from tokenizer.get_vocab()).
        decomp_method : str
            'radicals' (IDS components) or 'morphemes' (jieba fallback to chars).
        """
        tokens: List[str] = []
        for ch in text:
            if ch in tokenizer_vocab:
                tokens.append(ch)
            else:
                if decomp_method == 'radicals':
                    decomp = self.ids_map.get(ch, ch)
                    components = [c for c in decomp if c not in self._IDS_LAYOUT_MARKERS]
                    tokens.extend(components if components else [ch])
                else:
                    # morpheme fallback: try jieba on the char, else keep as-is
                    segs = list(jieba.cut(ch, cut_all=False))
                    tokens.extend(segs if segs else [ch])
        return ' '.join(tokens)

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
        if strategy == 'byte':
            return self.to_bytes(text).split()
        if strategy == 'random_index':
            return self.to_random_index(text).split()
        if strategy == 'sentencepiece':
            return self.to_sentencepiece(text).split()
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
