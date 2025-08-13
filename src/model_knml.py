import os

import kenlm
import sentencepiece as spm
from tokenizers import Regex, normalizers

# Borrowed from Jomleh dataset code
char_map = {
    # Arabic Letter Hamza
    # "\u": "\u0621",
    # Arabic Letter Alef with Hamza Above
    "\ufe83": "\u0623",
    "\ufe84": "\u0623",
    # Arabic Letter Yeh with Hamza Above
    "\ufe89": "\u0626",
    "\ufe8a": "\u0626",
    "\ufe8b": "\u0626",
    "\ufe8c": "\u0626",
    # Arabic Letter Waw with Hamza Above
    "\ufe85": "\u0624",
    "\ufe86": "\u0624",
    "\u0676": "\u0624",
    # Arabic Letter Alef with Madda Above
    "\ufe81": "\u0622",  # Arabic letter Alef final form
    "\ufe82": "\u0622",  # Arabic letter Alef isolated form
    # Alef
    "\ufb50": "\u0627",  # Arabic letter Alef wasla
    "\ufe87": "\u0627",
    "\u0675": "\u0627",
    "\u0625": "\u0627",
    "\ufe8d": "\u0627",
    "\ufe8e": "\u0627",
    "\u1ee00": "\u0627",
    "\u1ee80": "\u0627",
    # Beh
    "\ufe8f": "\u0628",
    "\ufe90": "\u0628",
    "\ufe91": "\u0628",
    "\ufe92": "\u0628",
    "\u1ee01": "\u0628",
    "\u1ee21": "\u0628",
    "\u1ee61": "\u0628",
    "\u1ee81": "\u0628",
    "\u1eea1": "\u0628",
    # Pe
    "\ufb56": "\u067e",
    "\ufb57": "\u067e",
    "\ufb58": "\u067e",
    "\ufb59": "\u067e",
    # Teh
    "\ufe95": "\u062a",
    "\ufe96": "\u062a",
    "\ufe97": "\u062a",
    "\ufe98": "\u062a",
    "\u1ee15": "\u062a",
    "\u1ee35": "\u062a",
    "\u1ee75": "\u062a",
    "\u1ee95": "\u062a",
    "\u1eeb5": "\u062a",
    # Theh
    "\ufe99": "\u062b",
    "\ufe9a": "\u062b",
    "\ufe9b": "\u062b",
    "\ufe9c": "\u062b",
    "\u1ee16": "\u062b",
    "\u1ee36": "\u062b",
    "\u1ee76": "\u062b",
    "\u1ee96": "\u062b",
    "\u1eeb6": "\u062b",
    # Jim
    "\ufe9d": "\u062c",
    "\ufe9e": "\u062c",
    "\ufe9f": "\u062c",
    "\ufea0": "\u062c",
    "\u1ee02": "\u062c",
    "\u1ee22": "\u062c",
    "\u1ee42": "\u062c",
    "\u1ee62": "\u062c",
    "\u1ee82": "\u062c",
    "\u1eea2": "\u062c",
    # Cheh
    "\ufb7a": "\u0686",
    "\ufb7b": "\u0686",
    "\ufb7c": "\u0686",
    "\ufb7d": "\u0686",
    # Hah
    "\ufea1": "\u062d",
    "\ufea2": "\u062d",
    "\ufea3": "\u062d",
    "\ufea4": "\u062d",
    "\u1ee07": "\u062d",
    "\u1ee27": "\u062d",
    "\u1ee47": "\u062d",
    "\u1ee67": "\u062d",
    "\u1ee87": "\u062d",
    "\u1eea7": "\u062d",
    # Khah
    "\ufea5": "\u062e",
    "\ufea6": "\u062e",
    "\ufea7": "\u062e",
    "\ufea8": "\u062e",
    "\u1ee17": "\u062e",
    "\u1ee37": "\u062e",
    "\u1ee57": "\u062e",
    "\u1ee77": "\u062e",
    "\u1ee97": "\u062e",
    "\u1eeb7": "\u062e",
    # Dal
    "\ufea9": "\u062f",
    "\ufeaa": "\u062f",
    "\u1ee03": "\u062f",
    "\u1ee83": "\u062f",
    "\u1eea3": "\u062f",
    # Zal
    "\ufeab": "\u0630",
    "\ufeac": "\u0630",
    "\u1ee18": "\u0630",
    "\u1ee98": "\u0630",
    "\u1eeb8": "\u0630",
    # Reh
    "\ufeae": "\u0631",  # Arabic letter Reh isolated form
    "\ufead": "\u0631",  # Arabic letter Reh final form
    "\u0692": "\u0631",
    "\u1ee13": "\u0631",
    "\u1ee93": "\u0631",
    "\u1eeb3": "\u0631",
    # Ze
    "\ufeaf": "\u0632",  #
    "\ufeb0": "\u0632",  #
    "\u1ee06": "\u0632",  #
    "\u1ee86": "\u0632",  #
    "\u1eea6": "\u0632",  #
    # Jhe
    "\ufb8a": "\u0698",
    "\ufb8b": "\u0698",
    # Seen
    "\ufeb1": "\u0633",  #
    "\ufeb2": "\u0633",  #
    "\ufeb3": "\u0633",  #
    "\ufeb4": "\u0633",  #
    "\u1ee0E": "\u0633",  #
    "\u1ee2E": "\u0633",  #
    "\u1ee4E": "\u0633",  #
    "\u1ee6E": "\u0633",  #
    "\u1ee8E": "\u0633",  #
    "\u1eeaE": "\u0633",  #
    # Sheen
    "\ufeb5": "\u0634",  #
    "\ufeb6": "\u0634",  #
    "\ufeb7": "\u0634",  #
    "\ufeb8": "\u0634",  #
    "\u1ee14": "\u0634",  #
    "\u1ee34": "\u0634",  #
    "\u1ee54": "\u0634",  #
    "\u1ee74": "\u0634",  #
    "\u1ee94": "\u0634",  #
    "\u1eeb4": "\u0634",  #
    # Sad
    "\ufeb9": "\u0635",  #
    "\ufeba": "\u0635",  #
    "\ufebb": "\u0635",  #
    "\ufebc": "\u0635",  #
    "\u1ee11": "\u0635",  #
    "\u1ee31": "\u0635",  #
    "\u1ee51": "\u0635",  #
    "\u1ee71": "\u0635",  #
    "\u1ee91": "\u0635",  #
    "\u1eeb1": "\u0635",  #
    # Zad
    "\ufebd": "\u0636",  #
    "\ufebe": "\u0636",  #
    "\ufebf": "\u0636",  #
    "\ufec0": "\u0636",  #
    "\u1ee19": "\u0636",  #
    "\u1ee39": "\u0636",  #
    "\u1ee59": "\u0636",  #
    "\u1ee79": "\u0636",  #
    "\u1ee99": "\u0636",  #
    "\u1eeb9": "\u0636",  #
    # Ta
    "\ufec1": "\u0637",  #
    "\ufec2": "\u0637",  #
    "\ufec3": "\u0637",  #
    "\ufec4": "\u0637",  #
    "\u1ee08": "\u0637",  #
    "\u1ee68": "\u0637",  #
    "\u1ee88": "\u0637",  #
    "\u1eea8": "\u0637",  #
    # Za
    "\ufec5": "\u0638",  #
    "\ufec6": "\u0638",  #
    "\ufec7": "\u0638",  #
    "\ufec8": "\u0638",  #
    "\u1ee1A": "\u0638",  #
    "\u1ee7A": "\u0638",  #
    "\u1ee9A": "\u0638",  #
    "\u1eebA": "\u0638",  #
    # Ain
    "\ufec9": "\u0639",  #
    "\ufeca": "\u0639",  #
    "\ufecb": "\u0639",  #
    "\ufecc": "\u0639",  #
    "\u1ee0F": "\u0639",  #
    "\u1ee2F": "\u0639",  #
    "\u1ee4F": "\u0639",  #
    "\u1ee6F": "\u0639",  #
    "\u1ee8F": "\u0639",  #
    "\u1eeaF": "\u0639",  #
    # Ghain
    "\ufecd": "\u063a",  #
    "\ufece": "\u063a",  #
    "\ufecf": "\u063a",  #
    "\ufed0": "\u063a",  #
    "\u1ee1B": "\u063a",  #
    "\u1ee3B": "\u063a",  #
    "\u1ee5B": "\u063a",  #
    "\u1ee7B": "\u063a",  #
    "\u1ee9B": "\u063a",  #
    "\u1eebB": "\u063a",  #
    # Fa
    "\ufed1": "\u0641",  #
    "\ufed2": "\u0641",  #
    "\ufed3": "\u0641",  #
    "\ufed4": "\u0641",  #
    "\u1ee10": "\u0641",  #
    "\u1ee30": "\u0641",  #
    "\u1ee70": "\u0641",  #
    "\u1ee90": "\u0641",  #
    "\u1eeb0": "\u0641",  #
    # Qaf
    "\ufed5": "\u0642",  #
    "\ufed6": "\u0642",  #
    "\ufed7": "\u0642",  #
    "\ufed8": "\u0642",  #
    "\u1ee12": "\u0642",  #
    "\u1ee32": "\u0642",  #
    "\u1ee52": "\u0642",  #
    "\u1ee72": "\u0642",  #
    "\u1ee92": "\u0642",  #
    "\u1eeb2": "\u0642",  #
    # Kaf
    "\ufb8e": "\u06a9",  # Arabic letter Kaf isolated form
    "\ufb8f": "\u06a9",  # Arabic letter Kaf final form
    "\ufb90": "\u06a9",  # Arabic letter Kaf initial form
    "\ufb91": "\u06a9",  # Arabic letter Kaf medial form
    "\ufcc8": "\u06a9",  # Arabic ligature Dal with Alef final form
    "\u0643": "\u06a9",
    "\ufed9": "\u06a9",
    "\ufeda": "\u06a9",  # Arabic Letter Kaf Final Form
    "\ufedb": "\u06a9",  #
    "\ufedc": "\u06a9",  #
    "\u1ee0A": "\u06a9",  #
    "\u1ee2A": "\u06a9",  #
    "\u1ee6A": "\u06a9",  #
    # Gaf
    "\ufb92": "\u06af",  # Arabic letter Gaf isolated form
    "\ufb93": "\u06af",  # Arabic letter Gaf final form
    "\ufb94": "\u06af",  # Arabic letter Gaf initial form
    "\ufb95": "\u06af",  # Arabic letter Gaf medial form
    # Lam
    "\ufcc9": "\u0644",  # Arabic Ligature Lam with Jeem Initial Form
    "\ufedd": "\u0644",  # Arabic Letter Lam Isolated Form
    "\ufede": "\u0644",  # Arabic Letter Lam Final Form
    "\ufedf": "\u0644",  # Arabic Letter Lam Initial Form
    "\ufee0": "\u0644",  # Arabic Letter Lam Medial Form
    "\u1ee0B": "\u0644",  # Arabic Mathematical Lam
    "\u1ee2B": "\u0644",  # Arabic Mathematical Initial Lam
    "\u1ee4B": "\u0644",  # Arabic Mathematical Tailed Lam
    "\u1ee8B": "\u0644",  # Arabic Mathematical Looped Lam
    "\u1eeaB": "\u0644",  # Arabic Mathematical Double-Struck Lam
    # Mim
    "\ufee1": "\u0645",  # Arabic Letter Meem Isolated Form
    "\ufee2": "\u0645",  # Arabic Letter Meem Final Form
    "\ufee3": "\u0645",  # Arabic Letter Meem Initial Form
    "\ufee4": "\u0645",  # Arabic Letter Meem Medial Form
    "\u1ee0C": "\u0645",  # Arabic Mathematical Meem
    "\u1ee2C": "\u0645",  # Arabic Mathematical Initial Meem
    "\u1ee6C": "\u0645",  # Arabic Mathematical Stretched Meem
    "\u1ee8C": "\u0645",  # Arabic Mathematical Looped Meem
    "\u1eeaC": "\u0645",  # Arabic Mathematical Double-Struck Meem
    # Nun
    "\ufee5": "\u0646",  # Arabic Letter Noon Isolated Form
    "\ufee6": "\u0646",  # Arabic Letter Noon Final Form
    "\ufee7": "\u0646",  # Arabic Letter Noon Initial Form
    "\ufee8": "\u0646",  # Arabic Letter Noon Medial Form
    "\u1ee0D": "\u0646",  # Arabic Mathematical Noon
    "\u1ee2D": "\u0646",  # Arabic Mathematical Initial Noon
    "\u1ee4D": "\u0646",  # Arabic Mathematical Tailed Noon
    "\u1ee6D": "\u0646",  # Arabic Mathematical Stretched Noon
    "\u1ee8D": "\u0646",  # Arabic Mathematical Looped Noon
    "\u1eeaD": "\u0646",  # Arabic Mathematical Double-Struck Noon
    # Vav
    "\u0677": "\u0648",  # Arabic letter Mid hamza on waw
    "\ufeed": "\u0648",  # Arabic Letter Waw Isolated Form
    "\ufeee": "\u0648",  # Arabic Letter Waw Final Form
    "\u06c6": "\u0648",  # Arabic Letter Oe
    "\u06c7": "\u0648",  # Arabic Letter U
    # He
    "\u06c0": "\u0647",  # Arabic letter Heh with yeh above
    "\u0629": "\u0647",  # Arabic Letter Teh Marbuta
    "\u06be": "\u0647",  # Arabic Letter Heh Doachashmee
    "\ufe93": "\u0647",  # Arabic Letter Teh Marbuta Isolated Form
    "\u06d5": "\u0647",  # Arabic Letter Ae
    "\ufee9": "\u0647",  # Arabic Letter Heh Isolated Form
    "\ufeea": "\u0647",  # Arabic Letter Heh Final Form
    "\ufeeb": "\u0647",  # Arabic Letter Heh Initial Form
    "\ufeec": "\u0647",  # Arabic Letter Heh Medial Form
    "\u1ee24": "\u0647",  # Arabic Mathematical Initial Heh
    "\u1ee64": "\u0647",  # Arabic Mathematical Stretched Heh
    "\u1ee84": "\u0647",  # Arabic Mathematical Looped Heh
    # Yeh
    "\u06d0": "\u06cc",  # Arabic letter Yeh with dot below
    "\ufeef": "\u06cc",  # Arabic Letter Alef Maksura Isolated Form
    "\ufef3": "\u06cc",  # Arabic Letter Yeh Initial Form
    "\ufef4": "\u06cc",  # Arabic Letter Yeh Medial Form
    "\u064a": "\u06cc",  # Arabic Letter Yeh
    "\ufef1": "\u06cc",  # Arabic Letter Yeh Isolated Form
    "\u06ce": "\u06cc",  # Arabic Letter Yeh with Small V
    "\ufbfd": "\u06cc",  # Arabic Letter Farsi Yeh Final Form
    "\ufbfc": "\u06cc",  # Arabic Letter Farsi Yeh Isolated Form
    "\ufbfe": "\u06cc",  # Arabic Letter Farsi Yeh Initial Form
    "\ufbff": "\u06cc",  # Arabic Letter Farsi Yeh Medial Form
    "\ufef0": "\u06cc",  # Arabic letter Lam final form
    "\ufef2": "\u06cc",  # Arabic letter Lam medial form
    "\u063d": "\u06cc",
    "\u063e": "\u06cc",
    "\u063f": "\u06cc",
    "\u06d2": "\u06cc",  # Arabic Letter Yeh Barree
    "\u064e": "",
    "\u064b": "",
    "\u064f": "",
    "\u064c": "",
    "\u0650": "",
    "\u064d": "",
    "\u0652": "",
    "\u0651": "",
    "\u0654": "",
    "0": "۰",
    "1": "۱",
    "2": "۲",
    "3": "۳",
    "4": "۴",
    "5": "۵",
    "6": "۶",
    "7": "۷",
    "8": "۸",
    "9": "۹",
    "٠": "۰",
    "١": "۱",
    "٢": "۲",
    "٣": "۳",
    "٤": "۴",
    "٥": "۵",
    "٦": "۶",
    "٧": "۷",
    "٨": "۸",
    "٩": "۹",
    "٬": "،",
    ",": "،",
    ";": "؛",
    "?": "؟",
    "\\": " ",
    "…": " غیره ",
    "%": " درصد ",
    "\u200e": " ",  # LEFT-TO-RIGHT
    "\u200f": " ",  # RIGHT-TO-LEFT
    "\u202a": " ",  # LEFT-TO-RIGHT EMBEDDING
    "\u202b": " ",  # RIGHT-TO-LEFT EMBEDDING
    "\u2066": " ",  # LEFT-TO-RIGHT ISOLATE
    "\u2067": " ",  # RIGHT-TO-LEFT ISOLATE
    "\u2069": " ",  # POP DIRECTIONAL ISOLATE
    "\ufdef": " ",  # Non-standard
    "\u00b7": ".",  # MIDDLE DOT
    "\u2022": " ",  # BULLET POINT
    "'": " ",
    "“": " ",
    "”": " ",
    "\u00ad": " ",
    "\u005f": " ",
    "\u002b": " ",
    "\u200b": " ",
    # ©
    "\u00a9": " ",
    "\u2014": " ",  # Em Dash
    "\u2019": " ",  # Right Single Quotation Mark
    "\ufe0f": "",  # Variation Selector-16 (VS16)
    "\u007c": " ",  # Vertical Line
}


class KenlmModel:
    def __init__(
        self,
        vocabulary_size: str,
        ngram: str,
        pruning: str,
        map_to_farsi_alphabet: bool = True,
        normalize_nfd: bool = True,
        normalize_numbers: bool = True,
        remove_puctuation: bool = True,
        remove_non_farsi: bool = True,
    ):
        self.model = kenlm.Model(
            os.path.join(
                "files", f"jomleh-sp-{vocabulary_size}-o{ngram}-prune{pruning}.probing"
            )
        )
        self.tokenizer = spm.SentencePieceProcessor(
            os.path.join("files", f"jomleh-sp-{vocabulary_size}.model")
        )

        norm_list = []
        if map_to_farsi_alphabet:
            norm_list += [
                normalizers.Replace(key, value) for key, value in char_map.items()
            ]
        if normalize_nfd:
            norm_list += [normalizers.NFD()]
        if normalize_numbers:
            norm_list += [normalizers.Replace(Regex("[۱۲۳۴۵۶۷۸۹]"), "۰")]
        if remove_puctuation:
            norm_list += [normalizers.Replace(Regex("[\\.!؛،؟]"), "")]
        if remove_non_farsi:
            norm_list += [
                normalizers.Replace(
                    Regex(
                        "[^\u060c\u061b\u061f\u0622\u0623\u0624\u0626\u0627"
                        "\u0628\u062a\u062b\u062c\u062d\u062e\u062f\u0630\u0631"
                        "\u0632\u0633\u0634\u0635\u0636\u0637\u0638\u0639\u063a"
                        "\u0641\u0642\u0644\u0645\u0646\u0647\u0648\u067e\u0686"
                        "\u0698\u06a9\u06af\u06cc\u06f0\u06f1\u06f2\u06f3\u06f4"
                        "\u06f5\u06f6\u06f7\u06f8\u06f9\\s\u200c\\.\\!]"
                    ),
                    "",
                )
            ]
        norm_list += [normalizers.Strip()]

        self.normalizer = normalizers.Sequence(norm_list)

    @classmethod
    def from_pretrained(
        cls,
        vocabulary_size: str,
        ngram: str,
        pruning: str,
        map_to_farsi_alphabet: bool = True,
        normalize_nfd: bool = True,
        normalize_numbers: bool = True,
        remove_puctuation: bool = True,
        remove_non_farsi: bool = True,
    ):
        return cls(
            vocabulary_size,
            ngram,
            pruning,
            map_to_farsi_alphabet,
            normalize_nfd,
            normalize_numbers,
            remove_puctuation,
            remove_non_farsi,
        )

    def score(self, doc: str):
        doc = self.normalizer.normalize_str(doc)
        doc = " ".join(self.tokenizer.encode(doc, out_type=str))
        return self.model.score(doc)

    def perplexity(self, doc: str):
        doc = self.normalizer.normalize_str(doc)
        doc = " ".join(self.tokenizer.encode(doc, out_type=str))
        log_score = self.model.score(doc)
        length = len(doc.split()) + 1
        return round(10.0 ** (-log_score / length), 1)
