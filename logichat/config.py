from dataclasses import dataclass, field

@dataclass
class LogiChatConfig:
    gates: tuple = ("BUF", "NOT", "AND", "NAND", "OR", "NOR", "XOR", "XNOR", "ANDNOT", "ORNOT",
            "MUX", "NMUX", "AOI3", "OAI3", "AOI4", "OAI4", "LUT2", "LUT3", "LUT4", "LUT5", "LUT6")
    idx_base: int = 8
    max_nbits: int = 2
    shift_altgr: bool = False
    with_words: bool = False
    with_dbls: bool = False
    with_tris: bool = False

cfg_large = LogiChatConfig(
    max_nbits = 4,
    with_words = True,
    with_dbls = True,
    with_tris = True
)

cfg_medium = LogiChatConfig(
    max_nbits = 4,
)

cfg_small = LogiChatConfig(
    # default
)

cfg_tiny = LogiChatConfig(
    shift_altgr = True
)
