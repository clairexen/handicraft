from dataclasses import dataclass, field

@dataclass
class _LogiChatConfig:
    name: str
    gates: tuple = ("BUF", "NOT", "AND", "NAND", "OR", "NOR", "XOR", "XNOR", "ANDNOT", "ORNOT",
            "MUX", "NMUX", "AOI3", "OAI3", "AOI4", "OAI4", "LUT2", "LUT3", "LUT4", "LUT5", "LUT6")
    idx_base: int = 8
    max_nbits: int = 2
    shift_altgr: bool = False
    with_words: bool = False
    with_dbls: bool = False
    with_tris: bool = False
    tokenizer = None

    def lex(self):
        if self.tokenizer is None:
            import tokens
            self.tokenizer = tokens.Tokenizer(self)
        return self.tokenizer

cfg_large = _LogiChatConfig(
    "large",
    max_nbits = 4,
    with_words = True,
    with_dbls = True,
    with_tris = True,
)

cfg_medium = _LogiChatConfig(
    "medium",
    max_nbits = 4,
)

cfg_small = _LogiChatConfig(
    "small",
)

cfg_tiny = _LogiChatConfig(
    "small",
    shift_altgr = True
)

# active (default) config
cfg = cfg_large

cfgs = {
     "large": (cfg_large,  'Large'  + (" (default)" if cfg == cfg_large  else "")),
    "medium": (cfg_medium, 'Medium' + (" (default)" if cfg == cfg_medium else "")),
     "small": (cfg_small,  'Small'  + (" (default)" if cfg == cfg_small  else "")),
      "tiny": (cfg_tiny,   'Tiny'   + (" (default)" if cfg == cfg_tiny   else "")),
}
