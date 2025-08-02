from dataclasses import dataclass, field

@dataclass
class _LogiChatConfig:
    name: str
    gates: tuple = ("BUF", "NOT", "AND", "NAND", "OR", "NOR", "XOR", "XNOR", "ANDNOT", "ORNOT",
            "MUX", "NMUX", "AOI3", "OAI3", "AOI4", "OAI4", "LUT2", "LUT3", "LUT4", "LUT5", "LUT6")
    idx_base: int = 8
    max_nbits: int = 2
    with_words: bool = False

    # baby GPT model :)
    gpt_n_layer = 6
    gpt_n_head = 6
    gpt_n_embd = 384
    gpt_dropout = 0.2
    gpt_bias = True

    _lex = None
    _gptcfg = None

    def lex(self):
        if self._lex is None:
            import tokens
            self._lex = tokens.Tokenizer(self)
        return self._lex

    def gptcfg(self):
        if self._gptcfg is None:
            from nanoGPT.model import GPTConfig
            self._gptcfg = tokens.Tokenizer(self)
            self._gptcfg.vocab_size = 1234
            self._gptcfg.modif_size = 5 + idx_base
            self._gptcfg.n_layer = gpt_n_layer
            self._gptcfg.n_head  = gpt_n_head
            self._gptcfg.n_embd  = gpt_n_embd
            self._gptcfg.dropout = gpt_dropout
            self._gptcfg.bias    = gpt_bias
        return self._gptcfg

cfg_small = _LogiChatConfig(
    "small",
)

cfg_large = _LogiChatConfig(
    "large",
    idx_base = 64,
    max_nbits = 4,
    with_words = True,
)

# active (default) config
cfg = cfg_large

cfgs = {
     "large": (cfg_large,  'Large'  + (" (default)" if cfg == cfg_large  else "")),
     "small": (cfg_small,  'Small'  + (" (default)" if cfg == cfg_small  else "")),
}
