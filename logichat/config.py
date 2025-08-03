from dataclasses import dataclass, field

@dataclass
class _LogiChatConfig:
    name: str
    gates: tuple = ("BUF", "NOT", "AND", "NAND", "OR", "NOR", "XOR", "XNOR", "ANDNOT", "ORNOT",
            "MUX", "NMUX", "AOI3", "OAI3", "AOI4", "OAI4", "LUT2", "LUT3", "LUT4", "LUT5", "LUT6")
    idx_base: int = 10
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
            lex = self.lex()
            from nanoGPT.model import GPTConfig
            self._gptcfg = GPTConfig()
            self._gptcfg.vocab_size = lex.meta.vocab_size
            self._gptcfg.modif_size = lex.meta.modif_size
            self._gptcfg.n_layer = self.gpt_n_layer
            self._gptcfg.n_head  = self.gpt_n_head
            self._gptcfg.n_embd  = self.gpt_n_embd
            self._gptcfg.dropout = self.gpt_dropout
            self._gptcfg.bias    = self.gpt_bias
        return self._gptcfg

    def gptname(self):
        return f"GPT2M-L{self.gpt_n_layer}-H{self.gpt_n_head}-E{self.gpt_n_embd}-" + \
                        f"D{f'{dropout:.2f}'[2:]}-B{1 if self.gpt_bias else 0}"

cfg_small = _LogiChatConfig(
    "small",
)

cfg_large = _LogiChatConfig(
    "large",
    idx_base = 100,
    max_nbits = 4,
    with_words = True,
)

# active (default) config
cfg = cfg_large

cfgs = {
     "large": (cfg_large,  'Large'  + (" (default)" if cfg == cfg_large  else "")),
     "small": (cfg_small,  'Small'  + (" (default)" if cfg == cfg_small  else "")),
}
