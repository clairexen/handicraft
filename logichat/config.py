from dataclasses import dataclass, field

@dataclass
class LogiChatConfig:
    name: str
    gates: tuple = ("BUF", "NOT", "AND", "NAND", "OR", "NOR", "XOR", "XNOR", "ANDNOT", "ORNOT",
            "MUX", "NMUX", "AOI3", "OAI3", "AOI4", "OAI4", "LUT2", "LUT3", "LUT4", "LUT5", "LUT6")
    idx_digits: int = 2
    bits_blksz: int = 2
    with_words: bool = False

    # baby GPT model :)
    gpt_blksz = 1024
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
            self._gptcfg = GPTConfig()
            self._gptcfg.block_size = self.gpt_blksz
            self._gptcfg.vocab_size = self.lex().meta["vocab_size"]
            self._gptcfg.modif_size = self.lex().meta["modif_size"]
            self._gptcfg.n_layer = self.gpt_n_layer
            self._gptcfg.n_head  = self.gpt_n_head
            self._gptcfg.n_embd  = self.gpt_n_embd
            self._gptcfg.dropout = self.gpt_dropout
            self._gptcfg.bias    = self.gpt_bias
        return self._gptcfg

    def gptname(self):
        return "-".join([
            f"GPT2{'M' if self.gptcfg().modif_size else ''}",
            f"B{self.gpt_blksz}",
            f"L{self.gpt_n_layer}",
            f"H{self.gpt_n_head}",
            f"E{self.gpt_n_embd}",
            f"D{f'{self.gpt_dropout:.2f}'[2:]}",
            f"B{1 if self.gpt_bias else 0}"
        ])

cfg_small = LogiChatConfig(
    "small",
)

cfg_large = LogiChatConfig(
    "large",
    idx_digits = 3,
    bits_blksz = 4,
    with_words = True,
)

# active (default) config
cfg = cfg_large

cfgs = {
     "large": (cfg_large,  'Large'  + (" (default)" if cfg == cfg_large  else "")),
     "small": (cfg_small,  'Small'  + (" (default)" if cfg == cfg_small  else "")),
}

if __name__ == "__main__":
    print(cfg_large)
    print(cfg_large.gptcfg())
    print(cfg_large.gptname())
