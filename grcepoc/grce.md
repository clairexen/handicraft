# Gradiend-less Recurrent Context Encoding (GRCE)

Prompt:

Ich lese "Attention is All you need". Ich frage mich was wäre, wenn man zusätzlich zu den right shifted predicted output symbols auch noch teile des internen states von output symbol N, an den Eingang von symbol N+1 anhängt. Ich denke da speziell an den eingang zum letzten feed-forward netzwerks für symbol N. Man könnte ein gleich dimensioniertes feed-forward Netzwerk an den eingang von symbol N+1 anhängen, und damit eine weitere lineare komponente errechnen, die wie das positional encoding zum input embedding dazugezählt wird. Ich nenne diese weitere lineare komponente "kontext encoding".

Eine erweiterung/generalisierung davon, wäre, Input so wie output zu behandeln, nur mit dem unterschied, dass die predicted output symbols verworfen werden, aber "kontext encoding" wird berechnet und durchgereicht. Damit würde das Netzwerk den Input-Text dann nicht nur "sehen" sondern "lesen".

Wichtige Punkte:
- Das ziel ist explizit nicht besseres "Langzeitgedächtnis" sondern besserer und stabilerer "Fokus" und "halten der Rolle", ein bisschen in Anlehnung an die funktion von SST und VIP Interneuronen im cortex. Also weitergeben von jenen Informationen, von einem Symbol zum nächsten, die immer "verfügbar" sein sollten, und daher keine "resource-pressure/conflicts" haben sollte, mit zB attention heads.
- Ich postuliere, dass Lernen über die Symbol-Boundary ist explizit _nicht_ notwendig ist. Der input-vektor zum letzten letzten feed-forward network zum letzten symbol sollte bereits alle notwendigen Informationen beinhalten, inklusive das vorhergehende kontext encoding, am eingang vom letzten symbol, das zum input encoding dazu addiert wurde.

In diesem Sinne, insbesondere wenn nicht über die Symbol-Boundary gelernt wird, ist es natürlich sinnvoll, den "Kontext Vektor" in der Implementierung explizit mitzuführen, und den neuen "Kontext Vektor" mittels Gating über den alten Vektor zu implementieren.

Formalisiere diese Idee. Schreibe eine Zusammenfassung im richtigen Fachjargon. Führe eine Literaturrecherche durch.

---

## Formalisierung: “Context Encoding” als expliziter rekurrenter Zustandskanal

Wir betrachten einen decoder-only Transformer (wie in *Attention Is All You Need*) mit Modellbreite (d), Layern (\ell=1..L). Für Tokenposition (t) sei

* (e_t \in \mathbb{R}^d) das Token-Embedding (right-shifted Teacher Forcing),
* (p_t \in \mathbb{R}^d) das Positional Encoding (oder RoPE/relativ),
* (x_t = e_t + p_t) der Standard-Input pro Position.

Du fügst einen zusätzlichen, expliziten Zustandsvektor (c_t \in \mathbb{R}^{d_c}) ein (oft (d_c=d)), der *zwischen* Zeitschritten weitergereicht wird und als additive “Kontext-Bias”-Komponente (dein “kontext encoding”) in den nächsten Token-Input eingeht.

### 1) Input-Augmentation (Kontext-Encoding additiv wie Position)

Definiere eine Projektion (\Pi:\mathbb{R}^{d_c}\to\mathbb{R}^d), z.B. linear:
[
k_t = \Pi(c_{t-1}) \in \mathbb{R}^{d}.
]
Dann wird der Eingangsvektor:
[
x_t = e_t + p_t + k_t.
]
Interpretation: (k_t) ist ein *persistenter*, positionsunabhängiger Bias, der global “fokussierende / rollenhaltende” Information trägt.

### 2) “Write”-Pfad: Kontext aus internem State extrahieren

Wähle den internen Vektor, aus dem du schreibst, z.B. **Pre-FFN** des letzten Blocks an Position (t):

* Standardblock (Pre-LN): (h^{\ell}_t = h^{\ell-1}_t + \mathrm{MHA}(\mathrm{LN}(h^{\ell-1}_t))), dann (h^{\ell}_t = h^{\ell}_t + \mathrm{FFN}(\mathrm{LN}(h^{\ell}_t))).
  Nimm den Input in das letzte FFN, z.B.
  [
  u_t := \mathrm{LN}(h^{L}_t) \in \mathbb{R}^d.
  ]
  Dann berechne einen Kandidatenzustand (\tilde c_t) über ein (gleichdimensionales) MLP/FFN:
  [
  \tilde c_t = W_2 ,\sigma(W_1 u_t) \in \mathbb{R}^{d_c}.
  ]
  (Genau das ist dein “gleich dimensioniertes feed-forward Netzwerk”.)

### 3) Gating / Leak-Integration (SST/VIP-Analogie als “Stabilitätskanal”)

Update über ein Gate (g_t\in(0,1)^{d_c}):
[
g_t = \sigma(W_g u_t + b_g),
]
[
c_t = g_t \odot c_{t-1} + (1-g_t)\odot \tilde c_t.
]
Das ist eine per-Dimension “leaky integrator”-Dynamik: Inhalte, die “immer verfügbar” sein sollen, können im Gate in Richtung *retain* laufen (hohes (g_t)), während kurzfristige Anpassungen über (\tilde c_t) einfließen.

### 4) Kein Lernen “über die Symbol-Boundary” (Stop-Gradient / Truncated BPTT)

Deine Postulierung entspricht einer expliziten Entkopplung der Gradienten über den rekurrenten Kanal:
[
x_t = e_t + p_t + \Pi(\mathrm{stopgrad}(c_{t-1})).
]
Damit kann der Forward-Pfad weiterhin *Fokus* halten, aber der Backprop muss nicht durch die Zeit über (c) laufen (stabiler, weniger Credit-Assignment über Zeit). Das ist konzeptionell nahe an “truncated BPTT” – nur dass du den RNN-artigen Teil bewusst als *steuernden Bias-Kanal* nutzt, nicht als Hauptgedächtnis.

> Wichtig: Auch ohne Gradienten “durch die Zeit” kann (c) **nützlich** sein, weil (u_t) ja bereits (per Attention) den bisherigen Kontext sieht und lernen kann, was in (c_t) stabil abgelegt werden sollte. Du verlierst “über-Zeit Credit Assignment” speziell für den (c)-Kanal, nicht die normale Kontextverarbeitung innerhalb des Transformers.

---

## Generalisierung: “Input lesen” (Streaming-Encoder über denselben Mechanismus)

Dein “Input wie Output behandeln, aber predicted output symbols verwerfen” ist effektiv ein **streaming / recurrent preprocessing pass**, der aus dem Eingabetext einen laufenden Zustand (c_t) destilliert.

Eine saubere Formulierung ist ein zweiphasiger Modus:

1. **Read-Pass (keine Token-Emission):**
   Du fütterst echte Eingabetokens (s_1..s_T) als (e_t), berechnest fortlaufend (c_t) wie oben, aber ignorierst Logits. Ergebnis: Endzustand (c_T) (oder eine Sequenz (c_{1..T})).

2. **Generate-Pass (normal autoregressiv):**
   Initialisiere (c_0 := c_T) (oder eine Funktion davon) und generiere Ausgabetokens (y_1..y_N) mit (x_t = e_t+p_t+\Pi(c_{t-1})).

Das ist ein **expliziter “state channel”** parallel zur Attention, der für Rollen-/Fokus-Information reserviert ist (also genau dein “keine resource-pressure/conflicts mit attention heads”).

---

## Fachjargon-Zusammenfassung (kurz, “paper-style”)

Du schlägst einen **explicit recurrent context state** (c_t) vor, der **additiv** als kontextuelles Encoding in die Token-Repräsentation eingespeist wird (ähnlich Positional Encoding), und der pro Zeitschritt aus einem späten internen Repräsentationspunkt (z.B. Pre-FFN im letzten Block) über ein **separates FFN** geschrieben wird. Ein **gated update** implementiert eine *slow-changing control subspace* (Rollen-/Fokus-Track), während die reguläre Self-Attention weiterhin den “working memory / content”-Track trägt. Optional wird der Gradientenfluss über den rekurrenten Kanal via **stop-gradient** unterbunden, um Credit Assignment über Symbolgrenzen zu vermeiden und Stabilität zu erhöhen.

---

## Literaturrecherche: Wo das schon ähnlich auftaucht (und wie es sich unterscheidet)

### A) Segment-/State-Rekurrenz bei Transformers (nächste Nachbarn)

* **Transformer-XL** führt *Segment-level recurrence* ein (Memory aus vorherigem Segment wird als Keys/Values wiederverwendet) und adressiert “context fragmentation”. Das ist recurrence, aber *über Attention/KV*, nicht als additiver Bias-Kanal wie bei dir. ([arXiv][1])
* **Recurrent Memory Transformer (RMT)** nutzt dedizierte **Memory Tokens**, die zwischen Segmenten weitergereicht werden. Das ist deinem Ziel (“konfliktfreie Ressource”) ziemlich nah – nur dass die Memory als *Tokens* im Attention-Raum lebt, nicht als separater Vektor, der wie PosEnc addiert wird. ([arXiv][2])
* **Block-Recurrent Transformers**: rekurrente Zustände über Blöcke, um lineare Komplexität zu bekommen (und eine RNN-artige Induktionsbias). ([arXiv][3])

**Dein Unterschied:** Du willst keinen “mehr Kontext” primär, sondern einen *kontrollierten persistenten Fokus-Kanal*, und du positionierst ihn als **additiven** Term am Input (ähnlich “global bias”), plus optionalem stop-grad über Zeit.

### B) “Recurrence ohne Quadratik”: Retention / RNN-Transformer-Hybride

* **RetNet (Retentive Networks)** ersetzt Attention durch eine Retention-Form, die parallel trainierbar und rekurrent inferierbar ist; dort ist der “State” zentraler Bestandteil der Architektur. ([arXiv][4])
  Das ist konzeptionell verwandt (explizite Rekurrenz), aber RetNet ist eher “neue Hauptmechanik” statt “zusätzlicher Fokus-Kanal”.

### C) “Stabilität/Fokus”-Phänomene in heutigen LLMs: Attention Sinks & Persistenz-Tricks

* **Attention Sinks / StreamingLLM-Idee**: Es gibt empirische Arbeiten, dass frühe Tokens (z.B. BOS) als “Sink” massiv Aufmerksamkeit ziehen und dass man das für **stabile Streaming-Inferenz** nutzen kann. Das ist *ein natürlicher, emergenter Fokus-Anker* im Attention-System. ([arXiv][5])
  Dein Ansatz wäre ein *konstruktiver* Anker statt ein emergentes Artefakt.
* Arbeiten zur Entstehung/Analyse von Attention Sinks (“When attention sink emerges …”) legen nahe, dass FFN/Outlier-Dynamiken hier eine Rolle spielen – interessant, weil du den Schreibpunkt explizit beim FFN siehst. ([smcnus.comp.nus.edu.sg][6])

### D) Soft-Prompts / Prefix / “always-on context”

* **Prompt-/Prefix-Tuning** verwendet trainierbare **virtuelle Tokens/Embeddings** als dauerhaftes Conditioning. ([IBM][7])
  Das ist persistent, aber **statisch pro Task** (oder pro Beispiel, wenn man es dynamisch lernt). Dein (c_t) ist **dynamisch** und wird online geschrieben.

---

## Ein sinnvoller Name / Einordnung (damit’s zitierfähig klingt)

Wenn du das “paper-ready” labeln willst:

**Gradient-less Recurrent Context Encoding (GRCE)**:
ein separater, gated, explizit rekurrenter Zustandskanal, der als additiver Kontextbias in die Token-Inputs eingespeist wird; optional mit **stop-gradient across time**.

---

[1]: https://arxiv.org/abs/1901.02860?utm_source=chatgpt.com "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context"
[2]: https://arxiv.org/abs/2207.06881?utm_source=chatgpt.com "[2207.06881] Recurrent Memory Transformer"
[3]: https://arxiv.org/html/2512.25026v1?utm_source=chatgpt.com "Modeling Language as a Sequence of Thoughts"
[4]: https://arxiv.org/pdf/2307.08621?utm_source=chatgpt.com "Retentive Network"
[5]: https://arxiv.org/html/2309.17453v3?utm_source=chatgpt.com "Efficient Streaming Language Models with Attention Sinks"
[6]: https://smcnus.comp.nus.edu.sg/archive/pdf/2025/2025_when_attention.pdf?utm_source=chatgpt.com "when attention sink emerges"
[7]: https://www.ibm.com/think/topics/prompt-tuning?utm_source=chatgpt.com "What is prompt tuning?"

- ## Proof-of-Concept: picoGPT + SimpleWiki + GRCE
-
- Der Ordner enthält jetzt ein kleines **picoGPT-inspiriertes Demo** (`grce_pico_poc.py`), das die Idee oben praktisch macht:
-
- **Architektur:** GPT-ähnliches Decoder-Modell mit 8 Layern, 8 Köpfen, 512 Embedding-Dimensionen (vgl. GPT-2 base mit 12/12/768). Der explizite GRCE-Kanal bleibt wie beschrieben und nutzt `context_dim = n_embd`; sein Schreib-FFN verarbeitet den aktuellen Token-Input (inkl. Kontext-Bias) plus **alle FFN-Ausgänge der vorherigen Position**. Die Geometrie folgt `(n_layer+1)*n_embd → n_grce → 4·n_embd → n_embd`; hier endet der Gradientenfluss (Stop-Grad), sodass das rekurrente Signal als explizit kontrollierte Überlagerung geführt wird.
- **Tokenisierung:** Eine GPT-2-Style Byte-Level-BPE (ByteLevel + BPE-Trainer) wird aus dem Trainings-Text gelernt (`--tokenizer-vocab` bestimmt die Vokabulargröße). Das resultierende Tokenizer-JSON landet in `tokenizer/` und wird wiederverwendet.
- **Daten:** Default ist das bereitgestellte Simple English Wikipedia Split (`data/simplewiki-train.asc`, `data/simplewiki-test.asc`). Für Quick-Tests kann die Menge via `--train-chars`/`--test-chars` begrenzt oder auf den alten Shakespeare-Schnipsel umgebogen werden.
- **Streaming + Regionen:** Zu Beginn jedes Trainingszyklus wird genau ein neues Chunk aus Train/Test eingelesen; beim Erreichen des Endes wird auf den Anfang “gewrappt”. Die zuletzt gelesenen Token- und Byte-Offsets werden im Modell-Checkpoint abgelegt, so dass der nächste Lauf nahtlos weiterliest. Beim Lesen printet das Skript die entsprechenden Byte-Bereiche (mit farblicher Hervorhebung im Terminal).
- **Regelmäßiges Testing:** Der Trainingsloop evaluiert nach jedem `--eval-interval`-ten Schritt sowohl auf `train` als auch `test` (mehrfach gemittelt über `--eval-iters` Batches). Damit ist klar sichtbar, wie stabil der zusätzliche Kontextkanal lernt.
- **Ausgabe:** Nach jedem Evaluationszyklus wird mit einem SimpleWiki-kompatiblen Prompt (default `"Bigotry is"`) eine einzeilige Probe erzeugt; Länge gesteuert durch `--generate`. Das Terminal nutzt Farben (weiß + fett für die Completion), während das `.log` farbfrei bleibt.
- **Persistenz & Logging:** Für jede Modellkonfiguration entsteht in `model/` ein Checkpoint (`*.pt`) plus ein gleichnamiges `.log`, das automatisch alle Konsolenausgaben spiegelt, oben Timestamp + komplette Kommandozeile schreibt und am Ende Wall/CPU-Zeit ergänzt. Bei erneutem Lauf mit denselben Parametern (inkl. Tokenizer) wird das gespeicherte Modell geladen, inklusive der Streaming-Offsets.

Verwendung (z.B. in einer lokalen venv):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python grce_pico_poc.py \
  --steps 80 --block-size 512 --batch-size 8 \
  --generate 300 --eval-interval 20 --eval-iters 5
```

Wichtige Flags:

- `--train-path` / `--test-path`: Textdateien für Training/Test (Default: SimpleWiki Split, auch in `.gitignore`).
- `--train-chars` / `--test-chars`: begrenzen die geladenen Zeichen (Suffixe wie `10k`, `2M` sind erlaubt).
- `--vocab-chars`: separater Ausschnitt für den Tokenizer (damit lässt sich ein kleines Vokabular mit denselben Kürzeln wie oben erzeugen).
- `--eval-interval`, `--eval-iters`: wie oft und wie intensiv der Testing-Pass läuft.
- `--cycles`: Anzahl der kompletten Trainingszyklen, die nacheinander ausgeführt werden.
- `--tokenizer-vocab`: Größe der GPT-2-Style BPE, die einmalig aus dem Trainingstext gelernt und in `tokenizer/` gecacht wird.
- `--context-dim`: Dimensionalität des GRCE-Zustands (Default 512, typischerweise = `n_embd`).
- `--prompt`: Starttext für die Generierung (Default `"Bigotry is"`).

Damit lässt sich experimentell nachvollziehen, wie der zusätzliche, gefensterte Kontextkanal das Modellverhalten beeinflusst (z.B. durch Variation der Kontextdimension, des Gatings oder durch Abschalten von `stop_grad` in der `GRCEContextChannel.project`-Methode).

### Hypothesen zur Dynamik des Kontextvektors

- **Überlagerte stabile vs. schnelle Komponenten:** Das rekurrente GRCE-Signal dürfte sich als Überlagerung zweier Einbettungsklassen äußern: (a) robuste, langsam variierende Komponenten, die globale Kontextmerkmale (Sprache, Rolle, Tonalität) tragen, und (b) schnell schaltende Komponenten, die lokale Informationen wie Satzposition oder grammatische Anforderungen für das nächste Token repräsentieren.
- **Mismatch-Erkennung:** Trifft das nächste Token nicht zu den verhergesagten Kontextmerkmalen, entstehen Inkompatibilitäten zwischen Kontextvektor und Token-Embedding, und dann verliert die Überlagerung dieser Vektoren an Bedeutung. Das Modell könnte diese Situation erkennen, um einen speziellen Alarm-Token zu emittieren. Training auf Eingaben mit absichtlich falschem Letzt-Token würde solche Fälle markieren und das Netzwerk auf diese Art konsistenzkritische Positionen sensibilisieren.
