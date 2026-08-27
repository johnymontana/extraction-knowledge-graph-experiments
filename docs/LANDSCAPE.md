# Extraction and Entity Resolution for Knowledge Graphs: The Options

A reference survey of what you can actually use to get entities, relations, and canonical
nodes out of text, for two use cases that pull in different directions: **agent memory**
(incremental, conversational, latency-sensitive, read on every turn) and **document
intelligence** (batch, long documents, auditable, precision-sensitive).

Assumes you know graph databases. Does not assume you have picked an extractor.

**On numbers in this document.** Every figure is linked to its source. Figures published by
the vendor of the thing being measured, with no independent replication, are marked
**[vendor]**. Figures I could not find at all are marked **[no published number]** rather
than estimated. Benchmark F1 is a weak proxy for your corpus; treat every number here as a
prior to be overwritten by a measurement on your own data.

---

## 1. The decision, framed

There are three axes that actually determine the answer. Everything else — framework
choice, database choice, chunking strategy — is downstream.

### Axis 1: Open vs closed vocabulary

Is your entity/relation type set fixed and known, or does it change per corpus, per tenant,
per question?

- **Closed and small (≤ ~20 types), stable for years.** Classical supervised NER is still
  the strongest option and it isn't close. spaCy's `en_core_web_trf` gets
  [89.9 F1 on OntoNotes](https://github.com/explosion/spacy-models/blob/master/meta/en_core_web_trf-3.8.0.json)
  across its 18 fixed labels. No zero-shot model touches that on those labels.
- **Open, or closed-but-changing.** The type set is a runtime argument. This is the GLiNER
  family's entire reason to exist, and it is where LLMs are the obvious alternative.
- **Open and enormous** (hundreds to thousands of types — product taxonomies, ontology
  classes, ICD codes). A uni-encoder that concatenates labels with text degrades
  catastrophically; you need a bi-encoder with precomputed label embeddings, or retrieval
  over the label set first.

### Axis 2: Cost and latency budget, per document and per turn

Extraction cost is not one number. It's two:

- **Write-path cost** — dollars and seconds per document ingested. Batch document
  intelligence can absorb seconds and cents per document. Agent memory usually cannot,
  because ingestion sits between the user's message and the agent's reply.
- **Read-path cost** — for agent memory, the graph is queried on *every turn*. This
  usually dominates in aggregate and is the reason retrieval shape matters more than
  extraction quality past a certain point.

An encoder model at ~200M params runs on a CPU with no API key and no per-token billing.
A frontier LLM call is [$5/$25 per million input/output
tokens](https://docs.claude.com/en/docs/about-claude/pricing) for Claude Opus 5, $2/$10 for
Sonnet 5, halved again by the [Batch API](https://docs.claude.com/en/docs/build-with-claude/batch-processing),
and cache reads are ~0.1× input price. The gap is real but it is not the 1000× that
"local model vs API" rhetoric implies once you apply batching and caching — it is more like
one to two orders of magnitude, and it shrinks as your prompt prefix stabilizes.

### Axis 3: Do you need implicit facts?

This is the axis people skip, and it is the one that decides whether an encoder can do
your job at all.

> *"Alice was promoted last quarter, so she now reports to Bob."*

An extractive model finds spans. `reports_to(Alice, Bob)` is not a span — it is an
inference over the sentence. Every span-based extractor, GLiNER included, will miss it.
There is no threshold to tune, no label phrasing that recovers it, and no fine-tuning
budget that fixes it in general: the architecture answers "which characters in this text
are a `person`?", not "what is true?"

If your graph needs implicit edges, you need a generative model somewhere in the pipeline.

**Measured correction (notebook 03).** This axis is real but the line is not where the argument above puts
it. On a five-case probe, GLiNER2.5 recovered an implied relation, a cross-sentence syllogism and a bridged
referent — a relation classifier infers relations between given spans, and that is its job. What it cannot do
is emit an entity that appears nowhere in the text. The boundary is about **arguments**, not inference:
relations can be inferred, entity arguments must be anchored. The same probe found the encoder asserting a
relation an explicitly *negated* sentence denies, which the LLM did not — so the trade runs in both
directions.
If it needs only *stated* facts — which is most document intelligence and a surprising
amount of agent memory — you don't.

### Decision guide

| If… | Use |
|---|---|
| Types are fixed, English, PERSON/ORG/GPE-shaped | spaCy `en_core_web_trf` or Flair. Stop here. |
| Types vary at runtime, only stated facts needed, cost/privacy matter | GLiNER2.5 JointIE + a real ER stage |
| Types vary, you need implicit/inferential edges | LLM with strict tool schemas |
| High volume + you need implicit edges on *some* of it | Hybrid: encoder first pass, LLM escalation on flagged chunks |
| High volume, one domain, you have or can label a corpus | LLM-as-teacher → fine-tuned encoder |
| >100 entity types | GLiNER bi-encoder with precomputed label embeddings |
| Conversational, pronoun-dense input | Whatever you picked **plus** a coreference layer — see §4 |
| You need stable IDs across corpora, not just clusters | Add entity linking (ReFinED / Wikidata) on top |

---

## 2. Extraction options

| Option | Size | Quality signal | Cost/latency | Open vocab | Implicit facts | Typed-edge guarantee |
|---|---|---|---|---|---|---|
| GLiNER2.5-base | 194M | 54.87 macro-F1 / 16 tasks **[vendor]**; Few-NERD 55.14 | CPU, no API **[no published latency]** | yes | **no** | **yes** (JointIE) |
| GLiNER2.5-small | 74M | not independently measured | fastest | yes | no | yes |
| GLiNER2.5-multi | 287M | 56.17 macro **[vendor]**; XNLI 62.30 | ~1.5× base | yes | no | yes |
| GLiNER2 (2.0) | 205M / 340M | CoNLL04 32.9 / DocRED 11.7 e2e RE | CPU | yes | no | no — independent triples |
| GLiNER v1 (`gliner_large-v2.5`) | 300M | **CrossNER 60.9** | collapses at many labels (0.03 ex/s @1024) | yes | no | n/a (NER only) |
| GLiNER bi-encoder (large) | ~340M | **CrossNER 61.5** | **2.64 ex/s @1024 labels** (88× uni) | yes | no | n/a |
| GLiNER-Relex | DeBERTa-v3-large | **25.6 avg e2e RE micro-F1** | fast | yes | no | joint |
| NuNER Zero | 125M | +3.1 token-F1 vs GLiNER-large-v2.1 **[vendor]** | fast | yes | no | n/a |
| spaCy `en_core_web_trf` | ~110M+RoBERTa | **89.9 F1 OntoNotes** | GPU-preferred | **no** (18 types) | no | n/a |
| Flair / Stanza | 100–500M | CoNLL-03 SOTA class | slower than spaCy | no | no | n/a |
| LLM + strict tool schema | — | best available; no clean zero-shot RE benchmark | $2–5/MTok in, 1–5 s/chunk | yes | **yes** | schema-valid, not graph-valid |
| Hybrid escalation | — | — | 1× on the encoder half | yes | yes on escalated chunks | yes on the encoder half |
| LLM-as-teacher → fine-tuned encoder | 200M | approaches teacher on-domain | encoder speed after training | fixed at train time | partially distilled | yes |

### Encoder-based zero-shot

**GLiNER v1** ([paper](https://arxiv.org/abs/2311.08526), [repo](https://github.com/urchade/GLiNER))
is the origin: a bidirectional encoder that embeds entity-type *labels* alongside the text
and scores spans against them, so the type set is an inference-time argument. It reaches
[60.9 average F1 on CrossNER](https://arxiv.org/html/2602.18487v1) zero-shot. What it cannot
do is scale in the number of labels — the labels are concatenated into the same sequence as
the text, so cost grows with the *product* of text and label budget. At 1024 labels
`gliner_large-v2.5` drops to [0.03 examples/second](https://arxiv.org/html/2602.18487v1).

**Bi-encoder GLiNER** ([knowledgator/gliner-bi-*](https://huggingface.co/knowledgator/gliner-bi-large-v2.0))
splits the label encoder from the text encoder so label embeddings can be precomputed and
cached. `gliner-bi-large-v2.0` hits [61.5 CrossNER — slightly *better* than the uni-encoder's
60.9 — while sustaining 2.64 ex/s at 1024 labels, an 88× throughput
improvement](https://arxiv.org/html/2602.18487v1); the smaller `bi-edge` variant reaches
18.3 ex/s at the same label count, and degradation from 1→1024 labels is 5.2% versus 98.7%
for the uni-encoder. If your type vocabulary exceeds roughly a hundred entries, this is not
an optimization, it is the only architecture that works. The tradeoff: labels can no longer
attend to the text, so genuinely context-dependent type distinctions get weaker.

**GLiNER2** ([paper](https://arxiv.org/abs/2507.18546),
[repo](https://github.com/fastino-ai/GLiNER2)) generalized the architecture into a
multi-task schema interface: NER, classification, JSON record extraction, and relation
extraction composed into one schema and one forward pass. Its relation extraction is the
weak point — triples are scored independently, so a relation can reference an entity that
fell below threshold, cardinality is unenforced, and hierarchies can contain cycles. It also
enumerates spans up to a fixed width (~12 words), which makes longer entities — addresses,
contract clauses — *structurally invisible* rather than merely low-scoring.

**GLiNER2.5** ([blog](https://fastino.ai/blog/gliner2-5-span-free-information-extraction))
replaces span enumeration with boundary prediction (score start and end positions
directly), which removes the width cap entirely and makes cost linear in sequence length,
enabling 4,096-word inputs. It adds **JointIE**: entities and relations decoded together
under schema constraints (`unique_head`, `no_self_loops`, `acyclic`) enforced during beam
search, so the returned object is a well-formed typed graph rather than a bag of triples
you must validate. Three Apache-2.0 checkpoints: `small-v1` (74M), `base-v1` (194M),
`multi-v1` (287M). See §7 for the candid assessment.

**NuNER** ([paper](https://aclanthology.org/2024.emnlp-main.660.pdf),
[repo](https://github.com/Serega6678/NuNER)) is the same idea trained on an LLM-annotated
pretraining corpus; `NuNER_Zero` uses the GLiNER architecture as a token classifier
(arbitrary-length entities) and reports [+3.1 token-level F1 over
GLiNER-large-v2.1](https://github.com/Serega6678/NuNER) **[vendor]**. Its practical value
is less zero-shot performance than being an unusually strong *starting point for
fine-tuning* — its pretraining objective is explicitly "be a good NER encoder to adapt."

**Calibration warning on zero-shot relation extraction.** Zero-shot *NER* numbers in the
55–61 F1 range are respectable. Zero-shot end-to-end *relation* extraction numbers are not
in that range for anyone. [GLiNER-Relex](https://arxiv.org/html/2605.10108v1) reports an
average of **25.6 micro-F1** across CoNLL04/DocRED/FewRel/CrossRE, against 17.8 for GLiNER2
and 22.1 for GPT-5-mini. Strict triple-level zero-shot extraction is a hard, unsolved task
regardless of which family you pick — which is a strong argument for constraining the
schema hard and for treating extraction output as candidate evidence, not as truth.

### Closed-vocabulary classical

**spaCy `en_core_web_trf`** fine-tunes RoBERTa-base and scores
[P 0.897 / R 0.901 / F 0.899 on OntoNotes](https://github.com/explosion/spacy-models/blob/master/meta/en_core_web_trf-3.8.0.json)
across exactly 18 labels (`PERSON`, `ORG`, `GPE`, `DATE`, `MONEY`, `LAW`, `WORK_OF_ART`, …).
The CNN pipelines trade accuracy for speed: `en_core_web_lg` 0.855, `en_core_web_sm` 0.843.
**Flair** and **Stanza** occupy the same niche with slightly different accuracy/speed
points — Flair's contextual string embeddings were CoNLL-03 SOTA for a while and remain
competitive; Stanza is the option when you need many languages with consistent tokenization
and dependency parses. The honest framing: if your ontology is PERSON/ORG/GPE plus a couple
of regex-shaped types, these beat every zero-shot model by 25–30 F1 points and cost nothing
to run. **What they cannot do** is accept a new type at runtime. Adding one label means a
labeling project.

### LLM structured output

**Claude with strict tool schemas.** Define an extraction tool with `strict: true` on the
tool definition (requires `additionalProperties: false` and a complete `required` list) and
the returned `tool_use.input` is guaranteed to validate against your JSON Schema. That
guarantee is *schema* validity, not *graph* validity — nothing stops the model from emitting
a `works_for` edge whose tail is a city, or two employers where your ontology allows one.
Encode graph constraints in the schema where you can (enums for endpoint types, `maxItems`
for cardinality) and post-validate the rest. Cost levers, in order of impact:
[prompt caching](https://docs.claude.com/en/docs/build-with-claude/prompt-caching) (put the
ontology and few-shot examples in a stable prefix; cache reads are ~0.1× input price), the
[Batch API](https://docs.claude.com/en/docs/build-with-claude/batch-processing) (50% off,
asynchronous — perfect for document intelligence, useless for agent-memory write paths),
and model tiering (Haiku 4.5 at $1/$5 for easy chunks).

**instructor** wraps Pydantic models around provider-native structured output with
validation and retries. It is the lowest-friction option in a Python codebase and its
failure mode is honest: strict JSON parsing, retry on validation error.
**[BAML](https://boundaryml.com/blog/structured-output-from-llms)** takes a different bet —
a DSL for prompt-as-typed-function plus *Schema-Aligned Parsing*, which recovers structured
data from output that is not valid JSON (chain-of-thought before the object, markdown
fences, trailing commas). BAML publishes accuracy and 2–4× latency advantages over
FC-strict JSON tools **[vendor, and the comparison is against OpenAI function calling, not
against Anthropic strict tools]**. The practical read: if your model provider has real
constrained decoding, `strict: true` plus instructor is enough; BAML's parser earns its
keep with weaker or self-hosted models, and its cross-language codegen is a genuine
differentiator if your stack isn't Python.

**The thing LLM extraction cannot do**: run for free, run offline, run in 150 ms, or be
deterministic. Two runs over the same paragraph will differ. For a knowledge graph this is
not a cosmetic problem — it means your graph is not reproducible from your corpus, and
re-ingestion silently changes the node set. Budget for it (store extraction outputs, not
just the graph; version the prompt alongside the data).

### Hybrid escalation and distillation

**Hybrid escalation** is the recommended production shape and it is underused. Run the
encoder on everything; escalate to the LLM only on chunks that trip a trigger — low mean
entity confidence, zero relations extracted from a paragraph dense with entities, presence
of inferential discourse markers (*so*, *therefore*, *as a result*, *which means*), or a
schema slot that came back empty on a document type that always fills it. The economics
work because the escalation *rate* is the only thing that matters: at a 20% escalation rate
you pay 20% of the LLM bill and recover most of the implicit facts, because implicit facts
are not uniformly distributed across your corpus. I have not found a published,
independently-replicated study of escalation rate vs. recall recovery; measure it on your
own corpus (§8).

**LLM-as-teacher distillation** is the strongest quality-per-dollar point *if you have a
corpus and a stable domain*. Label a few hundred examples by hand as a calibration set,
have a frontier model label 10⁵–10⁶ documents in Batch mode, fine-tune the encoder on the
result, and serve the encoder. The pattern is well-established (see e.g. NVIDIA's
[financial-data distillation blueprint](https://developer.nvidia.com/blog/build-efficient-financial-data-workflows-with-ai-model-distillation/),
where student F1 converges toward the teacher as dataset size grows). What it distills is
*teacher behavior on your distribution*, including some inferential patterns that are
regular enough to be learned as surface cues — but not general reasoning. And you trade
back the open vocabulary: the fine-tuned model is now good at your labels specifically.

---

## 3. Entity resolution options

Extraction gives you document-local mentions. ER is what makes them a graph. It is
consistently the part teams under-build, and it is where graphs fail visibly — one
over-merged node poisons every query that touches it.

| Option | Stage | Scale | The tradeoff |
|---|---|---|---|
| Exact match on `(type, normalized)` | normalize | free | Collapses a large fraction at zero cost. Always first. |
| rapidfuzz | score | ~10³ pairs/s/core | Bit-parallel Jaro-Winkler/token-sort. **Scoring only, never blocking.** Breaks on low-entropy names. |
| Embedding + ANN | block + score | to 10⁶ with FAISS/HNSW | Catches acronym↔expansion with no rules. **Must be type-constrained.** |
| MinHash + LSH | block | very large | Deterministic, cheap, no model. What Graphiti uses. |
| Connected components | cluster | trivial | What everyone ships. **Transitively greedy.** |
| Hierarchical / correlation clustering | cluster | costlier | Survives one bad edge; global threshold tunable post hoc. |
| Splink 4 | full pipeline | 1M/laptop/min; 100M+ on Spark | Fellegi–Sunter, EM-estimated, auditable. **Wants multi-field records.** |
| dedupe | full pipeline | small–medium | Active learning; documented memory limits at scale. |
| Zingg | full pipeline | lakehouse | JVM/Spark ops burden. |
| LLM adjudication | final tier only | volume = blocking looseness | Good judgment; non-idempotent; needs provenance. |
| AnyMatch (fine-tuned GPT-2) | scoring | tiny | 81.96 mean F1 across 9 EM benchmarks at ~3,899× lower inference cost than MatchGPT. |
| ReFinED / entity linking | canonical IDs | Wikidata-scale | Real KB ids, not just clusters. >60× faster than comparable systems. 2022-era. |

### The pipeline that works

**Normalize.** NFKC, whitespace collapse, casefold for the key while keeping the original
surface for display. Type-specific rules: move corporate legal suffixes (`Inc`, `Ltd`,
`GmbH`, `N.V.`) into a separate field rather than deleting them — they are *evidence*, and
"Acme Inc" vs "Acme LLC" is a real distinction. Split person names into components and
record whether a mention was first-name-only. Expand within-document acronyms by initial
matching. Exact match on `(type, normalized)` at this stage is free and does most of the
work.

**Block.** Never do an all-pairs sweep — it's quadratic and rapidfuzz at ~10³ full-record
pairs per second per core makes that fatal above a few thousand mentions. Two blockers,
unioned: a deterministic key (phonetic code of the head token, sorted-token-set hash) and
embedding ANN. Embed `"<type>: <normalized> || <sentence context>"` rather than the bare
name — context is what separates *Apple* the company from *Apple* the product, and those
two strings are identical.

> **Blocking recall is the hard ceiling on the entire system.** A pair that never enters the
> candidate set can never be matched, and no amount of downstream scoring, clustering, or
> LLM adjudication recovers it. Measure **pair completeness** (fraction of true-match pairs
> surviving blocking) separately from end-to-end F1, and report it next to **reduction
> ratio**. If you report only one ER number, report this one.

**Score.** Combine string similarity (rapidfuzz `token_sort_ratio` / `WRatio` /
`partial_ratio`), embedding cosine, type agreement, and context overlap into a single score
with visible weights.

> **Low-entropy names break string similarity.** Short, repetitive, or acronym-like names
> produce spuriously high fuzzy scores against each other — "Ann"/"Anna"/"Anne", "ACME"/"ACM".
> Graphiti's production answer is exactly this: compute
> [approximate Shannon entropy over the characters of the normalized
> name](https://blog.getzep.com/graphiti-hits-20k-stars-mcp-server-1-0/); below a threshold,
> skip the heuristics entirely and send the pair to the LLM; above it, use deterministic
> MinHash-over-3-gram-shingles + LSH banding and accept Jaccard ≥ 0.9. That inversion — cheap
> path for *hard-to-confuse* names, expensive path for *easy-to-confuse* ones — is the
> correct shape and worth copying.

**Cluster.** Connected components (`networkx`) is what almost everyone ships, and it is
**transitively greedy**: a single 0.91 edge between two 40-mention clusters merges 80
mentions. Two non-optional guards: a max-cluster-size alarm that prints the component's
edges sorted by score ascending (the weakest edge is nearly always the culprit), and a
threshold sweep. Both are cheap *if you logged every scored pair before clustering* —
re-clustering is free, re-scoring is not. Hierarchical or correlation clustering is more
robust to one bad edge and lets you move the global threshold after the fact; it costs more
and is worth it once your graph is load-bearing.

**Evaluate with B-cubed, not pairwise F1.** Pairwise F1 weights clusters *quadratically* in
size, so a single large-cluster error dominates the metric and hundreds of correct singleton
decisions are invisible. [B-cubed weights each record
linearly](https://arxiv.org/pdf/1509.04238) and gives a balanced view of over-linking vs
under-linking. It's ~40 lines of numpy against a small gold clustering; 40 gold mentions is
enough to be instructive, a few hundred to be trustworthy.

### The full-pipeline libraries

**[Splink 4](https://moj-analytical-services.github.io/splink/index.html)** is the strongest
open-source probabilistic linkage engine: Fellegi–Sunter with EM-estimated m/u parameters,
SQL backends (DuckDB, Spark, Athena, Postgres), interactive diagnostics, and
["a million records on a laptop in approximately one
minute"](https://moj-analytical-services.github.io/splink/index.html). The catch for our use
case is stated plainly in its own docs: it *"performs best with input data containing
multiple columns that are not highly correlated"* and is *"not designed for linking a single
column containing a 'bag of words'."* Text-extracted entities are exactly that — one name and
a pile of context. Splink is the right tool when you are reconciling extracted entities
**against a structured master table** (customers, suppliers, securities), and the wrong tool
for deduplicating a mention list against itself.

**[dedupe](https://github.com/dedupeio/dedupe)** offers active learning over Python-native
records and is pleasant at small scale; it has documented memory pressure well before the
scales a KG reaches. **[Zingg](https://github.com/zinggAI/zingg)** does active-learning ER
with auto-learned blocking on Spark — the right answer if you already run a lakehouse and
wrong if you don't, because the JVM/Spark operational surface dwarfs the problem otherwise.

### LLM adjudication as a tier, not a strategy

Reserve the LLM for the review band (say 0.85–0.95) where deterministic scoring is genuinely
ambiguous. It has good judgment on exactly the cases heuristics fail. Three objections you
must design around: **non-idempotence** (the same pair can be decided differently across
runs — your graph is no longer a pure function of your corpus); **provenance** (store the
inputs, the verdict, the confidence, and the model version on the merge edge, or the graph
is unauditable); and **explanation unreliability** (the stated rationale is not a faithful
account of the decision — do not build downstream logic on it).

Also, calibrate before reaching for a frontier model: **[AnyMatch](https://arxiv.org/abs/2409.04073)**,
a fine-tuned **GPT-2**, achieves **81.96 mean F1 across nine entity-matching benchmarks —
within 4.4% of a trillion-parameter MatchGPT at a 3,899× lower inference cost per 1k
tokens**. Entity matching is a task where small purpose-built models are startlingly
competitive.

### Entity linking to a KB

Clustering gives you *internally* consistent ids. Linking gives you *globally* stable ones.
**[ReFinED](https://arxiv.org/abs/2207.04108)** does mention detection, fine-grained typing,
and disambiguation for a whole document in a single forward pass — ["more than 60 times
faster than competitive existing approaches"](https://arxiv.org/abs/2207.04108), +3.7 F1
average over the then-SOTA, generalizing to Wikidata (15× more entities than Wikipedia) and
capable of zero-shot linking to entities unseen in training. If your entities are
world-knowledge entities (public companies, people, places), linking to Wikidata QIDs gives
you cross-corpus joins, alias lists, and type hierarchies for free, and turns "did we already
see this company?" from a fuzzy-match question into a lookup. It does nothing for
domain-private entities (internal projects, your customers' SKUs) — those still need
clustering. Caveat: ReFinED is a 2022 model with limited maintenance activity; verify before
depending on it.

---

## 4. Coreference

**An encoder-only pipeline has no coreference escape hatch.** This is the single most
common way a GLiNER-based agent-memory pipeline appears broken. On pronoun-dense
conversational text, most facts attach to *I*, *me*, *she*, *it*, *that repo* — none of
which are entity mentions in any useful sense. Recall looks catastrophic and the extractor
gets the blame. An LLM extractor resolves most of this implicitly, for free, as a side
effect of reading the passage; an encoder cannot, because there is no span to point at.

Three deterministic layers get you most of the way, and they are worth building before you
reach for a model:

1. **First person → a singleton user node.** Any mention in `{I, me, my, mine, myself}`, and
   any entity extracted from a user-authored turn typed as `user`, maps to one canonical
   node. Prefix every turn with an explicit speaker label (`"User: …"` / `"Assistant: …"`)
   *before* extraction so the model has a surface to latch onto. In agent memory this single
   rule recovers the majority of edges.
2. **Speaker-window binding.** Maintain a per-conversation "most recent mention by type"
   stack. Bare pronouns matching in number/gender, and definite descriptions
   (*"the manager"*, *"that repo"*), bind to the most recent same-type mention within K
   turns. Crude, inspectable, and measurable on a labeled handful of turns.
3. **First-name-only mentions.** *"Sarah"* binds to a registry `person` whose first token is
   Sarah **iff exactly one such person exists in the conversation-local set**. If two do,
   emit a review-band `same_as` edge rather than guessing. Ambiguity is data.

When you need a real model, there are two:

| Model | Accuracy | Speed | License |
|---|---|---|---|
| [Maverick](https://aclanthology.org/2024.acl-long.722/) | **83.6 CoNLL-2012 F1** (OntoNotes) | 170× faster inference, up to 0.006× memory vs prior SOTA | **CC BY-NC-SA 4.0 — non-commercial** |
| [fastcoref / F-coref](https://arxiv.org/abs/2209.04280) | 78.5 avg F1 (LingMess mode: 81.4) | 2.8K OntoNotes docs in **25 s on a V100** (LingMess: 6 min; AllenNLP: 12 min) | MIT |

**The licensing caveat matters and is easy to miss.** Maverick's
[repository states](https://github.com/SapienzaNLP/maverick-coref) that the data and software
are licensed **CC BY-NC-SA 4.0** — non-commercial. It is the better model (83.6 vs 78.5,
and it outperforms 13B-parameter systems with ~500M parameters), and you probably cannot
ship it in a commercial product without separate permission. fastcoref is the throughput and
licensing choice; ~5 F1 apart for a large speed gain and no legal question.

Also note both are trained on OntoNotes-style newswire and literature. Neither is trained on
chat transcripts, and their accuracy on *"can you redo that but with the other config?"* is
not what the benchmark number suggests. Measure on your own turns.

---

## 5. KG-from-text frameworks

Reaching for one of these instead of hand-rolling is a real option. The pattern to watch
for: nearly all of them solve *extraction* and leave *entity resolution* to you, which is
the harder half.

**[LangChain `LLMGraphTransformer`](https://github.com/langchain-ai/langchain-experimental/blob/main/libs/experimental/langchain_experimental/graph_transformers/llm.py)**
— one LLM call per chunk returning nodes and relationships, with `allowed_nodes`,
`allowed_relationships`, `node_properties`, and `strict_mode` (default on) filtering
off-schema output. It is the fastest path from text to a populated Neo4j instance and it is
genuinely useful for that. **It has no entity resolution at all** — "Acme", "Acme Corp", and
"ACME" become three nodes, and deduplication is an
[open feature request](https://github.com/langchain4j/langchain4j/issues/2543) rather than a
component. Reach for it for a demo, a prototype, or a corpus small enough that you'll fix
the duplicates by hand.

**[LlamaIndex `PropertyGraphIndex`](https://www.llamaindex.ai/blog/introducing-the-property-graph-index-a-powerful-new-way-to-build-knowledge-graphs-with-llms)**
— the most composable of the batch. Extraction is a list of `kg_extractors`
(`SimpleLLMPathExtractor`, `SchemaLLMPathExtractor` for a constrained ontology,
`DynamicLLMPathExtractor` for schema discovery, `ImplicitPathExtractor` which needs no model
at all and just materializes existing node relationships), and they compose. That list is
the natural seam for dropping in a GLiNER2.5 extractor next to an LLM one and running the
hybrid from §2. Same ER gap as LangChain. Reach for it when you want extractor pluggability
and a retrieval layer in the same package.

**[Microsoft GraphRAG](https://microsoft.github.io/graphrag/)** — the most complete pipeline:
entity/relationship extraction, claim extraction, Leiden community detection, and LLM-written
community summaries that power global "what are the themes?" queries no vector store can
answer. It is also the most expensive thing on this list, because every one of those steps is
an LLM call over the whole corpus; third-party estimates put indexing on the order of
[$20–50 per million tokens](https://medium.com/graph-praxis/the-graphrag-cost-cliff-how-33-000-became-33-in-eighteen-months-be1b0fbe37e4)
depending on model, and the number is highly sensitive to configuration.
**[LazyGraphRAG](https://www.microsoft.com/en-us/research/blog/lazygraphrag-setting-a-new-standard-for-quality-and-cost/)**
is the answer to that: defer all LLM summarization to query time, so
["data indexing costs are identical to vector RAG and 0.1% of the costs of full
GraphRAG"](https://www.microsoft.com/en-us/research/blog/lazygraphrag-setting-a-new-standard-for-quality-and-cost/)
with comparable global-query quality at >700× lower query cost **[vendor/Microsoft
Research, on their own benchmark set]**. For streaming or exploratory corpora, LazyGraphRAG
is the default and full GraphRAG is the exception.

**[Neo4j `graphrag-python` / `SimpleKGPipeline`](https://neo4j.com/docs/neo4j-graphrag-python/current/user_guide_kg_builder.html)**
— PDF→chunk→embed→extract→write, with a schema. Its distinguishing feature is the best
mainstream **entity resolution** story of any framework here: it actually ships resolvers —
`SinglePropertyExactMatchResolver`, `FuzzyMatchResolver` (RapidFuzz), and
`SpaCySemanticMatchResolver` (embedding similarity) — as pipeline components rather than as
homework. They are simple (same label + similar text properties), but they exist, they run
post-write, and they are the right three tiers. Reach for it if you're on Neo4j and want ER
in the box.

**[Graphiti / Zep](https://github.com/getzep/graphiti)** — the only one built for *agent
memory* rather than document batch. Three subgraphs (episode / semantic entity / community),
a **bi-temporal** model where every edge carries both when the fact was true and when it was
ingested, and incremental resolution of each new episode against the live graph. Contradiction
handling is first-class: new knowledge invalidates old edges rather than overwriting them,
so you can query the graph as of a past time. Its ER is described in §3 and is the most
sophisticated of any framework here. The [Zep paper](https://arxiv.org/abs/2501.13956)
reports DMR 94.8% vs MemGPT's 93.4% and up to 18.5% accuracy improvement with ~90% latency
reduction on LongMemEval **[vendor]**. Cost: the write path is the heaviest here — multiple
model calls per episode — which is exactly the thing agent memory can least afford, and the
reason their heuristics-first dedupe work exists.

**[mem0](https://github.com/mem0ai/mem0)** — the lightest option, and deliberately not a
knowledge graph in the typed-edge sense. It extracts salient facts from conversation and
maintains them with ADD/UPDATE/DELETE operations against a vector store (with an optional
graph store). Reports [92.5 on LoCoMo at under 7,000 tokens per retrieval versus 25,000+ for
full-context](https://mem0.ai/blog/mem0-the-token-efficient-memory-algorithm) **[vendor]**.
Reach for it when you want conversational memory with minimal ceremony and do *not* need to
traverse typed relations.

**[Cognee](https://github.com/topoteretes/cognee)** — an "ECL" (Extract, Cognify, Load)
pipeline: classify → chunk → LLM entity/relation extraction → summarize → embed → commit,
with **ontology-based entity validation** in the cognify step, which is its differentiator —
an explicit attempt to stop the LLM from emitting "car manufacturer" / "automobile maker" /
"vehicle producer" as three types. Plus `memify`, which post-processes the graph (prune stale
nodes, reweight edges by usage). Positioned between mem0's simplicity and Graphiti's rigor.
Younger and less battle-tested than either.

**[iText2KG](https://arxiv.org/abs/2409.03284)** — a research implementation rather than a
product, and the cleanest published reference for *incremental, resolution-aware*
construction: Document Distiller → Incremental Entity Extractor → Incremental Relation
Extractor → Graph Integrator, matching each document's local entities against a global set by
cosine similarity with a threshold, topic-independent and requiring no predefined ontology.
Read it for the architecture even if you don't run it; the two-set (local/global) framing is
the correct mental model for cross-document ER and is what you'll end up rebuilding.

---

## 6. Agent memory vs document intelligence

These are not the same problem with different data. Six requirements genuinely diverge.

| | Agent memory | Document intelligence |
|---|---|---|
| **ER mode** | Incremental — each episode resolved against a live registry, plus an intra-batch pass | Batch — the whole corpus is resolvable at once, globally optimal clustering is possible |
| **Time** | Bi-temporal and load-bearing: facts expire, get contradicted, get superseded | Usually document-dated; validity intervals rarely matter |
| **Provenance** | Episode-level: *which turn, which session, said by whom* | Span-level: *which document, which character offset* |
| **Write latency** | Sits in the user's turn — hundreds of ms, or moved off the critical path | Seconds to minutes per document is fine |
| **Read frequency** | **Every turn.** Retrieval cost dominates total cost | Per query, sporadic |
| **Failure mode** | Confidently recalling a stale preference | Silently dropping a fact |

**Incremental vs batch resolution.** Batch ER can look at all mentions at once and pick a
globally consistent clustering. Incremental ER must decide *now*, with only the past
available, and live with it. That asymmetry is why incremental systems need two passes: match
the new episode's mentions against the registry, *then* re-run resolution within the batch —
because an episode that mentions "Acme" and "Acme Corp" for the first time has no registry
entry to anchor either to. One pass is not enough, and this is the single most common bug in
hand-rolled agent memory.

**Temporal validity and invalidation.** *"I use npm"* followed three weeks later by *"we
moved to pnpm"* is not a contradiction to be resolved by confidence — both were true, at
different times. A graph that overwrites loses the history; a graph that appends without
validity intervals answers "what package manager does the user prefer?" with two equally
confident answers. This needs a bi-temporal edge model (valid-from/valid-to plus
ingested-at), and it is a *store-layer* concern — no extractor, GLiNER or LLM, gives it to
you. It is the main reason to consider Graphiti rather than build.

**Episode provenance.** Document intelligence wants character offsets so a human can verify
a claim against the source. Agent memory wants episode ids so the agent can say "you
mentioned this on Tuesday" and so a user can delete a session and have the derived facts
disappear with it. Design the evidence model for the one you're building; retrofitting is
painful.

**The read-path asymmetry is the one people underestimate.** Document-intelligence graphs are
written once and queried occasionally. Agent-memory graphs are queried on *every single
turn*, which means retrieval latency is added to every response and retrieval tokens are
added to every prompt. A memory system that produces a beautiful graph and returns 20K tokens
per turn is worse than a mediocre one that returns 3K. Optimize the read path first; it is
where both the latency and the cost live.

---

## 7. Where GLiNER2.5 fits

### What it genuinely wins at

- **Cost.** No API key, no per-token billing, no data leaving the machine. A 194M-parameter
  encoder on a CPU. For high-volume ingestion this is not a marginal saving.
- **Latency.** Single forward pass, no autoregressive decoding. **[No published
  latency number exists for GLiNER2.5 — the vendor blog and model cards contain none, and I
  could not find one. Measure it yourself before quoting anything.]** The architectural
  argument (linear in sequence length, one pass, no token-by-token generation) is sound; the
  number is not published.
- **Local and private.** Apache 2.0, runs offline, no third party sees the text. For
  regulated corpora this is often the deciding factor rather than a nice-to-have.
- **Typed-graph decoding via JointIE.** This is the real differentiator versus every other
  encoder. Entities and relations are decoded together under constraints enforced *during
  beam search*, so `unique_head`, `no_self_loops`, and `acyclic` hold by construction. The
  output is a well-formed graph, not a bag of triples plus a validation TODO. Nothing else
  in the encoder world offers this, and even LLM `strict: true` only guarantees schema
  validity, not graph validity.
- **The schema is a runtime argument.** Change the ontology, get different extraction, no
  retraining. This is the property that makes multi-tenant and exploratory work possible.

### What it structurally cannot do

- **Implicit and inferential facts.** *"Alice was promoted, so she now reports to Bob."* There
  is no span. No threshold, label phrasing, or fine-tune fixes this in general. If your
  ontology contains inferential predicates, GLiNER is the wrong primary extractor for them.
- **Coreference.** No pronoun resolution, no bridging anaphora, no cross-sentence entity
  tracking. See §4 — this is not a bug, it is the absence of a component, and on
  conversational input it dominates measured recall.
- **Temporal reasoning.** No notion of validity, sequence, or contradiction. *"we may face
  supply disruptions if tariffs rise"* extracts identically to a disruption that happened.
  You need a separate modality/qualifier pass (span attributes work well for this: tag each
  event `asserted` / `forecast` / `hypothetical` / `negated` and filter downstream) and a
  store layer that models time.
- **Node and edge attributes in `JointSchema`.** JointIE declares entity types, relation
  types, and structural constraints. It does not carry properties on nodes or edges. Getting
  qualified entities means a **second pass** with the `AttributeGroup` span-attribute API on
  the same text, joined back on character offsets. Workable, but it is two forward passes and
  a join, not one call.
- **Long-range relations.** Chunked long documents drop any relation whose endpoints land in
  different chunks. Silently. This is a property of chunking, not of GLiNER, but it bites
  hard on documents where the subject is named once at the top and referred to
  for twenty paragraphs after.

### On the published numbers

GLiNER2.5's benchmark claims deserve explicit qualification. They are **[vendor-published,
one release cycle old, and unreplicated]**. The comparison set is *only GLiNER2 at matched
sizes* — no LLM baselines, no other encoder families. The multilingual overall delta is
**56.17 vs 56.09**, which is noise, not an improvement; the meaningful gains are on
classification (XNLI +24.75) rather than NER. And the training data was
[generated by an LLM agent rather than assembled from public
datasets](https://fastino.ai/blog/gliner2-5-span-free-information-extraction), which is a
real contamination consideration for any "zero-shot" claim — if the generator saw the task
distribution, "zero-shot" means something weaker than it usually does. None of this makes
the model bad. It makes the numbers a weak prior.

Independent triangulation is available and is more sobering: on end-to-end zero-shot relation
extraction, [GLiNER2 averages 17.8 micro-F1](https://arxiv.org/html/2605.10108v1) across four
benchmarks. GLiNER2.5's joint decoding should improve on that, but nobody has published the
comparison.

### Recommended production shape

1. **GLiNER2.5 JointIE as the primary extractor**, with a hand-written ontology whose type
   descriptions read like annotation guidelines (including negative examples). Treat label
   strings as hyperparameters and sweep them.
2. **A second span-attribute pass** for modality and qualifiers, joined on offsets. In
   document intelligence, filtering on `modality == "asserted"` is the difference between a
   graph of facts and a graph of speculation.
3. **A real ER stage** — normalize → block (embedding ANN, type-constrained) → score
   (rapidfuzz + embeddings, entropy-gated) → cluster (with a max-size alarm) → canonicalize
   into a persistent registry. Log every scored pair.
4. **A coreference layer** appropriate to the input: deterministic rules for conversational
   text, fastcoref if you need more and Maverick's license blocks you.
5. **LLM escalation on a trigger**, not on everything. Chunks with inferential markers, empty
   relation sets, or low mean confidence.
6. **A store layer that models time**, if you're doing agent memory. GLiNER gives you none of it.

If steps 1–3 don't reach your quality bar on your corpus, the next move is *fine-tuning on a
labeled slice*, not switching to an LLM. That is the cheapest large quality jump available
and it is systematically skipped.

---

## 8. Open questions for this sandbox

Experiments worth actually running here, roughly in order of information-per-hour.

- [ ] **Label-phrasing sensitivity.** Run the same corpus against systematically varied label
      strings for the same concept (`date` vs `year` vs `calendar date`; `company` vs
      `organization` vs `employer`) and plot the confidence distribution per variant. If the
      swing is as large as it appears in spot checks, **label strings are the single highest-leverage
      hyperparameter in the system** and every downstream number is meaningless until they're tuned.
- [ ] **Threshold sweeps, all six of them.** GLiNER2.5 exposes at least six interacting
      thresholds (`extract(threshold=)`, per-entity thresholds, `AttributeGroup.threshold`, and
      JointIE's `candidate_threshold` / `relation_role_threshold` / `entity_threshold`), plus
      values baked into the checkpoint. Sweep the two that matter — entity and relation — and
      plot node/edge count against each. Publish no default without this.
- [ ] **GLiNER2.5 vs Claude head-to-head on the same corpus.** Same paragraphs, same ontology,
      same evaluation. Report three things separately: (a) F1 on *explicit* facts, (b) recall on
      a hand-labeled set of *implicit* facts, (c) wall-clock and dollars per document. (a) is
      probably closer than expected; (b) is the whole argument; (c) is the whole argument in
      the other direction.
- [ ] **Escalation rate vs recall recovery.** Define escalation triggers, sweep the rate from
      0% to 100%, and plot recovered-implicit-fact recall against LLM spend. The shape of that
      curve is the single most useful number for a production decision and I could not find it
      published anywhere.
- [ ] **Fine-tune on a labeled slice.** Take 200–500 hand-corrected examples from one ontology
      and LoRA-tune the base checkpoint. Measure the delta on held-out data. Hypothesis: this
      beats every threshold-tuning and prompt-engineering effort combined.
- [ ] **Bi-encoder for large type vocabularies.** Construct a 200+ type ontology and compare
      uni-encoder GLiNER2.5 against a bi-encoder GLiNER with precomputed label embeddings on
      both throughput and F1. Confirm on CPU whether the [published H100
      numbers](https://arxiv.org/html/2602.18487v1) translate.
- [ ] **Cross-encoder reranking for ER.** Add a cross-encoder (or AnyMatch-style small
      fine-tuned model) as a scoring tier between fuzzy/embedding scoring and LLM adjudication.
      Measure B-cubed lift and the reduction in LLM-adjudicated pairs.
- [ ] **Blocking recall on a gold pair set.** Hand-label a few hundred true-match pairs, then
      measure pair completeness and reduction ratio for each blocker (deterministic key,
      embedding ANN, MinHash/LSH) and their union. Establishes the ceiling everything else
      operates under.
- [ ] **Coreference ablation.** Measure agent-memory edge recall with (a) no coref, (b) the
      first-person rule only, (c) all three deterministic layers, (d) fastcoref. The delta from
      (a) to (b) is likely to be the largest single number in the whole notebook.
- [ ] **Chunk-boundary relation loss.** Run one long document at several chunk sizes and count
      edges. Quantify what chunking silently costs you.
- [ ] **Reproducibility.** Extract the same corpus twice and diff the graphs. Report node,
      edge, and cluster instability. Do this for both GLiNER and an LLM extractor; the numbers
      will not be zero for either.

---

## Sources

Extraction: [GLiNER](https://arxiv.org/abs/2311.08526) ·
[GLiNER2](https://arxiv.org/abs/2507.18546) ·
[GLiNER2.5](https://fastino.ai/blog/gliner2-5-span-free-information-extraction) ·
[GLiNER2 repo](https://github.com/fastino-ai/GLiNER2) ·
[GLiNER-Relex](https://arxiv.org/html/2605.10108v1) ·
[GLiNER bi-encoder](https://arxiv.org/html/2602.18487v1) ·
[gliner-bi-large-v2.0](https://huggingface.co/knowledgator/gliner-bi-large-v2.0) ·
[NuNER](https://aclanthology.org/2024.emnlp-main.660.pdf) ·
[spaCy model metadata](https://github.com/explosion/spacy-models/blob/master/meta/en_core_web_trf-3.8.0.json) ·
[Claude pricing](https://docs.claude.com/en/docs/about-claude/pricing) ·
[prompt caching](https://docs.claude.com/en/docs/build-with-claude/prompt-caching) ·
[Batch API](https://docs.claude.com/en/docs/build-with-claude/batch-processing) ·
[BAML](https://boundaryml.com/blog/structured-output-from-llms) ·
[NVIDIA distillation blueprint](https://developer.nvidia.com/blog/build-efficient-financial-data-workflows-with-ai-model-distillation/)

Entity resolution: [Splink](https://moj-analytical-services.github.io/splink/index.html) ·
[Zingg](https://github.com/zinggAI/zingg) ·
[AnyMatch](https://arxiv.org/abs/2409.04073) ·
[ReFinED](https://arxiv.org/abs/2207.04108) ·
[ER evaluation primer (B-cubed)](https://arxiv.org/pdf/1509.04238) ·
[How to Evaluate ER Systems](https://arxiv.org/pdf/2404.05622) ·
[Awesome Entity Resolution](https://github.com/OlivierBinette/Awesome-Entity-Resolution)

Coreference: [Maverick](https://aclanthology.org/2024.acl-long.722/) ·
[maverick-coref repo (license)](https://github.com/SapienzaNLP/maverick-coref) ·
[F-coref](https://arxiv.org/abs/2209.04280)

Frameworks: [LLMGraphTransformer](https://github.com/langchain-ai/langchain-experimental/blob/main/libs/experimental/langchain_experimental/graph_transformers/llm.py) ·
[PropertyGraphIndex](https://www.llamaindex.ai/blog/introducing-the-property-graph-index-a-powerful-new-way-to-build-knowledge-graphs-with-llms) ·
[LazyGraphRAG](https://www.microsoft.com/en-us/research/blog/lazygraphrag-setting-a-new-standard-for-quality-and-cost/) ·
[Neo4j KG Builder](https://neo4j.com/docs/neo4j-graphrag-python/current/user_guide_kg_builder.html) ·
[Zep/Graphiti paper](https://arxiv.org/abs/2501.13956) ·
[Graphiti dedupe internals](https://blog.getzep.com/graphiti-hits-20k-stars-mcp-server-1-0/) ·
[mem0 memory algorithm](https://mem0.ai/blog/mem0-the-token-efficient-memory-algorithm) ·
[Cognee](https://www.cognee.ai/blog/fundamentals/how-cognee-builds-ai-memory) ·
[iText2KG](https://arxiv.org/abs/2409.03284)
