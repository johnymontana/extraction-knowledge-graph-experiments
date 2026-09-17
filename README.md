# extraction-sandbox

**Exploring entity extraction and resolution for knowledge graph construction** — for agent memory and for
document intelligence, which turn out to be genuinely different problems.

The first experiment is [GLiNER2.5](https://fastino.ai/models/gliner2-5): a 194M-parameter encoder that does
joint entity *and* relation extraction against a schema supplied at runtime, on a laptop CPU, with no API key.

## The notebooks

| | | needs |
|---|---|---|
| **[01](notebooks/01_gliner25_knowledge_graphs.ipynb)** | **Extraction end to end** — two ontologies, two corpora, two graphs, entity resolution, a temporal layer | — |
| **[02](notebooks/02_neo4j_graphrag_retrieval.ipynb)** | **The graph in Neo4j** — loading, Cypher, all six `neo4j-graphrag` retrievers, NVL visualization | Neo4j |
| **[03](notebooks/03_extractors_head_to_head.ipynb)** | **Extractors compared** — GLiNER vs Claude vs spaCy vs escalation, scored on gold triples | `claude` CLI |
| **[04](notebooks/04_agent_memory_in_neo4j.ipynb)** | **Agent memory in Neo4j** — bi-temporal facts, incremental write-back, prompt assembly | Neo4j |
| **[05](notebooks/05_entity_resolution_bakeoff.ipynb)** | **Entity resolution compared** — the hand-rolled pipeline vs Splink, dedupe, cross-encoders, LLM adjudication | `claude` CLI |
| **[06](notebooks/06_finetuning_distillation.ipynb)** | **Distillation** — LLM labels fine-tuned into a 74M encoder, and whether that pays | `claude` CLI |
| **[07](notebooks/07_kg_framework_comparison.ipynb)** | **Frameworks compared** — SimpleKGPipeline, LLMGraphTransformer, PropertyGraphIndex on one corpus | Neo4j, `claude` CLI |
| **[08](notebooks/08_coreference.ipynb)** | **Coreference** — four neural engines against the deterministic rules | — |
| **[09](notebooks/09_three_domains.ipynb)** | **Three domains, no LLM** — travel, customer service and ecommerce end to end, with gazetteer linking and NVL | Neo4j |
| **[10](notebooks/10_typesafe_system_one.ipynb)** | **Typed judgments** — TypeSafe System One for assertion gating, review-band adjudication, typing before blocking, a must-not-merge test, and judgments as graph data | `TYPESAFE_API_KEY` |
| **[11](notebooks/11_typesafe_relation_selection.ipynb)** | **Selection, not generation** — relation extraction as one `Choice` per enumerated entity pair, scored against joint decoding; the recall experiment | `TYPESAFE_API_KEY` |
| **[12](notebooks/12_typesafe_contradictions.ipynb)** | **What contradicts what** — a `Noul` as the temporal layer's `alternative_fn`, against embeddings and `replaces` edges, on the planted switches | `TYPESAFE_API_KEY` |
| **[13](notebooks/13_typesafe_cascade.ipynb)** | **Escalate on confidence** — the uncertain edge judgments and `review` pairs sent to Claude Haiku, swept from 0% to 100%, with cost | `TYPESAFE_API_KEY`, `claude` CLI |
| **[14](notebooks/14_typesafe_document_judgments.ipynb)** | **The boundary of the no-LLM position** — intent, priority and resolution for the eight support threads, as three questions and then as graph properties | `TYPESAFE_API_KEY` |

Notebooks 01–09 need no API key. 03, 05, 06 and 07 use the `claude` CLI (already authenticated if you use
Claude Code) through `kgx.llm.ClaudeCLI`; **notebook 09 uses no LLM at all**, by design.

**Notebooks 10–14 are the exception** and need a hosted API key, `TYPESAFE_API_KEY` (13 also needs the
`claude` CLI). They are kept separate for that reason: `kgx.typesafe` is not exported from `kgx`, so
importing it is a deliberate act and nothing else in the repo acquires the dependency. Like `kgx.llm.ClaudeCLI`, it caches every response to disk
content-addressed on `(model, state, questions)` — a re-run is free, byte-identical, and needs no key.

---

## Quickstart

```bash
uv sync
uv run jupyter lab notebooks/01_gliner25_knowledge_graphs.ipynb
```

First run downloads ~400 MB (GLiNER2.5-base) plus ~90 MB (MiniLM, for entity resolution). Everything after
that is local; the only hosted dependency in the repo is notebooks 10–14's.

```python
import kgx

extractor = kgx.GlinerExtractor()                       # fastino/gliner2.5-base-v1
graphs    = extractor.extract_batch(docs, kgx.BUSINESS_NEWS)

resolver  = kgx.EntityResolver().learn_aliases(d["text"] for d in docs)
mentions  = [m for g in graphs for m in g.mentions]
kg        = kgx.build_graph(graphs, resolver.resolve(mentions), kgx.BUSINESS_NEWS)

print(kg.summary())
print(kgx.to_cypher(kg, min_support=2))
```

---

## The pipeline

```
ontology  →  extract  →  [coref]  →  resolve  →  graph  →  [temporal]
```

| module | what it does |
|---|---|
| `kgx.ontology` | The graph model as data — node labels, edge types, legal `(head, REL, tail)` triples. Compiles to a GLiNER `JointSchema`; also drives validation and Cypher export. JSON round-trippable. |
| `kgx.extract` | Joint entity+relation decoding, plus a span-attribute pass for qualifiers (modality, direction) that typed edges cannot carry. |
| `kgx.coref` | Deterministic conversation preprocessing. Encoder models have no coreference; this is what stops that from looking like an extraction failure. |
| `kgx.resolve` | `normalize → block → score → cluster → canonicalize`, plus an incremental registry for episode-at-a-time memory, corpus alias mining, and B-cubed scoring. |
| `kgx.graph` | Canonical, evidence-backed graph assembly. matplotlib, pyvis, and Cypher output. |
| `kgx.temporal` | Bi-temporal fact store — supersede contradictions instead of accumulating them. |
| `kgx.neo4j_io` | Idempotent loading into Neo4j via `$()` dynamic labels, with schema introspection. |
| `kgx.llm` | The `claude` CLI as a cached LLM backend, plus an `LLMExtractor` that returns the same `DocGraph` as GLiNER. |
| `kgx.typesafe` | TypeSafe System One as a pipeline stage — a disk-cached client; assertion gating, pair adjudication, referent typing and soft blocking, relation selection over enumerated pairs, and an `alternative_fn` for the temporal layer. The one module that needs a hosted API key. |
| `kgx.evaluate` | Triple-level P/R/F1 with an explicit, auditable matching policy. |
| `kgx.baselines` | spaCy as the closed-vocabulary floor, with the ontology-coverage gap made explicit. |
| `kgx.frameworks` | Adapters so LangChain / LlamaIndex / graphrag can run on a subprocess-backed LLM. |
| `kgx.gazetteer` | Dictionary linking against a controlled vocabulary — stable ids, not clusters. |
| `kgx.domains` | Travel, customer-service and shopping ontologies. |

Two ontologies ship: `kgx.AGENT_MEMORY` (13 node labels, 16 edge types) and `kgx.BUSINESS_NEWS` (13 / 15).
Both are ordinary data — write your own in Python, JSON, or YAML.

---

## The two use cases, and why they diverge

| | **Agent memory** | **Document intelligence** |
|---|---|---|
| input | conversation episodes, one at a time | a corpus, all at once |
| resolution | incremental — link each episode into what's known | batch |
| time | facts expire; contradictions must supersede | facts are stamped, not superseded |
| hard part | coreference (*"I"*, *"he"*, *"the migration"*) | alias variation (*NWL* / *Northwind* / *the Company*) |
| read pattern | every turn, latency-critical | analytical, offline |

Same model, same joint decoding, different everything else.

---

## Findings

Measured on this repo's corpora, not asserted. The notebook shows the working for each.

**Joint decoding is the reason to use GLiNER2.5 over GLiNER2.** Across both corpora, **zero** extracted edges
violated the ontology's endpoint types. Extracting entities and relations separately gives you no such
guarantee — you get a post-filter instead of a constraint.

**Relation recall falls off a cliff past ~400 words, and lands on zero.** A 518-word transcript decoded to
**0 relations** with `feasible=True` — indistinguishable from "no facts here". Entity recall over the same
window was fine. Window the input: extract per episode, or `extract_long(chunk_size=256..384)`, or
`JointIEConfig(max_len=512)`. `kgx` warns on this.

**`symmetric=True` is broken in `gliner2` 2.0.0.** It compiles to a constraint set that rejects every
candidate edge — the relation silently returns nothing, with `feasible=True`. Use a directed relation plus
`inverse=`, which works and emits the mirror edge tagged `derived=True`.

**First-person substitution is worth ~500× on conversational text.** Edges recovered about the user, over the
same 5 sessions: 1 with raw turns → 210 with speaker labels → 539 with the full preprocessing stack. Not a
model limitation; a preprocessing requirement of any coreference-free extractor.

**Entity resolution: B-cubed F1 0.85 → 0.95**, precision 1.00 throughout — all the movement is in recall.
Embeddings buy the most F1, but mining aliases from the corpus text (`Northwind Logistics Inc. (NASDAQ: NWL)`)
buys the most *blocking recall*, which is the harder ceiling: embeddings raise the score of pairs that are
already candidates, alias mining creates candidates nothing else would propose. Complementary, not redundant.

**Deciding what contradicts what needs world knowledge these models don't have.** Embedding similarity does
not separate genuine alternatives (npm/pnpm) from unrelated pairs (npm/Berlin) — the distributions overlap
under every template tried. Asking the extractor to classify tools into categories fails too. What *does*
work: declare `replaces(tool → tool)` in the ontology and let joint decoding find the switch the user
announced in the text. Extracted at 0.99 confidence, with the sentence attached.

**Errors move between stages wearing a disguise.** A mistyped entity in extraction and an under-merge in
resolution both surface as "the user changed their mind" in the temporal layer. A subject that flip-flops
back to a value it already held is the tell. Carrying evidence on every edge is what makes them separable.

**A prefix match is not an identity.** `Northwind` / `Northwind Logistics` should merge; `Apple` / `Apple Bank`
should not, and the string evidence is identical. Requiring context agreement splits them at no measured cost
to F1 — the kind of rule that is invisible on a corpus without the trap and expensive on one with it.

---

## Layout

```
src/kgx/            the library
  ontology.py       graph model as data; AGENT_MEMORY + BUSINESS_NEWS
  extract.py        GLiNER2.5 joint extraction + qualifier pass
  coref.py          conversation preprocessing (3 ablatable layers)
  resolve.py        entity resolution + incremental registry + B-cubed
  graph.py          canonical graph, evidence, matplotlib/pyvis/Cypher
  temporal.py       bi-temporal facts, supersession
  neo4j_io.py       idempotent Neo4j loading, schema introspection
  llm.py            cached claude-CLI backend + LLM extractor
  typesafe.py       cached TypeSafe System One client; edge gating + pair adjudication
  evaluate.py       triple scoring against gold
  baselines.py      spaCy closed-vocab baseline
  frameworks.py     LangChain / LlamaIndex / graphrag adapters
  gazetteer.py      controlled-vocabulary entity linking
  domains.py        travel / customer service / shopping ontologies
  data/             synthetic corpora with gold labels
notebooks/          01-14, see the table above
docs/LANDSCAPE.md   survey of the alternatives at every stage
output/             generated graphs, Cypher, CSVs (gitignored)
```

`src/kgx/data/` ships 5 conversation sessions (80 turns) and 10 business-news documents (~2,300 words), both
synthetic, both written with deliberate alias variation, planted contradictions, coreference stress, and
modality traps — plus gold labels for entity resolution and a gold triple set. All companies, people, and
events are fictional.

## Neo4j

Notebook 02 needs a Neo4j 5.26+ instance. Nothing else — the embeddings are a local MiniLM and the two
retrievers that genuinely need an LLM run against a deterministic stub (with a live path if
`ANTHROPIC_API_KEY` is set).

```bash
scripts/neo4j_up.sh          # docker if available, a local tarball under .neo4j/ if not
uv run python scripts/neo4j_restore.py   # reload notebook 02's graph + the four indexes
```

Both are idempotent. `neo4j_up.sh` prefers Docker and falls back to a native install, so the notebooks keep
working when Docker Desktop is down. `neo4j_restore.py` rebuilds the database from
`output/business_news_kg.json` — the graph is fully reproducible, so a lost container costs a minute, not a
re-extraction.

Override the connection with `NEO4J_URI` / `NEO4J_USER` / `NEO4J_PASSWORD`. Browse at <http://localhost:7476>.

**More findings, from the Neo4j and comparison notebooks:**

**Dynamic labels work in 5.26 — you don't need APOC.** `MERGE (n:$($label))` takes the label as data.
`MERGE (n:$label)` is a syntax error. APOC's `apoc.merge.node(labels, ident, onCreate, onMatch)` is a trap:
pass properties only as `onMatch` and a *first* load writes nothing but the id — invisible on any database
you have already loaded once.

**`CREATE INDEX ... IF NOT EXISTS` is satisfied by an equivalent index under a different name.** It succeeds,
creates nothing, and the retriever fails several cells later with "No index with name … found".

**Give every entity one shared `:__Entity__` label.** Cypher indexes are per label, so 13 ontology labels
would mean 13 vector indexes. The shortcut `create_fulltext_index(label="Company|Person")` silently creates
an index on one literal label named `Company|Person` that matches nothing, forever.

**Entity lookup is a lexical task; rank it that way.** `HybridCypherRetriever` with the default ranker
returns *Dresden* for the query `NWL`. `ranker="linear", alpha=0.2` fixes it.

**Provenance filtering fixes multi-hop queries.** A hallucinated `subsidiary_of` edge propagates into a
two-hop ownership chain that never existed; filtering on `r.support >= 2` inside the quantified path pattern
prunes it before the path is built.

**The "encoder cannot do implicit facts" claim was wrong** (notebook 03). GLiNER2.5 recovered an implied
relation, a cross-sentence syllogism and a bridged referent. The real boundary is about *arguments*: a
relation can be inferred from context, an entity argument has to be anchored in the text. The same probe
caught the encoder asserting a relation an explicitly negated sentence denies — which the LLM did not.

**Splink beats the hand-rolled resolver, and so does an LLM tier** (notebook 05). Both reach B³ F1 1.000
against the repo's 0.946, and both do it the same way: by fixing blocking recall (0.879 → 1.000), not by
scoring better. Splink's own machinery contributes little here — EM training is worthless with one
informative column, and its match probability is uncalibrated (optimal threshold 0.1, not 0.9).

**Distillation worked and did not help** (notebook 06). A full fine-tune learned the teacher's labelling
function far better than its starting point (mention fidelity 0.453 → 0.641) and scored *below* its own
zero-shot baseline on the benchmark. Seed variance inside one config (0.080 F1) exceeded the gap between the
74M and 194M models the experiment set out to close.

**Coreference models lose to four regexes on chat** (notebook 08). fastcoref added exactly zero gold facts
over the deterministic layers; its small F1 edge came from emitting fewer spurious edges. All four engines
fail on first person without a speaker prefix — and writing that prefix is what the rule layer does.

**Frameworks mostly skip entity resolution** (notebook 07). Bolting `kgx.EntityResolver` onto their output,
with zero LLM calls, recovered ~96% of the gap between strict and alias-tolerant scoring.

**Each NVL `render()` inlines an ~8.5 MB bundle**, and `from_neo4j` copies every property — including your
384-float embeddings — into it. Strip them in Python; a map projection does not help.

**"Precision 1.000" was a property of the corpus, not the resolver** (notebook 09). Notebooks 01 and 05 both
report perfect B-cubed precision on business news. Given an ecommerce corpus where `Aurora 14` and
`Aurora 14 Pro` are different products, the same code at the same threshold merges them — along with
`N600`/`N600X` and `Halcyon Buds`/`Halcyon Buds Pro`. The distinguishing token is exactly the kind of short
suffix normalisation is built to ignore.

**Where a controlled vocabulary exists, stop computing similarity.** A gazetteer links `LHR` to
`London Heathrow` exactly, where Jaro-Winkler scores 0.45 and no threshold reaches it — and it yields a
*stable id* that survives a rerun and a change of corpus, which clustering cannot. But validate any type
constraint you put on it: matching the extractor's type against the vocabulary's kind cost 20 points of
coverage and prevented zero errors, because both sides reasonably disagreed about whether an airport is a place.

**Document-level judgement is the boundary of the no-LLM position.** GLiNER2.5 classified support-ticket
intent near chance and priority *at* chance — a near-constant predictor emitting `high` for seven of eight
threads at 0.75–1.00 confidence. Intent is written down; severity is not, so a span model has nothing to key
on. Of every task in notebook 09, that is the one worth escalating.

**An ontology is a hypothesis about the text.** `replaces` fires at 0.99 on agent memory ("I've switched to
pnpm" — one clause, two named tools, an explicit verb) and never fires on support threads, where the same
supersession is spread across a four-turn negotiation. Same relation, same model, different discourse shape.

**More findings, from the typed-judgment notebook:**

**An assertion gate raised precision 0.279 → 0.404 and did not cost a single point of recall** (notebook
10). A `Noul` for *does the document connect these two things* plus a `Choice` over four assertion statuses
dropped 56 of GLiNER's 170 edges — 31 the document never related at all, 25 it related and then hedged or
denied — and not one of the 56 was a gold triple. GLiNER's own confidence does not separate the two groups,
because span confidence is about the decoding and not about the claim.

**A yes/no question that hides a second reading gets a confident answer to the wrong one.** The first
wording of a `Noul` caught 8 of 11 modality traps; a `Choice` in the same request called 11 of 11. Every miss
is a denial — *"Management has no plans to divest…"* — where the sentence really does assert a fact, about an
intention not to act. The misses come back at 0.66–0.74, not at 0.5: ambiguity in the *question* does not
surface as an uncertain answer. The `Choice` never had the problem because `negated` is one of its outcomes;
rewording the `Noul` to say which reading to take — *a sentence saying something will NOT happen does not
make it a fact* — took it to 11 of 11 with the controls intact. Keep the distinction in the answer type, or
state it in the question.

**A perfect judge on the wrong queue is worth exactly nothing.** Adjudicating the entity-resolution review
band produced 28 merges, all correct, none contradicting the gold labels — and moved B-cubed by zero to four
decimal places. All 28 were already co-clustered by transitivity, and the 10 merges that did change
clusters were on mentions outside the gold set. This is notebook 05's finding with the scoring hypothesis
eliminated: the judge was flawless and it still bought nothing.

**The pairs it never saw were a type disagreement, not a string-similarity failure.** 41 gold-same pairs
stayed split and *none had ever been proposed* — `HLCN` against every spelling of Halcyon, and `Torrent
Microsystems` against itself. The extractor types `HLCN` as `security` and `Halcyon Semiconductor` as
`company`, and types the identical string `Torrent Microsystems` both ways in different documents; blocking
is type-scoped, so none of it is ever a candidate. Handed those 8 pairs directly, the same adjudicator
reaches **B-cubed F1 1.000 at precision 1.000** — where notebook 05's best configurations also land, from
the opposite direction. The type constraint in notebook 09's gazetteer cost 20 points of coverage; here it
cost the entire remaining recall gap.

**The type wall comes down when blocking reads the distribution, not the label.** Re-typing every mention
by *referent* against the ontology's own descriptions agrees with GLiNER on 192 of 214 and moves recall
0.897 → 0.914 — but `HLCN` stays `security` at 0.56, `company` at 0.41, because the ontology defines a
ticker as a security and the model reports that ambiguity faithfully. Cloning each ambiguous mention into
every type block with ≥ 25% of the mass (17 clones) takes the *unchanged* resolver to B-cubed 1.000 at
precision 1.000 — no pairs hand-fed, no scoring rule touched. A `Choice` at 0.56/0.41 is a request to be
allowed both readings; type-scoped blocking that reads only the label makes a wall of it.

**It refuses, too.** On the shopping corpus's must-not-merge pairs (`Aurora 14` / `Aurora 14 Pro`,
`N600` / `N600X`) the repo's resolver merges three of twelve; the adjudicator merges none, while holding
every must-merge pair — zero false merges, zero false rejects. Putting the shopping ontology's own rule
(*"'Aurora 14' and 'Aurora 14 Pro' are two products, not one"*) into the question turns its one hedge
into `reject` at ≤ 0.06, at the cost of moving one cross-type must-merge pair from `merge` to `review`.
Nothing wrong, something curated.

**`pair_completeness` is not a ceiling on B-cubed recall.** 0.879 against a baseline recall of 0.897 and a
recovered recall of 1.000. Clustering is transitive, so it unites pairs blocking never proposed. What it
bounds is what an adjudicator can be *asked* about — which is the constraint that actually bit.

**Selection over enumerated pairs is the best relation extractor measured here — but only combined and
confidence-filtered** (notebook 11). Given GLiNER's entities, code enumerates every ontology-legal ordered pair
in a document and one `Choice` picks the relation or `none`. Alone it is not a better extractor: the same
recall as joint decoding at half the precision, because `none` (rightly) takes 64–71% of candidates and the
residue includes low-probability picks of the ontology's most permissive relation. But the two find
*different* things — six gold triples each that the other misses at paragraph scope — and at document scope
selection reaches recall 0.596, the highest on this corpus. Kept at `p ≥ 0.95`, gated, and unioned with
GLiNER's gated edges: **F1 0.442**, against 0.404 for notebook 10's best. Three high-confidence relabels of
GLiNER edges are all gold-correct.

**130 of GLiNER's 170 edges cross a paragraph.** The first draft of notebook 11 asserted zero, from a probe
with a precedence bug; the notebook's own check said 130 and reframed the experiment. Paragraph-scoped
candidate enumeration is a handicap, document scope is the fair comparison, and it costs 3× the requests.
Measure the scope.

**Whether two things are alternatives is a world-knowledge question, and one `Noul` answers it where
similarity cannot** (notebook 12). Over every pair of the things the agent-memory gold facts name, *are `a`
and `b` alternatives?* puts the two planted switches at 0.92 (`npm`/`pnpm`) and 0.63 (`Python`/`Go`) and
all 26 other pairs at ≤ 0.21. MiniLM cosine puts `Python`/`Go` at 0.166 — below most non-alternatives —
because it measures how alike two names are, and competing for the same role is not that. Plugged in as
`TemporalGraph(alternative_fn=JevAlternatives(ts))`, it is the whole integration.

**In the pipeline it never got to show it.** Four different `alternative_fn`s produced identical
supersessions, because extraction delivered exactly one real switch (`uses_tool npm → pnpm`) and even
embeddings cleared 0.55 on it, by 0.018. `Go` was never extracted; `prefers pnpm` was extracted a session
before the user switched. The judge is only as good as its queue — the third stage in a row to say so.

**Escalating the fast judge's uncertain calls to Claude Haiku bought nothing, and the sweep proves it**
(notebook 13). F1 0.404 with the System One gate; 0.400 sending its 31 least-confident edge judgments to
Haiku; 0.388 sending all 170 — recall identical throughout, precision falling as escalation rises, no width
of the band at which the slow judge helps. Haiku agreed with System One on 28 of the 31; the three flips
were a coin-flip acquisition the source itself hedges, and one spurious edge Haiku asserted. On the fourteen
`review` pairs, Haiku got **none** of the six gold-same right (*"different legal names"*), where the
adjudicator had calibratedly declined to decide. A larger model is not a curator. 20× the latency, $0.12.

**What is written down, the model reads; what is a definition, it has to be given** (notebook 14). On the
eight support threads where GLiNER classified priority at chance, a `Noul` on *is this resolved* — which is
in the customer's last turn — scores 8 of 8 with a clean gap (open 0.05–0.08, resolved 0.64–0.97). Intent
goes from 3 of 8 to 5 of 8; the three misses are the question reading the thread's end (*"Do it"* on a
refund) where the gold reads its opening ask. Priority stays at 3 of 8 with the misses *confident*, because
the `Score` levels were the notebook's and the gold's scale was never shown to it. Confidence routing sent
the three hedged calls to a person and let every confident disagreement through — it routes model
uncertainty, not definitional disagreement. Fix the question, not the threshold.

**Batching is a property of the state, not a flag.** 627 questions in 98 requests at ~180 ms each; edge
gating alone was 340 questions in 35, because one request carries a dozen edges over one shared document.
One question per request would re-send each document 34 times — ~128k tokens of document text against
~6.8k. The cost is paid in how the questions are written: every question in a request sees the same state
and question ids are never sent to the model, so each has to name its own edge.

## Notes

- Requires Python ≥3.10 (this repo pins 3.12). `sentencepiece` and `protobuf` are required —
  GLiNER2.5 uses a DeBERTa-v3 SPM tokenizer and will fail confusingly without them.
- Load with `AutoExtractor`, never `GLiNER2.from_pretrained` — the latter is the legacy span loader and
  raises `ArchitectureMismatchError` on a 2.5 checkpoint.
- Extraction is not bit-reproducible; exact counts will shift slightly between runs.
- GLiNER2.5's published benchmarks are vendor-reported and unreplicated. Nothing here depends on them.
