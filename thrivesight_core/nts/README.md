# nts: the element model

The periodic table of the soul from the Number Theory of the Soul v7, as data, record types and a reader prompt. It models the state of the elements in a person. It does not predict, and it trains no model: a model reads entries and proposes, the person confirms, and code keeps the record.

| File | Holds |
|---|---|
| `elements.json` | The 18 elements, 9 categories, v7's three planes (Interior S, Exterior TS, Transcendental T) and plane roles, prime squares, development stages, and the four Reality Field axes. The source of truth. |
| `composites.json` | Composites from v7 Parts XI and XX, each with a list of factor paths (Axiom 3 allows more than one), and the v7 composites left out because they use Memory or Image as a factor. |
| `table.py` | Loads both files and refuses a table that breaks v7's structure. `composite_id` names an unordered set of primes by their product; `factor` reverses it. |
| `records.py` | `ScopePath` (ordered, so `emotion[will]` is not `will[emotion]`), `RealityTags`, `Proposal` (a model's reading, which must quote the entry) and `LogEntry` (one confirmed record in Memory, owned by a person). |
| `prompts/reader.md` | The reader's system prompt. The category and element list is filled in from `elements.json`, so the prompt never drifts from the table. |
| `reader.py` | Builds the prompt and checks a model's answer. Citation fails closed: an unknown element, an identity offered as a finding, a bad scope path or a quote not in the entry is refused. |
| `agreement.py` | Per-prime agreement between the model and the person's own labels, with Cohen's kappa. This replaces training as the measure of whether the prompt works. |
| `run_reader.py` | Reads a file of entries with a model and prints readings and agreement. `--ollama <name>` uses a local Ollama server; `--model <id>` uses the Anthropic API and needs `ANTHROPIC_API_KEY`. |
| `fixtures/sample_entries.json` | Eight synthetic entries, unlabelled. Not anyone's journal. |

## Guided investigation (cues, then questions)

The first reader labelled entries; it read each one at the surface, and a journal entry is only a fragment. The investigation replaces labelling with asking:

| File | Holds |
|---|---|
| `../../lean/` | The Lean model: elements, scopes (ordered paths, rules built in, depth capped), the only two things a model may output (a cue or an open question), descriptive and normative records, and frames. `lake build` checks the proofs; `lake exe nts-export` writes `structure.json`. |
| `structure.json` | The Lean export the harness follows: primes, frames, the Will facet order, coercion sources, and each facet's stance. Regenerate it after changing the Lean model. |
| `prompts/cues.md` | The cue prompt: quote words worth asking about and say what they leave open. Never a feeling, motive or advice. |
| `questions/will.json` | Open questions for the decision frame, one per Will facet and one per coercion source. |
| `investigate.py` | Parses cues (fail closed), builds the question tree in the Lean order, and refuses leading or prescriptive questions. |

`reader.py` and `prompts/reader.md` are the first version (labels with strength) and stay for comparison.

## Journal, profile and beliefs

| File | Holds |
|---|---|
| `journal.py` | The journal. No model: a fixed "tell me more" after each thing the person writes, until they say they are done. Cues stay hidden unless the person asks, entry by entry (`show_cues`). |
| `person.py` | The profile (dated facts: a new value closes the old one, never overwrites it), the beliefs store, and grants. A fact or belief is either stated by the person or taken from a journal entry under a live grant for that store, quoting the entry; taken items are proposals until confirmed, and revoking a grant returns everything taken under it. |
| `records.py` `Act` | A will as weighted drivers (the person's weights, 0 to 10) plus what it was aimed at: whose, and which element. |
| `../../lean/NTS/Person.lean` | The same rules in Lean: two sources only, a revoked grant allows nothing, a grant feeds one store. |

## Decisions in the code

- Memory (0) is the append-only log of `LogEntry` records and Image (1) is the person id on each one. Neither is a factor in a product: Memory would make it 0 and Image changes nothing. So v7's Wisdom, Nostalgia and Purpose factorizations are listed as excluded until the theory decides.
- A product of primes cannot carry order, so a scoped experience is a `ScopePath`, and the product is only a name for which primes are involved.
- Memory may only end a scope path (Emotion[Memory] is remembering a feeling) and Image never appears in one, per v7 Part XXIII.

## Test

    python -m unittest thrivesight_core.nts.test_nts thrivesight_core.nts.test_investigate thrivesight_core.nts.test_person

## Run the reader on Ollama

    ollama pull qwen3.8
    python -m thrivesight_core.nts.run_reader --ollama qwen3.8 > readings.json

On Ollama Cloud, set `OLLAMA_HOST=https://ollama.com` and `OLLAMA_API_KEY`, and use a model from `https://ollama.com/api/tags`.
