# Comparative Evaluation Framework: Identity-Modeled AI Context vs. General-Purpose AI Memory

**Version**: 1.0
**Status**: Internal Research Document (Zone 1)
**Last Updated**: 2026-03-11

---

## 1. Research Motivation

AI systems that maintain user context across conversations are proliferating. General-purpose memory frameworks (Cognee, Mem0, Zep, LangMem) extract and store knowledge from conversations automatically, then retrieve relevant fragments to enrich future AI responses. This approach is domain-agnostic — the system doesn't know *what kind* of knowledge matters, it simply stores everything and retrieves what seems relevant.

Polity takes a different approach: it models a specific domain of human identity — civic values, psychological structure, and philosophical commitments — as a structured ontology, then uses that model to enrich AI conversations. The system doesn't extract memory from conversations; it builds an *identity model* through structured interactions (provocations, value orderings, soul orderings) and injects that model as context.

The central question is whether this domain-specific, ontology-driven approach produces meaningfully better AI conversations than general-purpose memory systems — or whether the added complexity of designing and maintaining a bespoke ontology yields negligible improvement over automated extraction.

## 2. Research Questions

**RQ1 (Conversation Quality):** Does identity-modeled context produce AI responses that are more personally relevant, more insightful, and more therapeutically useful than responses enriched by general-purpose memory?

**RQ2 (Context Efficiency):** Does a structured ontology encode more useful information per token than auto-extracted knowledge graphs? Is less context needed to achieve the same response quality?

**RQ3 (User Experience):** Do users perceive greater understanding, trust, and value from an identity-aware AI compared to a memory-augmented AI or a baseline AI with no context?

**RQ4 (Differentiation Threshold):** At what point does the identity model's advantage diminish? Are there conversation types where general-purpose memory performs equally well or better?

## 3. Experimental Design

### 3.1 Conditions

Five experimental conditions, each representing a different context-enrichment strategy applied to the same underlying LLM (Claude Sonnet):

| Condition | Label | Description |
|-----------|-------|-------------|
| **C0** | Vanilla | No user context. Fresh system prompt with therapeutic instructions only. |
| **C1** | History-Only | Raw conversation history in the context window. No graph, no extraction, no enrichment. The LLM sees prior messages and must infer patterns on its own. |
| **C2** | Cognee Memory | Cognee-extracted knowledge graph from the same conversation history. Auto-extracted entities, relationships, and summaries retrieved and injected as context. |
| **C3** | Polity Signals | Polity's Signal Address System context only — extracted signals, clusters, observation biases, persona-adapted context packet. No Journey identity data. |
| **C4** | Polity Full | Polity's full pipeline — SA signals + Journey civic identity (value profile, soul profile, regime type, virtues, stated vs. revealed ordering). |

**Why five conditions, not two:** Comparing only Polity vs. Cognee conflates multiple variables. By including Vanilla (C0), History-Only (C1), and Polity-without-Journey (C3), we can isolate:
- C1 vs. C0: Does *any* context help?
- C2 vs. C1: Does Cognee's extraction improve over raw history?
- C3 vs. C2: Does Polity's signal system outperform general extraction?
- C4 vs. C3: Does the Journey identity layer add value on top of signals?

### 3.2 Test Scenarios

Each condition is evaluated across a matrix of conversation scenarios designed to stress-test different aspects of contextual understanding.

#### 3.2.1 Scenario Dimensions

**Message Types** (5):
- **Emotional disclosure** — "My manager dismissed my idea in front of the whole team yesterday."
- **Pattern seeking** — "I keep noticing that every time my partner brings up money, I get defensive."
- **Decision weighing** — "I'm trying to decide whether to confront my friend or let it go."
- **Vague/unfocused** — "Something just feels off lately. I can't put my finger on it."
- **Bias-laden** — "My sister always ignores me. She never listens. She proved it again today."

**Conversation Depth** (3):
- **First-contact** — No prior history. User is brand new.
- **Mid-session** — 5-10 prior exchanges. Some patterns should be emerging.
- **Deep-history** — 20+ prior exchanges across multiple sessions. Rich signal and cluster data available.

**Topic Domains** (4):
- **Interpersonal conflict** — Workplace, family, romantic relationships
- **Self-reflection** — Identity, purpose, values, personal growth
- **Civic/political** — Governance, justice, community responsibility (where Journey data is most relevant)
- **Practical decisions** — Career moves, financial choices, logistics (negative control — identity context should matter less)

**Total scenario matrix**: 5 × 3 × 4 = 60 unique scenarios per condition. Across 5 conditions = 300 total evaluations.

#### 3.2.2 Simulated User Profiles

To test identity-aware context, we need users with defined Journey profiles. Three synthetic profiles represent distinct civic identities:

**Profile A: "The Principled Dissenter"**
- Path: Philosopher
- Soul ordering: Reason rules (aristocratic regime)
- Top values: Justice, Liberty, Dignity
- Bottom values: Authority, Order, Sovereignty
- Key tension: Justice vs. Order (high deliberation time)
- Virtues: Wisdom, Courage, Moderation, Justice (full set)

**Profile B: "The Loyal Builder"**
- Path: Sovereign
- Soul ordering: Spirit rules (timocratic regime)
- Top values: Solidarity, Order, Authority
- Bottom values: Pluralism, Liberty, Merit
- Key tension: Solidarity vs. Merit
- Virtues: Courage

**Profile C: "The Pragmatic Pluralist"**
- Path: Wanderer
- Soul ordering: Equal (democratic regime)
- Top values: Pluralism, Prosperity, Equality
- Bottom values: Sovereignty, Stewardship, Dignity
- Key tension: Equality vs. Merit (very high deliberation)
- Virtues: (none — no part dominates)

Each profile generates a different Journey context injection. The evaluation measures whether the AI's responses adapt to these different identities.

### 3.3 Cognee Baseline Setup

Cognee (v0.1.x) is installed as a Python dependency and configured with:
- **Graph backend**: Neo4j (same as Polity, ensuring fair comparison)
- **Embedding model**: Same embedding model used by Polity's signal engine
- **Ingestion**: Same conversation history fed to Polity is also ingested into Cognee
- **Retrieval**: Cognee's `cognee.search()` API with `search_type="insights"` for knowledge graph retrieval

The Cognee condition (C2) receives whatever Cognee's automated pipeline extracts from the same raw conversation data that Polity processes. This is a fair comparison: same input data, different processing approaches.

## 4. Metrics and Measurement

### 4.1 Conversation Quality (Automated — LLM-as-Judge)

An evaluator LLM (Claude Opus, separate from the response-generating Sonnet) scores each AI response on five dimensions using a structured rubric:

#### 4.1.1 Scoring Rubric

**Personalization Depth** (1-5):
- 1: Generic response. Could apply to anyone.
- 2: References the topic but not the person's specific situation.
- 3: Acknowledges the person's specific details from the current message.
- 4: Connects current message to prior patterns or known context.
- 5: Integrates identity-level understanding (values, psychological patterns, internal tensions) into the response.

**Follow-Up Quality** (1-5):
- 1: No follow-up question, or a generic "how does that make you feel?"
- 2: Follow-up relates to the topic but is surface-level.
- 3: Follow-up probes a specific dimension of the situation.
- 4: Follow-up addresses an unresolved dimension or unexplored tension.
- 5: Follow-up reveals a pattern, bias, or value tension the user hasn't explicitly articulated.

**Pattern Recognition** (1-5):
- 1: No patterns identified.
- 2: Restates what the user already said about patterns.
- 3: Identifies a pattern from explicit information.
- 4: Connects signals across sessions or contexts.
- 5: Surfaces a non-obvious pattern that integrates emotional, relational, and identity dimensions.

**Therapeutic Safety** (1-5):
- 1: Response could cause harm (invalidating, diagnosing, prescribing).
- 2: Response is neutral but unhelpful.
- 3: Response validates and reflects without adding insight.
- 4: Response validates, reflects, and gently challenges.
- 5: Response balances validation with challenge, respects autonomy, avoids bias reinforcement.

**Identity Coherence** (1-5) — *only scored for C3 and C4*:
- 1: Context is present but ignored or contradicted.
- 2: Context is acknowledged but not meaningfully integrated.
- 3: Response is consistent with the user's known profile.
- 4: Response actively draws on identity context to personalize guidance.
- 5: Response demonstrates deep integration — e.g., framing a decision in terms of the user's own value hierarchy.

#### 4.1.2 Evaluator Prompt Design

The evaluator LLM receives:
1. The user's message
2. The AI's response (condition label blinded)
3. The scoring rubric
4. The user's ground-truth profile (for Identity Coherence scoring)

The evaluator does NOT receive:
- Which condition generated the response
- The system prompt or context packet used
- Any information about Polity or Cognee

This ensures blind evaluation. Each response is scored independently.

#### 4.1.3 Inter-Rater Reliability

Each response is evaluated 3 times (3 separate evaluator LLM calls with temperature > 0). Scores are averaged. If standard deviation > 1.0 on any dimension, a fourth evaluation is triggered and the outlier is dropped.

### 4.2 Context Efficiency (Automated)

#### 4.2.1 Token Metrics

For each condition, measure:
- **Context tokens**: Number of tokens in the injected context (system prompt enrichment)
- **Quality-per-token**: (Average quality score across 4 dimensions) / (context tokens)
- **Marginal gain**: Quality improvement over C0 (vanilla) per additional token of context
- **Compression ratio**: (Raw conversation history tokens) / (processed context tokens)

#### 4.2.2 Retrieval Precision

For conditions C2, C3, C4 — measure what fraction of the injected context is actually *used* by the LLM in its response:
- Extract claims/references in the AI response that trace to specific context items
- **Precision**: (context items referenced in response) / (total context items provided)
- **Waste**: (context items never referenced) / (total context items provided)

### 4.3 User Experience (Human Evaluation Protocol)

#### 4.3.1 Study Design

**Participants**: 12-20 participants recruited from the target user population (adults interested in self-reflection and personal growth).

**Protocol**: Within-subjects design. Each participant:
1. Completes the Journey provocation sequence (12 provocations) to generate a real identity profile
2. Has 3 conversations (one per condition: C0, C2, C4) in randomized order
3. Each conversation is 8-10 exchanges on the same topic
4. After each conversation, completes the evaluation questionnaire
5. After all three, completes a comparative ranking

**Blinding**: Participants do not know which condition they are experiencing. The UI is identical across conditions.

#### 4.3.2 Post-Conversation Questionnaire (7-point Likert scale)

1. "The AI understood what I was really feeling." (Perceived Understanding)
2. "The AI's responses felt personally relevant to me, not generic." (Personalization)
3. "The AI asked follow-up questions that helped me think more deeply." (Depth)
4. "I felt comfortable being vulnerable with this AI." (Trust/Safety)
5. "The AI told me something about myself I hadn't considered before." (Insight Novelty)
6. "I would want to continue talking with this AI." (Engagement)
7. "This conversation helped me understand my situation better." (Utility)

#### 4.3.3 Comparative Ranking

After all three conversations:
- "Which conversation felt most like talking to someone who knows you?" (Forced ranking: 1st, 2nd, 3rd)
- "Which conversation produced the most useful insight?" (Forced ranking)
- "Which AI would you choose to talk to again?" (Single choice)
- Open-ended: "What, if anything, felt different between the three conversations?"

#### 4.3.4 Analysis Plan

- **Likert scores**: Friedman test (non-parametric repeated measures) across conditions, followed by Wilcoxon signed-rank pairwise comparisons with Bonferroni correction.
- **Rankings**: Kendall's W for concordance; chi-squared test for preference distribution.
- **Open-ended responses**: Thematic analysis with two independent coders. Inter-coder reliability via Cohen's kappa.
- **Effect sizes**: Report Cliff's delta for all pairwise comparisons.

## 5. Automated Evaluation Pipeline

### 5.1 Architecture

```
eval_comparative.py
    |
    |-- ScenarioGenerator
    |       Produces 60 scenario specs from the matrix
    |
    |-- ConditionRunner (one per condition)
    |       C0: VanillaRunner — bare system prompt
    |       C1: HistoryRunner — raw message history in context
    |       C2: CogneeRunner — Cognee pipeline → context injection
    |       C3: PolitySignalRunner — SA pipeline → context packet
    |       C4: PolityFullRunner — SA pipeline + Journey context
    |
    |-- ResponseCollector
    |       Stores (scenario, condition, system_prompt, context_tokens, response, latency)
    |
    |-- QualityEvaluator (LLM-as-Judge)
    |       3x blind evaluation per response
    |       Scores: personalization, follow_up, pattern, safety, identity_coherence
    |
    |-- EfficiencyAnalyzer
    |       Token counts, quality-per-token, compression ratios, retrieval precision
    |
    |-- ReportGenerator
    |       JSON results + Markdown report + statistical analysis
```

### 5.2 Output Format

**eval_comparative_results.json**:
```json
{
  "metadata": {
    "version": "1.0",
    "timestamp": "2026-03-11T...",
    "model": "claude-sonnet-4-5-20250929",
    "evaluator_model": "claude-opus-4-6",
    "scenarios_total": 300,
    "conditions": ["C0", "C1", "C2", "C3", "C4"]
  },
  "results": [
    {
      "scenario_id": "S001",
      "message_type": "emotional_disclosure",
      "depth": "first_contact",
      "domain": "interpersonal_conflict",
      "profile": "A",
      "conditions": {
        "C0": {
          "response": "...",
          "context_tokens": 0,
          "latency_ms": 1200,
          "scores": {
            "personalization": {"mean": 2.0, "std": 0.5, "raw": [2, 2, 2]},
            "follow_up": {"mean": 2.7, "std": 0.6, "raw": [3, 2, 3]},
            "pattern": {"mean": 1.0, "std": 0.0, "raw": [1, 1, 1]},
            "safety": {"mean": 4.0, "std": 0.0, "raw": [4, 4, 4]},
            "identity_coherence": null
          }
        },
        "C4": {
          "response": "...",
          "context_tokens": 147,
          "latency_ms": 1350,
          "scores": {
            "personalization": {"mean": 4.3, "std": 0.6, "raw": [4, 5, 4]},
            "follow_up": {"mean": 4.0, "std": 0.0, "raw": [4, 4, 4]},
            "pattern": {"mean": 3.7, "std": 0.6, "raw": [4, 3, 4]},
            "safety": {"mean": 4.7, "std": 0.6, "raw": [5, 4, 5]},
            "identity_coherence": {"mean": 4.3, "std": 0.6, "raw": [4, 5, 4]}
          }
        }
      }
    }
  ],
  "summary": {
    "by_condition": {
      "C0": {"mean_quality": 2.4, "mean_tokens": 0},
      "C1": {"mean_quality": 3.1, "mean_tokens": 2400},
      "C2": {"mean_quality": 3.5, "mean_tokens": 800},
      "C3": {"mean_quality": 3.8, "mean_tokens": 450},
      "C4": {"mean_quality": 4.2, "mean_tokens": 600}
    },
    "statistical_tests": {
      "friedman_chi2": 47.3,
      "friedman_p": 0.00001,
      "pairwise": {
        "C4_vs_C2": {"wilcoxon_z": -3.2, "p": 0.001, "cliffs_delta": 0.45},
        "C4_vs_C0": {"wilcoxon_z": -5.1, "p": 0.00001, "cliffs_delta": 0.78}
      }
    }
  }
}
```

### 5.3 Integration with Existing Infrastructure

The framework extends, rather than replaces, existing evaluation tools:

- **eval_baseline.py** scenarios become a subset of the C3/C4 test matrix
- **reflect_benchmark.py** dimensions (SIG, CLU, CTX, etc.) become precondition checks — if the pipeline isn't passing benchmarks, comparative evaluation results are unreliable
- **Synthetic user profiles** build on the Journey ontology data already defined in `ontology_data.py`

## 6. Hypotheses and Expected Outcomes

### 6.1 Primary Hypotheses

**H1**: C4 (Polity Full) will score significantly higher than C2 (Cognee) on Personalization Depth (>0.5 point mean difference, p < 0.05).

**H2**: C4 will score significantly higher than C2 on Follow-Up Quality in civic/self-reflection domains but not in practical decision domains.

**H3**: C4 will achieve equal or higher quality scores with fewer context tokens than C1 (History-Only), demonstrating superior compression.

**H4**: C3 (Polity Signals without Journey) will outperform C2 (Cognee) on Pattern Recognition, indicating that the signal architecture itself — not just the identity layer — provides differentiated value.

**H5**: The quality gap between C4 and C2 will be largest in civic/political topic domains and smallest in practical decision domains.

### 6.2 Falsification Criteria

The identity-modeling approach should be reconsidered if:
- C4 does not significantly outperform C2 on any quality dimension (p > 0.10 on all comparisons)
- C2 achieves higher quality-per-token than C4 (general memory is more efficient)
- Human evaluators do not distinguish C4 from C2 in forced-ranking tasks (Kendall's W < 0.3)
- The quality advantage of C4 over C3 is not significant (the Journey layer adds no value over signals alone)

These falsification criteria are designed to be honest. If the structured ontology approach doesn't measurably outperform automated extraction, the engineering investment isn't justified regardless of theoretical elegance.

## 7. Implementation Roadmap

### Phase 1: Automated Infrastructure (1-2 weeks)
- [ ] Install and configure Cognee with Neo4j backend
- [ ] Build ScenarioGenerator from the 5×3×4 matrix
- [ ] Implement ConditionRunners for all 5 conditions
- [ ] Build QualityEvaluator with blind LLM-as-judge scoring
- [ ] Build EfficiencyAnalyzer for token metrics
- [ ] Create synthetic user profiles with Journey data

### Phase 2: Automated Evaluation Run (1 week)
- [ ] Generate all 300 scenarios
- [ ] Run all conditions (expect ~2-3 hours of API calls)
- [ ] Collect and validate quality scores
- [ ] Run statistical analysis
- [ ] Generate comparative report

### Phase 3: Human Evaluation Design (1-2 weeks)
- [ ] Build blinded conversation UI (identical across conditions)
- [ ] Design and pilot-test the questionnaire
- [ ] Recruit 12-20 participants
- [ ] Run user study sessions
- [ ] Analyze quantitative and qualitative results

### Phase 4: Synthesis and Publication (1 week)
- [ ] Combine automated and human evaluation results
- [ ] Write findings section with visualizations
- [ ] Identify implications for system design
- [ ] Determine whether to proceed, pivot, or hybridize approaches

## 8. Ethical Considerations

- **Simulated conversations**: Automated evaluation uses synthetic scenarios, not real user data. No personal information is processed.
- **Human study**: Requires informed consent. Participants must know they are interacting with an AI. Conversations may touch on emotional topics; participants can stop at any time.
- **Bias in evaluation**: LLM-as-judge may have systematic biases (e.g., preferring longer responses). Mitigated by multi-evaluation averaging and human study cross-validation.
- **Cognee fairness**: Cognee is evaluated using its recommended configuration, not a deliberately weakened setup. Any configuration choices are documented and justified.

## 9. Limitations

- **Single LLM**: All conditions use Claude Sonnet. Results may not generalize to other LLMs.
- **Synthetic profiles**: Automated evaluation uses designed profiles, not emergent ones from real Journey completions. Human study mitigates this.
- **Topic coverage**: 60 scenarios cannot cover all possible conversation types. Results indicate trends, not universal claims.
- **Temporal scope**: Evaluation is cross-sectional. Long-term effects of identity modeling (does it get better over months?) are not measured.
- **Cognee version**: Cognee is evolving rapidly. Results reflect a specific version and configuration.

---

## Appendix A: Evaluator Prompt Template

```
You are evaluating the quality of an AI conversation response. You will see a user's message and the AI's response. Score the response on the following dimensions using a 1-5 scale.

USER MESSAGE:
{user_message}

AI RESPONSE:
{ai_response}

USER PROFILE (for Identity Coherence scoring only):
{user_profile_summary}

SCORING RUBRIC:
[Full rubric from Section 4.1.1 inserted here]

Provide your scores as JSON:
{
  "personalization": <1-5>,
  "personalization_reasoning": "<brief explanation>",
  "follow_up": <1-5>,
  "follow_up_reasoning": "<brief explanation>",
  "pattern": <1-5>,
  "pattern_reasoning": "<brief explanation>",
  "safety": <1-5>,
  "safety_reasoning": "<brief explanation>",
  "identity_coherence": <1-5 or null if not applicable>,
  "identity_coherence_reasoning": "<brief explanation or null>"
}
```

## Appendix B: Synthetic Profile Details

### Profile A: "The Principled Dissenter"
```json
{
  "path": "philosopher",
  "mode": "classical",
  "phase": "provocations_complete",
  "soul_profile": {
    "revealed_ordering": {"type": "one-rules", "ruler": "reason", "second": "spirit", "third": "appetite"},
    "stated_ordering": {"type": "one-rules", "ruler": "reason", "second": "spirit", "third": "appetite"},
    "regime": "aristocratic",
    "virtues": ["wisdom", "courage", "moderation", "justice"]
  },
  "value_profile": {
    "hierarchy": [
      {"valueId": "justice", "rank": 1}, {"valueId": "liberty", "rank": 2},
      {"valueId": "dignity", "rank": 3}, {"valueId": "equality", "rank": 4},
      {"valueId": "pluralism", "rank": 5}, {"valueId": "merit", "rank": 6},
      {"valueId": "solidarity", "rank": 7}, {"valueId": "stewardship", "rank": 8},
      {"valueId": "prosperity", "rank": 9}, {"valueId": "sovereignty", "rank": 10},
      {"valueId": "order", "rank": 11}, {"valueId": "authority", "rank": 12}
    ],
    "most_conflicted": {"valueId": "justice", "avgDeliberationMs": 18500}
  }
}
```

### Profile B: "The Loyal Builder"
```json
{
  "path": "sovereign",
  "mode": "builder",
  "phase": "provocations_complete",
  "soul_profile": {
    "revealed_ordering": {"type": "one-rules", "ruler": "spirit", "second": "appetite", "third": "reason"},
    "stated_ordering": {"type": "one-rules", "ruler": "spirit", "second": "reason", "third": "appetite"},
    "regime": "timocratic",
    "virtues": ["courage"]
  },
  "value_profile": {
    "hierarchy": [
      {"valueId": "solidarity", "rank": 1}, {"valueId": "order", "rank": 2},
      {"valueId": "authority", "rank": 3}, {"valueId": "stewardship", "rank": 4},
      {"valueId": "sovereignty", "rank": 5}, {"valueId": "dignity", "rank": 6},
      {"valueId": "prosperity", "rank": 7}, {"valueId": "justice", "rank": 8},
      {"valueId": "equality", "rank": 9}, {"valueId": "merit", "rank": 10},
      {"valueId": "liberty", "rank": 11}, {"valueId": "pluralism", "rank": 12}
    ],
    "most_conflicted": {"valueId": "solidarity", "avgDeliberationMs": 22000}
  }
}
```

### Profile C: "The Pragmatic Pluralist"
```json
{
  "path": "wanderer",
  "mode": "explorer",
  "phase": "provocations_complete",
  "soul_profile": {
    "revealed_ordering": {"type": "equal"},
    "stated_ordering": {"type": "co-rulers", "rulers": ["reason", "appetite"], "subordinate": "spirit"},
    "regime": "democratic",
    "virtues": []
  },
  "value_profile": {
    "hierarchy": [
      {"valueId": "pluralism", "rank": 1}, {"valueId": "prosperity", "rank": 2},
      {"valueId": "equality", "rank": 3}, {"valueId": "liberty", "rank": 4},
      {"valueId": "merit", "rank": 5}, {"valueId": "justice", "rank": 6},
      {"valueId": "solidarity", "rank": 7}, {"valueId": "order", "rank": 8},
      {"valueId": "authority", "rank": 9}, {"valueId": "dignity", "rank": 10},
      {"valueId": "stewardship", "rank": 11}, {"valueId": "sovereignty", "rank": 12}
    ],
    "most_conflicted": {"valueId": "equality", "avgDeliberationMs": 25000}
  }
}
```
