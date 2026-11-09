import NTS.Output
/-!
# Frames

A frame is what the person wants to investigate. It picks the root scope and the order in
which facets are asked. The introductory reflection starts with embodied life before the
mind (Torian, 2026-11-09): a mind asked to evaluate itself produces its own evidence,
whereas embodied answers are evidence produced by reality. v7 agrees in three places: the
write order of Part VI starts with the Matrices, the Reality-Governance Principle gives the
body default authority, and Part XIX has the Matrices present before birth.
-/
namespace NTS

inductive Frame where
  | introduction  -- the first reflection: embodied life first
  | feeling       -- why did this make me feel this way?
  | decision      -- why did I do this, or why do I make this kind of decision?
  | belief        -- why do I hold this to be true?
  | identity      -- who am I in this?
  deriving DecidableEq, Repr

def Frame.root : Frame → Prime
  | .introduction => .embodiment
  | .feeling => .emotion
  | .decision => .will
  | .belief => .reason
  | .identity => .selfDifferentiation

/-- Facet order inside the decision frame's root, Will. The Reality question comes first
(was it chosen at all?), then what moved it, who it involved, what else was possible, why
this, whether it ought to have been, and when. [P] -/
def willOrder : List Facet :=
  [.axis .voluntaryInvoluntary, .prime .desire, .prime .attachment, .prime .imagination,
   .prime .reason, .prime .conscience, .prime .temporality, .prime .emotion,
   .prime .embodiment, .prime .intuition, .prime .selfDifferentiation, .prime .attention,
   .prime .awareness, .prime .logos, .prime .aestheticSense, .prime .numinousSense,
   .prime .will]

/-- When an act was coerced, the next level asks where the force came from: a person, the
body, a duty, or circumstance in shared reality. [P] -/
def coercionSources : List Facet :=
  [.prime .attachment, .prime .embodiment, .prime .conscience, .axis .sharedInner]

/-- The introduction's top-level order: embodied life, then time, then bonds, then the mind. [P] -/
def introductionOrder : List Prime :=
  [.embodiment, .temporality, .attachment, .will, .desire, .emotion, .intuition,
   .selfDifferentiation, .attention, .awareness, .imagination, .logos, .reason,
   .conscience, .aestheticSense, .numinousSense]

/-- The Will order asks about every prime exactly once, plus the will axis. -/
theorem willOrder_covers (p : Prime) : Facet.prime p ∈ willOrder := by cases p <;> decide
theorem willOrder_nodup : willOrder.Nodup := by decide

/-- The introduction covers all 16 primes once. -/
theorem introduction_covers (p : Prime) : p ∈ introductionOrder := by cases p <;> decide
theorem introduction_nodup : introductionOrder.Nodup := by decide

/-- The introduction asks about the Matrices before any Regulator or Illuminator: the body
and time before the mind evaluates itself. -/
theorem body_before_mind :
    introductionOrder.idxOf .embodiment < introductionOrder.idxOf .reason ∧
    introductionOrder.idxOf .embodiment < introductionOrder.idxOf .conscience ∧
    introductionOrder.idxOf .temporality < introductionOrder.idxOf .awareness ∧
    introductionOrder.idxOf .temporality < introductionOrder.idxOf .attention := by decide

end NTS
