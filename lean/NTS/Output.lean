import NTS.Scope
/-!
# What the model may produce, and what the person's answers become

The model reads language and asks. It never prescribes and never predicts. That is
enforced by shape: `ModelOutput` has two constructors, a cue and an open question, and
nothing else. There is no constructor for "you should" or "you will". Wording (a question
that smuggles an assumption, such as "weren't you disappointed?") cannot be ruled out by a
type, so the harness checks question text as well.

A person's answer becomes a record, and a record is descriptive or normative by the facet
it answers, never by the model's choice. So a description can never be stored as a
belief about what ought to be, or the reverse.
-/
namespace NTS

/-- The four Reality Field axes of v7 Part VI. [D] -/
inductive Axis where
  | sharedInner | embodiedImagined | presentRepresented | voluntaryInvoluntary
  deriving DecidableEq, Repr

/-- Answers on the will axis. v7's axis is binary; `coerced` is Torian's extension
(2026-11-09): an act can be involuntary because something forced it. [S] False if every
coerced act turns out to be either voluntary or involuntary on reflection. -/
inductive WillPole where
  | voluntary | involuntary | coerced
  deriving DecidableEq, Repr

/-- What a question at a scope asks about: one of the primes as an aspect of that scope,
or one of the Reality axes. [D] -/
inductive Facet where
  | prime (p : Prime)
  | axis (a : Axis)
  deriving DecidableEq, Repr

/-- Descriptive: what is or was, as the person sees it. Normative: what the person holds
ought to be. [D] -/
inductive Stance where
  | descriptive | normative
  deriving DecidableEq, Repr

/-- A facet's stance. Conscience is the faculty that applies the moral standard (Part IV),
so its questions ask what the person holds ought to be; every other facet asks what is. [S]
False if some other prime's questions turn out to elicit normative answers. -/
def Facet.stance : Facet → Stance
  | .prime .conscience => .normative
  | _ => .descriptive

/-- A cue: an element the words suggest, with the exact words. A cue is a reason to ask,
not a finding. [D] -/
structure Cue where
  element : Prime
  quote : String
  deriving Repr

/-- An open question at a scope, about one facet. [D] -/
structure Question where
  scope : Scope
  facet : Facet
  text : String
  deriving Repr

/-- Everything the model may output. [D] -/
inductive ModelOutput where
  | cue (c : Cue)
  | ask (q : Question)
  deriving Repr

/-- A record made from the person's own answer. Its stance is computed from the facet. [D] -/
structure Record where
  address : Address
  facet : Facet
  answer : String
  confirmedBy : String

def Record.stance (r : Record) : Stance := r.facet.stance

/-- An answer about whether one wanted something is always descriptive. -/
example (a : Address) (s p : String) : (Record.mk a (.prime .desire) s p).stance = .descriptive := rfl

/-- An answer about what one ought to have done is always normative. -/
example (a : Address) (s p : String) : (Record.mk a (.prime .conscience) s p).stance = .normative := rfl

/-- The model's output is a cue or a question: there is no third kind. -/
theorem output_is_cue_or_ask (o : ModelOutput) : (∃ c, o = .cue c) ∨ (∃ q, o = .ask q) := by
  cases o with
  | cue c => exact .inl ⟨c, rfl⟩
  | ask q => exact .inr ⟨q, rfl⟩

end NTS
