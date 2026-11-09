/-!
# Elements

The 18 elements of the Number Theory of the Soul (v7, Part IV). The two identities are
kept out of `Prime` on purpose: Memory (0) and Image (1) are not things a question can
be about in the way a prime is, and neither has an interior to scope into (Part XXIII).
-/
namespace NTS

/-- The 16 primes, in number order. [D] -/
inductive Prime where
  | will | desire | imagination | emotion | conscience | attention | intuition
  | selfDifferentiation | attachment | reason | awareness | logos
  | embodiment | temporality | aestheticSense | numinousSense
  deriving DecidableEq, Repr, Inhabited

/-- The 18 elements: the two identities and the 16 primes. [D] -/
inductive Element where
  | memory
  | image
  | prime (p : Prime)
  deriving DecidableEq, Repr

namespace Prime

def all : List Prime :=
  [will, desire, imagination, emotion, conscience, attention, intuition,
   selfDifferentiation, attachment, reason, awareness, logos,
   embodiment, temporality, aestheticSense, numinousSense]

/-- The prime number each element carries in v7. [D] -/
def number : Prime → Nat
  | will => 2 | desire => 3 | imagination => 5 | emotion => 7 | conscience => 11
  | attention => 13 | intuition => 17 | selfDifferentiation => 19 | attachment => 23
  | reason => 29 | awareness => 31 | logos => 37 | embodiment => 41
  | temporality => 43 | aestheticSense => 47 | numinousSense => 53

def id : Prime → String
  | will => "will" | desire => "desire" | imagination => "imagination"
  | emotion => "emotion" | conscience => "conscience" | attention => "attention"
  | intuition => "intuition" | selfDifferentiation => "self-differentiation"
  | attachment => "attachment" | reason => "reason" | awareness => "awareness"
  | logos => "logos" | embodiment => "embodiment" | temporality => "temporality"
  | aestheticSense => "aesthetic-sense" | numinousSense => "numinous-sense"

/-- The nine categories of v7 Part V. [D] -/
inductive Category where
  | drives | generators | sensors | regulators | illuminators | binders | matrices | apprehensions
  deriving DecidableEq, Repr

def category : Prime → Category
  | will | desire => .drives
  | imagination | logos => .generators
  | emotion | intuition => .sensors
  | conscience | reason => .regulators
  | attention | awareness => .illuminators
  | selfDifferentiation | attachment => .binders
  | embodiment | temporality => .matrices
  | aestheticSense | numinousSense => .apprehensions

theorem all_length : all.length = 16 := rfl
theorem all_nodup : all.Nodup := by decide
theorem mem_all (p : Prime) : p ∈ all := by cases p <;> decide

/-- Every prime number is distinct, so a prime's number names it. -/
theorem number_injective : ∀ p q : Prime, p.number = q.number → p = q := by
  intro p q h; cases p <;> cases q <;> first | rfl | (simp [number] at h)

/-- Each category holds exactly two primes (v7 Part V). -/
theorem two_per_category (c : Category) : (all.filter (fun p => p.category == c)).length = 2 := by
  cases c <;> decide

end Prime
end NTS
