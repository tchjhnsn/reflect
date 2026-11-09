import NTS.Elements
/-!
# Scope

A scope is an ordered path into the fractal (v7 Parts VIII and XXIII). `root will` is
Will; `inner (root will) emotion` is Will[Emotion], Emotion as it operates inside the
volitional domain. The rules of Part XXIII hold by construction, because a path that
breaks them cannot be written:

* a path starts at a prime, since neither identity has an interior;
* Image never appears in a path, since it is the person at every depth;
* Memory appears only as the register of a scope (`Address.register`), never inside one.

An investigation is a tree of scopes. Its depth is capped (finite embodiment, Axiom 12),
so every investigation ends.
-/
namespace NTS

/-- An ordered path of primes, outermost first. [D] -/
inductive Scope where
  | root (p : Prime)
  | inner (outer : Scope) (p : Prime)
  deriving DecidableEq, Repr

namespace Scope

def depth : Scope → Nat
  | root _ => 1
  | inner s _ => s.depth + 1

def outermost : Scope → Prime
  | root p => p
  | inner s _ => s.outermost

def innermost : Scope → Prime
  | root p => p
  | inner _ p => p

def path : Scope → List Prime
  | root p => [p]
  | inner s p => s.path ++ [p]

/-- v7's notation: `will[emotion]`. -/
def render : Scope → String
  | root p => p.id
  | inner s p => s.render ++ "[" ++ p.id ++ "]"

theorem path_length (s : Scope) : s.path.length = s.depth := by
  induction s with
  | root p => rfl
  | inner s p ih => simp [path, depth, ih]

/-- Order matters (Part XXIII, phenomenological asymmetry): Will[Emotion] is not
Emotion[Will], although 2 × 7 = 7 × 2. -/
theorem asymmetric : inner (root .will) .emotion ≠ inner (root .emotion) .will := by decide

/-- The scopes one level inside `s`: one for each prime (the fractal principle). -/
def children (s : Scope) : List Scope := Prime.all.map (inner s)

/-- Every scope inside `s` down to `d` more levels, breadth first. Structural on `d`,
so it always ends. -/
def expand (s : Scope) : Nat → List Scope
  | 0 => []
  | d + 1 => s.children ++ (s.children.flatMap (fun c => c.expand d))

theorem children_length (s : Scope) : s.children.length = 16 := by
  simp [children, Prime.all_length]

theorem children_depth (s c : Scope) (h : c ∈ s.children) : c.depth = s.depth + 1 := by
  simp [children] at h
  obtain ⟨p, _, rfl⟩ := h
  rfl

end Scope

/-- Where a record is written: a scope, or the register (Memory) of a scope.
`⟨inner (root emotion) will, true⟩` is Emotion[Will][Memory]. [D] -/
structure Address where
  scope : Scope
  register : Bool := false
  deriving DecidableEq, Repr

/-- The depth cap that stands in for finite embodiment. [P] A parameter: v7 says the
architecture has no floor but embodied experience does; the value is a product choice. -/
def maxDepth : Nat := 3

/-- An investigation from a root down to the cap visits a finite, known number of
scopes: 16 + 16² for two levels below the root. -/
example : ((Scope.root .will).expand 2).length = 16 + 16 * 16 := by native_decide

end NTS
