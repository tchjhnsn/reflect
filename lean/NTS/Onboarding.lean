import NTS.Person
/-!
# Onboarding

The first thing a person does in Reflect is answer five questions: their name, their date of
birth, their work, their closest friend, and the puzzle of their life (Torian, 2026-11-09).
Every answer is stated by the person and may be skipped. No model runs.

The order follows the introduction frame: shared and embodied life first, then bonds, then
the mind. The puzzle asks the mind to look at itself, so it comes last.
-/
namespace NTS

/-- The five onboarding questions. [D] -/
inductive OnboardingQuestion where
  | name | birthDate | work | closestFriend | puzzle
  deriving DecidableEq, Repr

/-- Where an answer is filed. [D] -/
inductive Destination where
  | profile (key : String)
  | relationship (kind : RelationKind)
  | puzzles
  deriving DecidableEq, Repr

def OnboardingQuestion.id : OnboardingQuestion → String
  | .name => "name" | .birthDate => "date_of_birth" | .work => "occupation"
  | .closestFriend => "closest_friend" | .puzzle => "puzzle"

def OnboardingQuestion.destination : OnboardingQuestion → Destination
  | .name => .profile "name"
  | .birthDate => .profile "date_of_birth"
  | .work => .profile "occupation"
  | .closestFriend => .relationship .closestFriend
  | .puzzle => .puzzles

/-- The order the questions are asked in. [P] -/
def onboardingOrder : List OnboardingQuestion := [.name, .birthDate, .work, .closestFriend, .puzzle]

theorem onboarding_covers (q : OnboardingQuestion) : q ∈ onboardingOrder := by cases q <;> decide
theorem onboarding_nodup : onboardingOrder.Nodup := by decide

/-- The puzzle is asked last, after every question about shared life and bonds. -/
theorem puzzle_last : onboardingOrder.getLast? = some .puzzle := by decide

/-- What one answered question writes. Skipped questions write nothing. [D] -/
inductive Written where
  | fact (f : Fact)
  | relationship (r : Relationship)
  | puzzle (p : Puzzle)
  deriving Repr

/-- Turn an answer into the record it writes. The source is always `stated`: onboarding
has no other source to give. [D] -/
def OnboardingQuestion.write (q : OnboardingQuestion) (answer at_ : String) : Written :=
  match q.destination with
  | .profile key => .fact ⟨key, answer, at_, none, at_, .stated, .confirmed⟩
  | .relationship kind => .relationship ⟨answer, kind, at_, none, at_, .stated⟩
  | .puzzles => .puzzle ⟨answer, none, at_, .stated⟩

def Written.source : Written → Source
  | .fact f => f.source
  | .relationship r => r.source
  | .puzzle p => p.source

/-- Every onboarding answer is stated by the person. -/
theorem onboarding_stated (q : OnboardingQuestion) (a t : String) : (q.write a t).source = .stated := by
  cases q <;> rfl

end NTS
