import NTS.Person
import NTS.Output
/-!
# Onboarding

The first thing a person does in Reflect is answer a few questions: their name, their date of
birth, their work, and their closest friend (Torian, 2026-11-09); then their marital status,
whether they are seeing anyone, whether they have children, and how they hold partnership and
children: desire, will, duty or the sacred (Torian, 2026-11-09). Every answer is stated by the
person and may be skipped. No model runs.

The order follows the introduction frame: shared and embodied life first, then bonds. The
puzzle of a person's life is not an onboarding question: it has its own schema, like the
founder puzzle, and its own flow (Torian, 2026-11-09).
-/
namespace NTS

/-- The onboarding questions. [D] -/
inductive OnboardingQuestion where
  | name | birthDate | work | closestFriend
  | maritalStatus | seeingSomeone | children
  | holdPartnership | holdChildren
  deriving DecidableEq, Repr

/-- How a person holds something they may or may not have: a set of these, chosen by them.
Each names the element it speaks for. [D] -/
inductive Hold where
  | desire   -- I want it
  | will     -- I am working toward it
  | duty     -- I feel I ought to
  | sacred   -- it is sacred to me
  | notForMe
  | unsure
  deriving DecidableEq, Repr

def Hold.id : Hold → String
  | .desire => "desire" | .will => "will" | .duty => "duty" | .sacred => "sacred"
  | .notForMe => "not_for_me" | .unsure => "unsure"

/-- The facet a hold speaks for, if any. "Not for me" and "unsure" name no element. [S]
False if, for example, duty toward family turned out not to be Conscience. -/
def Hold.facet : Hold → Option Facet
  | .desire => some (.prime .desire)
  | .will => some (.prime .will)
  | .duty => some (.prime .conscience)
  | .sacred => some (.prime .numinousSense)
  | .notForMe => none
  | .unsure => none

/-- A hold's stance comes from its facet, like every record's. Duty is normative. -/
def Hold.stance (h : Hold) : Stance := (h.facet.map Facet.stance).getD .descriptive

theorem duty_normative : Hold.duty.stance = .normative := rfl
theorem desire_descriptive : Hold.desire.stance = .descriptive := rfl

/-- Where an answer is filed. Holds go to the beliefs and values store, never the profile. [D] -/
inductive Destination where
  | profile (key : String)
  | relationship (kind : RelationKind)
  | hold (subject : String)
  deriving DecidableEq, Repr

def OnboardingQuestion.id : OnboardingQuestion → String
  | .name => "name" | .birthDate => "date_of_birth" | .work => "occupation"
  | .closestFriend => "closest_friend" | .maritalStatus => "marital_status"
  | .seeingSomeone => "seeing_someone" | .children => "children"
  | .holdPartnership => "hold_partnership" | .holdChildren => "hold_children"

def OnboardingQuestion.destination : OnboardingQuestion → Destination
  | .name => .profile "name"
  | .birthDate => .profile "date_of_birth"
  | .work => .profile "occupation"
  | .closestFriend => .relationship .closestFriend
  | .maritalStatus => .profile "marital_status"
  | .seeingSomeone => .profile "seeing_someone"
  | .children => .profile "children"
  | .holdPartnership => .hold "partnership"
  | .holdChildren => .hold "children"

/-- The order the questions are asked in: shared and embodied life, then bonds, then how the
person holds them, so what is comes before what they want or ought. [P] -/
def onboardingOrder : List OnboardingQuestion :=
  [.name, .birthDate, .work, .closestFriend, .maritalStatus, .seeingSomeone, .children,
   .holdPartnership, .holdChildren]

theorem onboarding_covers (q : OnboardingQuestion) : q ∈ onboardingOrder := by cases q <;> decide
theorem onboarding_nodup : onboardingOrder.Nodup := by decide

def OnboardingQuestion.isHold (q : OnboardingQuestion) : Bool :=
  match q.destination with | .hold _ => true | _ => false

/-- Facts and bonds come before holds: every hold question is asked after every other one. -/
theorem holds_last : onboardingOrder.dropWhile (fun q => !q.isHold) = [.holdPartnership, .holdChildren] := by decide

/-- Bonds come after shared and embodied life. -/
theorem friend_after_work :
    onboardingOrder.idxOf .work < onboardingOrder.idxOf .closestFriend := by decide

/-- What one answered question writes. Skipped questions write nothing. [D] -/
inductive Written where
  | fact (f : Fact)
  | relationship (r : Relationship)
  | hold (subject : String) (b : Belief)
  deriving Repr

/-- Turn an answer into the record it writes. The source is always `stated`: onboarding
has no other source to give. [D] -/
def OnboardingQuestion.write (q : OnboardingQuestion) (answer at_ : String) : Written :=
  match q.destination with
  | .profile key => .fact ⟨key, answer, at_, none, at_, .stated, .confirmed⟩
  | .relationship kind => .relationship ⟨answer, kind, at_, none, at_, .stated⟩
  | .hold subject => .hold subject ⟨answer, at_, .stated, .confirmed⟩

def Written.source : Written → Source
  | .fact f => f.source
  | .relationship r => r.source
  | .hold _ b => b.source

/-- Every onboarding answer is stated by the person. -/
theorem onboarding_stated (q : OnboardingQuestion) (a t : String) : (q.write a t).source = .stated := by
  cases q <;> rfl

end NTS
