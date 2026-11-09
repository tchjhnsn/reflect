import NTS
/-! Prints the structure the Python harness follows, as JSON: primes, frames, orders. -/
open NTS

def facetJson : Facet → String
  | .prime p => "{\"prime\":\"" ++ p.id ++ "\"}"
  | .axis a =>
    let n := match a with
      | .sharedInner => "shared_inner" | .embodiedImagined => "embodied_imagined"
      | .presentRepresented => "present_represented" | .voluntaryInvoluntary => "voluntary_involuntary"
    "{\"axis\":\"" ++ n ++ "\"}"

def stanceJson (f : Facet) : String :=
  match f.stance with | .descriptive => "\"descriptive\"" | .normative => "\"normative\""

def list (xs : List String) : String := "[" ++ ",".intercalate xs ++ "]"

def destJson : Destination → String
  | .profile k => "{\"store\":\"profile\",\"key\":\"" ++ k ++ "\"}"
  | .relationship .closestFriend => "{\"store\":\"relationships\",\"kind\":\"closest_friend\"}"

def main : IO Unit := do
  let primes := list (Prime.all.map fun p => "{\"id\":\"" ++ p.id ++ "\",\"number\":" ++ toString p.number ++ "}")
  let frames := list ([Frame.introduction, .feeling, .decision, .belief, .identity].map fun f =>
    let name := match f with
      | .introduction => "introduction" | .feeling => "feeling" | .decision => "decision"
      | .belief => "belief" | .identity => "identity"
    "{\"id\":\"" ++ name ++ "\",\"root\":\"" ++ f.root.id ++ "\"}")
  let facets (xs : List Facet) := list (xs.map fun f => "{\"facet\":" ++ facetJson f ++ ",\"stance\":" ++ stanceJson f ++ "}")
  let onboarding := list (onboardingOrder.map fun q => "{\"id\":\"" ++ q.id ++ "\",\"destination\":" ++ destJson q.destination ++ "}")
  IO.println ("{\"max_depth\":" ++ toString maxDepth ++ ",\"primes\":" ++ primes ++ ",\"frames\":" ++ frames
    ++ ",\"will_order\":" ++ facets willOrder ++ ",\"coercion_sources\":" ++ facets coercionSources
    ++ ",\"introduction_order\":" ++ list (introductionOrder.map fun p => "\"" ++ p.id ++ "\"") ++ ",\"onboarding\":" ++ onboarding ++ "}")
