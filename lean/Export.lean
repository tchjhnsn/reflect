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
  | .hold s => "{\"store\":\"holds\",\"subject\":\"" ++ s ++ "\"}"

def main : IO Unit := do
  let primes := list (Prime.all.map fun p => "{\"id\":\"" ++ p.id ++ "\",\"number\":" ++ toString p.number ++ "}")
  let frames := list ([Frame.introduction, .feeling, .decision, .belief, .identity].map fun f =>
    let name := match f with
      | .introduction => "introduction" | .feeling => "feeling" | .decision => "decision"
      | .belief => "belief" | .identity => "identity"
    "{\"id\":\"" ++ name ++ "\",\"root\":\"" ++ f.root.id ++ "\"}")
  let facets (xs : List Facet) := list (xs.map fun f => "{\"facet\":" ++ facetJson f ++ ",\"stance\":" ++ stanceJson f ++ "}")
  let onboarding := list (onboardingOrder.map fun q => "{\"id\":\"" ++ q.id ++ "\",\"destination\":" ++ destJson q.destination ++ "}")
  let holds := list ([Hold.desire, .will, .duty, .sacred, .notForMe, .unsure].map fun h =>
    let f := match h.facet with | some x => facetJson x | none => "null"
    let st := match h.stance with | .descriptive => "\"descriptive\"" | .normative => "\"normative\""
    "{\"id\":\"" ++ h.id ++ "\",\"facet\":" ++ f ++ ",\"stance\":" ++ st ++ "}")
  let cats : List Prime.Category := [.drives, .generators, .sensors, .regulators, .illuminators, .binders, .matrices, .apprehensions]
  let catName : Prime.Category → String
    | .drives => "drives" | .generators => "generators" | .sensors => "sensors" | .regulators => "regulators"
    | .illuminators => "illuminators" | .binders => "binders" | .matrices => "matrices" | .apprehensions => "apprehensions"
  let categories := list (cats.map fun c => "{\"id\":\"" ++ catName c ++ "\",\"pair\":[\"" ++ c.pair.1.id ++ "\",\"" ++ c.pair.2.id ++ "\"]}")
  let domains := list (Domain.all.map fun d => "{\"id\":\"" ++ d.id ++ "\",\"governor\":\"" ++ d.governor.id ++ "\",\"proper\":" ++ list (d.proper.map fun p => "\"" ++ p.id ++ "\"") ++ "}")
  let state := "{\"arrangements\":[\"first_leads\",\"second_leads\",\"joined\",\"opposed\"],\"amplitudes\":[\"low\",\"raised\",\"dominant\",\"absorbed\"],\"absorbs\":{\"category\":\"apprehensions\",\"leader\":\"" ++ Prime.Category.apprehensions.pair.2.id ++ "\"},\"categories\":" ++ categories ++ ",\"domains\":" ++ domains ++ "}"
  IO.println ("{\"max_depth\":" ++ toString maxDepth ++ ",\"primes\":" ++ primes ++ ",\"frames\":" ++ frames
    ++ ",\"will_order\":" ++ facets willOrder ++ ",\"coercion_sources\":" ++ facets coercionSources
    ++ ",\"introduction_order\":" ++ list (introductionOrder.map fun p => "\"" ++ p.id ++ "\"") ++ ",\"onboarding\":" ++ onboarding ++ ",\"holds\":" ++ holds ++ ",\"state\":" ++ state ++ "}")
