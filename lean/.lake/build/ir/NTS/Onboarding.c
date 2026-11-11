// Lean compiler output
// Module: NTS.Onboarding
// Imports: public import Init public meta import Init public import NTS.Person public import NTS.Output
#include <lean/lean.h>
#if defined(__clang__)
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wunused-label"
#elif defined(__GNUC__) && !defined(__CLANG__)
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-label"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#endif
#ifdef __cplusplus
extern "C" {
#endif
uint8_t lean_string_dec_eq(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
uint8_t lp_reflect_x2dnts_NTS_Facet_stance(lean_object*);
lean_object* lp_reflect_x2dnts_NTS_instReprRelationKind_repr___redArg(lean_object*);
lean_object* lp_reflect_x2dnts_NTS_instReprFact_repr___redArg(lean_object*);
lean_object* l_Repr_addAppParen(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
lean_object* lean_nat_to_int(lean_object*);
lean_object* lp_reflect_x2dnts_NTS_instReprRelationship_repr___redArg(lean_object*);
lean_object* l_String_quote(lean_object*);
lean_object* lp_reflect_x2dnts_NTS_instReprBelief_repr___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_ctorIdx(uint8_t);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_ctorIdx___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_toCtorIdx(uint8_t);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_toCtorIdx___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_ctorElim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_ctorElim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_ctorElim(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_name_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_name_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_name_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_name_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_birthDate_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_birthDate_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_birthDate_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_birthDate_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_work_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_work_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_work_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_work_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_closestFriend_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_closestFriend_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_closestFriend_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_closestFriend_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_maritalStatus_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_maritalStatus_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_maritalStatus_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_maritalStatus_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_seeingSomeone_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_seeingSomeone_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_seeingSomeone_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_seeingSomeone_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_children_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_children_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_children_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_children_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_holdPartnership_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_holdPartnership_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_holdPartnership_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_holdPartnership_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_holdChildren_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_holdChildren_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_holdChildren_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_holdChildren_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_reflect_x2dnts_NTS_OnboardingQuestion_ofNat(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_ofNat___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_reflect_x2dnts_NTS_instDecidableEqOnboardingQuestion(uint8_t, uint8_t);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instDecidableEqOnboardingQuestion___boxed(lean_object*, lean_object*);
static const lean_string_object lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 28, .m_capacity = 28, .m_length = 27, .m_data = "NTS.OnboardingQuestion.name"};
static const lean_object* lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__0 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__0_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__0_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__1 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__1_value;
static const lean_string_object lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 33, .m_capacity = 33, .m_length = 32, .m_data = "NTS.OnboardingQuestion.birthDate"};
static const lean_object* lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__2 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__2_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__2_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__3 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__3_value;
static const lean_string_object lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 28, .m_capacity = 28, .m_length = 27, .m_data = "NTS.OnboardingQuestion.work"};
static const lean_object* lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__4 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__4_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__4_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__5 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__5_value;
static const lean_string_object lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 37, .m_capacity = 37, .m_length = 36, .m_data = "NTS.OnboardingQuestion.closestFriend"};
static const lean_object* lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__6 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__6_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__6_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__7 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__7_value;
static const lean_string_object lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 37, .m_capacity = 37, .m_length = 36, .m_data = "NTS.OnboardingQuestion.maritalStatus"};
static const lean_object* lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__8 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__8_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__8_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__9 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__9_value;
static const lean_string_object lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 37, .m_capacity = 37, .m_length = 36, .m_data = "NTS.OnboardingQuestion.seeingSomeone"};
static const lean_object* lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__10 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__10_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__10_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__11 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__11_value;
static const lean_string_object lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 32, .m_capacity = 32, .m_length = 31, .m_data = "NTS.OnboardingQuestion.children"};
static const lean_object* lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__12 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__12_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__12_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__13 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__13_value;
static const lean_string_object lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 39, .m_capacity = 39, .m_length = 38, .m_data = "NTS.OnboardingQuestion.holdPartnership"};
static const lean_object* lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__14 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__14_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__14_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__15 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__15_value;
static const lean_string_object lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 36, .m_capacity = 36, .m_length = 35, .m_data = "NTS.OnboardingQuestion.holdChildren"};
static const lean_object* lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__16 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__16_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__16_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__17 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__17_value;
static lean_once_cell_t lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__18_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__18;
static lean_once_cell_t lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__19_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__19;
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr(uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_reflect_x2dnts_NTS_instReprOnboardingQuestion___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_reflect_x2dnts_NTS_instReprOnboardingQuestion___closed__0 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprOnboardingQuestion___closed__0_value;
LEAN_EXPORT const lean_object* lp_reflect_x2dnts_NTS_instReprOnboardingQuestion = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprOnboardingQuestion___closed__0_value;
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Hold_ctorIdx(uint8_t);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Hold_ctorIdx___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Hold_toCtorIdx(uint8_t);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Hold_toCtorIdx___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Hold_ctorElim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Hold_ctorElim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Hold_ctorElim(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Hold_ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Hold_desire_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Hold_desire_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Hold_desire_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Hold_desire_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Hold_will_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Hold_will_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Hold_will_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Hold_will_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Hold_duty_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Hold_duty_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Hold_duty_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Hold_duty_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Hold_sacred_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Hold_sacred_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Hold_sacred_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Hold_sacred_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Hold_notForMe_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Hold_notForMe_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Hold_notForMe_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Hold_notForMe_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Hold_unsure_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Hold_unsure_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Hold_unsure_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Hold_unsure_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_reflect_x2dnts_NTS_Hold_ofNat(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Hold_ofNat___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_reflect_x2dnts_NTS_instDecidableEqHold(uint8_t, uint8_t);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instDecidableEqHold___boxed(lean_object*, lean_object*);
static const lean_string_object lp_reflect_x2dnts_NTS_instReprHold_repr___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "NTS.Hold.desire"};
static const lean_object* lp_reflect_x2dnts_NTS_instReprHold_repr___closed__0 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprHold_repr___closed__0_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_instReprHold_repr___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_reflect_x2dnts_NTS_instReprHold_repr___closed__0_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_instReprHold_repr___closed__1 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprHold_repr___closed__1_value;
static const lean_string_object lp_reflect_x2dnts_NTS_instReprHold_repr___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "NTS.Hold.will"};
static const lean_object* lp_reflect_x2dnts_NTS_instReprHold_repr___closed__2 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprHold_repr___closed__2_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_instReprHold_repr___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_reflect_x2dnts_NTS_instReprHold_repr___closed__2_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_instReprHold_repr___closed__3 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprHold_repr___closed__3_value;
static const lean_string_object lp_reflect_x2dnts_NTS_instReprHold_repr___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "NTS.Hold.duty"};
static const lean_object* lp_reflect_x2dnts_NTS_instReprHold_repr___closed__4 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprHold_repr___closed__4_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_instReprHold_repr___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_reflect_x2dnts_NTS_instReprHold_repr___closed__4_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_instReprHold_repr___closed__5 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprHold_repr___closed__5_value;
static const lean_string_object lp_reflect_x2dnts_NTS_instReprHold_repr___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "NTS.Hold.sacred"};
static const lean_object* lp_reflect_x2dnts_NTS_instReprHold_repr___closed__6 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprHold_repr___closed__6_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_instReprHold_repr___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_reflect_x2dnts_NTS_instReprHold_repr___closed__6_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_instReprHold_repr___closed__7 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprHold_repr___closed__7_value;
static const lean_string_object lp_reflect_x2dnts_NTS_instReprHold_repr___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "NTS.Hold.notForMe"};
static const lean_object* lp_reflect_x2dnts_NTS_instReprHold_repr___closed__8 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprHold_repr___closed__8_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_instReprHold_repr___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_reflect_x2dnts_NTS_instReprHold_repr___closed__8_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_instReprHold_repr___closed__9 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprHold_repr___closed__9_value;
static const lean_string_object lp_reflect_x2dnts_NTS_instReprHold_repr___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "NTS.Hold.unsure"};
static const lean_object* lp_reflect_x2dnts_NTS_instReprHold_repr___closed__10 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprHold_repr___closed__10_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_instReprHold_repr___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_reflect_x2dnts_NTS_instReprHold_repr___closed__10_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_instReprHold_repr___closed__11 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprHold_repr___closed__11_value;
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instReprHold_repr(uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instReprHold_repr___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_reflect_x2dnts_NTS_instReprHold___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_reflect_x2dnts_NTS_instReprHold_repr___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_reflect_x2dnts_NTS_instReprHold___closed__0 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprHold___closed__0_value;
LEAN_EXPORT const lean_object* lp_reflect_x2dnts_NTS_instReprHold = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprHold___closed__0_value;
static const lean_string_object lp_reflect_x2dnts_NTS_Hold_id___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "desire"};
static const lean_object* lp_reflect_x2dnts_NTS_Hold_id___closed__0 = (const lean_object*)&lp_reflect_x2dnts_NTS_Hold_id___closed__0_value;
static const lean_string_object lp_reflect_x2dnts_NTS_Hold_id___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "will"};
static const lean_object* lp_reflect_x2dnts_NTS_Hold_id___closed__1 = (const lean_object*)&lp_reflect_x2dnts_NTS_Hold_id___closed__1_value;
static const lean_string_object lp_reflect_x2dnts_NTS_Hold_id___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "duty"};
static const lean_object* lp_reflect_x2dnts_NTS_Hold_id___closed__2 = (const lean_object*)&lp_reflect_x2dnts_NTS_Hold_id___closed__2_value;
static const lean_string_object lp_reflect_x2dnts_NTS_Hold_id___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "sacred"};
static const lean_object* lp_reflect_x2dnts_NTS_Hold_id___closed__3 = (const lean_object*)&lp_reflect_x2dnts_NTS_Hold_id___closed__3_value;
static const lean_string_object lp_reflect_x2dnts_NTS_Hold_id___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "not_for_me"};
static const lean_object* lp_reflect_x2dnts_NTS_Hold_id___closed__4 = (const lean_object*)&lp_reflect_x2dnts_NTS_Hold_id___closed__4_value;
static const lean_string_object lp_reflect_x2dnts_NTS_Hold_id___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "unsure"};
static const lean_object* lp_reflect_x2dnts_NTS_Hold_id___closed__5 = (const lean_object*)&lp_reflect_x2dnts_NTS_Hold_id___closed__5_value;
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Hold_id(uint8_t);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Hold_id___boxed(lean_object*);
static const lean_ctor_object lp_reflect_x2dnts_NTS_Hold_facet___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*0 + 8, .m_other = 0, .m_tag = 0}, .m_objs = {LEAN_SCALAR_PTR_LITERAL(1, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_reflect_x2dnts_NTS_Hold_facet___closed__0 = (const lean_object*)&lp_reflect_x2dnts_NTS_Hold_facet___closed__0_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_Hold_facet___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)&lp_reflect_x2dnts_NTS_Hold_facet___closed__0_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_Hold_facet___closed__1 = (const lean_object*)&lp_reflect_x2dnts_NTS_Hold_facet___closed__1_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_Hold_facet___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*0 + 8, .m_other = 0, .m_tag = 0}, .m_objs = {LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_reflect_x2dnts_NTS_Hold_facet___closed__2 = (const lean_object*)&lp_reflect_x2dnts_NTS_Hold_facet___closed__2_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_Hold_facet___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)&lp_reflect_x2dnts_NTS_Hold_facet___closed__2_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_Hold_facet___closed__3 = (const lean_object*)&lp_reflect_x2dnts_NTS_Hold_facet___closed__3_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_Hold_facet___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*0 + 8, .m_other = 0, .m_tag = 0}, .m_objs = {LEAN_SCALAR_PTR_LITERAL(4, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_reflect_x2dnts_NTS_Hold_facet___closed__4 = (const lean_object*)&lp_reflect_x2dnts_NTS_Hold_facet___closed__4_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_Hold_facet___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)&lp_reflect_x2dnts_NTS_Hold_facet___closed__4_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_Hold_facet___closed__5 = (const lean_object*)&lp_reflect_x2dnts_NTS_Hold_facet___closed__5_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_Hold_facet___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*0 + 8, .m_other = 0, .m_tag = 0}, .m_objs = {LEAN_SCALAR_PTR_LITERAL(15, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_reflect_x2dnts_NTS_Hold_facet___closed__6 = (const lean_object*)&lp_reflect_x2dnts_NTS_Hold_facet___closed__6_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_Hold_facet___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)&lp_reflect_x2dnts_NTS_Hold_facet___closed__6_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_Hold_facet___closed__7 = (const lean_object*)&lp_reflect_x2dnts_NTS_Hold_facet___closed__7_value;
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Hold_facet(uint8_t);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Hold_facet___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_reflect_x2dnts_NTS_Hold_stance(uint8_t);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Hold_stance___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Destination_ctorIdx(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Destination_ctorIdx___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Destination_ctorElim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Destination_ctorElim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Destination_ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Destination_profile_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Destination_profile_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Destination_relationship_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Destination_relationship_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Destination_hold_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Destination_hold_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_reflect_x2dnts_NTS_instDecidableEqDestination_decEq(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instDecidableEqDestination_decEq___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_reflect_x2dnts_NTS_instDecidableEqDestination(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instDecidableEqDestination___boxed(lean_object*, lean_object*);
static const lean_string_object lp_reflect_x2dnts_NTS_instReprDestination_repr___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 29, .m_capacity = 29, .m_length = 28, .m_data = "NTS.Destination.relationship"};
static const lean_object* lp_reflect_x2dnts_NTS_instReprDestination_repr___closed__0 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprDestination_repr___closed__0_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_instReprDestination_repr___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_reflect_x2dnts_NTS_instReprDestination_repr___closed__0_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_instReprDestination_repr___closed__1 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprDestination_repr___closed__1_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_instReprDestination_repr___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_reflect_x2dnts_NTS_instReprDestination_repr___closed__1_value),((lean_object*)(((size_t)(1) << 1) | 1))}};
static const lean_object* lp_reflect_x2dnts_NTS_instReprDestination_repr___closed__2 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprDestination_repr___closed__2_value;
static lean_once_cell_t lp_reflect_x2dnts_NTS_instReprDestination_repr___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_reflect_x2dnts_NTS_instReprDestination_repr___closed__3;
static lean_once_cell_t lp_reflect_x2dnts_NTS_instReprDestination_repr___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_reflect_x2dnts_NTS_instReprDestination_repr___closed__4;
static const lean_string_object lp_reflect_x2dnts_NTS_instReprDestination_repr___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 24, .m_capacity = 24, .m_length = 23, .m_data = "NTS.Destination.profile"};
static const lean_object* lp_reflect_x2dnts_NTS_instReprDestination_repr___closed__5 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprDestination_repr___closed__5_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_instReprDestination_repr___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_reflect_x2dnts_NTS_instReprDestination_repr___closed__5_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_instReprDestination_repr___closed__6 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprDestination_repr___closed__6_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_instReprDestination_repr___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_reflect_x2dnts_NTS_instReprDestination_repr___closed__6_value),((lean_object*)(((size_t)(1) << 1) | 1))}};
static const lean_object* lp_reflect_x2dnts_NTS_instReprDestination_repr___closed__7 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprDestination_repr___closed__7_value;
static const lean_string_object lp_reflect_x2dnts_NTS_instReprDestination_repr___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = "NTS.Destination.hold"};
static const lean_object* lp_reflect_x2dnts_NTS_instReprDestination_repr___closed__8 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprDestination_repr___closed__8_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_instReprDestination_repr___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_reflect_x2dnts_NTS_instReprDestination_repr___closed__8_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_instReprDestination_repr___closed__9 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprDestination_repr___closed__9_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_instReprDestination_repr___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_reflect_x2dnts_NTS_instReprDestination_repr___closed__9_value),((lean_object*)(((size_t)(1) << 1) | 1))}};
static const lean_object* lp_reflect_x2dnts_NTS_instReprDestination_repr___closed__10 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprDestination_repr___closed__10_value;
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instReprDestination_repr(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instReprDestination_repr___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_reflect_x2dnts_NTS_instReprDestination___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_reflect_x2dnts_NTS_instReprDestination_repr___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_reflect_x2dnts_NTS_instReprDestination___closed__0 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprDestination___closed__0_value;
LEAN_EXPORT const lean_object* lp_reflect_x2dnts_NTS_instReprDestination = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprDestination___closed__0_value;
static const lean_string_object lp_reflect_x2dnts_NTS_OnboardingQuestion_id___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "name"};
static const lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_id___closed__0 = (const lean_object*)&lp_reflect_x2dnts_NTS_OnboardingQuestion_id___closed__0_value;
static const lean_string_object lp_reflect_x2dnts_NTS_OnboardingQuestion_id___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "date_of_birth"};
static const lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_id___closed__1 = (const lean_object*)&lp_reflect_x2dnts_NTS_OnboardingQuestion_id___closed__1_value;
static const lean_string_object lp_reflect_x2dnts_NTS_OnboardingQuestion_id___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "occupation"};
static const lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_id___closed__2 = (const lean_object*)&lp_reflect_x2dnts_NTS_OnboardingQuestion_id___closed__2_value;
static const lean_string_object lp_reflect_x2dnts_NTS_OnboardingQuestion_id___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "closest_friend"};
static const lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_id___closed__3 = (const lean_object*)&lp_reflect_x2dnts_NTS_OnboardingQuestion_id___closed__3_value;
static const lean_string_object lp_reflect_x2dnts_NTS_OnboardingQuestion_id___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "marital_status"};
static const lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_id___closed__4 = (const lean_object*)&lp_reflect_x2dnts_NTS_OnboardingQuestion_id___closed__4_value;
static const lean_string_object lp_reflect_x2dnts_NTS_OnboardingQuestion_id___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "seeing_someone"};
static const lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_id___closed__5 = (const lean_object*)&lp_reflect_x2dnts_NTS_OnboardingQuestion_id___closed__5_value;
static const lean_string_object lp_reflect_x2dnts_NTS_OnboardingQuestion_id___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "children"};
static const lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_id___closed__6 = (const lean_object*)&lp_reflect_x2dnts_NTS_OnboardingQuestion_id___closed__6_value;
static const lean_string_object lp_reflect_x2dnts_NTS_OnboardingQuestion_id___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "hold_partnership"};
static const lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_id___closed__7 = (const lean_object*)&lp_reflect_x2dnts_NTS_OnboardingQuestion_id___closed__7_value;
static const lean_string_object lp_reflect_x2dnts_NTS_OnboardingQuestion_id___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "hold_children"};
static const lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_id___closed__8 = (const lean_object*)&lp_reflect_x2dnts_NTS_OnboardingQuestion_id___closed__8_value;
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_id(uint8_t);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_id___boxed(lean_object*);
static const lean_ctor_object lp_reflect_x2dnts_NTS_OnboardingQuestion_destination___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_reflect_x2dnts_NTS_OnboardingQuestion_id___closed__0_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_destination___closed__0 = (const lean_object*)&lp_reflect_x2dnts_NTS_OnboardingQuestion_destination___closed__0_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_OnboardingQuestion_destination___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_reflect_x2dnts_NTS_OnboardingQuestion_id___closed__1_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_destination___closed__1 = (const lean_object*)&lp_reflect_x2dnts_NTS_OnboardingQuestion_destination___closed__1_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_OnboardingQuestion_destination___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_reflect_x2dnts_NTS_OnboardingQuestion_id___closed__2_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_destination___closed__2 = (const lean_object*)&lp_reflect_x2dnts_NTS_OnboardingQuestion_destination___closed__2_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_OnboardingQuestion_destination___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_destination___closed__3 = (const lean_object*)&lp_reflect_x2dnts_NTS_OnboardingQuestion_destination___closed__3_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_OnboardingQuestion_destination___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_reflect_x2dnts_NTS_OnboardingQuestion_id___closed__4_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_destination___closed__4 = (const lean_object*)&lp_reflect_x2dnts_NTS_OnboardingQuestion_destination___closed__4_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_OnboardingQuestion_destination___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_reflect_x2dnts_NTS_OnboardingQuestion_id___closed__5_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_destination___closed__5 = (const lean_object*)&lp_reflect_x2dnts_NTS_OnboardingQuestion_destination___closed__5_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_OnboardingQuestion_destination___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_reflect_x2dnts_NTS_OnboardingQuestion_id___closed__6_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_destination___closed__6 = (const lean_object*)&lp_reflect_x2dnts_NTS_OnboardingQuestion_destination___closed__6_value;
static const lean_string_object lp_reflect_x2dnts_NTS_OnboardingQuestion_destination___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "partnership"};
static const lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_destination___closed__7 = (const lean_object*)&lp_reflect_x2dnts_NTS_OnboardingQuestion_destination___closed__7_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_OnboardingQuestion_destination___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 2}, .m_objs = {((lean_object*)&lp_reflect_x2dnts_NTS_OnboardingQuestion_destination___closed__7_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_destination___closed__8 = (const lean_object*)&lp_reflect_x2dnts_NTS_OnboardingQuestion_destination___closed__8_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_OnboardingQuestion_destination___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 2}, .m_objs = {((lean_object*)&lp_reflect_x2dnts_NTS_OnboardingQuestion_id___closed__6_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_destination___closed__9 = (const lean_object*)&lp_reflect_x2dnts_NTS_OnboardingQuestion_destination___closed__9_value;
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_destination(uint8_t);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_destination___boxed(lean_object*);
static const lean_ctor_object lp_reflect_x2dnts_NTS_onboardingOrder___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(8) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_reflect_x2dnts_NTS_onboardingOrder___closed__0 = (const lean_object*)&lp_reflect_x2dnts_NTS_onboardingOrder___closed__0_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_onboardingOrder___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(7) << 1) | 1)),((lean_object*)&lp_reflect_x2dnts_NTS_onboardingOrder___closed__0_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_onboardingOrder___closed__1 = (const lean_object*)&lp_reflect_x2dnts_NTS_onboardingOrder___closed__1_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_onboardingOrder___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(6) << 1) | 1)),((lean_object*)&lp_reflect_x2dnts_NTS_onboardingOrder___closed__1_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_onboardingOrder___closed__2 = (const lean_object*)&lp_reflect_x2dnts_NTS_onboardingOrder___closed__2_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_onboardingOrder___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(5) << 1) | 1)),((lean_object*)&lp_reflect_x2dnts_NTS_onboardingOrder___closed__2_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_onboardingOrder___closed__3 = (const lean_object*)&lp_reflect_x2dnts_NTS_onboardingOrder___closed__3_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_onboardingOrder___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(4) << 1) | 1)),((lean_object*)&lp_reflect_x2dnts_NTS_onboardingOrder___closed__3_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_onboardingOrder___closed__4 = (const lean_object*)&lp_reflect_x2dnts_NTS_onboardingOrder___closed__4_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_onboardingOrder___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(3) << 1) | 1)),((lean_object*)&lp_reflect_x2dnts_NTS_onboardingOrder___closed__4_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_onboardingOrder___closed__5 = (const lean_object*)&lp_reflect_x2dnts_NTS_onboardingOrder___closed__5_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_onboardingOrder___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(2) << 1) | 1)),((lean_object*)&lp_reflect_x2dnts_NTS_onboardingOrder___closed__5_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_onboardingOrder___closed__6 = (const lean_object*)&lp_reflect_x2dnts_NTS_onboardingOrder___closed__6_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_onboardingOrder___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(1) << 1) | 1)),((lean_object*)&lp_reflect_x2dnts_NTS_onboardingOrder___closed__6_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_onboardingOrder___closed__7 = (const lean_object*)&lp_reflect_x2dnts_NTS_onboardingOrder___closed__7_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_onboardingOrder___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_reflect_x2dnts_NTS_onboardingOrder___closed__7_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_onboardingOrder___closed__8 = (const lean_object*)&lp_reflect_x2dnts_NTS_onboardingOrder___closed__8_value;
LEAN_EXPORT const lean_object* lp_reflect_x2dnts_NTS_onboardingOrder = (const lean_object*)&lp_reflect_x2dnts_NTS_onboardingOrder___closed__8_value;
LEAN_EXPORT uint8_t lp_reflect_x2dnts_NTS_OnboardingQuestion_isHold(uint8_t);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_isHold___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Written_ctorIdx(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Written_ctorIdx___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Written_ctorElim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Written_ctorElim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Written_ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Written_fact_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Written_fact_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Written_relationship_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Written_relationship_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Written_hold_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Written_hold_elim(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_reflect_x2dnts_NTS_instReprWritten_repr___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "NTS.Written.fact"};
static const lean_object* lp_reflect_x2dnts_NTS_instReprWritten_repr___closed__0 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprWritten_repr___closed__0_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_instReprWritten_repr___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_reflect_x2dnts_NTS_instReprWritten_repr___closed__0_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_instReprWritten_repr___closed__1 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprWritten_repr___closed__1_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_instReprWritten_repr___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_reflect_x2dnts_NTS_instReprWritten_repr___closed__1_value),((lean_object*)(((size_t)(1) << 1) | 1))}};
static const lean_object* lp_reflect_x2dnts_NTS_instReprWritten_repr___closed__2 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprWritten_repr___closed__2_value;
static const lean_string_object lp_reflect_x2dnts_NTS_instReprWritten_repr___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 25, .m_capacity = 25, .m_length = 24, .m_data = "NTS.Written.relationship"};
static const lean_object* lp_reflect_x2dnts_NTS_instReprWritten_repr___closed__3 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprWritten_repr___closed__3_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_instReprWritten_repr___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_reflect_x2dnts_NTS_instReprWritten_repr___closed__3_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_instReprWritten_repr___closed__4 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprWritten_repr___closed__4_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_instReprWritten_repr___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_reflect_x2dnts_NTS_instReprWritten_repr___closed__4_value),((lean_object*)(((size_t)(1) << 1) | 1))}};
static const lean_object* lp_reflect_x2dnts_NTS_instReprWritten_repr___closed__5 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprWritten_repr___closed__5_value;
static const lean_string_object lp_reflect_x2dnts_NTS_instReprWritten_repr___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "NTS.Written.hold"};
static const lean_object* lp_reflect_x2dnts_NTS_instReprWritten_repr___closed__6 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprWritten_repr___closed__6_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_instReprWritten_repr___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_reflect_x2dnts_NTS_instReprWritten_repr___closed__6_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_instReprWritten_repr___closed__7 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprWritten_repr___closed__7_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_instReprWritten_repr___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_reflect_x2dnts_NTS_instReprWritten_repr___closed__7_value),((lean_object*)(((size_t)(1) << 1) | 1))}};
static const lean_object* lp_reflect_x2dnts_NTS_instReprWritten_repr___closed__8 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprWritten_repr___closed__8_value;
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instReprWritten_repr(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instReprWritten_repr___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_reflect_x2dnts_NTS_instReprWritten___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_reflect_x2dnts_NTS_instReprWritten_repr___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_reflect_x2dnts_NTS_instReprWritten___closed__0 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprWritten___closed__0_value;
LEAN_EXPORT const lean_object* lp_reflect_x2dnts_NTS_instReprWritten = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprWritten___closed__0_value;
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_write(uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_write___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Written_source(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Written_source___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_ctorIdx(uint8_t v_x_1_){
_start:
{
switch(v_x_1_)
{
case 0:
{
lean_object* v___x_2_; 
v___x_2_ = lean_unsigned_to_nat(0u);
return v___x_2_;
}
case 1:
{
lean_object* v___x_3_; 
v___x_3_ = lean_unsigned_to_nat(1u);
return v___x_3_;
}
case 2:
{
lean_object* v___x_4_; 
v___x_4_ = lean_unsigned_to_nat(2u);
return v___x_4_;
}
case 3:
{
lean_object* v___x_5_; 
v___x_5_ = lean_unsigned_to_nat(3u);
return v___x_5_;
}
case 4:
{
lean_object* v___x_6_; 
v___x_6_ = lean_unsigned_to_nat(4u);
return v___x_6_;
}
case 5:
{
lean_object* v___x_7_; 
v___x_7_ = lean_unsigned_to_nat(5u);
return v___x_7_;
}
case 6:
{
lean_object* v___x_8_; 
v___x_8_ = lean_unsigned_to_nat(6u);
return v___x_8_;
}
case 7:
{
lean_object* v___x_9_; 
v___x_9_ = lean_unsigned_to_nat(7u);
return v___x_9_;
}
default: 
{
lean_object* v___x_10_; 
v___x_10_ = lean_unsigned_to_nat(8u);
return v___x_10_;
}
}
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_ctorIdx___boxed(lean_object* v_x_11_){
_start:
{
uint8_t v_x_boxed_12_; lean_object* v_res_13_; 
v_x_boxed_12_ = lean_unbox(v_x_11_);
v_res_13_ = lp_reflect_x2dnts_NTS_OnboardingQuestion_ctorIdx(v_x_boxed_12_);
return v_res_13_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_toCtorIdx(uint8_t v_x_14_){
_start:
{
lean_object* v___x_15_; 
v___x_15_ = lp_reflect_x2dnts_NTS_OnboardingQuestion_ctorIdx(v_x_14_);
return v___x_15_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_toCtorIdx___boxed(lean_object* v_x_16_){
_start:
{
uint8_t v_x_4__boxed_17_; lean_object* v_res_18_; 
v_x_4__boxed_17_ = lean_unbox(v_x_16_);
v_res_18_ = lp_reflect_x2dnts_NTS_OnboardingQuestion_toCtorIdx(v_x_4__boxed_17_);
return v_res_18_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_ctorElim___redArg(lean_object* v_k_19_){
_start:
{
lean_inc(v_k_19_);
return v_k_19_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_ctorElim___redArg___boxed(lean_object* v_k_20_){
_start:
{
lean_object* v_res_21_; 
v_res_21_ = lp_reflect_x2dnts_NTS_OnboardingQuestion_ctorElim___redArg(v_k_20_);
lean_dec(v_k_20_);
return v_res_21_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_ctorElim(lean_object* v_motive_22_, lean_object* v_ctorIdx_23_, uint8_t v_t_24_, lean_object* v_h_25_, lean_object* v_k_26_){
_start:
{
lean_inc(v_k_26_);
return v_k_26_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_ctorElim___boxed(lean_object* v_motive_27_, lean_object* v_ctorIdx_28_, lean_object* v_t_29_, lean_object* v_h_30_, lean_object* v_k_31_){
_start:
{
uint8_t v_t_boxed_32_; lean_object* v_res_33_; 
v_t_boxed_32_ = lean_unbox(v_t_29_);
v_res_33_ = lp_reflect_x2dnts_NTS_OnboardingQuestion_ctorElim(v_motive_27_, v_ctorIdx_28_, v_t_boxed_32_, v_h_30_, v_k_31_);
lean_dec(v_k_31_);
lean_dec(v_ctorIdx_28_);
return v_res_33_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_name_elim___redArg(lean_object* v_name_34_){
_start:
{
lean_inc(v_name_34_);
return v_name_34_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_name_elim___redArg___boxed(lean_object* v_name_35_){
_start:
{
lean_object* v_res_36_; 
v_res_36_ = lp_reflect_x2dnts_NTS_OnboardingQuestion_name_elim___redArg(v_name_35_);
lean_dec(v_name_35_);
return v_res_36_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_name_elim(lean_object* v_motive_37_, uint8_t v_t_38_, lean_object* v_h_39_, lean_object* v_name_40_){
_start:
{
lean_inc(v_name_40_);
return v_name_40_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_name_elim___boxed(lean_object* v_motive_41_, lean_object* v_t_42_, lean_object* v_h_43_, lean_object* v_name_44_){
_start:
{
uint8_t v_t_boxed_45_; lean_object* v_res_46_; 
v_t_boxed_45_ = lean_unbox(v_t_42_);
v_res_46_ = lp_reflect_x2dnts_NTS_OnboardingQuestion_name_elim(v_motive_41_, v_t_boxed_45_, v_h_43_, v_name_44_);
lean_dec(v_name_44_);
return v_res_46_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_birthDate_elim___redArg(lean_object* v_birthDate_47_){
_start:
{
lean_inc(v_birthDate_47_);
return v_birthDate_47_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_birthDate_elim___redArg___boxed(lean_object* v_birthDate_48_){
_start:
{
lean_object* v_res_49_; 
v_res_49_ = lp_reflect_x2dnts_NTS_OnboardingQuestion_birthDate_elim___redArg(v_birthDate_48_);
lean_dec(v_birthDate_48_);
return v_res_49_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_birthDate_elim(lean_object* v_motive_50_, uint8_t v_t_51_, lean_object* v_h_52_, lean_object* v_birthDate_53_){
_start:
{
lean_inc(v_birthDate_53_);
return v_birthDate_53_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_birthDate_elim___boxed(lean_object* v_motive_54_, lean_object* v_t_55_, lean_object* v_h_56_, lean_object* v_birthDate_57_){
_start:
{
uint8_t v_t_boxed_58_; lean_object* v_res_59_; 
v_t_boxed_58_ = lean_unbox(v_t_55_);
v_res_59_ = lp_reflect_x2dnts_NTS_OnboardingQuestion_birthDate_elim(v_motive_54_, v_t_boxed_58_, v_h_56_, v_birthDate_57_);
lean_dec(v_birthDate_57_);
return v_res_59_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_work_elim___redArg(lean_object* v_work_60_){
_start:
{
lean_inc(v_work_60_);
return v_work_60_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_work_elim___redArg___boxed(lean_object* v_work_61_){
_start:
{
lean_object* v_res_62_; 
v_res_62_ = lp_reflect_x2dnts_NTS_OnboardingQuestion_work_elim___redArg(v_work_61_);
lean_dec(v_work_61_);
return v_res_62_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_work_elim(lean_object* v_motive_63_, uint8_t v_t_64_, lean_object* v_h_65_, lean_object* v_work_66_){
_start:
{
lean_inc(v_work_66_);
return v_work_66_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_work_elim___boxed(lean_object* v_motive_67_, lean_object* v_t_68_, lean_object* v_h_69_, lean_object* v_work_70_){
_start:
{
uint8_t v_t_boxed_71_; lean_object* v_res_72_; 
v_t_boxed_71_ = lean_unbox(v_t_68_);
v_res_72_ = lp_reflect_x2dnts_NTS_OnboardingQuestion_work_elim(v_motive_67_, v_t_boxed_71_, v_h_69_, v_work_70_);
lean_dec(v_work_70_);
return v_res_72_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_closestFriend_elim___redArg(lean_object* v_closestFriend_73_){
_start:
{
lean_inc(v_closestFriend_73_);
return v_closestFriend_73_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_closestFriend_elim___redArg___boxed(lean_object* v_closestFriend_74_){
_start:
{
lean_object* v_res_75_; 
v_res_75_ = lp_reflect_x2dnts_NTS_OnboardingQuestion_closestFriend_elim___redArg(v_closestFriend_74_);
lean_dec(v_closestFriend_74_);
return v_res_75_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_closestFriend_elim(lean_object* v_motive_76_, uint8_t v_t_77_, lean_object* v_h_78_, lean_object* v_closestFriend_79_){
_start:
{
lean_inc(v_closestFriend_79_);
return v_closestFriend_79_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_closestFriend_elim___boxed(lean_object* v_motive_80_, lean_object* v_t_81_, lean_object* v_h_82_, lean_object* v_closestFriend_83_){
_start:
{
uint8_t v_t_boxed_84_; lean_object* v_res_85_; 
v_t_boxed_84_ = lean_unbox(v_t_81_);
v_res_85_ = lp_reflect_x2dnts_NTS_OnboardingQuestion_closestFriend_elim(v_motive_80_, v_t_boxed_84_, v_h_82_, v_closestFriend_83_);
lean_dec(v_closestFriend_83_);
return v_res_85_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_maritalStatus_elim___redArg(lean_object* v_maritalStatus_86_){
_start:
{
lean_inc(v_maritalStatus_86_);
return v_maritalStatus_86_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_maritalStatus_elim___redArg___boxed(lean_object* v_maritalStatus_87_){
_start:
{
lean_object* v_res_88_; 
v_res_88_ = lp_reflect_x2dnts_NTS_OnboardingQuestion_maritalStatus_elim___redArg(v_maritalStatus_87_);
lean_dec(v_maritalStatus_87_);
return v_res_88_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_maritalStatus_elim(lean_object* v_motive_89_, uint8_t v_t_90_, lean_object* v_h_91_, lean_object* v_maritalStatus_92_){
_start:
{
lean_inc(v_maritalStatus_92_);
return v_maritalStatus_92_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_maritalStatus_elim___boxed(lean_object* v_motive_93_, lean_object* v_t_94_, lean_object* v_h_95_, lean_object* v_maritalStatus_96_){
_start:
{
uint8_t v_t_boxed_97_; lean_object* v_res_98_; 
v_t_boxed_97_ = lean_unbox(v_t_94_);
v_res_98_ = lp_reflect_x2dnts_NTS_OnboardingQuestion_maritalStatus_elim(v_motive_93_, v_t_boxed_97_, v_h_95_, v_maritalStatus_96_);
lean_dec(v_maritalStatus_96_);
return v_res_98_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_seeingSomeone_elim___redArg(lean_object* v_seeingSomeone_99_){
_start:
{
lean_inc(v_seeingSomeone_99_);
return v_seeingSomeone_99_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_seeingSomeone_elim___redArg___boxed(lean_object* v_seeingSomeone_100_){
_start:
{
lean_object* v_res_101_; 
v_res_101_ = lp_reflect_x2dnts_NTS_OnboardingQuestion_seeingSomeone_elim___redArg(v_seeingSomeone_100_);
lean_dec(v_seeingSomeone_100_);
return v_res_101_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_seeingSomeone_elim(lean_object* v_motive_102_, uint8_t v_t_103_, lean_object* v_h_104_, lean_object* v_seeingSomeone_105_){
_start:
{
lean_inc(v_seeingSomeone_105_);
return v_seeingSomeone_105_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_seeingSomeone_elim___boxed(lean_object* v_motive_106_, lean_object* v_t_107_, lean_object* v_h_108_, lean_object* v_seeingSomeone_109_){
_start:
{
uint8_t v_t_boxed_110_; lean_object* v_res_111_; 
v_t_boxed_110_ = lean_unbox(v_t_107_);
v_res_111_ = lp_reflect_x2dnts_NTS_OnboardingQuestion_seeingSomeone_elim(v_motive_106_, v_t_boxed_110_, v_h_108_, v_seeingSomeone_109_);
lean_dec(v_seeingSomeone_109_);
return v_res_111_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_children_elim___redArg(lean_object* v_children_112_){
_start:
{
lean_inc(v_children_112_);
return v_children_112_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_children_elim___redArg___boxed(lean_object* v_children_113_){
_start:
{
lean_object* v_res_114_; 
v_res_114_ = lp_reflect_x2dnts_NTS_OnboardingQuestion_children_elim___redArg(v_children_113_);
lean_dec(v_children_113_);
return v_res_114_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_children_elim(lean_object* v_motive_115_, uint8_t v_t_116_, lean_object* v_h_117_, lean_object* v_children_118_){
_start:
{
lean_inc(v_children_118_);
return v_children_118_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_children_elim___boxed(lean_object* v_motive_119_, lean_object* v_t_120_, lean_object* v_h_121_, lean_object* v_children_122_){
_start:
{
uint8_t v_t_boxed_123_; lean_object* v_res_124_; 
v_t_boxed_123_ = lean_unbox(v_t_120_);
v_res_124_ = lp_reflect_x2dnts_NTS_OnboardingQuestion_children_elim(v_motive_119_, v_t_boxed_123_, v_h_121_, v_children_122_);
lean_dec(v_children_122_);
return v_res_124_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_holdPartnership_elim___redArg(lean_object* v_holdPartnership_125_){
_start:
{
lean_inc(v_holdPartnership_125_);
return v_holdPartnership_125_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_holdPartnership_elim___redArg___boxed(lean_object* v_holdPartnership_126_){
_start:
{
lean_object* v_res_127_; 
v_res_127_ = lp_reflect_x2dnts_NTS_OnboardingQuestion_holdPartnership_elim___redArg(v_holdPartnership_126_);
lean_dec(v_holdPartnership_126_);
return v_res_127_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_holdPartnership_elim(lean_object* v_motive_128_, uint8_t v_t_129_, lean_object* v_h_130_, lean_object* v_holdPartnership_131_){
_start:
{
lean_inc(v_holdPartnership_131_);
return v_holdPartnership_131_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_holdPartnership_elim___boxed(lean_object* v_motive_132_, lean_object* v_t_133_, lean_object* v_h_134_, lean_object* v_holdPartnership_135_){
_start:
{
uint8_t v_t_boxed_136_; lean_object* v_res_137_; 
v_t_boxed_136_ = lean_unbox(v_t_133_);
v_res_137_ = lp_reflect_x2dnts_NTS_OnboardingQuestion_holdPartnership_elim(v_motive_132_, v_t_boxed_136_, v_h_134_, v_holdPartnership_135_);
lean_dec(v_holdPartnership_135_);
return v_res_137_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_holdChildren_elim___redArg(lean_object* v_holdChildren_138_){
_start:
{
lean_inc(v_holdChildren_138_);
return v_holdChildren_138_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_holdChildren_elim___redArg___boxed(lean_object* v_holdChildren_139_){
_start:
{
lean_object* v_res_140_; 
v_res_140_ = lp_reflect_x2dnts_NTS_OnboardingQuestion_holdChildren_elim___redArg(v_holdChildren_139_);
lean_dec(v_holdChildren_139_);
return v_res_140_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_holdChildren_elim(lean_object* v_motive_141_, uint8_t v_t_142_, lean_object* v_h_143_, lean_object* v_holdChildren_144_){
_start:
{
lean_inc(v_holdChildren_144_);
return v_holdChildren_144_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_holdChildren_elim___boxed(lean_object* v_motive_145_, lean_object* v_t_146_, lean_object* v_h_147_, lean_object* v_holdChildren_148_){
_start:
{
uint8_t v_t_boxed_149_; lean_object* v_res_150_; 
v_t_boxed_149_ = lean_unbox(v_t_146_);
v_res_150_ = lp_reflect_x2dnts_NTS_OnboardingQuestion_holdChildren_elim(v_motive_145_, v_t_boxed_149_, v_h_147_, v_holdChildren_148_);
lean_dec(v_holdChildren_148_);
return v_res_150_;
}
}
LEAN_EXPORT uint8_t lp_reflect_x2dnts_NTS_OnboardingQuestion_ofNat(lean_object* v_n_151_){
_start:
{
lean_object* v___x_152_; uint8_t v___x_153_; 
v___x_152_ = lean_unsigned_to_nat(3u);
v___x_153_ = lean_nat_dec_le(v_n_151_, v___x_152_);
if (v___x_153_ == 0)
{
lean_object* v___x_154_; uint8_t v___x_155_; 
v___x_154_ = lean_unsigned_to_nat(5u);
v___x_155_ = lean_nat_dec_le(v_n_151_, v___x_154_);
if (v___x_155_ == 0)
{
lean_object* v___x_156_; uint8_t v___x_157_; 
v___x_156_ = lean_unsigned_to_nat(6u);
v___x_157_ = lean_nat_dec_le(v_n_151_, v___x_156_);
if (v___x_157_ == 0)
{
lean_object* v___x_158_; uint8_t v___x_159_; 
v___x_158_ = lean_unsigned_to_nat(7u);
v___x_159_ = lean_nat_dec_le(v_n_151_, v___x_158_);
if (v___x_159_ == 0)
{
uint8_t v___x_160_; 
v___x_160_ = 8;
return v___x_160_;
}
else
{
uint8_t v___x_161_; 
v___x_161_ = 7;
return v___x_161_;
}
}
else
{
uint8_t v___x_162_; 
v___x_162_ = 6;
return v___x_162_;
}
}
else
{
lean_object* v___x_163_; uint8_t v___x_164_; 
v___x_163_ = lean_unsigned_to_nat(4u);
v___x_164_ = lean_nat_dec_le(v_n_151_, v___x_163_);
if (v___x_164_ == 0)
{
uint8_t v___x_165_; 
v___x_165_ = 5;
return v___x_165_;
}
else
{
uint8_t v___x_166_; 
v___x_166_ = 4;
return v___x_166_;
}
}
}
else
{
lean_object* v___x_167_; uint8_t v___x_168_; 
v___x_167_ = lean_unsigned_to_nat(1u);
v___x_168_ = lean_nat_dec_le(v_n_151_, v___x_167_);
if (v___x_168_ == 0)
{
lean_object* v___x_169_; uint8_t v___x_170_; 
v___x_169_ = lean_unsigned_to_nat(2u);
v___x_170_ = lean_nat_dec_le(v_n_151_, v___x_169_);
if (v___x_170_ == 0)
{
uint8_t v___x_171_; 
v___x_171_ = 3;
return v___x_171_;
}
else
{
uint8_t v___x_172_; 
v___x_172_ = 2;
return v___x_172_;
}
}
else
{
lean_object* v___x_173_; uint8_t v___x_174_; 
v___x_173_ = lean_unsigned_to_nat(0u);
v___x_174_ = lean_nat_dec_le(v_n_151_, v___x_173_);
if (v___x_174_ == 0)
{
uint8_t v___x_175_; 
v___x_175_ = 1;
return v___x_175_;
}
else
{
uint8_t v___x_176_; 
v___x_176_ = 0;
return v___x_176_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_ofNat___boxed(lean_object* v_n_177_){
_start:
{
uint8_t v_res_178_; lean_object* v_r_179_; 
v_res_178_ = lp_reflect_x2dnts_NTS_OnboardingQuestion_ofNat(v_n_177_);
lean_dec(v_n_177_);
v_r_179_ = lean_box(v_res_178_);
return v_r_179_;
}
}
LEAN_EXPORT uint8_t lp_reflect_x2dnts_NTS_instDecidableEqOnboardingQuestion(uint8_t v_x_180_, uint8_t v_y_181_){
_start:
{
lean_object* v___x_182_; lean_object* v___x_183_; uint8_t v___x_184_; 
v___x_182_ = lp_reflect_x2dnts_NTS_OnboardingQuestion_ctorIdx(v_x_180_);
v___x_183_ = lp_reflect_x2dnts_NTS_OnboardingQuestion_ctorIdx(v_y_181_);
v___x_184_ = lean_nat_dec_eq(v___x_182_, v___x_183_);
lean_dec(v___x_183_);
lean_dec(v___x_182_);
return v___x_184_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instDecidableEqOnboardingQuestion___boxed(lean_object* v_x_185_, lean_object* v_y_186_){
_start:
{
uint8_t v_x_13__boxed_187_; uint8_t v_y_14__boxed_188_; uint8_t v_res_189_; lean_object* v_r_190_; 
v_x_13__boxed_187_ = lean_unbox(v_x_185_);
v_y_14__boxed_188_ = lean_unbox(v_y_186_);
v_res_189_ = lp_reflect_x2dnts_NTS_instDecidableEqOnboardingQuestion(v_x_13__boxed_187_, v_y_14__boxed_188_);
v_r_190_ = lean_box(v_res_189_);
return v_r_190_;
}
}
static lean_object* _init_lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__18(void){
_start:
{
lean_object* v___x_218_; lean_object* v___x_219_; 
v___x_218_ = lean_unsigned_to_nat(2u);
v___x_219_ = lean_nat_to_int(v___x_218_);
return v___x_219_;
}
}
static lean_object* _init_lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__19(void){
_start:
{
lean_object* v___x_220_; lean_object* v___x_221_; 
v___x_220_ = lean_unsigned_to_nat(1u);
v___x_221_ = lean_nat_to_int(v___x_220_);
return v___x_221_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr(uint8_t v_x_222_, lean_object* v_prec_223_){
_start:
{
lean_object* v___y_225_; lean_object* v___y_232_; lean_object* v___y_239_; lean_object* v___y_246_; lean_object* v___y_253_; lean_object* v___y_260_; lean_object* v___y_267_; lean_object* v___y_274_; lean_object* v___y_281_; 
switch(v_x_222_)
{
case 0:
{
lean_object* v___x_287_; uint8_t v___x_288_; 
v___x_287_ = lean_unsigned_to_nat(1024u);
v___x_288_ = lean_nat_dec_le(v___x_287_, v_prec_223_);
if (v___x_288_ == 0)
{
lean_object* v___x_289_; 
v___x_289_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__18, &lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__18_once, _init_lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__18);
v___y_225_ = v___x_289_;
goto v___jp_224_;
}
else
{
lean_object* v___x_290_; 
v___x_290_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__19, &lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__19_once, _init_lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__19);
v___y_225_ = v___x_290_;
goto v___jp_224_;
}
}
case 1:
{
lean_object* v___x_291_; uint8_t v___x_292_; 
v___x_291_ = lean_unsigned_to_nat(1024u);
v___x_292_ = lean_nat_dec_le(v___x_291_, v_prec_223_);
if (v___x_292_ == 0)
{
lean_object* v___x_293_; 
v___x_293_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__18, &lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__18_once, _init_lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__18);
v___y_232_ = v___x_293_;
goto v___jp_231_;
}
else
{
lean_object* v___x_294_; 
v___x_294_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__19, &lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__19_once, _init_lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__19);
v___y_232_ = v___x_294_;
goto v___jp_231_;
}
}
case 2:
{
lean_object* v___x_295_; uint8_t v___x_296_; 
v___x_295_ = lean_unsigned_to_nat(1024u);
v___x_296_ = lean_nat_dec_le(v___x_295_, v_prec_223_);
if (v___x_296_ == 0)
{
lean_object* v___x_297_; 
v___x_297_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__18, &lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__18_once, _init_lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__18);
v___y_239_ = v___x_297_;
goto v___jp_238_;
}
else
{
lean_object* v___x_298_; 
v___x_298_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__19, &lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__19_once, _init_lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__19);
v___y_239_ = v___x_298_;
goto v___jp_238_;
}
}
case 3:
{
lean_object* v___x_299_; uint8_t v___x_300_; 
v___x_299_ = lean_unsigned_to_nat(1024u);
v___x_300_ = lean_nat_dec_le(v___x_299_, v_prec_223_);
if (v___x_300_ == 0)
{
lean_object* v___x_301_; 
v___x_301_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__18, &lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__18_once, _init_lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__18);
v___y_246_ = v___x_301_;
goto v___jp_245_;
}
else
{
lean_object* v___x_302_; 
v___x_302_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__19, &lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__19_once, _init_lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__19);
v___y_246_ = v___x_302_;
goto v___jp_245_;
}
}
case 4:
{
lean_object* v___x_303_; uint8_t v___x_304_; 
v___x_303_ = lean_unsigned_to_nat(1024u);
v___x_304_ = lean_nat_dec_le(v___x_303_, v_prec_223_);
if (v___x_304_ == 0)
{
lean_object* v___x_305_; 
v___x_305_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__18, &lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__18_once, _init_lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__18);
v___y_253_ = v___x_305_;
goto v___jp_252_;
}
else
{
lean_object* v___x_306_; 
v___x_306_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__19, &lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__19_once, _init_lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__19);
v___y_253_ = v___x_306_;
goto v___jp_252_;
}
}
case 5:
{
lean_object* v___x_307_; uint8_t v___x_308_; 
v___x_307_ = lean_unsigned_to_nat(1024u);
v___x_308_ = lean_nat_dec_le(v___x_307_, v_prec_223_);
if (v___x_308_ == 0)
{
lean_object* v___x_309_; 
v___x_309_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__18, &lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__18_once, _init_lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__18);
v___y_260_ = v___x_309_;
goto v___jp_259_;
}
else
{
lean_object* v___x_310_; 
v___x_310_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__19, &lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__19_once, _init_lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__19);
v___y_260_ = v___x_310_;
goto v___jp_259_;
}
}
case 6:
{
lean_object* v___x_311_; uint8_t v___x_312_; 
v___x_311_ = lean_unsigned_to_nat(1024u);
v___x_312_ = lean_nat_dec_le(v___x_311_, v_prec_223_);
if (v___x_312_ == 0)
{
lean_object* v___x_313_; 
v___x_313_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__18, &lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__18_once, _init_lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__18);
v___y_267_ = v___x_313_;
goto v___jp_266_;
}
else
{
lean_object* v___x_314_; 
v___x_314_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__19, &lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__19_once, _init_lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__19);
v___y_267_ = v___x_314_;
goto v___jp_266_;
}
}
case 7:
{
lean_object* v___x_315_; uint8_t v___x_316_; 
v___x_315_ = lean_unsigned_to_nat(1024u);
v___x_316_ = lean_nat_dec_le(v___x_315_, v_prec_223_);
if (v___x_316_ == 0)
{
lean_object* v___x_317_; 
v___x_317_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__18, &lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__18_once, _init_lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__18);
v___y_274_ = v___x_317_;
goto v___jp_273_;
}
else
{
lean_object* v___x_318_; 
v___x_318_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__19, &lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__19_once, _init_lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__19);
v___y_274_ = v___x_318_;
goto v___jp_273_;
}
}
default: 
{
lean_object* v___x_319_; uint8_t v___x_320_; 
v___x_319_ = lean_unsigned_to_nat(1024u);
v___x_320_ = lean_nat_dec_le(v___x_319_, v_prec_223_);
if (v___x_320_ == 0)
{
lean_object* v___x_321_; 
v___x_321_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__18, &lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__18_once, _init_lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__18);
v___y_281_ = v___x_321_;
goto v___jp_280_;
}
else
{
lean_object* v___x_322_; 
v___x_322_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__19, &lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__19_once, _init_lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__19);
v___y_281_ = v___x_322_;
goto v___jp_280_;
}
}
}
v___jp_224_:
{
lean_object* v___x_226_; lean_object* v___x_227_; uint8_t v___x_228_; lean_object* v___x_229_; lean_object* v___x_230_; 
v___x_226_ = ((lean_object*)(lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__1));
lean_inc(v___y_225_);
v___x_227_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_227_, 0, v___y_225_);
lean_ctor_set(v___x_227_, 1, v___x_226_);
v___x_228_ = 0;
v___x_229_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_229_, 0, v___x_227_);
lean_ctor_set_uint8(v___x_229_, sizeof(void*)*1, v___x_228_);
v___x_230_ = l_Repr_addAppParen(v___x_229_, v_prec_223_);
return v___x_230_;
}
v___jp_231_:
{
lean_object* v___x_233_; lean_object* v___x_234_; uint8_t v___x_235_; lean_object* v___x_236_; lean_object* v___x_237_; 
v___x_233_ = ((lean_object*)(lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__3));
lean_inc(v___y_232_);
v___x_234_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_234_, 0, v___y_232_);
lean_ctor_set(v___x_234_, 1, v___x_233_);
v___x_235_ = 0;
v___x_236_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_236_, 0, v___x_234_);
lean_ctor_set_uint8(v___x_236_, sizeof(void*)*1, v___x_235_);
v___x_237_ = l_Repr_addAppParen(v___x_236_, v_prec_223_);
return v___x_237_;
}
v___jp_238_:
{
lean_object* v___x_240_; lean_object* v___x_241_; uint8_t v___x_242_; lean_object* v___x_243_; lean_object* v___x_244_; 
v___x_240_ = ((lean_object*)(lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__5));
lean_inc(v___y_239_);
v___x_241_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_241_, 0, v___y_239_);
lean_ctor_set(v___x_241_, 1, v___x_240_);
v___x_242_ = 0;
v___x_243_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_243_, 0, v___x_241_);
lean_ctor_set_uint8(v___x_243_, sizeof(void*)*1, v___x_242_);
v___x_244_ = l_Repr_addAppParen(v___x_243_, v_prec_223_);
return v___x_244_;
}
v___jp_245_:
{
lean_object* v___x_247_; lean_object* v___x_248_; uint8_t v___x_249_; lean_object* v___x_250_; lean_object* v___x_251_; 
v___x_247_ = ((lean_object*)(lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__7));
lean_inc(v___y_246_);
v___x_248_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_248_, 0, v___y_246_);
lean_ctor_set(v___x_248_, 1, v___x_247_);
v___x_249_ = 0;
v___x_250_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_250_, 0, v___x_248_);
lean_ctor_set_uint8(v___x_250_, sizeof(void*)*1, v___x_249_);
v___x_251_ = l_Repr_addAppParen(v___x_250_, v_prec_223_);
return v___x_251_;
}
v___jp_252_:
{
lean_object* v___x_254_; lean_object* v___x_255_; uint8_t v___x_256_; lean_object* v___x_257_; lean_object* v___x_258_; 
v___x_254_ = ((lean_object*)(lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__9));
lean_inc(v___y_253_);
v___x_255_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_255_, 0, v___y_253_);
lean_ctor_set(v___x_255_, 1, v___x_254_);
v___x_256_ = 0;
v___x_257_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_257_, 0, v___x_255_);
lean_ctor_set_uint8(v___x_257_, sizeof(void*)*1, v___x_256_);
v___x_258_ = l_Repr_addAppParen(v___x_257_, v_prec_223_);
return v___x_258_;
}
v___jp_259_:
{
lean_object* v___x_261_; lean_object* v___x_262_; uint8_t v___x_263_; lean_object* v___x_264_; lean_object* v___x_265_; 
v___x_261_ = ((lean_object*)(lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__11));
lean_inc(v___y_260_);
v___x_262_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_262_, 0, v___y_260_);
lean_ctor_set(v___x_262_, 1, v___x_261_);
v___x_263_ = 0;
v___x_264_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_264_, 0, v___x_262_);
lean_ctor_set_uint8(v___x_264_, sizeof(void*)*1, v___x_263_);
v___x_265_ = l_Repr_addAppParen(v___x_264_, v_prec_223_);
return v___x_265_;
}
v___jp_266_:
{
lean_object* v___x_268_; lean_object* v___x_269_; uint8_t v___x_270_; lean_object* v___x_271_; lean_object* v___x_272_; 
v___x_268_ = ((lean_object*)(lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__13));
lean_inc(v___y_267_);
v___x_269_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_269_, 0, v___y_267_);
lean_ctor_set(v___x_269_, 1, v___x_268_);
v___x_270_ = 0;
v___x_271_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_271_, 0, v___x_269_);
lean_ctor_set_uint8(v___x_271_, sizeof(void*)*1, v___x_270_);
v___x_272_ = l_Repr_addAppParen(v___x_271_, v_prec_223_);
return v___x_272_;
}
v___jp_273_:
{
lean_object* v___x_275_; lean_object* v___x_276_; uint8_t v___x_277_; lean_object* v___x_278_; lean_object* v___x_279_; 
v___x_275_ = ((lean_object*)(lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__15));
lean_inc(v___y_274_);
v___x_276_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_276_, 0, v___y_274_);
lean_ctor_set(v___x_276_, 1, v___x_275_);
v___x_277_ = 0;
v___x_278_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_278_, 0, v___x_276_);
lean_ctor_set_uint8(v___x_278_, sizeof(void*)*1, v___x_277_);
v___x_279_ = l_Repr_addAppParen(v___x_278_, v_prec_223_);
return v___x_279_;
}
v___jp_280_:
{
lean_object* v___x_282_; lean_object* v___x_283_; uint8_t v___x_284_; lean_object* v___x_285_; lean_object* v___x_286_; 
v___x_282_ = ((lean_object*)(lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__17));
lean_inc(v___y_281_);
v___x_283_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_283_, 0, v___y_281_);
lean_ctor_set(v___x_283_, 1, v___x_282_);
v___x_284_ = 0;
v___x_285_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_285_, 0, v___x_283_);
lean_ctor_set_uint8(v___x_285_, sizeof(void*)*1, v___x_284_);
v___x_286_ = l_Repr_addAppParen(v___x_285_, v_prec_223_);
return v___x_286_;
}
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___boxed(lean_object* v_x_323_, lean_object* v_prec_324_){
_start:
{
uint8_t v_x_513__boxed_325_; lean_object* v_res_326_; 
v_x_513__boxed_325_ = lean_unbox(v_x_323_);
v_res_326_ = lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr(v_x_513__boxed_325_, v_prec_324_);
lean_dec(v_prec_324_);
return v_res_326_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Hold_ctorIdx(uint8_t v_x_329_){
_start:
{
switch(v_x_329_)
{
case 0:
{
lean_object* v___x_330_; 
v___x_330_ = lean_unsigned_to_nat(0u);
return v___x_330_;
}
case 1:
{
lean_object* v___x_331_; 
v___x_331_ = lean_unsigned_to_nat(1u);
return v___x_331_;
}
case 2:
{
lean_object* v___x_332_; 
v___x_332_ = lean_unsigned_to_nat(2u);
return v___x_332_;
}
case 3:
{
lean_object* v___x_333_; 
v___x_333_ = lean_unsigned_to_nat(3u);
return v___x_333_;
}
case 4:
{
lean_object* v___x_334_; 
v___x_334_ = lean_unsigned_to_nat(4u);
return v___x_334_;
}
default: 
{
lean_object* v___x_335_; 
v___x_335_ = lean_unsigned_to_nat(5u);
return v___x_335_;
}
}
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Hold_ctorIdx___boxed(lean_object* v_x_336_){
_start:
{
uint8_t v_x_boxed_337_; lean_object* v_res_338_; 
v_x_boxed_337_ = lean_unbox(v_x_336_);
v_res_338_ = lp_reflect_x2dnts_NTS_Hold_ctorIdx(v_x_boxed_337_);
return v_res_338_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Hold_toCtorIdx(uint8_t v_x_339_){
_start:
{
lean_object* v___x_340_; 
v___x_340_ = lp_reflect_x2dnts_NTS_Hold_ctorIdx(v_x_339_);
return v___x_340_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Hold_toCtorIdx___boxed(lean_object* v_x_341_){
_start:
{
uint8_t v_x_4__boxed_342_; lean_object* v_res_343_; 
v_x_4__boxed_342_ = lean_unbox(v_x_341_);
v_res_343_ = lp_reflect_x2dnts_NTS_Hold_toCtorIdx(v_x_4__boxed_342_);
return v_res_343_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Hold_ctorElim___redArg(lean_object* v_k_344_){
_start:
{
lean_inc(v_k_344_);
return v_k_344_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Hold_ctorElim___redArg___boxed(lean_object* v_k_345_){
_start:
{
lean_object* v_res_346_; 
v_res_346_ = lp_reflect_x2dnts_NTS_Hold_ctorElim___redArg(v_k_345_);
lean_dec(v_k_345_);
return v_res_346_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Hold_ctorElim(lean_object* v_motive_347_, lean_object* v_ctorIdx_348_, uint8_t v_t_349_, lean_object* v_h_350_, lean_object* v_k_351_){
_start:
{
lean_inc(v_k_351_);
return v_k_351_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Hold_ctorElim___boxed(lean_object* v_motive_352_, lean_object* v_ctorIdx_353_, lean_object* v_t_354_, lean_object* v_h_355_, lean_object* v_k_356_){
_start:
{
uint8_t v_t_boxed_357_; lean_object* v_res_358_; 
v_t_boxed_357_ = lean_unbox(v_t_354_);
v_res_358_ = lp_reflect_x2dnts_NTS_Hold_ctorElim(v_motive_352_, v_ctorIdx_353_, v_t_boxed_357_, v_h_355_, v_k_356_);
lean_dec(v_k_356_);
lean_dec(v_ctorIdx_353_);
return v_res_358_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Hold_desire_elim___redArg(lean_object* v_desire_359_){
_start:
{
lean_inc(v_desire_359_);
return v_desire_359_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Hold_desire_elim___redArg___boxed(lean_object* v_desire_360_){
_start:
{
lean_object* v_res_361_; 
v_res_361_ = lp_reflect_x2dnts_NTS_Hold_desire_elim___redArg(v_desire_360_);
lean_dec(v_desire_360_);
return v_res_361_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Hold_desire_elim(lean_object* v_motive_362_, uint8_t v_t_363_, lean_object* v_h_364_, lean_object* v_desire_365_){
_start:
{
lean_inc(v_desire_365_);
return v_desire_365_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Hold_desire_elim___boxed(lean_object* v_motive_366_, lean_object* v_t_367_, lean_object* v_h_368_, lean_object* v_desire_369_){
_start:
{
uint8_t v_t_boxed_370_; lean_object* v_res_371_; 
v_t_boxed_370_ = lean_unbox(v_t_367_);
v_res_371_ = lp_reflect_x2dnts_NTS_Hold_desire_elim(v_motive_366_, v_t_boxed_370_, v_h_368_, v_desire_369_);
lean_dec(v_desire_369_);
return v_res_371_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Hold_will_elim___redArg(lean_object* v_will_372_){
_start:
{
lean_inc(v_will_372_);
return v_will_372_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Hold_will_elim___redArg___boxed(lean_object* v_will_373_){
_start:
{
lean_object* v_res_374_; 
v_res_374_ = lp_reflect_x2dnts_NTS_Hold_will_elim___redArg(v_will_373_);
lean_dec(v_will_373_);
return v_res_374_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Hold_will_elim(lean_object* v_motive_375_, uint8_t v_t_376_, lean_object* v_h_377_, lean_object* v_will_378_){
_start:
{
lean_inc(v_will_378_);
return v_will_378_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Hold_will_elim___boxed(lean_object* v_motive_379_, lean_object* v_t_380_, lean_object* v_h_381_, lean_object* v_will_382_){
_start:
{
uint8_t v_t_boxed_383_; lean_object* v_res_384_; 
v_t_boxed_383_ = lean_unbox(v_t_380_);
v_res_384_ = lp_reflect_x2dnts_NTS_Hold_will_elim(v_motive_379_, v_t_boxed_383_, v_h_381_, v_will_382_);
lean_dec(v_will_382_);
return v_res_384_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Hold_duty_elim___redArg(lean_object* v_duty_385_){
_start:
{
lean_inc(v_duty_385_);
return v_duty_385_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Hold_duty_elim___redArg___boxed(lean_object* v_duty_386_){
_start:
{
lean_object* v_res_387_; 
v_res_387_ = lp_reflect_x2dnts_NTS_Hold_duty_elim___redArg(v_duty_386_);
lean_dec(v_duty_386_);
return v_res_387_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Hold_duty_elim(lean_object* v_motive_388_, uint8_t v_t_389_, lean_object* v_h_390_, lean_object* v_duty_391_){
_start:
{
lean_inc(v_duty_391_);
return v_duty_391_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Hold_duty_elim___boxed(lean_object* v_motive_392_, lean_object* v_t_393_, lean_object* v_h_394_, lean_object* v_duty_395_){
_start:
{
uint8_t v_t_boxed_396_; lean_object* v_res_397_; 
v_t_boxed_396_ = lean_unbox(v_t_393_);
v_res_397_ = lp_reflect_x2dnts_NTS_Hold_duty_elim(v_motive_392_, v_t_boxed_396_, v_h_394_, v_duty_395_);
lean_dec(v_duty_395_);
return v_res_397_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Hold_sacred_elim___redArg(lean_object* v_sacred_398_){
_start:
{
lean_inc(v_sacred_398_);
return v_sacred_398_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Hold_sacred_elim___redArg___boxed(lean_object* v_sacred_399_){
_start:
{
lean_object* v_res_400_; 
v_res_400_ = lp_reflect_x2dnts_NTS_Hold_sacred_elim___redArg(v_sacred_399_);
lean_dec(v_sacred_399_);
return v_res_400_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Hold_sacred_elim(lean_object* v_motive_401_, uint8_t v_t_402_, lean_object* v_h_403_, lean_object* v_sacred_404_){
_start:
{
lean_inc(v_sacred_404_);
return v_sacred_404_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Hold_sacred_elim___boxed(lean_object* v_motive_405_, lean_object* v_t_406_, lean_object* v_h_407_, lean_object* v_sacred_408_){
_start:
{
uint8_t v_t_boxed_409_; lean_object* v_res_410_; 
v_t_boxed_409_ = lean_unbox(v_t_406_);
v_res_410_ = lp_reflect_x2dnts_NTS_Hold_sacred_elim(v_motive_405_, v_t_boxed_409_, v_h_407_, v_sacred_408_);
lean_dec(v_sacred_408_);
return v_res_410_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Hold_notForMe_elim___redArg(lean_object* v_notForMe_411_){
_start:
{
lean_inc(v_notForMe_411_);
return v_notForMe_411_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Hold_notForMe_elim___redArg___boxed(lean_object* v_notForMe_412_){
_start:
{
lean_object* v_res_413_; 
v_res_413_ = lp_reflect_x2dnts_NTS_Hold_notForMe_elim___redArg(v_notForMe_412_);
lean_dec(v_notForMe_412_);
return v_res_413_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Hold_notForMe_elim(lean_object* v_motive_414_, uint8_t v_t_415_, lean_object* v_h_416_, lean_object* v_notForMe_417_){
_start:
{
lean_inc(v_notForMe_417_);
return v_notForMe_417_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Hold_notForMe_elim___boxed(lean_object* v_motive_418_, lean_object* v_t_419_, lean_object* v_h_420_, lean_object* v_notForMe_421_){
_start:
{
uint8_t v_t_boxed_422_; lean_object* v_res_423_; 
v_t_boxed_422_ = lean_unbox(v_t_419_);
v_res_423_ = lp_reflect_x2dnts_NTS_Hold_notForMe_elim(v_motive_418_, v_t_boxed_422_, v_h_420_, v_notForMe_421_);
lean_dec(v_notForMe_421_);
return v_res_423_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Hold_unsure_elim___redArg(lean_object* v_unsure_424_){
_start:
{
lean_inc(v_unsure_424_);
return v_unsure_424_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Hold_unsure_elim___redArg___boxed(lean_object* v_unsure_425_){
_start:
{
lean_object* v_res_426_; 
v_res_426_ = lp_reflect_x2dnts_NTS_Hold_unsure_elim___redArg(v_unsure_425_);
lean_dec(v_unsure_425_);
return v_res_426_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Hold_unsure_elim(lean_object* v_motive_427_, uint8_t v_t_428_, lean_object* v_h_429_, lean_object* v_unsure_430_){
_start:
{
lean_inc(v_unsure_430_);
return v_unsure_430_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Hold_unsure_elim___boxed(lean_object* v_motive_431_, lean_object* v_t_432_, lean_object* v_h_433_, lean_object* v_unsure_434_){
_start:
{
uint8_t v_t_boxed_435_; lean_object* v_res_436_; 
v_t_boxed_435_ = lean_unbox(v_t_432_);
v_res_436_ = lp_reflect_x2dnts_NTS_Hold_unsure_elim(v_motive_431_, v_t_boxed_435_, v_h_433_, v_unsure_434_);
lean_dec(v_unsure_434_);
return v_res_436_;
}
}
LEAN_EXPORT uint8_t lp_reflect_x2dnts_NTS_Hold_ofNat(lean_object* v_n_437_){
_start:
{
lean_object* v___x_438_; uint8_t v___x_439_; 
v___x_438_ = lean_unsigned_to_nat(2u);
v___x_439_ = lean_nat_dec_le(v_n_437_, v___x_438_);
if (v___x_439_ == 0)
{
lean_object* v___x_440_; uint8_t v___x_441_; 
v___x_440_ = lean_unsigned_to_nat(3u);
v___x_441_ = lean_nat_dec_le(v_n_437_, v___x_440_);
if (v___x_441_ == 0)
{
lean_object* v___x_442_; uint8_t v___x_443_; 
v___x_442_ = lean_unsigned_to_nat(4u);
v___x_443_ = lean_nat_dec_le(v_n_437_, v___x_442_);
if (v___x_443_ == 0)
{
uint8_t v___x_444_; 
v___x_444_ = 5;
return v___x_444_;
}
else
{
uint8_t v___x_445_; 
v___x_445_ = 4;
return v___x_445_;
}
}
else
{
uint8_t v___x_446_; 
v___x_446_ = 3;
return v___x_446_;
}
}
else
{
lean_object* v___x_447_; uint8_t v___x_448_; 
v___x_447_ = lean_unsigned_to_nat(0u);
v___x_448_ = lean_nat_dec_le(v_n_437_, v___x_447_);
if (v___x_448_ == 0)
{
lean_object* v___x_449_; uint8_t v___x_450_; 
v___x_449_ = lean_unsigned_to_nat(1u);
v___x_450_ = lean_nat_dec_le(v_n_437_, v___x_449_);
if (v___x_450_ == 0)
{
uint8_t v___x_451_; 
v___x_451_ = 2;
return v___x_451_;
}
else
{
uint8_t v___x_452_; 
v___x_452_ = 1;
return v___x_452_;
}
}
else
{
uint8_t v___x_453_; 
v___x_453_ = 0;
return v___x_453_;
}
}
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Hold_ofNat___boxed(lean_object* v_n_454_){
_start:
{
uint8_t v_res_455_; lean_object* v_r_456_; 
v_res_455_ = lp_reflect_x2dnts_NTS_Hold_ofNat(v_n_454_);
lean_dec(v_n_454_);
v_r_456_ = lean_box(v_res_455_);
return v_r_456_;
}
}
LEAN_EXPORT uint8_t lp_reflect_x2dnts_NTS_instDecidableEqHold(uint8_t v_x_457_, uint8_t v_y_458_){
_start:
{
lean_object* v___x_459_; lean_object* v___x_460_; uint8_t v___x_461_; 
v___x_459_ = lp_reflect_x2dnts_NTS_Hold_ctorIdx(v_x_457_);
v___x_460_ = lp_reflect_x2dnts_NTS_Hold_ctorIdx(v_y_458_);
v___x_461_ = lean_nat_dec_eq(v___x_459_, v___x_460_);
lean_dec(v___x_460_);
lean_dec(v___x_459_);
return v___x_461_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instDecidableEqHold___boxed(lean_object* v_x_462_, lean_object* v_y_463_){
_start:
{
uint8_t v_x_13__boxed_464_; uint8_t v_y_14__boxed_465_; uint8_t v_res_466_; lean_object* v_r_467_; 
v_x_13__boxed_464_ = lean_unbox(v_x_462_);
v_y_14__boxed_465_ = lean_unbox(v_y_463_);
v_res_466_ = lp_reflect_x2dnts_NTS_instDecidableEqHold(v_x_13__boxed_464_, v_y_14__boxed_465_);
v_r_467_ = lean_box(v_res_466_);
return v_r_467_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instReprHold_repr(uint8_t v_x_486_, lean_object* v_prec_487_){
_start:
{
lean_object* v___y_489_; lean_object* v___y_496_; lean_object* v___y_503_; lean_object* v___y_510_; lean_object* v___y_517_; lean_object* v___y_524_; 
switch(v_x_486_)
{
case 0:
{
lean_object* v___x_530_; uint8_t v___x_531_; 
v___x_530_ = lean_unsigned_to_nat(1024u);
v___x_531_ = lean_nat_dec_le(v___x_530_, v_prec_487_);
if (v___x_531_ == 0)
{
lean_object* v___x_532_; 
v___x_532_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__18, &lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__18_once, _init_lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__18);
v___y_489_ = v___x_532_;
goto v___jp_488_;
}
else
{
lean_object* v___x_533_; 
v___x_533_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__19, &lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__19_once, _init_lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__19);
v___y_489_ = v___x_533_;
goto v___jp_488_;
}
}
case 1:
{
lean_object* v___x_534_; uint8_t v___x_535_; 
v___x_534_ = lean_unsigned_to_nat(1024u);
v___x_535_ = lean_nat_dec_le(v___x_534_, v_prec_487_);
if (v___x_535_ == 0)
{
lean_object* v___x_536_; 
v___x_536_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__18, &lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__18_once, _init_lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__18);
v___y_496_ = v___x_536_;
goto v___jp_495_;
}
else
{
lean_object* v___x_537_; 
v___x_537_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__19, &lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__19_once, _init_lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__19);
v___y_496_ = v___x_537_;
goto v___jp_495_;
}
}
case 2:
{
lean_object* v___x_538_; uint8_t v___x_539_; 
v___x_538_ = lean_unsigned_to_nat(1024u);
v___x_539_ = lean_nat_dec_le(v___x_538_, v_prec_487_);
if (v___x_539_ == 0)
{
lean_object* v___x_540_; 
v___x_540_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__18, &lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__18_once, _init_lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__18);
v___y_503_ = v___x_540_;
goto v___jp_502_;
}
else
{
lean_object* v___x_541_; 
v___x_541_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__19, &lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__19_once, _init_lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__19);
v___y_503_ = v___x_541_;
goto v___jp_502_;
}
}
case 3:
{
lean_object* v___x_542_; uint8_t v___x_543_; 
v___x_542_ = lean_unsigned_to_nat(1024u);
v___x_543_ = lean_nat_dec_le(v___x_542_, v_prec_487_);
if (v___x_543_ == 0)
{
lean_object* v___x_544_; 
v___x_544_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__18, &lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__18_once, _init_lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__18);
v___y_510_ = v___x_544_;
goto v___jp_509_;
}
else
{
lean_object* v___x_545_; 
v___x_545_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__19, &lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__19_once, _init_lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__19);
v___y_510_ = v___x_545_;
goto v___jp_509_;
}
}
case 4:
{
lean_object* v___x_546_; uint8_t v___x_547_; 
v___x_546_ = lean_unsigned_to_nat(1024u);
v___x_547_ = lean_nat_dec_le(v___x_546_, v_prec_487_);
if (v___x_547_ == 0)
{
lean_object* v___x_548_; 
v___x_548_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__18, &lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__18_once, _init_lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__18);
v___y_517_ = v___x_548_;
goto v___jp_516_;
}
else
{
lean_object* v___x_549_; 
v___x_549_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__19, &lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__19_once, _init_lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__19);
v___y_517_ = v___x_549_;
goto v___jp_516_;
}
}
default: 
{
lean_object* v___x_550_; uint8_t v___x_551_; 
v___x_550_ = lean_unsigned_to_nat(1024u);
v___x_551_ = lean_nat_dec_le(v___x_550_, v_prec_487_);
if (v___x_551_ == 0)
{
lean_object* v___x_552_; 
v___x_552_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__18, &lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__18_once, _init_lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__18);
v___y_524_ = v___x_552_;
goto v___jp_523_;
}
else
{
lean_object* v___x_553_; 
v___x_553_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__19, &lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__19_once, _init_lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__19);
v___y_524_ = v___x_553_;
goto v___jp_523_;
}
}
}
v___jp_488_:
{
lean_object* v___x_490_; lean_object* v___x_491_; uint8_t v___x_492_; lean_object* v___x_493_; lean_object* v___x_494_; 
v___x_490_ = ((lean_object*)(lp_reflect_x2dnts_NTS_instReprHold_repr___closed__1));
lean_inc(v___y_489_);
v___x_491_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_491_, 0, v___y_489_);
lean_ctor_set(v___x_491_, 1, v___x_490_);
v___x_492_ = 0;
v___x_493_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_493_, 0, v___x_491_);
lean_ctor_set_uint8(v___x_493_, sizeof(void*)*1, v___x_492_);
v___x_494_ = l_Repr_addAppParen(v___x_493_, v_prec_487_);
return v___x_494_;
}
v___jp_495_:
{
lean_object* v___x_497_; lean_object* v___x_498_; uint8_t v___x_499_; lean_object* v___x_500_; lean_object* v___x_501_; 
v___x_497_ = ((lean_object*)(lp_reflect_x2dnts_NTS_instReprHold_repr___closed__3));
lean_inc(v___y_496_);
v___x_498_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_498_, 0, v___y_496_);
lean_ctor_set(v___x_498_, 1, v___x_497_);
v___x_499_ = 0;
v___x_500_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_500_, 0, v___x_498_);
lean_ctor_set_uint8(v___x_500_, sizeof(void*)*1, v___x_499_);
v___x_501_ = l_Repr_addAppParen(v___x_500_, v_prec_487_);
return v___x_501_;
}
v___jp_502_:
{
lean_object* v___x_504_; lean_object* v___x_505_; uint8_t v___x_506_; lean_object* v___x_507_; lean_object* v___x_508_; 
v___x_504_ = ((lean_object*)(lp_reflect_x2dnts_NTS_instReprHold_repr___closed__5));
lean_inc(v___y_503_);
v___x_505_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_505_, 0, v___y_503_);
lean_ctor_set(v___x_505_, 1, v___x_504_);
v___x_506_ = 0;
v___x_507_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_507_, 0, v___x_505_);
lean_ctor_set_uint8(v___x_507_, sizeof(void*)*1, v___x_506_);
v___x_508_ = l_Repr_addAppParen(v___x_507_, v_prec_487_);
return v___x_508_;
}
v___jp_509_:
{
lean_object* v___x_511_; lean_object* v___x_512_; uint8_t v___x_513_; lean_object* v___x_514_; lean_object* v___x_515_; 
v___x_511_ = ((lean_object*)(lp_reflect_x2dnts_NTS_instReprHold_repr___closed__7));
lean_inc(v___y_510_);
v___x_512_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_512_, 0, v___y_510_);
lean_ctor_set(v___x_512_, 1, v___x_511_);
v___x_513_ = 0;
v___x_514_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_514_, 0, v___x_512_);
lean_ctor_set_uint8(v___x_514_, sizeof(void*)*1, v___x_513_);
v___x_515_ = l_Repr_addAppParen(v___x_514_, v_prec_487_);
return v___x_515_;
}
v___jp_516_:
{
lean_object* v___x_518_; lean_object* v___x_519_; uint8_t v___x_520_; lean_object* v___x_521_; lean_object* v___x_522_; 
v___x_518_ = ((lean_object*)(lp_reflect_x2dnts_NTS_instReprHold_repr___closed__9));
lean_inc(v___y_517_);
v___x_519_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_519_, 0, v___y_517_);
lean_ctor_set(v___x_519_, 1, v___x_518_);
v___x_520_ = 0;
v___x_521_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_521_, 0, v___x_519_);
lean_ctor_set_uint8(v___x_521_, sizeof(void*)*1, v___x_520_);
v___x_522_ = l_Repr_addAppParen(v___x_521_, v_prec_487_);
return v___x_522_;
}
v___jp_523_:
{
lean_object* v___x_525_; lean_object* v___x_526_; uint8_t v___x_527_; lean_object* v___x_528_; lean_object* v___x_529_; 
v___x_525_ = ((lean_object*)(lp_reflect_x2dnts_NTS_instReprHold_repr___closed__11));
lean_inc(v___y_524_);
v___x_526_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_526_, 0, v___y_524_);
lean_ctor_set(v___x_526_, 1, v___x_525_);
v___x_527_ = 0;
v___x_528_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_528_, 0, v___x_526_);
lean_ctor_set_uint8(v___x_528_, sizeof(void*)*1, v___x_527_);
v___x_529_ = l_Repr_addAppParen(v___x_528_, v_prec_487_);
return v___x_529_;
}
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instReprHold_repr___boxed(lean_object* v_x_554_, lean_object* v_prec_555_){
_start:
{
uint8_t v_x_341__boxed_556_; lean_object* v_res_557_; 
v_x_341__boxed_556_ = lean_unbox(v_x_554_);
v_res_557_ = lp_reflect_x2dnts_NTS_instReprHold_repr(v_x_341__boxed_556_, v_prec_555_);
lean_dec(v_prec_555_);
return v_res_557_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Hold_id(uint8_t v_x_566_){
_start:
{
switch(v_x_566_)
{
case 0:
{
lean_object* v___x_567_; 
v___x_567_ = ((lean_object*)(lp_reflect_x2dnts_NTS_Hold_id___closed__0));
return v___x_567_;
}
case 1:
{
lean_object* v___x_568_; 
v___x_568_ = ((lean_object*)(lp_reflect_x2dnts_NTS_Hold_id___closed__1));
return v___x_568_;
}
case 2:
{
lean_object* v___x_569_; 
v___x_569_ = ((lean_object*)(lp_reflect_x2dnts_NTS_Hold_id___closed__2));
return v___x_569_;
}
case 3:
{
lean_object* v___x_570_; 
v___x_570_ = ((lean_object*)(lp_reflect_x2dnts_NTS_Hold_id___closed__3));
return v___x_570_;
}
case 4:
{
lean_object* v___x_571_; 
v___x_571_ = ((lean_object*)(lp_reflect_x2dnts_NTS_Hold_id___closed__4));
return v___x_571_;
}
default: 
{
lean_object* v___x_572_; 
v___x_572_ = ((lean_object*)(lp_reflect_x2dnts_NTS_Hold_id___closed__5));
return v___x_572_;
}
}
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Hold_id___boxed(lean_object* v_x_573_){
_start:
{
uint8_t v_x_58__boxed_574_; lean_object* v_res_575_; 
v_x_58__boxed_574_ = lean_unbox(v_x_573_);
v_res_575_ = lp_reflect_x2dnts_NTS_Hold_id(v_x_58__boxed_574_);
return v_res_575_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Hold_facet(uint8_t v_x_592_){
_start:
{
switch(v_x_592_)
{
case 0:
{
lean_object* v___x_593_; 
v___x_593_ = ((lean_object*)(lp_reflect_x2dnts_NTS_Hold_facet___closed__1));
return v___x_593_;
}
case 1:
{
lean_object* v___x_594_; 
v___x_594_ = ((lean_object*)(lp_reflect_x2dnts_NTS_Hold_facet___closed__3));
return v___x_594_;
}
case 2:
{
lean_object* v___x_595_; 
v___x_595_ = ((lean_object*)(lp_reflect_x2dnts_NTS_Hold_facet___closed__5));
return v___x_595_;
}
case 3:
{
lean_object* v___x_596_; 
v___x_596_ = ((lean_object*)(lp_reflect_x2dnts_NTS_Hold_facet___closed__7));
return v___x_596_;
}
default: 
{
lean_object* v___x_597_; 
v___x_597_ = lean_box(0);
return v___x_597_;
}
}
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Hold_facet___boxed(lean_object* v_x_598_){
_start:
{
uint8_t v_x_92__boxed_599_; lean_object* v_res_600_; 
v_x_92__boxed_599_ = lean_unbox(v_x_598_);
v_res_600_ = lp_reflect_x2dnts_NTS_Hold_facet(v_x_92__boxed_599_);
return v_res_600_;
}
}
LEAN_EXPORT uint8_t lp_reflect_x2dnts_NTS_Hold_stance(uint8_t v_h_601_){
_start:
{
lean_object* v___x_602_; 
v___x_602_ = lp_reflect_x2dnts_NTS_Hold_facet(v_h_601_);
if (lean_obj_tag(v___x_602_) == 0)
{
uint8_t v___x_603_; 
v___x_603_ = 0;
return v___x_603_;
}
else
{
lean_object* v_val_604_; uint8_t v___x_605_; 
v_val_604_ = lean_ctor_get(v___x_602_, 0);
lean_inc(v_val_604_);
lean_dec_ref_known(v___x_602_, 1);
v___x_605_ = lp_reflect_x2dnts_NTS_Facet_stance(v_val_604_);
lean_dec(v_val_604_);
return v___x_605_;
}
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Hold_stance___boxed(lean_object* v_h_606_){
_start:
{
uint8_t v_h_boxed_607_; uint8_t v_res_608_; lean_object* v_r_609_; 
v_h_boxed_607_ = lean_unbox(v_h_606_);
v_res_608_ = lp_reflect_x2dnts_NTS_Hold_stance(v_h_boxed_607_);
v_r_609_ = lean_box(v_res_608_);
return v_r_609_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Destination_ctorIdx(lean_object* v_x_610_){
_start:
{
switch(lean_obj_tag(v_x_610_))
{
case 0:
{
lean_object* v___x_611_; 
v___x_611_ = lean_unsigned_to_nat(0u);
return v___x_611_;
}
case 1:
{
lean_object* v___x_612_; 
v___x_612_ = lean_unsigned_to_nat(1u);
return v___x_612_;
}
default: 
{
lean_object* v___x_613_; 
v___x_613_ = lean_unsigned_to_nat(2u);
return v___x_613_;
}
}
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Destination_ctorIdx___boxed(lean_object* v_x_614_){
_start:
{
lean_object* v_res_615_; 
v_res_615_ = lp_reflect_x2dnts_NTS_Destination_ctorIdx(v_x_614_);
lean_dec_ref(v_x_614_);
return v_res_615_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Destination_ctorElim___redArg(lean_object* v_t_616_, lean_object* v_k_617_){
_start:
{
if (lean_obj_tag(v_t_616_) == 1)
{
lean_object* v_kind_618_; lean_object* v___x_619_; 
v_kind_618_ = lean_ctor_get(v_t_616_, 0);
lean_inc(v_kind_618_);
lean_dec_ref_known(v_t_616_, 1);
v___x_619_ = lean_apply_1(v_k_617_, v_kind_618_);
return v___x_619_;
}
else
{
lean_object* v_key_620_; lean_object* v___x_621_; 
v_key_620_ = lean_ctor_get(v_t_616_, 0);
lean_inc_ref(v_key_620_);
lean_dec_ref(v_t_616_);
v___x_621_ = lean_apply_1(v_k_617_, v_key_620_);
return v___x_621_;
}
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Destination_ctorElim(lean_object* v_motive_622_, lean_object* v_ctorIdx_623_, lean_object* v_t_624_, lean_object* v_h_625_, lean_object* v_k_626_){
_start:
{
lean_object* v___x_627_; 
v___x_627_ = lp_reflect_x2dnts_NTS_Destination_ctorElim___redArg(v_t_624_, v_k_626_);
return v___x_627_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Destination_ctorElim___boxed(lean_object* v_motive_628_, lean_object* v_ctorIdx_629_, lean_object* v_t_630_, lean_object* v_h_631_, lean_object* v_k_632_){
_start:
{
lean_object* v_res_633_; 
v_res_633_ = lp_reflect_x2dnts_NTS_Destination_ctorElim(v_motive_628_, v_ctorIdx_629_, v_t_630_, v_h_631_, v_k_632_);
lean_dec(v_ctorIdx_629_);
return v_res_633_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Destination_profile_elim___redArg(lean_object* v_t_634_, lean_object* v_profile_635_){
_start:
{
lean_object* v___x_636_; 
v___x_636_ = lp_reflect_x2dnts_NTS_Destination_ctorElim___redArg(v_t_634_, v_profile_635_);
return v___x_636_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Destination_profile_elim(lean_object* v_motive_637_, lean_object* v_t_638_, lean_object* v_h_639_, lean_object* v_profile_640_){
_start:
{
lean_object* v___x_641_; 
v___x_641_ = lp_reflect_x2dnts_NTS_Destination_ctorElim___redArg(v_t_638_, v_profile_640_);
return v___x_641_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Destination_relationship_elim___redArg(lean_object* v_t_642_, lean_object* v_relationship_643_){
_start:
{
lean_object* v___x_644_; 
v___x_644_ = lp_reflect_x2dnts_NTS_Destination_ctorElim___redArg(v_t_642_, v_relationship_643_);
return v___x_644_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Destination_relationship_elim(lean_object* v_motive_645_, lean_object* v_t_646_, lean_object* v_h_647_, lean_object* v_relationship_648_){
_start:
{
lean_object* v___x_649_; 
v___x_649_ = lp_reflect_x2dnts_NTS_Destination_ctorElim___redArg(v_t_646_, v_relationship_648_);
return v___x_649_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Destination_hold_elim___redArg(lean_object* v_t_650_, lean_object* v_hold_651_){
_start:
{
lean_object* v___x_652_; 
v___x_652_ = lp_reflect_x2dnts_NTS_Destination_ctorElim___redArg(v_t_650_, v_hold_651_);
return v___x_652_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Destination_hold_elim(lean_object* v_motive_653_, lean_object* v_t_654_, lean_object* v_h_655_, lean_object* v_hold_656_){
_start:
{
lean_object* v___x_657_; 
v___x_657_ = lp_reflect_x2dnts_NTS_Destination_ctorElim___redArg(v_t_654_, v_hold_656_);
return v___x_657_;
}
}
LEAN_EXPORT uint8_t lp_reflect_x2dnts_NTS_instDecidableEqDestination_decEq(lean_object* v_x_658_, lean_object* v_x_659_){
_start:
{
switch(lean_obj_tag(v_x_658_))
{
case 0:
{
if (lean_obj_tag(v_x_659_) == 0)
{
lean_object* v_key_660_; lean_object* v_key_661_; uint8_t v___x_662_; 
v_key_660_ = lean_ctor_get(v_x_658_, 0);
v_key_661_ = lean_ctor_get(v_x_659_, 0);
v___x_662_ = lean_string_dec_eq(v_key_660_, v_key_661_);
return v___x_662_;
}
else
{
uint8_t v___x_663_; 
v___x_663_ = 0;
return v___x_663_;
}
}
case 1:
{
if (lean_obj_tag(v_x_659_) == 1)
{
uint8_t v___x_664_; 
v___x_664_ = 1;
return v___x_664_;
}
else
{
uint8_t v___x_665_; 
v___x_665_ = 0;
return v___x_665_;
}
}
default: 
{
if (lean_obj_tag(v_x_659_) == 2)
{
lean_object* v_subject_666_; lean_object* v_subject_667_; uint8_t v___x_668_; 
v_subject_666_ = lean_ctor_get(v_x_658_, 0);
v_subject_667_ = lean_ctor_get(v_x_659_, 0);
v___x_668_ = lean_string_dec_eq(v_subject_666_, v_subject_667_);
return v___x_668_;
}
else
{
uint8_t v___x_669_; 
v___x_669_ = 0;
return v___x_669_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instDecidableEqDestination_decEq___boxed(lean_object* v_x_670_, lean_object* v_x_671_){
_start:
{
uint8_t v_res_672_; lean_object* v_r_673_; 
v_res_672_ = lp_reflect_x2dnts_NTS_instDecidableEqDestination_decEq(v_x_670_, v_x_671_);
lean_dec_ref(v_x_671_);
lean_dec_ref(v_x_670_);
v_r_673_ = lean_box(v_res_672_);
return v_r_673_;
}
}
LEAN_EXPORT uint8_t lp_reflect_x2dnts_NTS_instDecidableEqDestination(lean_object* v_x_674_, lean_object* v_x_675_){
_start:
{
uint8_t v___x_676_; 
v___x_676_ = lp_reflect_x2dnts_NTS_instDecidableEqDestination_decEq(v_x_674_, v_x_675_);
return v___x_676_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instDecidableEqDestination___boxed(lean_object* v_x_677_, lean_object* v_x_678_){
_start:
{
uint8_t v_res_679_; lean_object* v_r_680_; 
v_res_679_ = lp_reflect_x2dnts_NTS_instDecidableEqDestination(v_x_677_, v_x_678_);
lean_dec_ref(v_x_678_);
lean_dec_ref(v_x_677_);
v_r_680_ = lean_box(v_res_679_);
return v_r_680_;
}
}
static lean_object* _init_lp_reflect_x2dnts_NTS_instReprDestination_repr___closed__3(void){
_start:
{
lean_object* v___x_687_; lean_object* v___x_688_; 
v___x_687_ = lean_unsigned_to_nat(1024u);
v___x_688_ = lp_reflect_x2dnts_NTS_instReprRelationKind_repr___redArg(v___x_687_);
return v___x_688_;
}
}
static lean_object* _init_lp_reflect_x2dnts_NTS_instReprDestination_repr___closed__4(void){
_start:
{
lean_object* v___x_689_; lean_object* v___x_690_; lean_object* v___x_691_; 
v___x_689_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprDestination_repr___closed__3, &lp_reflect_x2dnts_NTS_instReprDestination_repr___closed__3_once, _init_lp_reflect_x2dnts_NTS_instReprDestination_repr___closed__3);
v___x_690_ = ((lean_object*)(lp_reflect_x2dnts_NTS_instReprDestination_repr___closed__2));
v___x_691_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_691_, 0, v___x_690_);
lean_ctor_set(v___x_691_, 1, v___x_689_);
return v___x_691_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instReprDestination_repr(lean_object* v_x_704_, lean_object* v_prec_705_){
_start:
{
lean_object* v___y_707_; 
switch(lean_obj_tag(v_x_704_))
{
case 0:
{
lean_object* v_key_713_; lean_object* v___x_715_; uint8_t v_isShared_716_; uint8_t v_isSharedCheck_733_; 
v_key_713_ = lean_ctor_get(v_x_704_, 0);
v_isSharedCheck_733_ = !lean_is_exclusive(v_x_704_);
if (v_isSharedCheck_733_ == 0)
{
v___x_715_ = v_x_704_;
v_isShared_716_ = v_isSharedCheck_733_;
goto v_resetjp_714_;
}
else
{
lean_inc(v_key_713_);
lean_dec(v_x_704_);
v___x_715_ = lean_box(0);
v_isShared_716_ = v_isSharedCheck_733_;
goto v_resetjp_714_;
}
v_resetjp_714_:
{
lean_object* v___y_718_; lean_object* v___x_729_; uint8_t v___x_730_; 
v___x_729_ = lean_unsigned_to_nat(1024u);
v___x_730_ = lean_nat_dec_le(v___x_729_, v_prec_705_);
if (v___x_730_ == 0)
{
lean_object* v___x_731_; 
v___x_731_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__18, &lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__18_once, _init_lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__18);
v___y_718_ = v___x_731_;
goto v___jp_717_;
}
else
{
lean_object* v___x_732_; 
v___x_732_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__19, &lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__19_once, _init_lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__19);
v___y_718_ = v___x_732_;
goto v___jp_717_;
}
v___jp_717_:
{
lean_object* v___x_719_; lean_object* v___x_720_; lean_object* v___x_722_; 
v___x_719_ = ((lean_object*)(lp_reflect_x2dnts_NTS_instReprDestination_repr___closed__7));
v___x_720_ = l_String_quote(v_key_713_);
if (v_isShared_716_ == 0)
{
lean_ctor_set_tag(v___x_715_, 3);
lean_ctor_set(v___x_715_, 0, v___x_720_);
v___x_722_ = v___x_715_;
goto v_reusejp_721_;
}
else
{
lean_object* v_reuseFailAlloc_728_; 
v_reuseFailAlloc_728_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v_reuseFailAlloc_728_, 0, v___x_720_);
v___x_722_ = v_reuseFailAlloc_728_;
goto v_reusejp_721_;
}
v_reusejp_721_:
{
lean_object* v___x_723_; lean_object* v___x_724_; uint8_t v___x_725_; lean_object* v___x_726_; lean_object* v___x_727_; 
v___x_723_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_723_, 0, v___x_719_);
lean_ctor_set(v___x_723_, 1, v___x_722_);
lean_inc(v___y_718_);
v___x_724_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_724_, 0, v___y_718_);
lean_ctor_set(v___x_724_, 1, v___x_723_);
v___x_725_ = 0;
v___x_726_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_726_, 0, v___x_724_);
lean_ctor_set_uint8(v___x_726_, sizeof(void*)*1, v___x_725_);
v___x_727_ = l_Repr_addAppParen(v___x_726_, v_prec_705_);
return v___x_727_;
}
}
}
}
case 1:
{
lean_object* v___x_734_; uint8_t v___x_735_; 
lean_dec_ref_known(v_x_704_, 1);
v___x_734_ = lean_unsigned_to_nat(1024u);
v___x_735_ = lean_nat_dec_le(v___x_734_, v_prec_705_);
if (v___x_735_ == 0)
{
lean_object* v___x_736_; 
v___x_736_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__18, &lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__18_once, _init_lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__18);
v___y_707_ = v___x_736_;
goto v___jp_706_;
}
else
{
lean_object* v___x_737_; 
v___x_737_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__19, &lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__19_once, _init_lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__19);
v___y_707_ = v___x_737_;
goto v___jp_706_;
}
}
default: 
{
lean_object* v_subject_738_; lean_object* v___x_740_; uint8_t v_isShared_741_; uint8_t v_isSharedCheck_758_; 
v_subject_738_ = lean_ctor_get(v_x_704_, 0);
v_isSharedCheck_758_ = !lean_is_exclusive(v_x_704_);
if (v_isSharedCheck_758_ == 0)
{
v___x_740_ = v_x_704_;
v_isShared_741_ = v_isSharedCheck_758_;
goto v_resetjp_739_;
}
else
{
lean_inc(v_subject_738_);
lean_dec(v_x_704_);
v___x_740_ = lean_box(0);
v_isShared_741_ = v_isSharedCheck_758_;
goto v_resetjp_739_;
}
v_resetjp_739_:
{
lean_object* v___y_743_; lean_object* v___x_754_; uint8_t v___x_755_; 
v___x_754_ = lean_unsigned_to_nat(1024u);
v___x_755_ = lean_nat_dec_le(v___x_754_, v_prec_705_);
if (v___x_755_ == 0)
{
lean_object* v___x_756_; 
v___x_756_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__18, &lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__18_once, _init_lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__18);
v___y_743_ = v___x_756_;
goto v___jp_742_;
}
else
{
lean_object* v___x_757_; 
v___x_757_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__19, &lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__19_once, _init_lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__19);
v___y_743_ = v___x_757_;
goto v___jp_742_;
}
v___jp_742_:
{
lean_object* v___x_744_; lean_object* v___x_745_; lean_object* v___x_747_; 
v___x_744_ = ((lean_object*)(lp_reflect_x2dnts_NTS_instReprDestination_repr___closed__10));
v___x_745_ = l_String_quote(v_subject_738_);
if (v_isShared_741_ == 0)
{
lean_ctor_set_tag(v___x_740_, 3);
lean_ctor_set(v___x_740_, 0, v___x_745_);
v___x_747_ = v___x_740_;
goto v_reusejp_746_;
}
else
{
lean_object* v_reuseFailAlloc_753_; 
v_reuseFailAlloc_753_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v_reuseFailAlloc_753_, 0, v___x_745_);
v___x_747_ = v_reuseFailAlloc_753_;
goto v_reusejp_746_;
}
v_reusejp_746_:
{
lean_object* v___x_748_; lean_object* v___x_749_; uint8_t v___x_750_; lean_object* v___x_751_; lean_object* v___x_752_; 
v___x_748_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_748_, 0, v___x_744_);
lean_ctor_set(v___x_748_, 1, v___x_747_);
lean_inc(v___y_743_);
v___x_749_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_749_, 0, v___y_743_);
lean_ctor_set(v___x_749_, 1, v___x_748_);
v___x_750_ = 0;
v___x_751_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_751_, 0, v___x_749_);
lean_ctor_set_uint8(v___x_751_, sizeof(void*)*1, v___x_750_);
v___x_752_ = l_Repr_addAppParen(v___x_751_, v_prec_705_);
return v___x_752_;
}
}
}
}
}
v___jp_706_:
{
lean_object* v___x_708_; lean_object* v___x_709_; uint8_t v___x_710_; lean_object* v___x_711_; lean_object* v___x_712_; 
v___x_708_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprDestination_repr___closed__4, &lp_reflect_x2dnts_NTS_instReprDestination_repr___closed__4_once, _init_lp_reflect_x2dnts_NTS_instReprDestination_repr___closed__4);
lean_inc(v___y_707_);
v___x_709_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_709_, 0, v___y_707_);
lean_ctor_set(v___x_709_, 1, v___x_708_);
v___x_710_ = 0;
v___x_711_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_711_, 0, v___x_709_);
lean_ctor_set_uint8(v___x_711_, sizeof(void*)*1, v___x_710_);
v___x_712_ = l_Repr_addAppParen(v___x_711_, v_prec_705_);
return v___x_712_;
}
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instReprDestination_repr___boxed(lean_object* v_x_759_, lean_object* v_prec_760_){
_start:
{
lean_object* v_res_761_; 
v_res_761_ = lp_reflect_x2dnts_NTS_instReprDestination_repr(v_x_759_, v_prec_760_);
lean_dec(v_prec_760_);
return v_res_761_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_id(uint8_t v_x_773_){
_start:
{
switch(v_x_773_)
{
case 0:
{
lean_object* v___x_774_; 
v___x_774_ = ((lean_object*)(lp_reflect_x2dnts_NTS_OnboardingQuestion_id___closed__0));
return v___x_774_;
}
case 1:
{
lean_object* v___x_775_; 
v___x_775_ = ((lean_object*)(lp_reflect_x2dnts_NTS_OnboardingQuestion_id___closed__1));
return v___x_775_;
}
case 2:
{
lean_object* v___x_776_; 
v___x_776_ = ((lean_object*)(lp_reflect_x2dnts_NTS_OnboardingQuestion_id___closed__2));
return v___x_776_;
}
case 3:
{
lean_object* v___x_777_; 
v___x_777_ = ((lean_object*)(lp_reflect_x2dnts_NTS_OnboardingQuestion_id___closed__3));
return v___x_777_;
}
case 4:
{
lean_object* v___x_778_; 
v___x_778_ = ((lean_object*)(lp_reflect_x2dnts_NTS_OnboardingQuestion_id___closed__4));
return v___x_778_;
}
case 5:
{
lean_object* v___x_779_; 
v___x_779_ = ((lean_object*)(lp_reflect_x2dnts_NTS_OnboardingQuestion_id___closed__5));
return v___x_779_;
}
case 6:
{
lean_object* v___x_780_; 
v___x_780_ = ((lean_object*)(lp_reflect_x2dnts_NTS_OnboardingQuestion_id___closed__6));
return v___x_780_;
}
case 7:
{
lean_object* v___x_781_; 
v___x_781_ = ((lean_object*)(lp_reflect_x2dnts_NTS_OnboardingQuestion_id___closed__7));
return v___x_781_;
}
default: 
{
lean_object* v___x_782_; 
v___x_782_ = ((lean_object*)(lp_reflect_x2dnts_NTS_OnboardingQuestion_id___closed__8));
return v___x_782_;
}
}
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_id___boxed(lean_object* v_x_783_){
_start:
{
uint8_t v_x_85__boxed_784_; lean_object* v_res_785_; 
v_x_85__boxed_784_ = lean_unbox(v_x_783_);
v_res_785_ = lp_reflect_x2dnts_NTS_OnboardingQuestion_id(v_x_85__boxed_784_);
return v_res_785_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_destination(uint8_t v_x_805_){
_start:
{
switch(v_x_805_)
{
case 0:
{
lean_object* v___x_806_; 
v___x_806_ = ((lean_object*)(lp_reflect_x2dnts_NTS_OnboardingQuestion_destination___closed__0));
return v___x_806_;
}
case 1:
{
lean_object* v___x_807_; 
v___x_807_ = ((lean_object*)(lp_reflect_x2dnts_NTS_OnboardingQuestion_destination___closed__1));
return v___x_807_;
}
case 2:
{
lean_object* v___x_808_; 
v___x_808_ = ((lean_object*)(lp_reflect_x2dnts_NTS_OnboardingQuestion_destination___closed__2));
return v___x_808_;
}
case 3:
{
lean_object* v___x_809_; 
v___x_809_ = ((lean_object*)(lp_reflect_x2dnts_NTS_OnboardingQuestion_destination___closed__3));
return v___x_809_;
}
case 4:
{
lean_object* v___x_810_; 
v___x_810_ = ((lean_object*)(lp_reflect_x2dnts_NTS_OnboardingQuestion_destination___closed__4));
return v___x_810_;
}
case 5:
{
lean_object* v___x_811_; 
v___x_811_ = ((lean_object*)(lp_reflect_x2dnts_NTS_OnboardingQuestion_destination___closed__5));
return v___x_811_;
}
case 6:
{
lean_object* v___x_812_; 
v___x_812_ = ((lean_object*)(lp_reflect_x2dnts_NTS_OnboardingQuestion_destination___closed__6));
return v___x_812_;
}
case 7:
{
lean_object* v___x_813_; 
v___x_813_ = ((lean_object*)(lp_reflect_x2dnts_NTS_OnboardingQuestion_destination___closed__8));
return v___x_813_;
}
default: 
{
lean_object* v___x_814_; 
v___x_814_ = ((lean_object*)(lp_reflect_x2dnts_NTS_OnboardingQuestion_destination___closed__9));
return v___x_814_;
}
}
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_destination___boxed(lean_object* v_x_815_){
_start:
{
uint8_t v_x_130__boxed_816_; lean_object* v_res_817_; 
v_x_130__boxed_816_ = lean_unbox(v_x_815_);
v_res_817_ = lp_reflect_x2dnts_NTS_OnboardingQuestion_destination(v_x_130__boxed_816_);
return v_res_817_;
}
}
LEAN_EXPORT uint8_t lp_reflect_x2dnts_NTS_OnboardingQuestion_isHold(uint8_t v_q_855_){
_start:
{
lean_object* v___x_856_; 
v___x_856_ = lp_reflect_x2dnts_NTS_OnboardingQuestion_destination(v_q_855_);
if (lean_obj_tag(v___x_856_) == 2)
{
uint8_t v___x_857_; 
lean_dec_ref_known(v___x_856_, 1);
v___x_857_ = 1;
return v___x_857_;
}
else
{
uint8_t v___x_858_; 
lean_dec_ref(v___x_856_);
v___x_858_ = 0;
return v___x_858_;
}
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_isHold___boxed(lean_object* v_q_859_){
_start:
{
uint8_t v_q_boxed_860_; uint8_t v_res_861_; lean_object* v_r_862_; 
v_q_boxed_860_ = lean_unbox(v_q_859_);
v_res_861_ = lp_reflect_x2dnts_NTS_OnboardingQuestion_isHold(v_q_boxed_860_);
v_r_862_ = lean_box(v_res_861_);
return v_r_862_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Written_ctorIdx(lean_object* v_x_863_){
_start:
{
switch(lean_obj_tag(v_x_863_))
{
case 0:
{
lean_object* v___x_864_; 
v___x_864_ = lean_unsigned_to_nat(0u);
return v___x_864_;
}
case 1:
{
lean_object* v___x_865_; 
v___x_865_ = lean_unsigned_to_nat(1u);
return v___x_865_;
}
default: 
{
lean_object* v___x_866_; 
v___x_866_ = lean_unsigned_to_nat(2u);
return v___x_866_;
}
}
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Written_ctorIdx___boxed(lean_object* v_x_867_){
_start:
{
lean_object* v_res_868_; 
v_res_868_ = lp_reflect_x2dnts_NTS_Written_ctorIdx(v_x_867_);
lean_dec_ref(v_x_867_);
return v_res_868_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Written_ctorElim___redArg(lean_object* v_t_869_, lean_object* v_k_870_){
_start:
{
if (lean_obj_tag(v_t_869_) == 2)
{
lean_object* v_subject_871_; lean_object* v_b_872_; lean_object* v___x_873_; 
v_subject_871_ = lean_ctor_get(v_t_869_, 0);
lean_inc_ref(v_subject_871_);
v_b_872_ = lean_ctor_get(v_t_869_, 1);
lean_inc_ref(v_b_872_);
lean_dec_ref_known(v_t_869_, 2);
v___x_873_ = lean_apply_2(v_k_870_, v_subject_871_, v_b_872_);
return v___x_873_;
}
else
{
lean_object* v_f_874_; lean_object* v___x_875_; 
v_f_874_ = lean_ctor_get(v_t_869_, 0);
lean_inc_ref(v_f_874_);
lean_dec_ref(v_t_869_);
v___x_875_ = lean_apply_1(v_k_870_, v_f_874_);
return v___x_875_;
}
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Written_ctorElim(lean_object* v_motive_876_, lean_object* v_ctorIdx_877_, lean_object* v_t_878_, lean_object* v_h_879_, lean_object* v_k_880_){
_start:
{
lean_object* v___x_881_; 
v___x_881_ = lp_reflect_x2dnts_NTS_Written_ctorElim___redArg(v_t_878_, v_k_880_);
return v___x_881_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Written_ctorElim___boxed(lean_object* v_motive_882_, lean_object* v_ctorIdx_883_, lean_object* v_t_884_, lean_object* v_h_885_, lean_object* v_k_886_){
_start:
{
lean_object* v_res_887_; 
v_res_887_ = lp_reflect_x2dnts_NTS_Written_ctorElim(v_motive_882_, v_ctorIdx_883_, v_t_884_, v_h_885_, v_k_886_);
lean_dec(v_ctorIdx_883_);
return v_res_887_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Written_fact_elim___redArg(lean_object* v_t_888_, lean_object* v_fact_889_){
_start:
{
lean_object* v___x_890_; 
v___x_890_ = lp_reflect_x2dnts_NTS_Written_ctorElim___redArg(v_t_888_, v_fact_889_);
return v___x_890_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Written_fact_elim(lean_object* v_motive_891_, lean_object* v_t_892_, lean_object* v_h_893_, lean_object* v_fact_894_){
_start:
{
lean_object* v___x_895_; 
v___x_895_ = lp_reflect_x2dnts_NTS_Written_ctorElim___redArg(v_t_892_, v_fact_894_);
return v___x_895_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Written_relationship_elim___redArg(lean_object* v_t_896_, lean_object* v_relationship_897_){
_start:
{
lean_object* v___x_898_; 
v___x_898_ = lp_reflect_x2dnts_NTS_Written_ctorElim___redArg(v_t_896_, v_relationship_897_);
return v___x_898_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Written_relationship_elim(lean_object* v_motive_899_, lean_object* v_t_900_, lean_object* v_h_901_, lean_object* v_relationship_902_){
_start:
{
lean_object* v___x_903_; 
v___x_903_ = lp_reflect_x2dnts_NTS_Written_ctorElim___redArg(v_t_900_, v_relationship_902_);
return v___x_903_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Written_hold_elim___redArg(lean_object* v_t_904_, lean_object* v_hold_905_){
_start:
{
lean_object* v___x_906_; 
v___x_906_ = lp_reflect_x2dnts_NTS_Written_ctorElim___redArg(v_t_904_, v_hold_905_);
return v___x_906_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Written_hold_elim(lean_object* v_motive_907_, lean_object* v_t_908_, lean_object* v_h_909_, lean_object* v_hold_910_){
_start:
{
lean_object* v___x_911_; 
v___x_911_ = lp_reflect_x2dnts_NTS_Written_ctorElim___redArg(v_t_908_, v_hold_910_);
return v___x_911_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instReprWritten_repr(lean_object* v_x_930_, lean_object* v_prec_931_){
_start:
{
switch(lean_obj_tag(v_x_930_))
{
case 0:
{
lean_object* v_f_932_; lean_object* v___y_934_; lean_object* v___x_942_; uint8_t v___x_943_; 
v_f_932_ = lean_ctor_get(v_x_930_, 0);
lean_inc_ref(v_f_932_);
lean_dec_ref_known(v_x_930_, 1);
v___x_942_ = lean_unsigned_to_nat(1024u);
v___x_943_ = lean_nat_dec_le(v___x_942_, v_prec_931_);
if (v___x_943_ == 0)
{
lean_object* v___x_944_; 
v___x_944_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__18, &lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__18_once, _init_lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__18);
v___y_934_ = v___x_944_;
goto v___jp_933_;
}
else
{
lean_object* v___x_945_; 
v___x_945_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__19, &lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__19_once, _init_lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__19);
v___y_934_ = v___x_945_;
goto v___jp_933_;
}
v___jp_933_:
{
lean_object* v___x_935_; lean_object* v___x_936_; lean_object* v___x_937_; lean_object* v___x_938_; uint8_t v___x_939_; lean_object* v___x_940_; lean_object* v___x_941_; 
v___x_935_ = ((lean_object*)(lp_reflect_x2dnts_NTS_instReprWritten_repr___closed__2));
v___x_936_ = lp_reflect_x2dnts_NTS_instReprFact_repr___redArg(v_f_932_);
v___x_937_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_937_, 0, v___x_935_);
lean_ctor_set(v___x_937_, 1, v___x_936_);
lean_inc(v___y_934_);
v___x_938_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_938_, 0, v___y_934_);
lean_ctor_set(v___x_938_, 1, v___x_937_);
v___x_939_ = 0;
v___x_940_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_940_, 0, v___x_938_);
lean_ctor_set_uint8(v___x_940_, sizeof(void*)*1, v___x_939_);
v___x_941_ = l_Repr_addAppParen(v___x_940_, v_prec_931_);
return v___x_941_;
}
}
case 1:
{
lean_object* v_r_946_; lean_object* v___y_948_; lean_object* v___x_956_; uint8_t v___x_957_; 
v_r_946_ = lean_ctor_get(v_x_930_, 0);
lean_inc_ref(v_r_946_);
lean_dec_ref_known(v_x_930_, 1);
v___x_956_ = lean_unsigned_to_nat(1024u);
v___x_957_ = lean_nat_dec_le(v___x_956_, v_prec_931_);
if (v___x_957_ == 0)
{
lean_object* v___x_958_; 
v___x_958_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__18, &lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__18_once, _init_lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__18);
v___y_948_ = v___x_958_;
goto v___jp_947_;
}
else
{
lean_object* v___x_959_; 
v___x_959_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__19, &lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__19_once, _init_lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__19);
v___y_948_ = v___x_959_;
goto v___jp_947_;
}
v___jp_947_:
{
lean_object* v___x_949_; lean_object* v___x_950_; lean_object* v___x_951_; lean_object* v___x_952_; uint8_t v___x_953_; lean_object* v___x_954_; lean_object* v___x_955_; 
v___x_949_ = ((lean_object*)(lp_reflect_x2dnts_NTS_instReprWritten_repr___closed__5));
v___x_950_ = lp_reflect_x2dnts_NTS_instReprRelationship_repr___redArg(v_r_946_);
v___x_951_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_951_, 0, v___x_949_);
lean_ctor_set(v___x_951_, 1, v___x_950_);
lean_inc(v___y_948_);
v___x_952_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_952_, 0, v___y_948_);
lean_ctor_set(v___x_952_, 1, v___x_951_);
v___x_953_ = 0;
v___x_954_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_954_, 0, v___x_952_);
lean_ctor_set_uint8(v___x_954_, sizeof(void*)*1, v___x_953_);
v___x_955_ = l_Repr_addAppParen(v___x_954_, v_prec_931_);
return v___x_955_;
}
}
default: 
{
lean_object* v_subject_960_; lean_object* v_b_961_; lean_object* v___x_963_; uint8_t v_isShared_964_; uint8_t v_isSharedCheck_985_; 
v_subject_960_ = lean_ctor_get(v_x_930_, 0);
v_b_961_ = lean_ctor_get(v_x_930_, 1);
v_isSharedCheck_985_ = !lean_is_exclusive(v_x_930_);
if (v_isSharedCheck_985_ == 0)
{
v___x_963_ = v_x_930_;
v_isShared_964_ = v_isSharedCheck_985_;
goto v_resetjp_962_;
}
else
{
lean_inc(v_b_961_);
lean_inc(v_subject_960_);
lean_dec(v_x_930_);
v___x_963_ = lean_box(0);
v_isShared_964_ = v_isSharedCheck_985_;
goto v_resetjp_962_;
}
v_resetjp_962_:
{
lean_object* v___y_966_; lean_object* v___x_981_; uint8_t v___x_982_; 
v___x_981_ = lean_unsigned_to_nat(1024u);
v___x_982_ = lean_nat_dec_le(v___x_981_, v_prec_931_);
if (v___x_982_ == 0)
{
lean_object* v___x_983_; 
v___x_983_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__18, &lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__18_once, _init_lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__18);
v___y_966_ = v___x_983_;
goto v___jp_965_;
}
else
{
lean_object* v___x_984_; 
v___x_984_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__19, &lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__19_once, _init_lp_reflect_x2dnts_NTS_instReprOnboardingQuestion_repr___closed__19);
v___y_966_ = v___x_984_;
goto v___jp_965_;
}
v___jp_965_:
{
lean_object* v___x_967_; lean_object* v___x_968_; lean_object* v___x_969_; lean_object* v___x_970_; lean_object* v___x_972_; 
v___x_967_ = lean_box(1);
v___x_968_ = ((lean_object*)(lp_reflect_x2dnts_NTS_instReprWritten_repr___closed__8));
v___x_969_ = l_String_quote(v_subject_960_);
v___x_970_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_970_, 0, v___x_969_);
if (v_isShared_964_ == 0)
{
lean_ctor_set_tag(v___x_963_, 5);
lean_ctor_set(v___x_963_, 1, v___x_970_);
lean_ctor_set(v___x_963_, 0, v___x_968_);
v___x_972_ = v___x_963_;
goto v_reusejp_971_;
}
else
{
lean_object* v_reuseFailAlloc_980_; 
v_reuseFailAlloc_980_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v_reuseFailAlloc_980_, 0, v___x_968_);
lean_ctor_set(v_reuseFailAlloc_980_, 1, v___x_970_);
v___x_972_ = v_reuseFailAlloc_980_;
goto v_reusejp_971_;
}
v_reusejp_971_:
{
lean_object* v___x_973_; lean_object* v___x_974_; lean_object* v___x_975_; lean_object* v___x_976_; uint8_t v___x_977_; lean_object* v___x_978_; lean_object* v___x_979_; 
v___x_973_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_973_, 0, v___x_972_);
lean_ctor_set(v___x_973_, 1, v___x_967_);
v___x_974_ = lp_reflect_x2dnts_NTS_instReprBelief_repr___redArg(v_b_961_);
v___x_975_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_975_, 0, v___x_973_);
lean_ctor_set(v___x_975_, 1, v___x_974_);
lean_inc(v___y_966_);
v___x_976_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_976_, 0, v___y_966_);
lean_ctor_set(v___x_976_, 1, v___x_975_);
v___x_977_ = 0;
v___x_978_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_978_, 0, v___x_976_);
lean_ctor_set_uint8(v___x_978_, sizeof(void*)*1, v___x_977_);
v___x_979_ = l_Repr_addAppParen(v___x_978_, v_prec_931_);
return v___x_979_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instReprWritten_repr___boxed(lean_object* v_x_986_, lean_object* v_prec_987_){
_start:
{
lean_object* v_res_988_; 
v_res_988_ = lp_reflect_x2dnts_NTS_instReprWritten_repr(v_x_986_, v_prec_987_);
lean_dec(v_prec_987_);
return v_res_988_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_write(uint8_t v_q_991_, lean_object* v_answer_992_, lean_object* v_at___993_){
_start:
{
lean_object* v___x_994_; 
v___x_994_ = lp_reflect_x2dnts_NTS_OnboardingQuestion_destination(v_q_991_);
switch(lean_obj_tag(v___x_994_))
{
case 0:
{
lean_object* v_key_995_; lean_object* v___x_997_; uint8_t v_isShared_998_; uint8_t v_isSharedCheck_1006_; 
v_key_995_ = lean_ctor_get(v___x_994_, 0);
v_isSharedCheck_1006_ = !lean_is_exclusive(v___x_994_);
if (v_isSharedCheck_1006_ == 0)
{
v___x_997_ = v___x_994_;
v_isShared_998_ = v_isSharedCheck_1006_;
goto v_resetjp_996_;
}
else
{
lean_inc(v_key_995_);
lean_dec(v___x_994_);
v___x_997_ = lean_box(0);
v_isShared_998_ = v_isSharedCheck_1006_;
goto v_resetjp_996_;
}
v_resetjp_996_:
{
lean_object* v___x_999_; lean_object* v___x_1000_; uint8_t v___x_1001_; lean_object* v___x_1002_; lean_object* v___x_1004_; 
v___x_999_ = lean_box(0);
v___x_1000_ = lean_box(0);
v___x_1001_ = 1;
lean_inc_ref(v_at___993_);
v___x_1002_ = lean_alloc_ctor(0, 6, 1);
lean_ctor_set(v___x_1002_, 0, v_key_995_);
lean_ctor_set(v___x_1002_, 1, v_answer_992_);
lean_ctor_set(v___x_1002_, 2, v_at___993_);
lean_ctor_set(v___x_1002_, 3, v___x_999_);
lean_ctor_set(v___x_1002_, 4, v_at___993_);
lean_ctor_set(v___x_1002_, 5, v___x_1000_);
lean_ctor_set_uint8(v___x_1002_, sizeof(void*)*6, v___x_1001_);
if (v_isShared_998_ == 0)
{
lean_ctor_set(v___x_997_, 0, v___x_1002_);
v___x_1004_ = v___x_997_;
goto v_reusejp_1003_;
}
else
{
lean_object* v_reuseFailAlloc_1005_; 
v_reuseFailAlloc_1005_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1005_, 0, v___x_1002_);
v___x_1004_ = v_reuseFailAlloc_1005_;
goto v_reusejp_1003_;
}
v_reusejp_1003_:
{
return v___x_1004_;
}
}
}
case 1:
{
lean_object* v_kind_1007_; lean_object* v___x_1009_; uint8_t v_isShared_1010_; uint8_t v_isSharedCheck_1017_; 
v_kind_1007_ = lean_ctor_get(v___x_994_, 0);
v_isSharedCheck_1017_ = !lean_is_exclusive(v___x_994_);
if (v_isSharedCheck_1017_ == 0)
{
v___x_1009_ = v___x_994_;
v_isShared_1010_ = v_isSharedCheck_1017_;
goto v_resetjp_1008_;
}
else
{
lean_inc(v_kind_1007_);
lean_dec(v___x_994_);
v___x_1009_ = lean_box(0);
v_isShared_1010_ = v_isSharedCheck_1017_;
goto v_resetjp_1008_;
}
v_resetjp_1008_:
{
lean_object* v___x_1011_; lean_object* v___x_1012_; lean_object* v___x_1013_; lean_object* v___x_1015_; 
v___x_1011_ = lean_box(0);
v___x_1012_ = lean_box(0);
lean_inc_ref(v_at___993_);
v___x_1013_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v___x_1013_, 0, v_answer_992_);
lean_ctor_set(v___x_1013_, 1, v_kind_1007_);
lean_ctor_set(v___x_1013_, 2, v_at___993_);
lean_ctor_set(v___x_1013_, 3, v___x_1011_);
lean_ctor_set(v___x_1013_, 4, v_at___993_);
lean_ctor_set(v___x_1013_, 5, v___x_1012_);
if (v_isShared_1010_ == 0)
{
lean_ctor_set(v___x_1009_, 0, v___x_1013_);
v___x_1015_ = v___x_1009_;
goto v_reusejp_1014_;
}
else
{
lean_object* v_reuseFailAlloc_1016_; 
v_reuseFailAlloc_1016_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1016_, 0, v___x_1013_);
v___x_1015_ = v_reuseFailAlloc_1016_;
goto v_reusejp_1014_;
}
v_reusejp_1014_:
{
return v___x_1015_;
}
}
}
default: 
{
lean_object* v_subject_1018_; lean_object* v___x_1019_; uint8_t v___x_1020_; lean_object* v___x_1021_; lean_object* v___x_1022_; 
v_subject_1018_ = lean_ctor_get(v___x_994_, 0);
lean_inc_ref(v_subject_1018_);
lean_dec_ref_known(v___x_994_, 1);
v___x_1019_ = lean_box(0);
v___x_1020_ = 1;
v___x_1021_ = lean_alloc_ctor(0, 3, 1);
lean_ctor_set(v___x_1021_, 0, v_answer_992_);
lean_ctor_set(v___x_1021_, 1, v_at___993_);
lean_ctor_set(v___x_1021_, 2, v___x_1019_);
lean_ctor_set_uint8(v___x_1021_, sizeof(void*)*3, v___x_1020_);
v___x_1022_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1022_, 0, v_subject_1018_);
lean_ctor_set(v___x_1022_, 1, v___x_1021_);
return v___x_1022_;
}
}
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_OnboardingQuestion_write___boxed(lean_object* v_q_1023_, lean_object* v_answer_1024_, lean_object* v_at___1025_){
_start:
{
uint8_t v_q_boxed_1026_; lean_object* v_res_1027_; 
v_q_boxed_1026_ = lean_unbox(v_q_1023_);
v_res_1027_ = lp_reflect_x2dnts_NTS_OnboardingQuestion_write(v_q_boxed_1026_, v_answer_1024_, v_at___1025_);
return v_res_1027_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Written_source(lean_object* v_x_1028_){
_start:
{
if (lean_obj_tag(v_x_1028_) == 2)
{
lean_object* v_b_1029_; lean_object* v_source_1030_; 
v_b_1029_ = lean_ctor_get(v_x_1028_, 1);
v_source_1030_ = lean_ctor_get(v_b_1029_, 2);
lean_inc(v_source_1030_);
return v_source_1030_;
}
else
{
lean_object* v_f_1031_; lean_object* v_source_1032_; 
v_f_1031_ = lean_ctor_get(v_x_1028_, 0);
v_source_1032_ = lean_ctor_get(v_f_1031_, 5);
lean_inc(v_source_1032_);
return v_source_1032_;
}
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Written_source___boxed(lean_object* v_x_1033_){
_start:
{
lean_object* v_res_1034_; 
v_res_1034_ = lp_reflect_x2dnts_NTS_Written_source(v_x_1033_);
lean_dec_ref(v_x_1033_);
return v_res_1034_;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_reflect_x2dnts_NTS_Person(uint8_t builtin);
lean_object* initialize_reflect_x2dnts_NTS_Output(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_reflect_x2dnts_NTS_Onboarding(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_reflect_x2dnts_NTS_Person(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_reflect_x2dnts_NTS_Output(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
