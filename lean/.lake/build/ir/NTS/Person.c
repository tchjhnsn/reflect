// Lean compiler output
// Module: NTS.Person
// Imports: public import Init public meta import Init public import NTS.Output
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
lean_object* l_Repr_addAppParen(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
lean_object* lean_nat_to_int(lean_object*);
lean_object* l_String_quote(lean_object*);
lean_object* lp_reflect_x2dnts_NTS_instReprPrime_repr(uint8_t, lean_object*);
lean_object* l_List_reverse___redArg(lean_object*);
lean_object* l_Std_Format_joinSep___at___00Lean_Syntax_formatStxAux_spec__2(lean_object*, lean_object*);
lean_object* lean_string_length(lean_object*);
uint8_t lean_string_dec_eq(lean_object*, lean_object*);
lean_object* l_Nat_reprFast(lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* l_instDecidableEqString___boxed(lean_object*, lean_object*);
uint8_t l_instDecidableEqList___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_List_repr_x27___at___00Lean_Syntax_instReprPreresolved_repr_spec__0___redArg(lean_object*);
lean_object* l_Bool_repr___redArg(uint8_t);
lean_object* lp_reflect_x2dnts_NTS_instDecidableEqPrime___boxed(lean_object*, lean_object*);
uint8_t l_List_nodupDecidable___redArg(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Store_ctorIdx(uint8_t);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Store_ctorIdx___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Store_toCtorIdx(uint8_t);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Store_toCtorIdx___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Store_ctorElim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Store_ctorElim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Store_ctorElim(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Store_ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Store_profile_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Store_profile_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Store_profile_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Store_profile_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Store_beliefs_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Store_beliefs_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Store_beliefs_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Store_beliefs_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_reflect_x2dnts_NTS_Store_ofNat(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Store_ofNat___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_reflect_x2dnts_NTS_instDecidableEqStore(uint8_t, uint8_t);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instDecidableEqStore___boxed(lean_object*, lean_object*);
static const lean_string_object lp_reflect_x2dnts_NTS_instReprStore_repr___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "NTS.Store.profile"};
static const lean_object* lp_reflect_x2dnts_NTS_instReprStore_repr___closed__0 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprStore_repr___closed__0_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_instReprStore_repr___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_reflect_x2dnts_NTS_instReprStore_repr___closed__0_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_instReprStore_repr___closed__1 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprStore_repr___closed__1_value;
static const lean_string_object lp_reflect_x2dnts_NTS_instReprStore_repr___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "NTS.Store.beliefs"};
static const lean_object* lp_reflect_x2dnts_NTS_instReprStore_repr___closed__2 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprStore_repr___closed__2_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_instReprStore_repr___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_reflect_x2dnts_NTS_instReprStore_repr___closed__2_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_instReprStore_repr___closed__3 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprStore_repr___closed__3_value;
static lean_once_cell_t lp_reflect_x2dnts_NTS_instReprStore_repr___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_reflect_x2dnts_NTS_instReprStore_repr___closed__4;
static lean_once_cell_t lp_reflect_x2dnts_NTS_instReprStore_repr___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_reflect_x2dnts_NTS_instReprStore_repr___closed__5;
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instReprStore_repr(uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instReprStore_repr___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_reflect_x2dnts_NTS_instReprStore___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_reflect_x2dnts_NTS_instReprStore_repr___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_reflect_x2dnts_NTS_instReprStore___closed__0 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprStore___closed__0_value;
LEAN_EXPORT const lean_object* lp_reflect_x2dnts_NTS_instReprStore = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprStore___closed__0_value;
LEAN_EXPORT uint8_t lp_reflect_x2dnts_NTS_instDecidableEqGrant_decEq(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instDecidableEqGrant_decEq___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_reflect_x2dnts_NTS_instDecidableEqGrant(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instDecidableEqGrant___boxed(lean_object*, lean_object*);
static const lean_string_object lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "{ "};
static const lean_object* lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__0 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__0_value;
static const lean_string_object lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "id"};
static const lean_object* lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__1 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__1_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__1_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__2 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__2_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__2_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__3 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__3_value;
static const lean_string_object lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = " := "};
static const lean_object* lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__4 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__4_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__4_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__5 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__5_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__3_value),((lean_object*)&lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__5_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__6 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__6_value;
static lean_once_cell_t lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__7;
static const lean_string_object lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ","};
static const lean_object* lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__8 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__8_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__8_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__9 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__9_value;
static const lean_string_object lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "store"};
static const lean_object* lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__10 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__10_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__10_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__11 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__11_value;
static lean_once_cell_t lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__12;
static const lean_string_object lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "entries"};
static const lean_object* lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__13 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__13_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__13_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__14 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__14_value;
static lean_once_cell_t lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__15;
static const lean_string_object lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "revoked"};
static const lean_object* lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__16 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__16_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__16_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__17 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__17_value;
static const lean_string_object lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = " }"};
static const lean_object* lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__18 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__18_value;
static lean_once_cell_t lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__19_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__19;
static lean_once_cell_t lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__20_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__20;
static const lean_ctor_object lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__0_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__21 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__21_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__18_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__22 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__22_value;
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instReprGrant_repr(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instReprGrant_repr___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_reflect_x2dnts_NTS_instReprGrant___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_reflect_x2dnts_NTS_instReprGrant_repr___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_reflect_x2dnts_NTS_instReprGrant___closed__0 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprGrant___closed__0_value;
LEAN_EXPORT const lean_object* lp_reflect_x2dnts_NTS_instReprGrant = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprGrant___closed__0_value;
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Source_ctorIdx(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Source_ctorIdx___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Source_ctorElim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Source_ctorElim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Source_ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Source_stated_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Source_stated_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Source_fromJournal_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Source_fromJournal_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_reflect_x2dnts_NTS_instDecidableEqSource_decEq(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instDecidableEqSource_decEq___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_reflect_x2dnts_NTS_instDecidableEqSource(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instDecidableEqSource___boxed(lean_object*, lean_object*);
static const lean_string_object lp_reflect_x2dnts_NTS_instReprSource_repr___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "NTS.Source.stated"};
static const lean_object* lp_reflect_x2dnts_NTS_instReprSource_repr___closed__0 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprSource_repr___closed__0_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_instReprSource_repr___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_reflect_x2dnts_NTS_instReprSource_repr___closed__0_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_instReprSource_repr___closed__1 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprSource_repr___closed__1_value;
static const lean_string_object lp_reflect_x2dnts_NTS_instReprSource_repr___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 23, .m_capacity = 23, .m_length = 22, .m_data = "NTS.Source.fromJournal"};
static const lean_object* lp_reflect_x2dnts_NTS_instReprSource_repr___closed__2 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprSource_repr___closed__2_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_instReprSource_repr___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_reflect_x2dnts_NTS_instReprSource_repr___closed__2_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_instReprSource_repr___closed__3 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprSource_repr___closed__3_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_instReprSource_repr___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_reflect_x2dnts_NTS_instReprSource_repr___closed__3_value),((lean_object*)(((size_t)(1) << 1) | 1))}};
static const lean_object* lp_reflect_x2dnts_NTS_instReprSource_repr___closed__4 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprSource_repr___closed__4_value;
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instReprSource_repr(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instReprSource_repr___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_reflect_x2dnts_NTS_instReprSource___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_reflect_x2dnts_NTS_instReprSource_repr___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_reflect_x2dnts_NTS_instReprSource___closed__0 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprSource___closed__0_value;
LEAN_EXPORT const lean_object* lp_reflect_x2dnts_NTS_instReprSource = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprSource___closed__0_value;
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Status_ctorIdx(uint8_t);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Status_ctorIdx___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Status_toCtorIdx(uint8_t);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Status_toCtorIdx___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Status_ctorElim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Status_ctorElim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Status_ctorElim(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Status_ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Status_proposed_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Status_proposed_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Status_proposed_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Status_proposed_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Status_confirmed_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Status_confirmed_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Status_confirmed_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Status_confirmed_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_reflect_x2dnts_NTS_Status_ofNat(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Status_ofNat___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_reflect_x2dnts_NTS_instDecidableEqStatus(uint8_t, uint8_t);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instDecidableEqStatus___boxed(lean_object*, lean_object*);
static const lean_string_object lp_reflect_x2dnts_NTS_instReprStatus_repr___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "NTS.Status.proposed"};
static const lean_object* lp_reflect_x2dnts_NTS_instReprStatus_repr___closed__0 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprStatus_repr___closed__0_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_instReprStatus_repr___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_reflect_x2dnts_NTS_instReprStatus_repr___closed__0_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_instReprStatus_repr___closed__1 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprStatus_repr___closed__1_value;
static const lean_string_object lp_reflect_x2dnts_NTS_instReprStatus_repr___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = "NTS.Status.confirmed"};
static const lean_object* lp_reflect_x2dnts_NTS_instReprStatus_repr___closed__2 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprStatus_repr___closed__2_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_instReprStatus_repr___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_reflect_x2dnts_NTS_instReprStatus_repr___closed__2_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_instReprStatus_repr___closed__3 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprStatus_repr___closed__3_value;
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instReprStatus_repr(uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instReprStatus_repr___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_reflect_x2dnts_NTS_instReprStatus___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_reflect_x2dnts_NTS_instReprStatus_repr___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_reflect_x2dnts_NTS_instReprStatus___closed__0 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprStatus___closed__0_value;
LEAN_EXPORT const lean_object* lp_reflect_x2dnts_NTS_instReprStatus = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprStatus___closed__0_value;
static const lean_string_object lp_reflect_x2dnts_Option_repr___at___00NTS_instReprFact_repr_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "none"};
static const lean_object* lp_reflect_x2dnts_Option_repr___at___00NTS_instReprFact_repr_spec__0___closed__0 = (const lean_object*)&lp_reflect_x2dnts_Option_repr___at___00NTS_instReprFact_repr_spec__0___closed__0_value;
static const lean_ctor_object lp_reflect_x2dnts_Option_repr___at___00NTS_instReprFact_repr_spec__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_reflect_x2dnts_Option_repr___at___00NTS_instReprFact_repr_spec__0___closed__0_value)}};
static const lean_object* lp_reflect_x2dnts_Option_repr___at___00NTS_instReprFact_repr_spec__0___closed__1 = (const lean_object*)&lp_reflect_x2dnts_Option_repr___at___00NTS_instReprFact_repr_spec__0___closed__1_value;
static const lean_string_object lp_reflect_x2dnts_Option_repr___at___00NTS_instReprFact_repr_spec__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "some "};
static const lean_object* lp_reflect_x2dnts_Option_repr___at___00NTS_instReprFact_repr_spec__0___closed__2 = (const lean_object*)&lp_reflect_x2dnts_Option_repr___at___00NTS_instReprFact_repr_spec__0___closed__2_value;
static const lean_ctor_object lp_reflect_x2dnts_Option_repr___at___00NTS_instReprFact_repr_spec__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_reflect_x2dnts_Option_repr___at___00NTS_instReprFact_repr_spec__0___closed__2_value)}};
static const lean_object* lp_reflect_x2dnts_Option_repr___at___00NTS_instReprFact_repr_spec__0___closed__3 = (const lean_object*)&lp_reflect_x2dnts_Option_repr___at___00NTS_instReprFact_repr_spec__0___closed__3_value;
LEAN_EXPORT lean_object* lp_reflect_x2dnts_Option_repr___at___00NTS_instReprFact_repr_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_Option_repr___at___00NTS_instReprFact_repr_spec__0___boxed(lean_object*, lean_object*);
static const lean_string_object lp_reflect_x2dnts_NTS_instReprFact_repr___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "key"};
static const lean_object* lp_reflect_x2dnts_NTS_instReprFact_repr___redArg___closed__0 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprFact_repr___redArg___closed__0_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_instReprFact_repr___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_reflect_x2dnts_NTS_instReprFact_repr___redArg___closed__0_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_instReprFact_repr___redArg___closed__1 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprFact_repr___redArg___closed__1_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_instReprFact_repr___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_reflect_x2dnts_NTS_instReprFact_repr___redArg___closed__1_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_instReprFact_repr___redArg___closed__2 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprFact_repr___redArg___closed__2_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_instReprFact_repr___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_reflect_x2dnts_NTS_instReprFact_repr___redArg___closed__2_value),((lean_object*)&lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__5_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_instReprFact_repr___redArg___closed__3 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprFact_repr___redArg___closed__3_value;
static lean_once_cell_t lp_reflect_x2dnts_NTS_instReprFact_repr___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_reflect_x2dnts_NTS_instReprFact_repr___redArg___closed__4;
static const lean_string_object lp_reflect_x2dnts_NTS_instReprFact_repr___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "value"};
static const lean_object* lp_reflect_x2dnts_NTS_instReprFact_repr___redArg___closed__5 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprFact_repr___redArg___closed__5_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_instReprFact_repr___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_reflect_x2dnts_NTS_instReprFact_repr___redArg___closed__5_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_instReprFact_repr___redArg___closed__6 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprFact_repr___redArg___closed__6_value;
static const lean_string_object lp_reflect_x2dnts_NTS_instReprFact_repr___redArg___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "validFrom"};
static const lean_object* lp_reflect_x2dnts_NTS_instReprFact_repr___redArg___closed__7 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprFact_repr___redArg___closed__7_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_instReprFact_repr___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_reflect_x2dnts_NTS_instReprFact_repr___redArg___closed__7_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_instReprFact_repr___redArg___closed__8 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprFact_repr___redArg___closed__8_value;
static lean_once_cell_t lp_reflect_x2dnts_NTS_instReprFact_repr___redArg___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_reflect_x2dnts_NTS_instReprFact_repr___redArg___closed__9;
static const lean_string_object lp_reflect_x2dnts_NTS_instReprFact_repr___redArg___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "validTo"};
static const lean_object* lp_reflect_x2dnts_NTS_instReprFact_repr___redArg___closed__10 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprFact_repr___redArg___closed__10_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_instReprFact_repr___redArg___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_reflect_x2dnts_NTS_instReprFact_repr___redArg___closed__10_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_instReprFact_repr___redArg___closed__11 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprFact_repr___redArg___closed__11_value;
static const lean_string_object lp_reflect_x2dnts_NTS_instReprFact_repr___redArg___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "recordedAt"};
static const lean_object* lp_reflect_x2dnts_NTS_instReprFact_repr___redArg___closed__12 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprFact_repr___redArg___closed__12_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_instReprFact_repr___redArg___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_reflect_x2dnts_NTS_instReprFact_repr___redArg___closed__12_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_instReprFact_repr___redArg___closed__13 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprFact_repr___redArg___closed__13_value;
static lean_once_cell_t lp_reflect_x2dnts_NTS_instReprFact_repr___redArg___closed__14_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_reflect_x2dnts_NTS_instReprFact_repr___redArg___closed__14;
static const lean_string_object lp_reflect_x2dnts_NTS_instReprFact_repr___redArg___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "source"};
static const lean_object* lp_reflect_x2dnts_NTS_instReprFact_repr___redArg___closed__15 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprFact_repr___redArg___closed__15_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_instReprFact_repr___redArg___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_reflect_x2dnts_NTS_instReprFact_repr___redArg___closed__15_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_instReprFact_repr___redArg___closed__16 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprFact_repr___redArg___closed__16_value;
static lean_once_cell_t lp_reflect_x2dnts_NTS_instReprFact_repr___redArg___closed__17_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_reflect_x2dnts_NTS_instReprFact_repr___redArg___closed__17;
static const lean_string_object lp_reflect_x2dnts_NTS_instReprFact_repr___redArg___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "status"};
static const lean_object* lp_reflect_x2dnts_NTS_instReprFact_repr___redArg___closed__18 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprFact_repr___redArg___closed__18_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_instReprFact_repr___redArg___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_reflect_x2dnts_NTS_instReprFact_repr___redArg___closed__18_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_instReprFact_repr___redArg___closed__19 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprFact_repr___redArg___closed__19_value;
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instReprFact_repr___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instReprFact_repr(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instReprFact_repr___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_reflect_x2dnts_NTS_instReprFact___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_reflect_x2dnts_NTS_instReprFact_repr___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_reflect_x2dnts_NTS_instReprFact___closed__0 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprFact___closed__0_value;
LEAN_EXPORT const lean_object* lp_reflect_x2dnts_NTS_instReprFact = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprFact___closed__0_value;
static const lean_string_object lp_reflect_x2dnts_NTS_instReprBelief_repr___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "text"};
static const lean_object* lp_reflect_x2dnts_NTS_instReprBelief_repr___redArg___closed__0 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprBelief_repr___redArg___closed__0_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_instReprBelief_repr___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_reflect_x2dnts_NTS_instReprBelief_repr___redArg___closed__0_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_instReprBelief_repr___redArg___closed__1 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprBelief_repr___redArg___closed__1_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_instReprBelief_repr___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_reflect_x2dnts_NTS_instReprBelief_repr___redArg___closed__1_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_instReprBelief_repr___redArg___closed__2 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprBelief_repr___redArg___closed__2_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_instReprBelief_repr___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_reflect_x2dnts_NTS_instReprBelief_repr___redArg___closed__2_value),((lean_object*)&lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__5_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_instReprBelief_repr___redArg___closed__3 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprBelief_repr___redArg___closed__3_value;
static lean_once_cell_t lp_reflect_x2dnts_NTS_instReprBelief_repr___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_reflect_x2dnts_NTS_instReprBelief_repr___redArg___closed__4;
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instReprBelief_repr___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instReprBelief_repr(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instReprBelief_repr___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_reflect_x2dnts_NTS_instReprBelief___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_reflect_x2dnts_NTS_instReprBelief_repr___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_reflect_x2dnts_NTS_instReprBelief___closed__0 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprBelief___closed__0_value;
LEAN_EXPORT const lean_object* lp_reflect_x2dnts_NTS_instReprBelief = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprBelief___closed__0_value;
LEAN_EXPORT uint8_t lp_reflect_x2dnts_List_elem___at___00NTS_Source_allowed_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_List_elem___at___00NTS_Source_allowed_spec__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_reflect_x2dnts_List_any___at___00NTS_Source_allowed_spec__1(lean_object*, lean_object*, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_List_any___at___00NTS_Source_allowed_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_reflect_x2dnts_NTS_Source_allowed(lean_object*, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Source_allowed___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts___private_NTS_Person_0__NTS_instReprSource_repr_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts___private_NTS_Person_0__NTS_instReprSource_repr_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Whose_ctorIdx(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Whose_ctorIdx___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Whose_ctorElim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Whose_ctorElim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Whose_ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Whose_self_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Whose_self_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Whose_other_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Whose_other_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_reflect_x2dnts_NTS_instDecidableEqWhose_decEq(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instDecidableEqWhose_decEq___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_reflect_x2dnts_NTS_instDecidableEqWhose(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instDecidableEqWhose___boxed(lean_object*, lean_object*);
static const lean_string_object lp_reflect_x2dnts_NTS_instReprWhose_repr___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "NTS.Whose.self"};
static const lean_object* lp_reflect_x2dnts_NTS_instReprWhose_repr___closed__0 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprWhose_repr___closed__0_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_instReprWhose_repr___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_reflect_x2dnts_NTS_instReprWhose_repr___closed__0_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_instReprWhose_repr___closed__1 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprWhose_repr___closed__1_value;
static const lean_string_object lp_reflect_x2dnts_NTS_instReprWhose_repr___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "NTS.Whose.other"};
static const lean_object* lp_reflect_x2dnts_NTS_instReprWhose_repr___closed__2 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprWhose_repr___closed__2_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_instReprWhose_repr___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_reflect_x2dnts_NTS_instReprWhose_repr___closed__2_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_instReprWhose_repr___closed__3 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprWhose_repr___closed__3_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_instReprWhose_repr___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_reflect_x2dnts_NTS_instReprWhose_repr___closed__3_value),((lean_object*)(((size_t)(1) << 1) | 1))}};
static const lean_object* lp_reflect_x2dnts_NTS_instReprWhose_repr___closed__4 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprWhose_repr___closed__4_value;
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instReprWhose_repr(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instReprWhose_repr___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_reflect_x2dnts_NTS_instReprWhose___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_reflect_x2dnts_NTS_instReprWhose_repr___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_reflect_x2dnts_NTS_instReprWhose___closed__0 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprWhose___closed__0_value;
LEAN_EXPORT const lean_object* lp_reflect_x2dnts_NTS_instReprWhose = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprWhose___closed__0_value;
static const lean_string_object lp_reflect_x2dnts_Prod_repr___at___00List_repr___at___00NTS_instReprAct_repr_spec__0_spec__0___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "("};
static const lean_object* lp_reflect_x2dnts_Prod_repr___at___00List_repr___at___00NTS_instReprAct_repr_spec__0_spec__0___redArg___closed__0 = (const lean_object*)&lp_reflect_x2dnts_Prod_repr___at___00List_repr___at___00NTS_instReprAct_repr_spec__0_spec__0___redArg___closed__0_value;
static const lean_ctor_object lp_reflect_x2dnts_Prod_repr___at___00List_repr___at___00NTS_instReprAct_repr_spec__0_spec__0___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__9_value),((lean_object*)(((size_t)(1) << 1) | 1))}};
static const lean_object* lp_reflect_x2dnts_Prod_repr___at___00List_repr___at___00NTS_instReprAct_repr_spec__0_spec__0___redArg___closed__1 = (const lean_object*)&lp_reflect_x2dnts_Prod_repr___at___00List_repr___at___00NTS_instReprAct_repr_spec__0_spec__0___redArg___closed__1_value;
static const lean_string_object lp_reflect_x2dnts_Prod_repr___at___00List_repr___at___00NTS_instReprAct_repr_spec__0_spec__0___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ")"};
static const lean_object* lp_reflect_x2dnts_Prod_repr___at___00List_repr___at___00NTS_instReprAct_repr_spec__0_spec__0___redArg___closed__2 = (const lean_object*)&lp_reflect_x2dnts_Prod_repr___at___00List_repr___at___00NTS_instReprAct_repr_spec__0_spec__0___redArg___closed__2_value;
static lean_once_cell_t lp_reflect_x2dnts_Prod_repr___at___00List_repr___at___00NTS_instReprAct_repr_spec__0_spec__0___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_reflect_x2dnts_Prod_repr___at___00List_repr___at___00NTS_instReprAct_repr_spec__0_spec__0___redArg___closed__3;
static lean_once_cell_t lp_reflect_x2dnts_Prod_repr___at___00List_repr___at___00NTS_instReprAct_repr_spec__0_spec__0___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_reflect_x2dnts_Prod_repr___at___00List_repr___at___00NTS_instReprAct_repr_spec__0_spec__0___redArg___closed__4;
static const lean_ctor_object lp_reflect_x2dnts_Prod_repr___at___00List_repr___at___00NTS_instReprAct_repr_spec__0_spec__0___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_reflect_x2dnts_Prod_repr___at___00List_repr___at___00NTS_instReprAct_repr_spec__0_spec__0___redArg___closed__0_value)}};
static const lean_object* lp_reflect_x2dnts_Prod_repr___at___00List_repr___at___00NTS_instReprAct_repr_spec__0_spec__0___redArg___closed__5 = (const lean_object*)&lp_reflect_x2dnts_Prod_repr___at___00List_repr___at___00NTS_instReprAct_repr_spec__0_spec__0___redArg___closed__5_value;
static const lean_ctor_object lp_reflect_x2dnts_Prod_repr___at___00List_repr___at___00NTS_instReprAct_repr_spec__0_spec__0___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_reflect_x2dnts_Prod_repr___at___00List_repr___at___00NTS_instReprAct_repr_spec__0_spec__0___redArg___closed__2_value)}};
static const lean_object* lp_reflect_x2dnts_Prod_repr___at___00List_repr___at___00NTS_instReprAct_repr_spec__0_spec__0___redArg___closed__6 = (const lean_object*)&lp_reflect_x2dnts_Prod_repr___at___00List_repr___at___00NTS_instReprAct_repr_spec__0_spec__0___redArg___closed__6_value;
LEAN_EXPORT lean_object* lp_reflect_x2dnts_Prod_repr___at___00List_repr___at___00NTS_instReprAct_repr_spec__0_spec__0___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_List_foldl___at___00List_foldl___at___00Std_Format_joinSep___at___00List_repr___at___00NTS_instReprAct_repr_spec__0_spec__1_spec__2_spec__4(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_List_foldl___at___00Std_Format_joinSep___at___00List_repr___at___00NTS_instReprAct_repr_spec__0_spec__1_spec__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_Std_Format_joinSep___at___00List_repr___at___00NTS_instReprAct_repr_spec__0_spec__1(lean_object*, lean_object*);
static const lean_string_object lp_reflect_x2dnts_List_repr___at___00NTS_instReprAct_repr_spec__0___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "[]"};
static const lean_object* lp_reflect_x2dnts_List_repr___at___00NTS_instReprAct_repr_spec__0___redArg___closed__0 = (const lean_object*)&lp_reflect_x2dnts_List_repr___at___00NTS_instReprAct_repr_spec__0___redArg___closed__0_value;
static const lean_ctor_object lp_reflect_x2dnts_List_repr___at___00NTS_instReprAct_repr_spec__0___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_reflect_x2dnts_List_repr___at___00NTS_instReprAct_repr_spec__0___redArg___closed__0_value)}};
static const lean_object* lp_reflect_x2dnts_List_repr___at___00NTS_instReprAct_repr_spec__0___redArg___closed__1 = (const lean_object*)&lp_reflect_x2dnts_List_repr___at___00NTS_instReprAct_repr_spec__0___redArg___closed__1_value;
static const lean_string_object lp_reflect_x2dnts_List_repr___at___00NTS_instReprAct_repr_spec__0___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "["};
static const lean_object* lp_reflect_x2dnts_List_repr___at___00NTS_instReprAct_repr_spec__0___redArg___closed__2 = (const lean_object*)&lp_reflect_x2dnts_List_repr___at___00NTS_instReprAct_repr_spec__0___redArg___closed__2_value;
static const lean_string_object lp_reflect_x2dnts_List_repr___at___00NTS_instReprAct_repr_spec__0___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "]"};
static const lean_object* lp_reflect_x2dnts_List_repr___at___00NTS_instReprAct_repr_spec__0___redArg___closed__3 = (const lean_object*)&lp_reflect_x2dnts_List_repr___at___00NTS_instReprAct_repr_spec__0___redArg___closed__3_value;
static lean_once_cell_t lp_reflect_x2dnts_List_repr___at___00NTS_instReprAct_repr_spec__0___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_reflect_x2dnts_List_repr___at___00NTS_instReprAct_repr_spec__0___redArg___closed__4;
static lean_once_cell_t lp_reflect_x2dnts_List_repr___at___00NTS_instReprAct_repr_spec__0___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_reflect_x2dnts_List_repr___at___00NTS_instReprAct_repr_spec__0___redArg___closed__5;
static const lean_ctor_object lp_reflect_x2dnts_List_repr___at___00NTS_instReprAct_repr_spec__0___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_reflect_x2dnts_List_repr___at___00NTS_instReprAct_repr_spec__0___redArg___closed__2_value)}};
static const lean_object* lp_reflect_x2dnts_List_repr___at___00NTS_instReprAct_repr_spec__0___redArg___closed__6 = (const lean_object*)&lp_reflect_x2dnts_List_repr___at___00NTS_instReprAct_repr_spec__0___redArg___closed__6_value;
static const lean_ctor_object lp_reflect_x2dnts_List_repr___at___00NTS_instReprAct_repr_spec__0___redArg___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_reflect_x2dnts_List_repr___at___00NTS_instReprAct_repr_spec__0___redArg___closed__3_value)}};
static const lean_object* lp_reflect_x2dnts_List_repr___at___00NTS_instReprAct_repr_spec__0___redArg___closed__7 = (const lean_object*)&lp_reflect_x2dnts_List_repr___at___00NTS_instReprAct_repr_spec__0___redArg___closed__7_value;
LEAN_EXPORT lean_object* lp_reflect_x2dnts_List_repr___at___00NTS_instReprAct_repr_spec__0___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_Prod_repr___at___00Option_repr___at___00NTS_instReprAct_repr_spec__1_spec__3___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_Option_repr___at___00NTS_instReprAct_repr_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_Option_repr___at___00NTS_instReprAct_repr_spec__1___boxed(lean_object*, lean_object*);
static const lean_string_object lp_reflect_x2dnts_NTS_instReprAct_repr___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "drivers"};
static const lean_object* lp_reflect_x2dnts_NTS_instReprAct_repr___redArg___closed__0 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprAct_repr___redArg___closed__0_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_instReprAct_repr___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_reflect_x2dnts_NTS_instReprAct_repr___redArg___closed__0_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_instReprAct_repr___redArg___closed__1 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprAct_repr___redArg___closed__1_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_instReprAct_repr___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_reflect_x2dnts_NTS_instReprAct_repr___redArg___closed__1_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_instReprAct_repr___redArg___closed__2 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprAct_repr___redArg___closed__2_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_instReprAct_repr___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_reflect_x2dnts_NTS_instReprAct_repr___redArg___closed__2_value),((lean_object*)&lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__5_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_instReprAct_repr___redArg___closed__3 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprAct_repr___redArg___closed__3_value;
static const lean_string_object lp_reflect_x2dnts_NTS_instReprAct_repr___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "target"};
static const lean_object* lp_reflect_x2dnts_NTS_instReprAct_repr___redArg___closed__4 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprAct_repr___redArg___closed__4_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_instReprAct_repr___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_reflect_x2dnts_NTS_instReprAct_repr___redArg___closed__4_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_instReprAct_repr___redArg___closed__5 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprAct_repr___redArg___closed__5_value;
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instReprAct_repr___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instReprAct_repr(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instReprAct_repr___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_List_repr___at___00NTS_instReprAct_repr_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_List_repr___at___00NTS_instReprAct_repr_spec__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_Prod_repr___at___00List_repr___at___00NTS_instReprAct_repr_spec__0_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_Prod_repr___at___00List_repr___at___00NTS_instReprAct_repr_spec__0_spec__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_Prod_repr___at___00Option_repr___at___00NTS_instReprAct_repr_spec__1_spec__3(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_Prod_repr___at___00Option_repr___at___00NTS_instReprAct_repr_spec__1_spec__3___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_reflect_x2dnts_NTS_instReprAct___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_reflect_x2dnts_NTS_instReprAct_repr___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_reflect_x2dnts_NTS_instReprAct___closed__0 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprAct___closed__0_value;
LEAN_EXPORT const lean_object* lp_reflect_x2dnts_NTS_instReprAct = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprAct___closed__0_value;
LEAN_EXPORT lean_object* lp_reflect_x2dnts_List_mapTR_loop___at___00NTS_Act_wellFormed_spec__0(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_reflect_x2dnts_List_all___at___00NTS_Act_wellFormed_spec__1(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_List_all___at___00NTS_Act_wellFormed_spec__1___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_reflect_x2dnts_NTS_Act_wellFormed(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Act_wellFormed___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_RelationKind_toCtorIdx(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_RelationKind_ofNat(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_RelationKind_ofNat___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_reflect_x2dnts_NTS_instDecidableEqRelationKind(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instDecidableEqRelationKind___boxed(lean_object*, lean_object*);
static const lean_string_object lp_reflect_x2dnts_NTS_instReprRelationKind_repr___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 31, .m_capacity = 31, .m_length = 30, .m_data = "NTS.RelationKind.closestFriend"};
static const lean_object* lp_reflect_x2dnts_NTS_instReprRelationKind_repr___redArg___closed__0 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprRelationKind_repr___redArg___closed__0_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_instReprRelationKind_repr___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_reflect_x2dnts_NTS_instReprRelationKind_repr___redArg___closed__0_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_instReprRelationKind_repr___redArg___closed__1 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprRelationKind_repr___redArg___closed__1_value;
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instReprRelationKind_repr___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instReprRelationKind_repr___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instReprRelationKind_repr(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instReprRelationKind_repr___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_reflect_x2dnts_NTS_instReprRelationKind___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_reflect_x2dnts_NTS_instReprRelationKind_repr___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_reflect_x2dnts_NTS_instReprRelationKind___closed__0 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprRelationKind___closed__0_value;
LEAN_EXPORT const lean_object* lp_reflect_x2dnts_NTS_instReprRelationKind = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprRelationKind___closed__0_value;
static const lean_string_object lp_reflect_x2dnts_NTS_instReprRelationship_repr___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "name"};
static const lean_object* lp_reflect_x2dnts_NTS_instReprRelationship_repr___redArg___closed__0 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprRelationship_repr___redArg___closed__0_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_instReprRelationship_repr___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_reflect_x2dnts_NTS_instReprRelationship_repr___redArg___closed__0_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_instReprRelationship_repr___redArg___closed__1 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprRelationship_repr___redArg___closed__1_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_instReprRelationship_repr___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_reflect_x2dnts_NTS_instReprRelationship_repr___redArg___closed__1_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_instReprRelationship_repr___redArg___closed__2 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprRelationship_repr___redArg___closed__2_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_instReprRelationship_repr___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_reflect_x2dnts_NTS_instReprRelationship_repr___redArg___closed__2_value),((lean_object*)&lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__5_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_instReprRelationship_repr___redArg___closed__3 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprRelationship_repr___redArg___closed__3_value;
static const lean_string_object lp_reflect_x2dnts_NTS_instReprRelationship_repr___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "kind"};
static const lean_object* lp_reflect_x2dnts_NTS_instReprRelationship_repr___redArg___closed__4 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprRelationship_repr___redArg___closed__4_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_instReprRelationship_repr___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_reflect_x2dnts_NTS_instReprRelationship_repr___redArg___closed__4_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_instReprRelationship_repr___redArg___closed__5 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprRelationship_repr___redArg___closed__5_value;
static lean_once_cell_t lp_reflect_x2dnts_NTS_instReprRelationship_repr___redArg___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_reflect_x2dnts_NTS_instReprRelationship_repr___redArg___closed__6;
static lean_once_cell_t lp_reflect_x2dnts_NTS_instReprRelationship_repr___redArg___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_reflect_x2dnts_NTS_instReprRelationship_repr___redArg___closed__7;
static lean_once_cell_t lp_reflect_x2dnts_NTS_instReprRelationship_repr___redArg___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_reflect_x2dnts_NTS_instReprRelationship_repr___redArg___closed__8;
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instReprRelationship_repr___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instReprRelationship_repr(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instReprRelationship_repr___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_reflect_x2dnts_NTS_instReprRelationship___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_reflect_x2dnts_NTS_instReprRelationship_repr___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_reflect_x2dnts_NTS_instReprRelationship___closed__0 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprRelationship___closed__0_value;
LEAN_EXPORT const lean_object* lp_reflect_x2dnts_NTS_instReprRelationship = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprRelationship___closed__0_value;
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Store_ctorIdx(uint8_t v_x_1_){
_start:
{
if (v_x_1_ == 0)
{
lean_object* v___x_2_; 
v___x_2_ = lean_unsigned_to_nat(0u);
return v___x_2_;
}
else
{
lean_object* v___x_3_; 
v___x_3_ = lean_unsigned_to_nat(1u);
return v___x_3_;
}
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Store_ctorIdx___boxed(lean_object* v_x_4_){
_start:
{
uint8_t v_x_boxed_5_; lean_object* v_res_6_; 
v_x_boxed_5_ = lean_unbox(v_x_4_);
v_res_6_ = lp_reflect_x2dnts_NTS_Store_ctorIdx(v_x_boxed_5_);
return v_res_6_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Store_toCtorIdx(uint8_t v_x_7_){
_start:
{
lean_object* v___x_8_; 
v___x_8_ = lp_reflect_x2dnts_NTS_Store_ctorIdx(v_x_7_);
return v___x_8_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Store_toCtorIdx___boxed(lean_object* v_x_9_){
_start:
{
uint8_t v_x_4__boxed_10_; lean_object* v_res_11_; 
v_x_4__boxed_10_ = lean_unbox(v_x_9_);
v_res_11_ = lp_reflect_x2dnts_NTS_Store_toCtorIdx(v_x_4__boxed_10_);
return v_res_11_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Store_ctorElim___redArg(lean_object* v_k_12_){
_start:
{
lean_inc(v_k_12_);
return v_k_12_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Store_ctorElim___redArg___boxed(lean_object* v_k_13_){
_start:
{
lean_object* v_res_14_; 
v_res_14_ = lp_reflect_x2dnts_NTS_Store_ctorElim___redArg(v_k_13_);
lean_dec(v_k_13_);
return v_res_14_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Store_ctorElim(lean_object* v_motive_15_, lean_object* v_ctorIdx_16_, uint8_t v_t_17_, lean_object* v_h_18_, lean_object* v_k_19_){
_start:
{
lean_inc(v_k_19_);
return v_k_19_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Store_ctorElim___boxed(lean_object* v_motive_20_, lean_object* v_ctorIdx_21_, lean_object* v_t_22_, lean_object* v_h_23_, lean_object* v_k_24_){
_start:
{
uint8_t v_t_boxed_25_; lean_object* v_res_26_; 
v_t_boxed_25_ = lean_unbox(v_t_22_);
v_res_26_ = lp_reflect_x2dnts_NTS_Store_ctorElim(v_motive_20_, v_ctorIdx_21_, v_t_boxed_25_, v_h_23_, v_k_24_);
lean_dec(v_k_24_);
lean_dec(v_ctorIdx_21_);
return v_res_26_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Store_profile_elim___redArg(lean_object* v_profile_27_){
_start:
{
lean_inc(v_profile_27_);
return v_profile_27_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Store_profile_elim___redArg___boxed(lean_object* v_profile_28_){
_start:
{
lean_object* v_res_29_; 
v_res_29_ = lp_reflect_x2dnts_NTS_Store_profile_elim___redArg(v_profile_28_);
lean_dec(v_profile_28_);
return v_res_29_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Store_profile_elim(lean_object* v_motive_30_, uint8_t v_t_31_, lean_object* v_h_32_, lean_object* v_profile_33_){
_start:
{
lean_inc(v_profile_33_);
return v_profile_33_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Store_profile_elim___boxed(lean_object* v_motive_34_, lean_object* v_t_35_, lean_object* v_h_36_, lean_object* v_profile_37_){
_start:
{
uint8_t v_t_boxed_38_; lean_object* v_res_39_; 
v_t_boxed_38_ = lean_unbox(v_t_35_);
v_res_39_ = lp_reflect_x2dnts_NTS_Store_profile_elim(v_motive_34_, v_t_boxed_38_, v_h_36_, v_profile_37_);
lean_dec(v_profile_37_);
return v_res_39_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Store_beliefs_elim___redArg(lean_object* v_beliefs_40_){
_start:
{
lean_inc(v_beliefs_40_);
return v_beliefs_40_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Store_beliefs_elim___redArg___boxed(lean_object* v_beliefs_41_){
_start:
{
lean_object* v_res_42_; 
v_res_42_ = lp_reflect_x2dnts_NTS_Store_beliefs_elim___redArg(v_beliefs_41_);
lean_dec(v_beliefs_41_);
return v_res_42_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Store_beliefs_elim(lean_object* v_motive_43_, uint8_t v_t_44_, lean_object* v_h_45_, lean_object* v_beliefs_46_){
_start:
{
lean_inc(v_beliefs_46_);
return v_beliefs_46_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Store_beliefs_elim___boxed(lean_object* v_motive_47_, lean_object* v_t_48_, lean_object* v_h_49_, lean_object* v_beliefs_50_){
_start:
{
uint8_t v_t_boxed_51_; lean_object* v_res_52_; 
v_t_boxed_51_ = lean_unbox(v_t_48_);
v_res_52_ = lp_reflect_x2dnts_NTS_Store_beliefs_elim(v_motive_47_, v_t_boxed_51_, v_h_49_, v_beliefs_50_);
lean_dec(v_beliefs_50_);
return v_res_52_;
}
}
LEAN_EXPORT uint8_t lp_reflect_x2dnts_NTS_Store_ofNat(lean_object* v_n_53_){
_start:
{
lean_object* v___x_54_; uint8_t v___x_55_; 
v___x_54_ = lean_unsigned_to_nat(0u);
v___x_55_ = lean_nat_dec_le(v_n_53_, v___x_54_);
if (v___x_55_ == 0)
{
uint8_t v___x_56_; 
v___x_56_ = 1;
return v___x_56_;
}
else
{
uint8_t v___x_57_; 
v___x_57_ = 0;
return v___x_57_;
}
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Store_ofNat___boxed(lean_object* v_n_58_){
_start:
{
uint8_t v_res_59_; lean_object* v_r_60_; 
v_res_59_ = lp_reflect_x2dnts_NTS_Store_ofNat(v_n_58_);
lean_dec(v_n_58_);
v_r_60_ = lean_box(v_res_59_);
return v_r_60_;
}
}
LEAN_EXPORT uint8_t lp_reflect_x2dnts_NTS_instDecidableEqStore(uint8_t v_x_61_, uint8_t v_y_62_){
_start:
{
lean_object* v___x_63_; lean_object* v___x_64_; uint8_t v___x_65_; 
v___x_63_ = lp_reflect_x2dnts_NTS_Store_ctorIdx(v_x_61_);
v___x_64_ = lp_reflect_x2dnts_NTS_Store_ctorIdx(v_y_62_);
v___x_65_ = lean_nat_dec_eq(v___x_63_, v___x_64_);
lean_dec(v___x_64_);
lean_dec(v___x_63_);
return v___x_65_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instDecidableEqStore___boxed(lean_object* v_x_66_, lean_object* v_y_67_){
_start:
{
uint8_t v_x_13__boxed_68_; uint8_t v_y_14__boxed_69_; uint8_t v_res_70_; lean_object* v_r_71_; 
v_x_13__boxed_68_ = lean_unbox(v_x_66_);
v_y_14__boxed_69_ = lean_unbox(v_y_67_);
v_res_70_ = lp_reflect_x2dnts_NTS_instDecidableEqStore(v_x_13__boxed_68_, v_y_14__boxed_69_);
v_r_71_ = lean_box(v_res_70_);
return v_r_71_;
}
}
static lean_object* _init_lp_reflect_x2dnts_NTS_instReprStore_repr___closed__4(void){
_start:
{
lean_object* v___x_78_; lean_object* v___x_79_; 
v___x_78_ = lean_unsigned_to_nat(2u);
v___x_79_ = lean_nat_to_int(v___x_78_);
return v___x_79_;
}
}
static lean_object* _init_lp_reflect_x2dnts_NTS_instReprStore_repr___closed__5(void){
_start:
{
lean_object* v___x_80_; lean_object* v___x_81_; 
v___x_80_ = lean_unsigned_to_nat(1u);
v___x_81_ = lean_nat_to_int(v___x_80_);
return v___x_81_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instReprStore_repr(uint8_t v_x_82_, lean_object* v_prec_83_){
_start:
{
lean_object* v___y_85_; lean_object* v___y_92_; 
if (v_x_82_ == 0)
{
lean_object* v___x_98_; uint8_t v___x_99_; 
v___x_98_ = lean_unsigned_to_nat(1024u);
v___x_99_ = lean_nat_dec_le(v___x_98_, v_prec_83_);
if (v___x_99_ == 0)
{
lean_object* v___x_100_; 
v___x_100_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprStore_repr___closed__4, &lp_reflect_x2dnts_NTS_instReprStore_repr___closed__4_once, _init_lp_reflect_x2dnts_NTS_instReprStore_repr___closed__4);
v___y_85_ = v___x_100_;
goto v___jp_84_;
}
else
{
lean_object* v___x_101_; 
v___x_101_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprStore_repr___closed__5, &lp_reflect_x2dnts_NTS_instReprStore_repr___closed__5_once, _init_lp_reflect_x2dnts_NTS_instReprStore_repr___closed__5);
v___y_85_ = v___x_101_;
goto v___jp_84_;
}
}
else
{
lean_object* v___x_102_; uint8_t v___x_103_; 
v___x_102_ = lean_unsigned_to_nat(1024u);
v___x_103_ = lean_nat_dec_le(v___x_102_, v_prec_83_);
if (v___x_103_ == 0)
{
lean_object* v___x_104_; 
v___x_104_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprStore_repr___closed__4, &lp_reflect_x2dnts_NTS_instReprStore_repr___closed__4_once, _init_lp_reflect_x2dnts_NTS_instReprStore_repr___closed__4);
v___y_92_ = v___x_104_;
goto v___jp_91_;
}
else
{
lean_object* v___x_105_; 
v___x_105_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprStore_repr___closed__5, &lp_reflect_x2dnts_NTS_instReprStore_repr___closed__5_once, _init_lp_reflect_x2dnts_NTS_instReprStore_repr___closed__5);
v___y_92_ = v___x_105_;
goto v___jp_91_;
}
}
v___jp_84_:
{
lean_object* v___x_86_; lean_object* v___x_87_; uint8_t v___x_88_; lean_object* v___x_89_; lean_object* v___x_90_; 
v___x_86_ = ((lean_object*)(lp_reflect_x2dnts_NTS_instReprStore_repr___closed__1));
lean_inc(v___y_85_);
v___x_87_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_87_, 0, v___y_85_);
lean_ctor_set(v___x_87_, 1, v___x_86_);
v___x_88_ = 0;
v___x_89_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_89_, 0, v___x_87_);
lean_ctor_set_uint8(v___x_89_, sizeof(void*)*1, v___x_88_);
v___x_90_ = l_Repr_addAppParen(v___x_89_, v_prec_83_);
return v___x_90_;
}
v___jp_91_:
{
lean_object* v___x_93_; lean_object* v___x_94_; uint8_t v___x_95_; lean_object* v___x_96_; lean_object* v___x_97_; 
v___x_93_ = ((lean_object*)(lp_reflect_x2dnts_NTS_instReprStore_repr___closed__3));
lean_inc(v___y_92_);
v___x_94_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_94_, 0, v___y_92_);
lean_ctor_set(v___x_94_, 1, v___x_93_);
v___x_95_ = 0;
v___x_96_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_96_, 0, v___x_94_);
lean_ctor_set_uint8(v___x_96_, sizeof(void*)*1, v___x_95_);
v___x_97_ = l_Repr_addAppParen(v___x_96_, v_prec_83_);
return v___x_97_;
}
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instReprStore_repr___boxed(lean_object* v_x_106_, lean_object* v_prec_107_){
_start:
{
uint8_t v_x_121__boxed_108_; lean_object* v_res_109_; 
v_x_121__boxed_108_ = lean_unbox(v_x_106_);
v_res_109_ = lp_reflect_x2dnts_NTS_instReprStore_repr(v_x_121__boxed_108_, v_prec_107_);
lean_dec(v_prec_107_);
return v_res_109_;
}
}
LEAN_EXPORT uint8_t lp_reflect_x2dnts_NTS_instDecidableEqGrant_decEq(lean_object* v_x_112_, lean_object* v_x_113_){
_start:
{
lean_object* v_id_114_; uint8_t v_store_115_; lean_object* v_entries_116_; uint8_t v_revoked_117_; lean_object* v_id_118_; uint8_t v_store_119_; lean_object* v_entries_120_; uint8_t v_revoked_121_; uint8_t v___x_122_; 
v_id_114_ = lean_ctor_get(v_x_112_, 0);
lean_inc_ref(v_id_114_);
v_store_115_ = lean_ctor_get_uint8(v_x_112_, sizeof(void*)*2);
v_entries_116_ = lean_ctor_get(v_x_112_, 1);
lean_inc(v_entries_116_);
v_revoked_117_ = lean_ctor_get_uint8(v_x_112_, sizeof(void*)*2 + 1);
lean_dec_ref(v_x_112_);
v_id_118_ = lean_ctor_get(v_x_113_, 0);
lean_inc_ref(v_id_118_);
v_store_119_ = lean_ctor_get_uint8(v_x_113_, sizeof(void*)*2);
v_entries_120_ = lean_ctor_get(v_x_113_, 1);
lean_inc(v_entries_120_);
v_revoked_121_ = lean_ctor_get_uint8(v_x_113_, sizeof(void*)*2 + 1);
lean_dec_ref(v_x_113_);
v___x_122_ = lean_string_dec_eq(v_id_114_, v_id_118_);
lean_dec_ref(v_id_118_);
lean_dec_ref(v_id_114_);
if (v___x_122_ == 0)
{
lean_dec(v_entries_120_);
lean_dec(v_entries_116_);
return v___x_122_;
}
else
{
uint8_t v___x_123_; 
v___x_123_ = lp_reflect_x2dnts_NTS_instDecidableEqStore(v_store_115_, v_store_119_);
if (v___x_123_ == 0)
{
lean_dec(v_entries_120_);
lean_dec(v_entries_116_);
return v___x_123_;
}
else
{
lean_object* v___x_124_; uint8_t v___x_125_; 
v___x_124_ = lean_alloc_closure((void*)(l_instDecidableEqString___boxed), 2, 0);
v___x_125_ = l_instDecidableEqList___redArg(v___x_124_, v_entries_116_, v_entries_120_);
if (v___x_125_ == 0)
{
return v___x_125_;
}
else
{
if (v_revoked_117_ == 0)
{
if (v_revoked_121_ == 0)
{
return v___x_125_;
}
else
{
return v_revoked_117_;
}
}
else
{
return v_revoked_121_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instDecidableEqGrant_decEq___boxed(lean_object* v_x_126_, lean_object* v_x_127_){
_start:
{
uint8_t v_res_128_; lean_object* v_r_129_; 
v_res_128_ = lp_reflect_x2dnts_NTS_instDecidableEqGrant_decEq(v_x_126_, v_x_127_);
v_r_129_ = lean_box(v_res_128_);
return v_r_129_;
}
}
LEAN_EXPORT uint8_t lp_reflect_x2dnts_NTS_instDecidableEqGrant(lean_object* v_x_130_, lean_object* v_x_131_){
_start:
{
uint8_t v___x_132_; 
v___x_132_ = lp_reflect_x2dnts_NTS_instDecidableEqGrant_decEq(v_x_130_, v_x_131_);
return v___x_132_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instDecidableEqGrant___boxed(lean_object* v_x_133_, lean_object* v_x_134_){
_start:
{
uint8_t v_res_135_; lean_object* v_r_136_; 
v_res_135_ = lp_reflect_x2dnts_NTS_instDecidableEqGrant(v_x_133_, v_x_134_);
v_r_136_ = lean_box(v_res_135_);
return v_r_136_;
}
}
static lean_object* _init_lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__7(void){
_start:
{
lean_object* v___x_150_; lean_object* v___x_151_; 
v___x_150_ = lean_unsigned_to_nat(6u);
v___x_151_ = lean_nat_to_int(v___x_150_);
return v___x_151_;
}
}
static lean_object* _init_lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__12(void){
_start:
{
lean_object* v___x_158_; lean_object* v___x_159_; 
v___x_158_ = lean_unsigned_to_nat(9u);
v___x_159_ = lean_nat_to_int(v___x_158_);
return v___x_159_;
}
}
static lean_object* _init_lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__15(void){
_start:
{
lean_object* v___x_163_; lean_object* v___x_164_; 
v___x_163_ = lean_unsigned_to_nat(11u);
v___x_164_ = lean_nat_to_int(v___x_163_);
return v___x_164_;
}
}
static lean_object* _init_lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__19(void){
_start:
{
lean_object* v___x_169_; lean_object* v___x_170_; 
v___x_169_ = ((lean_object*)(lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__0));
v___x_170_ = lean_string_length(v___x_169_);
return v___x_170_;
}
}
static lean_object* _init_lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__20(void){
_start:
{
lean_object* v___x_171_; lean_object* v___x_172_; 
v___x_171_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__19, &lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__19_once, _init_lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__19);
v___x_172_ = lean_nat_to_int(v___x_171_);
return v___x_172_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg(lean_object* v_x_177_){
_start:
{
lean_object* v_id_178_; uint8_t v_store_179_; lean_object* v_entries_180_; uint8_t v_revoked_181_; lean_object* v___x_182_; lean_object* v___x_183_; lean_object* v___x_184_; lean_object* v___x_185_; lean_object* v___x_186_; lean_object* v___x_187_; uint8_t v___x_188_; lean_object* v___x_189_; lean_object* v___x_190_; lean_object* v___x_191_; lean_object* v___x_192_; lean_object* v___x_193_; lean_object* v___x_194_; lean_object* v___x_195_; lean_object* v___x_196_; lean_object* v___x_197_; lean_object* v___x_198_; lean_object* v___x_199_; lean_object* v___x_200_; lean_object* v___x_201_; lean_object* v___x_202_; lean_object* v___x_203_; lean_object* v___x_204_; lean_object* v___x_205_; lean_object* v___x_206_; lean_object* v___x_207_; lean_object* v___x_208_; lean_object* v___x_209_; lean_object* v___x_210_; lean_object* v___x_211_; lean_object* v___x_212_; lean_object* v___x_213_; lean_object* v___x_214_; lean_object* v___x_215_; lean_object* v___x_216_; lean_object* v___x_217_; lean_object* v___x_218_; lean_object* v___x_219_; lean_object* v___x_220_; lean_object* v___x_221_; lean_object* v___x_222_; lean_object* v___x_223_; lean_object* v___x_224_; lean_object* v___x_225_; lean_object* v___x_226_; lean_object* v___x_227_; lean_object* v___x_228_; lean_object* v___x_229_; 
v_id_178_ = lean_ctor_get(v_x_177_, 0);
lean_inc_ref(v_id_178_);
v_store_179_ = lean_ctor_get_uint8(v_x_177_, sizeof(void*)*2);
v_entries_180_ = lean_ctor_get(v_x_177_, 1);
lean_inc(v_entries_180_);
v_revoked_181_ = lean_ctor_get_uint8(v_x_177_, sizeof(void*)*2 + 1);
lean_dec_ref(v_x_177_);
v___x_182_ = ((lean_object*)(lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__5));
v___x_183_ = ((lean_object*)(lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__6));
v___x_184_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__7, &lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__7_once, _init_lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__7);
v___x_185_ = l_String_quote(v_id_178_);
v___x_186_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_186_, 0, v___x_185_);
v___x_187_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_187_, 0, v___x_184_);
lean_ctor_set(v___x_187_, 1, v___x_186_);
v___x_188_ = 0;
v___x_189_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_189_, 0, v___x_187_);
lean_ctor_set_uint8(v___x_189_, sizeof(void*)*1, v___x_188_);
v___x_190_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_190_, 0, v___x_183_);
lean_ctor_set(v___x_190_, 1, v___x_189_);
v___x_191_ = ((lean_object*)(lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__9));
v___x_192_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_192_, 0, v___x_190_);
lean_ctor_set(v___x_192_, 1, v___x_191_);
v___x_193_ = lean_box(1);
v___x_194_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_194_, 0, v___x_192_);
lean_ctor_set(v___x_194_, 1, v___x_193_);
v___x_195_ = ((lean_object*)(lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__11));
v___x_196_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_196_, 0, v___x_194_);
lean_ctor_set(v___x_196_, 1, v___x_195_);
v___x_197_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_197_, 0, v___x_196_);
lean_ctor_set(v___x_197_, 1, v___x_182_);
v___x_198_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__12, &lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__12_once, _init_lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__12);
v___x_199_ = lean_unsigned_to_nat(0u);
v___x_200_ = lp_reflect_x2dnts_NTS_instReprStore_repr(v_store_179_, v___x_199_);
v___x_201_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_201_, 0, v___x_198_);
lean_ctor_set(v___x_201_, 1, v___x_200_);
v___x_202_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_202_, 0, v___x_201_);
lean_ctor_set_uint8(v___x_202_, sizeof(void*)*1, v___x_188_);
v___x_203_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_203_, 0, v___x_197_);
lean_ctor_set(v___x_203_, 1, v___x_202_);
v___x_204_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_204_, 0, v___x_203_);
lean_ctor_set(v___x_204_, 1, v___x_191_);
v___x_205_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_205_, 0, v___x_204_);
lean_ctor_set(v___x_205_, 1, v___x_193_);
v___x_206_ = ((lean_object*)(lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__14));
v___x_207_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_207_, 0, v___x_205_);
lean_ctor_set(v___x_207_, 1, v___x_206_);
v___x_208_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_208_, 0, v___x_207_);
lean_ctor_set(v___x_208_, 1, v___x_182_);
v___x_209_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__15, &lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__15_once, _init_lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__15);
v___x_210_ = l_List_repr_x27___at___00Lean_Syntax_instReprPreresolved_repr_spec__0___redArg(v_entries_180_);
v___x_211_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_211_, 0, v___x_209_);
lean_ctor_set(v___x_211_, 1, v___x_210_);
v___x_212_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_212_, 0, v___x_211_);
lean_ctor_set_uint8(v___x_212_, sizeof(void*)*1, v___x_188_);
v___x_213_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_213_, 0, v___x_208_);
lean_ctor_set(v___x_213_, 1, v___x_212_);
v___x_214_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_214_, 0, v___x_213_);
lean_ctor_set(v___x_214_, 1, v___x_191_);
v___x_215_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_215_, 0, v___x_214_);
lean_ctor_set(v___x_215_, 1, v___x_193_);
v___x_216_ = ((lean_object*)(lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__17));
v___x_217_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_217_, 0, v___x_215_);
lean_ctor_set(v___x_217_, 1, v___x_216_);
v___x_218_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_218_, 0, v___x_217_);
lean_ctor_set(v___x_218_, 1, v___x_182_);
v___x_219_ = l_Bool_repr___redArg(v_revoked_181_);
v___x_220_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_220_, 0, v___x_209_);
lean_ctor_set(v___x_220_, 1, v___x_219_);
v___x_221_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_221_, 0, v___x_220_);
lean_ctor_set_uint8(v___x_221_, sizeof(void*)*1, v___x_188_);
v___x_222_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_222_, 0, v___x_218_);
lean_ctor_set(v___x_222_, 1, v___x_221_);
v___x_223_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__20, &lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__20_once, _init_lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__20);
v___x_224_ = ((lean_object*)(lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__21));
v___x_225_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_225_, 0, v___x_224_);
lean_ctor_set(v___x_225_, 1, v___x_222_);
v___x_226_ = ((lean_object*)(lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__22));
v___x_227_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_227_, 0, v___x_225_);
lean_ctor_set(v___x_227_, 1, v___x_226_);
v___x_228_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_228_, 0, v___x_223_);
lean_ctor_set(v___x_228_, 1, v___x_227_);
v___x_229_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_229_, 0, v___x_228_);
lean_ctor_set_uint8(v___x_229_, sizeof(void*)*1, v___x_188_);
return v___x_229_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instReprGrant_repr(lean_object* v_x_230_, lean_object* v_prec_231_){
_start:
{
lean_object* v___x_232_; 
v___x_232_ = lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg(v_x_230_);
return v___x_232_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instReprGrant_repr___boxed(lean_object* v_x_233_, lean_object* v_prec_234_){
_start:
{
lean_object* v_res_235_; 
v_res_235_ = lp_reflect_x2dnts_NTS_instReprGrant_repr(v_x_233_, v_prec_234_);
lean_dec(v_prec_234_);
return v_res_235_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Source_ctorIdx(lean_object* v_x_238_){
_start:
{
if (lean_obj_tag(v_x_238_) == 0)
{
lean_object* v___x_239_; 
v___x_239_ = lean_unsigned_to_nat(0u);
return v___x_239_;
}
else
{
lean_object* v___x_240_; 
v___x_240_ = lean_unsigned_to_nat(1u);
return v___x_240_;
}
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Source_ctorIdx___boxed(lean_object* v_x_241_){
_start:
{
lean_object* v_res_242_; 
v_res_242_ = lp_reflect_x2dnts_NTS_Source_ctorIdx(v_x_241_);
lean_dec(v_x_241_);
return v_res_242_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Source_ctorElim___redArg(lean_object* v_t_243_, lean_object* v_k_244_){
_start:
{
if (lean_obj_tag(v_t_243_) == 0)
{
return v_k_244_;
}
else
{
lean_object* v_entry_245_; lean_object* v_quote_246_; lean_object* v_grant_247_; lean_object* v___x_248_; 
v_entry_245_ = lean_ctor_get(v_t_243_, 0);
lean_inc_ref(v_entry_245_);
v_quote_246_ = lean_ctor_get(v_t_243_, 1);
lean_inc_ref(v_quote_246_);
v_grant_247_ = lean_ctor_get(v_t_243_, 2);
lean_inc_ref(v_grant_247_);
lean_dec_ref_known(v_t_243_, 3);
v___x_248_ = lean_apply_3(v_k_244_, v_entry_245_, v_quote_246_, v_grant_247_);
return v___x_248_;
}
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Source_ctorElim(lean_object* v_motive_249_, lean_object* v_ctorIdx_250_, lean_object* v_t_251_, lean_object* v_h_252_, lean_object* v_k_253_){
_start:
{
lean_object* v___x_254_; 
v___x_254_ = lp_reflect_x2dnts_NTS_Source_ctorElim___redArg(v_t_251_, v_k_253_);
return v___x_254_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Source_ctorElim___boxed(lean_object* v_motive_255_, lean_object* v_ctorIdx_256_, lean_object* v_t_257_, lean_object* v_h_258_, lean_object* v_k_259_){
_start:
{
lean_object* v_res_260_; 
v_res_260_ = lp_reflect_x2dnts_NTS_Source_ctorElim(v_motive_255_, v_ctorIdx_256_, v_t_257_, v_h_258_, v_k_259_);
lean_dec(v_ctorIdx_256_);
return v_res_260_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Source_stated_elim___redArg(lean_object* v_t_261_, lean_object* v_stated_262_){
_start:
{
lean_object* v___x_263_; 
v___x_263_ = lp_reflect_x2dnts_NTS_Source_ctorElim___redArg(v_t_261_, v_stated_262_);
return v___x_263_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Source_stated_elim(lean_object* v_motive_264_, lean_object* v_t_265_, lean_object* v_h_266_, lean_object* v_stated_267_){
_start:
{
lean_object* v___x_268_; 
v___x_268_ = lp_reflect_x2dnts_NTS_Source_ctorElim___redArg(v_t_265_, v_stated_267_);
return v___x_268_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Source_fromJournal_elim___redArg(lean_object* v_t_269_, lean_object* v_fromJournal_270_){
_start:
{
lean_object* v___x_271_; 
v___x_271_ = lp_reflect_x2dnts_NTS_Source_ctorElim___redArg(v_t_269_, v_fromJournal_270_);
return v___x_271_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Source_fromJournal_elim(lean_object* v_motive_272_, lean_object* v_t_273_, lean_object* v_h_274_, lean_object* v_fromJournal_275_){
_start:
{
lean_object* v___x_276_; 
v___x_276_ = lp_reflect_x2dnts_NTS_Source_ctorElim___redArg(v_t_273_, v_fromJournal_275_);
return v___x_276_;
}
}
LEAN_EXPORT uint8_t lp_reflect_x2dnts_NTS_instDecidableEqSource_decEq(lean_object* v_x_277_, lean_object* v_x_278_){
_start:
{
if (lean_obj_tag(v_x_277_) == 0)
{
if (lean_obj_tag(v_x_278_) == 0)
{
uint8_t v___x_279_; 
v___x_279_ = 1;
return v___x_279_;
}
else
{
uint8_t v___x_280_; 
v___x_280_ = 0;
return v___x_280_;
}
}
else
{
lean_object* v_entry_281_; lean_object* v_quote_282_; lean_object* v_grant_283_; uint8_t v___x_284_; 
v_entry_281_ = lean_ctor_get(v_x_277_, 0);
v_quote_282_ = lean_ctor_get(v_x_277_, 1);
v_grant_283_ = lean_ctor_get(v_x_277_, 2);
v___x_284_ = 0;
if (lean_obj_tag(v_x_278_) == 0)
{
return v___x_284_;
}
else
{
lean_object* v_entry_285_; lean_object* v_quote_286_; lean_object* v_grant_287_; uint8_t v___x_288_; 
v_entry_285_ = lean_ctor_get(v_x_278_, 0);
v_quote_286_ = lean_ctor_get(v_x_278_, 1);
v_grant_287_ = lean_ctor_get(v_x_278_, 2);
v___x_288_ = lean_string_dec_eq(v_entry_281_, v_entry_285_);
if (v___x_288_ == 0)
{
return v___x_284_;
}
else
{
uint8_t v___x_289_; 
v___x_289_ = lean_string_dec_eq(v_quote_282_, v_quote_286_);
if (v___x_289_ == 0)
{
return v___x_284_;
}
else
{
uint8_t v___x_290_; 
v___x_290_ = lean_string_dec_eq(v_grant_283_, v_grant_287_);
if (v___x_290_ == 0)
{
return v___x_284_;
}
else
{
return v___x_290_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instDecidableEqSource_decEq___boxed(lean_object* v_x_291_, lean_object* v_x_292_){
_start:
{
uint8_t v_res_293_; lean_object* v_r_294_; 
v_res_293_ = lp_reflect_x2dnts_NTS_instDecidableEqSource_decEq(v_x_291_, v_x_292_);
lean_dec(v_x_292_);
lean_dec(v_x_291_);
v_r_294_ = lean_box(v_res_293_);
return v_r_294_;
}
}
LEAN_EXPORT uint8_t lp_reflect_x2dnts_NTS_instDecidableEqSource(lean_object* v_x_295_, lean_object* v_x_296_){
_start:
{
uint8_t v___x_297_; 
v___x_297_ = lp_reflect_x2dnts_NTS_instDecidableEqSource_decEq(v_x_295_, v_x_296_);
return v___x_297_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instDecidableEqSource___boxed(lean_object* v_x_298_, lean_object* v_x_299_){
_start:
{
uint8_t v_res_300_; lean_object* v_r_301_; 
v_res_300_ = lp_reflect_x2dnts_NTS_instDecidableEqSource(v_x_298_, v_x_299_);
lean_dec(v_x_299_);
lean_dec(v_x_298_);
v_r_301_ = lean_box(v_res_300_);
return v_r_301_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instReprSource_repr(lean_object* v_x_311_, lean_object* v_prec_312_){
_start:
{
lean_object* v___y_314_; 
if (lean_obj_tag(v_x_311_) == 0)
{
lean_object* v___x_320_; uint8_t v___x_321_; 
v___x_320_ = lean_unsigned_to_nat(1024u);
v___x_321_ = lean_nat_dec_le(v___x_320_, v_prec_312_);
if (v___x_321_ == 0)
{
lean_object* v___x_322_; 
v___x_322_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprStore_repr___closed__4, &lp_reflect_x2dnts_NTS_instReprStore_repr___closed__4_once, _init_lp_reflect_x2dnts_NTS_instReprStore_repr___closed__4);
v___y_314_ = v___x_322_;
goto v___jp_313_;
}
else
{
lean_object* v___x_323_; 
v___x_323_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprStore_repr___closed__5, &lp_reflect_x2dnts_NTS_instReprStore_repr___closed__5_once, _init_lp_reflect_x2dnts_NTS_instReprStore_repr___closed__5);
v___y_314_ = v___x_323_;
goto v___jp_313_;
}
}
else
{
lean_object* v_entry_324_; lean_object* v_quote_325_; lean_object* v_grant_326_; lean_object* v___y_328_; lean_object* v___x_346_; uint8_t v___x_347_; 
v_entry_324_ = lean_ctor_get(v_x_311_, 0);
lean_inc_ref(v_entry_324_);
v_quote_325_ = lean_ctor_get(v_x_311_, 1);
lean_inc_ref(v_quote_325_);
v_grant_326_ = lean_ctor_get(v_x_311_, 2);
lean_inc_ref(v_grant_326_);
lean_dec_ref_known(v_x_311_, 3);
v___x_346_ = lean_unsigned_to_nat(1024u);
v___x_347_ = lean_nat_dec_le(v___x_346_, v_prec_312_);
if (v___x_347_ == 0)
{
lean_object* v___x_348_; 
v___x_348_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprStore_repr___closed__4, &lp_reflect_x2dnts_NTS_instReprStore_repr___closed__4_once, _init_lp_reflect_x2dnts_NTS_instReprStore_repr___closed__4);
v___y_328_ = v___x_348_;
goto v___jp_327_;
}
else
{
lean_object* v___x_349_; 
v___x_349_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprStore_repr___closed__5, &lp_reflect_x2dnts_NTS_instReprStore_repr___closed__5_once, _init_lp_reflect_x2dnts_NTS_instReprStore_repr___closed__5);
v___y_328_ = v___x_349_;
goto v___jp_327_;
}
v___jp_327_:
{
lean_object* v___x_329_; lean_object* v___x_330_; lean_object* v___x_331_; lean_object* v___x_332_; lean_object* v___x_333_; lean_object* v___x_334_; lean_object* v___x_335_; lean_object* v___x_336_; lean_object* v___x_337_; lean_object* v___x_338_; lean_object* v___x_339_; lean_object* v___x_340_; lean_object* v___x_341_; lean_object* v___x_342_; uint8_t v___x_343_; lean_object* v___x_344_; lean_object* v___x_345_; 
v___x_329_ = lean_box(1);
v___x_330_ = ((lean_object*)(lp_reflect_x2dnts_NTS_instReprSource_repr___closed__4));
v___x_331_ = l_String_quote(v_entry_324_);
v___x_332_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_332_, 0, v___x_331_);
v___x_333_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_333_, 0, v___x_330_);
lean_ctor_set(v___x_333_, 1, v___x_332_);
v___x_334_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_334_, 0, v___x_333_);
lean_ctor_set(v___x_334_, 1, v___x_329_);
v___x_335_ = l_String_quote(v_quote_325_);
v___x_336_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_336_, 0, v___x_335_);
v___x_337_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_337_, 0, v___x_334_);
lean_ctor_set(v___x_337_, 1, v___x_336_);
v___x_338_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_338_, 0, v___x_337_);
lean_ctor_set(v___x_338_, 1, v___x_329_);
v___x_339_ = l_String_quote(v_grant_326_);
v___x_340_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_340_, 0, v___x_339_);
v___x_341_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_341_, 0, v___x_338_);
lean_ctor_set(v___x_341_, 1, v___x_340_);
lean_inc(v___y_328_);
v___x_342_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_342_, 0, v___y_328_);
lean_ctor_set(v___x_342_, 1, v___x_341_);
v___x_343_ = 0;
v___x_344_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_344_, 0, v___x_342_);
lean_ctor_set_uint8(v___x_344_, sizeof(void*)*1, v___x_343_);
v___x_345_ = l_Repr_addAppParen(v___x_344_, v_prec_312_);
return v___x_345_;
}
}
v___jp_313_:
{
lean_object* v___x_315_; lean_object* v___x_316_; uint8_t v___x_317_; lean_object* v___x_318_; lean_object* v___x_319_; 
v___x_315_ = ((lean_object*)(lp_reflect_x2dnts_NTS_instReprSource_repr___closed__1));
lean_inc(v___y_314_);
v___x_316_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_316_, 0, v___y_314_);
lean_ctor_set(v___x_316_, 1, v___x_315_);
v___x_317_ = 0;
v___x_318_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_318_, 0, v___x_316_);
lean_ctor_set_uint8(v___x_318_, sizeof(void*)*1, v___x_317_);
v___x_319_ = l_Repr_addAppParen(v___x_318_, v_prec_312_);
return v___x_319_;
}
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instReprSource_repr___boxed(lean_object* v_x_350_, lean_object* v_prec_351_){
_start:
{
lean_object* v_res_352_; 
v_res_352_ = lp_reflect_x2dnts_NTS_instReprSource_repr(v_x_350_, v_prec_351_);
lean_dec(v_prec_351_);
return v_res_352_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Status_ctorIdx(uint8_t v_x_355_){
_start:
{
if (v_x_355_ == 0)
{
lean_object* v___x_356_; 
v___x_356_ = lean_unsigned_to_nat(0u);
return v___x_356_;
}
else
{
lean_object* v___x_357_; 
v___x_357_ = lean_unsigned_to_nat(1u);
return v___x_357_;
}
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Status_ctorIdx___boxed(lean_object* v_x_358_){
_start:
{
uint8_t v_x_boxed_359_; lean_object* v_res_360_; 
v_x_boxed_359_ = lean_unbox(v_x_358_);
v_res_360_ = lp_reflect_x2dnts_NTS_Status_ctorIdx(v_x_boxed_359_);
return v_res_360_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Status_toCtorIdx(uint8_t v_x_361_){
_start:
{
lean_object* v___x_362_; 
v___x_362_ = lp_reflect_x2dnts_NTS_Status_ctorIdx(v_x_361_);
return v___x_362_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Status_toCtorIdx___boxed(lean_object* v_x_363_){
_start:
{
uint8_t v_x_4__boxed_364_; lean_object* v_res_365_; 
v_x_4__boxed_364_ = lean_unbox(v_x_363_);
v_res_365_ = lp_reflect_x2dnts_NTS_Status_toCtorIdx(v_x_4__boxed_364_);
return v_res_365_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Status_ctorElim___redArg(lean_object* v_k_366_){
_start:
{
lean_inc(v_k_366_);
return v_k_366_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Status_ctorElim___redArg___boxed(lean_object* v_k_367_){
_start:
{
lean_object* v_res_368_; 
v_res_368_ = lp_reflect_x2dnts_NTS_Status_ctorElim___redArg(v_k_367_);
lean_dec(v_k_367_);
return v_res_368_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Status_ctorElim(lean_object* v_motive_369_, lean_object* v_ctorIdx_370_, uint8_t v_t_371_, lean_object* v_h_372_, lean_object* v_k_373_){
_start:
{
lean_inc(v_k_373_);
return v_k_373_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Status_ctorElim___boxed(lean_object* v_motive_374_, lean_object* v_ctorIdx_375_, lean_object* v_t_376_, lean_object* v_h_377_, lean_object* v_k_378_){
_start:
{
uint8_t v_t_boxed_379_; lean_object* v_res_380_; 
v_t_boxed_379_ = lean_unbox(v_t_376_);
v_res_380_ = lp_reflect_x2dnts_NTS_Status_ctorElim(v_motive_374_, v_ctorIdx_375_, v_t_boxed_379_, v_h_377_, v_k_378_);
lean_dec(v_k_378_);
lean_dec(v_ctorIdx_375_);
return v_res_380_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Status_proposed_elim___redArg(lean_object* v_proposed_381_){
_start:
{
lean_inc(v_proposed_381_);
return v_proposed_381_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Status_proposed_elim___redArg___boxed(lean_object* v_proposed_382_){
_start:
{
lean_object* v_res_383_; 
v_res_383_ = lp_reflect_x2dnts_NTS_Status_proposed_elim___redArg(v_proposed_382_);
lean_dec(v_proposed_382_);
return v_res_383_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Status_proposed_elim(lean_object* v_motive_384_, uint8_t v_t_385_, lean_object* v_h_386_, lean_object* v_proposed_387_){
_start:
{
lean_inc(v_proposed_387_);
return v_proposed_387_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Status_proposed_elim___boxed(lean_object* v_motive_388_, lean_object* v_t_389_, lean_object* v_h_390_, lean_object* v_proposed_391_){
_start:
{
uint8_t v_t_boxed_392_; lean_object* v_res_393_; 
v_t_boxed_392_ = lean_unbox(v_t_389_);
v_res_393_ = lp_reflect_x2dnts_NTS_Status_proposed_elim(v_motive_388_, v_t_boxed_392_, v_h_390_, v_proposed_391_);
lean_dec(v_proposed_391_);
return v_res_393_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Status_confirmed_elim___redArg(lean_object* v_confirmed_394_){
_start:
{
lean_inc(v_confirmed_394_);
return v_confirmed_394_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Status_confirmed_elim___redArg___boxed(lean_object* v_confirmed_395_){
_start:
{
lean_object* v_res_396_; 
v_res_396_ = lp_reflect_x2dnts_NTS_Status_confirmed_elim___redArg(v_confirmed_395_);
lean_dec(v_confirmed_395_);
return v_res_396_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Status_confirmed_elim(lean_object* v_motive_397_, uint8_t v_t_398_, lean_object* v_h_399_, lean_object* v_confirmed_400_){
_start:
{
lean_inc(v_confirmed_400_);
return v_confirmed_400_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Status_confirmed_elim___boxed(lean_object* v_motive_401_, lean_object* v_t_402_, lean_object* v_h_403_, lean_object* v_confirmed_404_){
_start:
{
uint8_t v_t_boxed_405_; lean_object* v_res_406_; 
v_t_boxed_405_ = lean_unbox(v_t_402_);
v_res_406_ = lp_reflect_x2dnts_NTS_Status_confirmed_elim(v_motive_401_, v_t_boxed_405_, v_h_403_, v_confirmed_404_);
lean_dec(v_confirmed_404_);
return v_res_406_;
}
}
LEAN_EXPORT uint8_t lp_reflect_x2dnts_NTS_Status_ofNat(lean_object* v_n_407_){
_start:
{
lean_object* v___x_408_; uint8_t v___x_409_; 
v___x_408_ = lean_unsigned_to_nat(0u);
v___x_409_ = lean_nat_dec_le(v_n_407_, v___x_408_);
if (v___x_409_ == 0)
{
uint8_t v___x_410_; 
v___x_410_ = 1;
return v___x_410_;
}
else
{
uint8_t v___x_411_; 
v___x_411_ = 0;
return v___x_411_;
}
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Status_ofNat___boxed(lean_object* v_n_412_){
_start:
{
uint8_t v_res_413_; lean_object* v_r_414_; 
v_res_413_ = lp_reflect_x2dnts_NTS_Status_ofNat(v_n_412_);
lean_dec(v_n_412_);
v_r_414_ = lean_box(v_res_413_);
return v_r_414_;
}
}
LEAN_EXPORT uint8_t lp_reflect_x2dnts_NTS_instDecidableEqStatus(uint8_t v_x_415_, uint8_t v_y_416_){
_start:
{
lean_object* v___x_417_; lean_object* v___x_418_; uint8_t v___x_419_; 
v___x_417_ = lp_reflect_x2dnts_NTS_Status_ctorIdx(v_x_415_);
v___x_418_ = lp_reflect_x2dnts_NTS_Status_ctorIdx(v_y_416_);
v___x_419_ = lean_nat_dec_eq(v___x_417_, v___x_418_);
lean_dec(v___x_418_);
lean_dec(v___x_417_);
return v___x_419_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instDecidableEqStatus___boxed(lean_object* v_x_420_, lean_object* v_y_421_){
_start:
{
uint8_t v_x_13__boxed_422_; uint8_t v_y_14__boxed_423_; uint8_t v_res_424_; lean_object* v_r_425_; 
v_x_13__boxed_422_ = lean_unbox(v_x_420_);
v_y_14__boxed_423_ = lean_unbox(v_y_421_);
v_res_424_ = lp_reflect_x2dnts_NTS_instDecidableEqStatus(v_x_13__boxed_422_, v_y_14__boxed_423_);
v_r_425_ = lean_box(v_res_424_);
return v_r_425_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instReprStatus_repr(uint8_t v_x_432_, lean_object* v_prec_433_){
_start:
{
lean_object* v___y_435_; lean_object* v___y_442_; 
if (v_x_432_ == 0)
{
lean_object* v___x_448_; uint8_t v___x_449_; 
v___x_448_ = lean_unsigned_to_nat(1024u);
v___x_449_ = lean_nat_dec_le(v___x_448_, v_prec_433_);
if (v___x_449_ == 0)
{
lean_object* v___x_450_; 
v___x_450_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprStore_repr___closed__4, &lp_reflect_x2dnts_NTS_instReprStore_repr___closed__4_once, _init_lp_reflect_x2dnts_NTS_instReprStore_repr___closed__4);
v___y_435_ = v___x_450_;
goto v___jp_434_;
}
else
{
lean_object* v___x_451_; 
v___x_451_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprStore_repr___closed__5, &lp_reflect_x2dnts_NTS_instReprStore_repr___closed__5_once, _init_lp_reflect_x2dnts_NTS_instReprStore_repr___closed__5);
v___y_435_ = v___x_451_;
goto v___jp_434_;
}
}
else
{
lean_object* v___x_452_; uint8_t v___x_453_; 
v___x_452_ = lean_unsigned_to_nat(1024u);
v___x_453_ = lean_nat_dec_le(v___x_452_, v_prec_433_);
if (v___x_453_ == 0)
{
lean_object* v___x_454_; 
v___x_454_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprStore_repr___closed__4, &lp_reflect_x2dnts_NTS_instReprStore_repr___closed__4_once, _init_lp_reflect_x2dnts_NTS_instReprStore_repr___closed__4);
v___y_442_ = v___x_454_;
goto v___jp_441_;
}
else
{
lean_object* v___x_455_; 
v___x_455_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprStore_repr___closed__5, &lp_reflect_x2dnts_NTS_instReprStore_repr___closed__5_once, _init_lp_reflect_x2dnts_NTS_instReprStore_repr___closed__5);
v___y_442_ = v___x_455_;
goto v___jp_441_;
}
}
v___jp_434_:
{
lean_object* v___x_436_; lean_object* v___x_437_; uint8_t v___x_438_; lean_object* v___x_439_; lean_object* v___x_440_; 
v___x_436_ = ((lean_object*)(lp_reflect_x2dnts_NTS_instReprStatus_repr___closed__1));
lean_inc(v___y_435_);
v___x_437_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_437_, 0, v___y_435_);
lean_ctor_set(v___x_437_, 1, v___x_436_);
v___x_438_ = 0;
v___x_439_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_439_, 0, v___x_437_);
lean_ctor_set_uint8(v___x_439_, sizeof(void*)*1, v___x_438_);
v___x_440_ = l_Repr_addAppParen(v___x_439_, v_prec_433_);
return v___x_440_;
}
v___jp_441_:
{
lean_object* v___x_443_; lean_object* v___x_444_; uint8_t v___x_445_; lean_object* v___x_446_; lean_object* v___x_447_; 
v___x_443_ = ((lean_object*)(lp_reflect_x2dnts_NTS_instReprStatus_repr___closed__3));
lean_inc(v___y_442_);
v___x_444_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_444_, 0, v___y_442_);
lean_ctor_set(v___x_444_, 1, v___x_443_);
v___x_445_ = 0;
v___x_446_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_446_, 0, v___x_444_);
lean_ctor_set_uint8(v___x_446_, sizeof(void*)*1, v___x_445_);
v___x_447_ = l_Repr_addAppParen(v___x_446_, v_prec_433_);
return v___x_447_;
}
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instReprStatus_repr___boxed(lean_object* v_x_456_, lean_object* v_prec_457_){
_start:
{
uint8_t v_x_117__boxed_458_; lean_object* v_res_459_; 
v_x_117__boxed_458_ = lean_unbox(v_x_456_);
v_res_459_ = lp_reflect_x2dnts_NTS_instReprStatus_repr(v_x_117__boxed_458_, v_prec_457_);
lean_dec(v_prec_457_);
return v_res_459_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_Option_repr___at___00NTS_instReprFact_repr_spec__0(lean_object* v_x_468_, lean_object* v_x_469_){
_start:
{
if (lean_obj_tag(v_x_468_) == 0)
{
lean_object* v___x_470_; 
v___x_470_ = ((lean_object*)(lp_reflect_x2dnts_Option_repr___at___00NTS_instReprFact_repr_spec__0___closed__1));
return v___x_470_;
}
else
{
lean_object* v_val_471_; lean_object* v___x_473_; uint8_t v_isShared_474_; uint8_t v_isSharedCheck_482_; 
v_val_471_ = lean_ctor_get(v_x_468_, 0);
v_isSharedCheck_482_ = !lean_is_exclusive(v_x_468_);
if (v_isSharedCheck_482_ == 0)
{
v___x_473_ = v_x_468_;
v_isShared_474_ = v_isSharedCheck_482_;
goto v_resetjp_472_;
}
else
{
lean_inc(v_val_471_);
lean_dec(v_x_468_);
v___x_473_ = lean_box(0);
v_isShared_474_ = v_isSharedCheck_482_;
goto v_resetjp_472_;
}
v_resetjp_472_:
{
lean_object* v___x_475_; lean_object* v___x_476_; lean_object* v___x_478_; 
v___x_475_ = ((lean_object*)(lp_reflect_x2dnts_Option_repr___at___00NTS_instReprFact_repr_spec__0___closed__3));
v___x_476_ = l_String_quote(v_val_471_);
if (v_isShared_474_ == 0)
{
lean_ctor_set_tag(v___x_473_, 3);
lean_ctor_set(v___x_473_, 0, v___x_476_);
v___x_478_ = v___x_473_;
goto v_reusejp_477_;
}
else
{
lean_object* v_reuseFailAlloc_481_; 
v_reuseFailAlloc_481_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v_reuseFailAlloc_481_, 0, v___x_476_);
v___x_478_ = v_reuseFailAlloc_481_;
goto v_reusejp_477_;
}
v_reusejp_477_:
{
lean_object* v___x_479_; lean_object* v___x_480_; 
v___x_479_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_479_, 0, v___x_475_);
lean_ctor_set(v___x_479_, 1, v___x_478_);
v___x_480_ = l_Repr_addAppParen(v___x_479_, v_x_469_);
return v___x_480_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_Option_repr___at___00NTS_instReprFact_repr_spec__0___boxed(lean_object* v_x_483_, lean_object* v_x_484_){
_start:
{
lean_object* v_res_485_; 
v_res_485_ = lp_reflect_x2dnts_Option_repr___at___00NTS_instReprFact_repr_spec__0(v_x_483_, v_x_484_);
lean_dec(v_x_484_);
return v_res_485_;
}
}
static lean_object* _init_lp_reflect_x2dnts_NTS_instReprFact_repr___redArg___closed__4(void){
_start:
{
lean_object* v___x_495_; lean_object* v___x_496_; 
v___x_495_ = lean_unsigned_to_nat(7u);
v___x_496_ = lean_nat_to_int(v___x_495_);
return v___x_496_;
}
}
static lean_object* _init_lp_reflect_x2dnts_NTS_instReprFact_repr___redArg___closed__9(void){
_start:
{
lean_object* v___x_503_; lean_object* v___x_504_; 
v___x_503_ = lean_unsigned_to_nat(13u);
v___x_504_ = lean_nat_to_int(v___x_503_);
return v___x_504_;
}
}
static lean_object* _init_lp_reflect_x2dnts_NTS_instReprFact_repr___redArg___closed__14(void){
_start:
{
lean_object* v___x_511_; lean_object* v___x_512_; 
v___x_511_ = lean_unsigned_to_nat(14u);
v___x_512_ = lean_nat_to_int(v___x_511_);
return v___x_512_;
}
}
static lean_object* _init_lp_reflect_x2dnts_NTS_instReprFact_repr___redArg___closed__17(void){
_start:
{
lean_object* v___x_516_; lean_object* v___x_517_; 
v___x_516_ = lean_unsigned_to_nat(10u);
v___x_517_ = lean_nat_to_int(v___x_516_);
return v___x_517_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instReprFact_repr___redArg(lean_object* v_x_521_){
_start:
{
lean_object* v_key_522_; lean_object* v_value_523_; lean_object* v_validFrom_524_; lean_object* v_validTo_525_; lean_object* v_recordedAt_526_; lean_object* v_source_527_; uint8_t v_status_528_; lean_object* v___x_529_; lean_object* v___x_530_; lean_object* v___x_531_; lean_object* v___x_532_; lean_object* v___x_533_; lean_object* v___x_534_; uint8_t v___x_535_; lean_object* v___x_536_; lean_object* v___x_537_; lean_object* v___x_538_; lean_object* v___x_539_; lean_object* v___x_540_; lean_object* v___x_541_; lean_object* v___x_542_; lean_object* v___x_543_; lean_object* v___x_544_; lean_object* v___x_545_; lean_object* v___x_546_; lean_object* v___x_547_; lean_object* v___x_548_; lean_object* v___x_549_; lean_object* v___x_550_; lean_object* v___x_551_; lean_object* v___x_552_; lean_object* v___x_553_; lean_object* v___x_554_; lean_object* v___x_555_; lean_object* v___x_556_; lean_object* v___x_557_; lean_object* v___x_558_; lean_object* v___x_559_; lean_object* v___x_560_; lean_object* v___x_561_; lean_object* v___x_562_; lean_object* v___x_563_; lean_object* v___x_564_; lean_object* v___x_565_; lean_object* v___x_566_; lean_object* v___x_567_; lean_object* v___x_568_; lean_object* v___x_569_; lean_object* v___x_570_; lean_object* v___x_571_; lean_object* v___x_572_; lean_object* v___x_573_; lean_object* v___x_574_; lean_object* v___x_575_; lean_object* v___x_576_; lean_object* v___x_577_; lean_object* v___x_578_; lean_object* v___x_579_; lean_object* v___x_580_; lean_object* v___x_581_; lean_object* v___x_582_; lean_object* v___x_583_; lean_object* v___x_584_; lean_object* v___x_585_; lean_object* v___x_586_; lean_object* v___x_587_; lean_object* v___x_588_; lean_object* v___x_589_; lean_object* v___x_590_; lean_object* v___x_591_; lean_object* v___x_592_; lean_object* v___x_593_; lean_object* v___x_594_; lean_object* v___x_595_; lean_object* v___x_596_; lean_object* v___x_597_; lean_object* v___x_598_; lean_object* v___x_599_; lean_object* v___x_600_; lean_object* v___x_601_; lean_object* v___x_602_; lean_object* v___x_603_; lean_object* v___x_604_; lean_object* v___x_605_; lean_object* v___x_606_; lean_object* v___x_607_; lean_object* v___x_608_; lean_object* v___x_609_; 
v_key_522_ = lean_ctor_get(v_x_521_, 0);
lean_inc_ref(v_key_522_);
v_value_523_ = lean_ctor_get(v_x_521_, 1);
lean_inc_ref(v_value_523_);
v_validFrom_524_ = lean_ctor_get(v_x_521_, 2);
lean_inc_ref(v_validFrom_524_);
v_validTo_525_ = lean_ctor_get(v_x_521_, 3);
lean_inc(v_validTo_525_);
v_recordedAt_526_ = lean_ctor_get(v_x_521_, 4);
lean_inc_ref(v_recordedAt_526_);
v_source_527_ = lean_ctor_get(v_x_521_, 5);
lean_inc(v_source_527_);
v_status_528_ = lean_ctor_get_uint8(v_x_521_, sizeof(void*)*6);
lean_dec_ref(v_x_521_);
v___x_529_ = ((lean_object*)(lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__5));
v___x_530_ = ((lean_object*)(lp_reflect_x2dnts_NTS_instReprFact_repr___redArg___closed__3));
v___x_531_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprFact_repr___redArg___closed__4, &lp_reflect_x2dnts_NTS_instReprFact_repr___redArg___closed__4_once, _init_lp_reflect_x2dnts_NTS_instReprFact_repr___redArg___closed__4);
v___x_532_ = l_String_quote(v_key_522_);
v___x_533_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_533_, 0, v___x_532_);
v___x_534_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_534_, 0, v___x_531_);
lean_ctor_set(v___x_534_, 1, v___x_533_);
v___x_535_ = 0;
v___x_536_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_536_, 0, v___x_534_);
lean_ctor_set_uint8(v___x_536_, sizeof(void*)*1, v___x_535_);
v___x_537_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_537_, 0, v___x_530_);
lean_ctor_set(v___x_537_, 1, v___x_536_);
v___x_538_ = ((lean_object*)(lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__9));
v___x_539_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_539_, 0, v___x_537_);
lean_ctor_set(v___x_539_, 1, v___x_538_);
v___x_540_ = lean_box(1);
v___x_541_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_541_, 0, v___x_539_);
lean_ctor_set(v___x_541_, 1, v___x_540_);
v___x_542_ = ((lean_object*)(lp_reflect_x2dnts_NTS_instReprFact_repr___redArg___closed__6));
v___x_543_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_543_, 0, v___x_541_);
lean_ctor_set(v___x_543_, 1, v___x_542_);
v___x_544_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_544_, 0, v___x_543_);
lean_ctor_set(v___x_544_, 1, v___x_529_);
v___x_545_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__12, &lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__12_once, _init_lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__12);
v___x_546_ = l_String_quote(v_value_523_);
v___x_547_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_547_, 0, v___x_546_);
v___x_548_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_548_, 0, v___x_545_);
lean_ctor_set(v___x_548_, 1, v___x_547_);
v___x_549_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_549_, 0, v___x_548_);
lean_ctor_set_uint8(v___x_549_, sizeof(void*)*1, v___x_535_);
v___x_550_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_550_, 0, v___x_544_);
lean_ctor_set(v___x_550_, 1, v___x_549_);
v___x_551_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_551_, 0, v___x_550_);
lean_ctor_set(v___x_551_, 1, v___x_538_);
v___x_552_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_552_, 0, v___x_551_);
lean_ctor_set(v___x_552_, 1, v___x_540_);
v___x_553_ = ((lean_object*)(lp_reflect_x2dnts_NTS_instReprFact_repr___redArg___closed__8));
v___x_554_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_554_, 0, v___x_552_);
lean_ctor_set(v___x_554_, 1, v___x_553_);
v___x_555_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_555_, 0, v___x_554_);
lean_ctor_set(v___x_555_, 1, v___x_529_);
v___x_556_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprFact_repr___redArg___closed__9, &lp_reflect_x2dnts_NTS_instReprFact_repr___redArg___closed__9_once, _init_lp_reflect_x2dnts_NTS_instReprFact_repr___redArg___closed__9);
v___x_557_ = l_String_quote(v_validFrom_524_);
v___x_558_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_558_, 0, v___x_557_);
v___x_559_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_559_, 0, v___x_556_);
lean_ctor_set(v___x_559_, 1, v___x_558_);
v___x_560_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_560_, 0, v___x_559_);
lean_ctor_set_uint8(v___x_560_, sizeof(void*)*1, v___x_535_);
v___x_561_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_561_, 0, v___x_555_);
lean_ctor_set(v___x_561_, 1, v___x_560_);
v___x_562_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_562_, 0, v___x_561_);
lean_ctor_set(v___x_562_, 1, v___x_538_);
v___x_563_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_563_, 0, v___x_562_);
lean_ctor_set(v___x_563_, 1, v___x_540_);
v___x_564_ = ((lean_object*)(lp_reflect_x2dnts_NTS_instReprFact_repr___redArg___closed__11));
v___x_565_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_565_, 0, v___x_563_);
lean_ctor_set(v___x_565_, 1, v___x_564_);
v___x_566_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_566_, 0, v___x_565_);
lean_ctor_set(v___x_566_, 1, v___x_529_);
v___x_567_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__15, &lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__15_once, _init_lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__15);
v___x_568_ = lean_unsigned_to_nat(0u);
v___x_569_ = lp_reflect_x2dnts_Option_repr___at___00NTS_instReprFact_repr_spec__0(v_validTo_525_, v___x_568_);
v___x_570_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_570_, 0, v___x_567_);
lean_ctor_set(v___x_570_, 1, v___x_569_);
v___x_571_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_571_, 0, v___x_570_);
lean_ctor_set_uint8(v___x_571_, sizeof(void*)*1, v___x_535_);
v___x_572_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_572_, 0, v___x_566_);
lean_ctor_set(v___x_572_, 1, v___x_571_);
v___x_573_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_573_, 0, v___x_572_);
lean_ctor_set(v___x_573_, 1, v___x_538_);
v___x_574_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_574_, 0, v___x_573_);
lean_ctor_set(v___x_574_, 1, v___x_540_);
v___x_575_ = ((lean_object*)(lp_reflect_x2dnts_NTS_instReprFact_repr___redArg___closed__13));
v___x_576_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_576_, 0, v___x_574_);
lean_ctor_set(v___x_576_, 1, v___x_575_);
v___x_577_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_577_, 0, v___x_576_);
lean_ctor_set(v___x_577_, 1, v___x_529_);
v___x_578_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprFact_repr___redArg___closed__14, &lp_reflect_x2dnts_NTS_instReprFact_repr___redArg___closed__14_once, _init_lp_reflect_x2dnts_NTS_instReprFact_repr___redArg___closed__14);
v___x_579_ = l_String_quote(v_recordedAt_526_);
v___x_580_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_580_, 0, v___x_579_);
v___x_581_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_581_, 0, v___x_578_);
lean_ctor_set(v___x_581_, 1, v___x_580_);
v___x_582_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_582_, 0, v___x_581_);
lean_ctor_set_uint8(v___x_582_, sizeof(void*)*1, v___x_535_);
v___x_583_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_583_, 0, v___x_577_);
lean_ctor_set(v___x_583_, 1, v___x_582_);
v___x_584_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_584_, 0, v___x_583_);
lean_ctor_set(v___x_584_, 1, v___x_538_);
v___x_585_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_585_, 0, v___x_584_);
lean_ctor_set(v___x_585_, 1, v___x_540_);
v___x_586_ = ((lean_object*)(lp_reflect_x2dnts_NTS_instReprFact_repr___redArg___closed__16));
v___x_587_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_587_, 0, v___x_585_);
lean_ctor_set(v___x_587_, 1, v___x_586_);
v___x_588_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_588_, 0, v___x_587_);
lean_ctor_set(v___x_588_, 1, v___x_529_);
v___x_589_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprFact_repr___redArg___closed__17, &lp_reflect_x2dnts_NTS_instReprFact_repr___redArg___closed__17_once, _init_lp_reflect_x2dnts_NTS_instReprFact_repr___redArg___closed__17);
v___x_590_ = lp_reflect_x2dnts_NTS_instReprSource_repr(v_source_527_, v___x_568_);
v___x_591_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_591_, 0, v___x_589_);
lean_ctor_set(v___x_591_, 1, v___x_590_);
v___x_592_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_592_, 0, v___x_591_);
lean_ctor_set_uint8(v___x_592_, sizeof(void*)*1, v___x_535_);
v___x_593_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_593_, 0, v___x_588_);
lean_ctor_set(v___x_593_, 1, v___x_592_);
v___x_594_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_594_, 0, v___x_593_);
lean_ctor_set(v___x_594_, 1, v___x_538_);
v___x_595_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_595_, 0, v___x_594_);
lean_ctor_set(v___x_595_, 1, v___x_540_);
v___x_596_ = ((lean_object*)(lp_reflect_x2dnts_NTS_instReprFact_repr___redArg___closed__19));
v___x_597_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_597_, 0, v___x_595_);
lean_ctor_set(v___x_597_, 1, v___x_596_);
v___x_598_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_598_, 0, v___x_597_);
lean_ctor_set(v___x_598_, 1, v___x_529_);
v___x_599_ = lp_reflect_x2dnts_NTS_instReprStatus_repr(v_status_528_, v___x_568_);
v___x_600_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_600_, 0, v___x_589_);
lean_ctor_set(v___x_600_, 1, v___x_599_);
v___x_601_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_601_, 0, v___x_600_);
lean_ctor_set_uint8(v___x_601_, sizeof(void*)*1, v___x_535_);
v___x_602_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_602_, 0, v___x_598_);
lean_ctor_set(v___x_602_, 1, v___x_601_);
v___x_603_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__20, &lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__20_once, _init_lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__20);
v___x_604_ = ((lean_object*)(lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__21));
v___x_605_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_605_, 0, v___x_604_);
lean_ctor_set(v___x_605_, 1, v___x_602_);
v___x_606_ = ((lean_object*)(lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__22));
v___x_607_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_607_, 0, v___x_605_);
lean_ctor_set(v___x_607_, 1, v___x_606_);
v___x_608_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_608_, 0, v___x_603_);
lean_ctor_set(v___x_608_, 1, v___x_607_);
v___x_609_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_609_, 0, v___x_608_);
lean_ctor_set_uint8(v___x_609_, sizeof(void*)*1, v___x_535_);
return v___x_609_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instReprFact_repr(lean_object* v_x_610_, lean_object* v_prec_611_){
_start:
{
lean_object* v___x_612_; 
v___x_612_ = lp_reflect_x2dnts_NTS_instReprFact_repr___redArg(v_x_610_);
return v___x_612_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instReprFact_repr___boxed(lean_object* v_x_613_, lean_object* v_prec_614_){
_start:
{
lean_object* v_res_615_; 
v_res_615_ = lp_reflect_x2dnts_NTS_instReprFact_repr(v_x_613_, v_prec_614_);
lean_dec(v_prec_614_);
return v_res_615_;
}
}
static lean_object* _init_lp_reflect_x2dnts_NTS_instReprBelief_repr___redArg___closed__4(void){
_start:
{
lean_object* v___x_627_; lean_object* v___x_628_; 
v___x_627_ = lean_unsigned_to_nat(8u);
v___x_628_ = lean_nat_to_int(v___x_627_);
return v___x_628_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instReprBelief_repr___redArg(lean_object* v_x_629_){
_start:
{
lean_object* v_text_630_; lean_object* v_recordedAt_631_; lean_object* v_source_632_; uint8_t v_status_633_; lean_object* v___x_634_; lean_object* v___x_635_; lean_object* v___x_636_; lean_object* v___x_637_; lean_object* v___x_638_; lean_object* v___x_639_; uint8_t v___x_640_; lean_object* v___x_641_; lean_object* v___x_642_; lean_object* v___x_643_; lean_object* v___x_644_; lean_object* v___x_645_; lean_object* v___x_646_; lean_object* v___x_647_; lean_object* v___x_648_; lean_object* v___x_649_; lean_object* v___x_650_; lean_object* v___x_651_; lean_object* v___x_652_; lean_object* v___x_653_; lean_object* v___x_654_; lean_object* v___x_655_; lean_object* v___x_656_; lean_object* v___x_657_; lean_object* v___x_658_; lean_object* v___x_659_; lean_object* v___x_660_; lean_object* v___x_661_; lean_object* v___x_662_; lean_object* v___x_663_; lean_object* v___x_664_; lean_object* v___x_665_; lean_object* v___x_666_; lean_object* v___x_667_; lean_object* v___x_668_; lean_object* v___x_669_; lean_object* v___x_670_; lean_object* v___x_671_; lean_object* v___x_672_; lean_object* v___x_673_; lean_object* v___x_674_; lean_object* v___x_675_; lean_object* v___x_676_; lean_object* v___x_677_; lean_object* v___x_678_; lean_object* v___x_679_; lean_object* v___x_680_; lean_object* v___x_681_; lean_object* v___x_682_; 
v_text_630_ = lean_ctor_get(v_x_629_, 0);
lean_inc_ref(v_text_630_);
v_recordedAt_631_ = lean_ctor_get(v_x_629_, 1);
lean_inc_ref(v_recordedAt_631_);
v_source_632_ = lean_ctor_get(v_x_629_, 2);
lean_inc(v_source_632_);
v_status_633_ = lean_ctor_get_uint8(v_x_629_, sizeof(void*)*3);
lean_dec_ref(v_x_629_);
v___x_634_ = ((lean_object*)(lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__5));
v___x_635_ = ((lean_object*)(lp_reflect_x2dnts_NTS_instReprBelief_repr___redArg___closed__3));
v___x_636_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprBelief_repr___redArg___closed__4, &lp_reflect_x2dnts_NTS_instReprBelief_repr___redArg___closed__4_once, _init_lp_reflect_x2dnts_NTS_instReprBelief_repr___redArg___closed__4);
v___x_637_ = l_String_quote(v_text_630_);
v___x_638_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_638_, 0, v___x_637_);
v___x_639_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_639_, 0, v___x_636_);
lean_ctor_set(v___x_639_, 1, v___x_638_);
v___x_640_ = 0;
v___x_641_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_641_, 0, v___x_639_);
lean_ctor_set_uint8(v___x_641_, sizeof(void*)*1, v___x_640_);
v___x_642_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_642_, 0, v___x_635_);
lean_ctor_set(v___x_642_, 1, v___x_641_);
v___x_643_ = ((lean_object*)(lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__9));
v___x_644_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_644_, 0, v___x_642_);
lean_ctor_set(v___x_644_, 1, v___x_643_);
v___x_645_ = lean_box(1);
v___x_646_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_646_, 0, v___x_644_);
lean_ctor_set(v___x_646_, 1, v___x_645_);
v___x_647_ = ((lean_object*)(lp_reflect_x2dnts_NTS_instReprFact_repr___redArg___closed__13));
v___x_648_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_648_, 0, v___x_646_);
lean_ctor_set(v___x_648_, 1, v___x_647_);
v___x_649_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_649_, 0, v___x_648_);
lean_ctor_set(v___x_649_, 1, v___x_634_);
v___x_650_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprFact_repr___redArg___closed__14, &lp_reflect_x2dnts_NTS_instReprFact_repr___redArg___closed__14_once, _init_lp_reflect_x2dnts_NTS_instReprFact_repr___redArg___closed__14);
v___x_651_ = l_String_quote(v_recordedAt_631_);
v___x_652_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_652_, 0, v___x_651_);
v___x_653_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_653_, 0, v___x_650_);
lean_ctor_set(v___x_653_, 1, v___x_652_);
v___x_654_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_654_, 0, v___x_653_);
lean_ctor_set_uint8(v___x_654_, sizeof(void*)*1, v___x_640_);
v___x_655_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_655_, 0, v___x_649_);
lean_ctor_set(v___x_655_, 1, v___x_654_);
v___x_656_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_656_, 0, v___x_655_);
lean_ctor_set(v___x_656_, 1, v___x_643_);
v___x_657_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_657_, 0, v___x_656_);
lean_ctor_set(v___x_657_, 1, v___x_645_);
v___x_658_ = ((lean_object*)(lp_reflect_x2dnts_NTS_instReprFact_repr___redArg___closed__16));
v___x_659_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_659_, 0, v___x_657_);
lean_ctor_set(v___x_659_, 1, v___x_658_);
v___x_660_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_660_, 0, v___x_659_);
lean_ctor_set(v___x_660_, 1, v___x_634_);
v___x_661_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprFact_repr___redArg___closed__17, &lp_reflect_x2dnts_NTS_instReprFact_repr___redArg___closed__17_once, _init_lp_reflect_x2dnts_NTS_instReprFact_repr___redArg___closed__17);
v___x_662_ = lean_unsigned_to_nat(0u);
v___x_663_ = lp_reflect_x2dnts_NTS_instReprSource_repr(v_source_632_, v___x_662_);
v___x_664_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_664_, 0, v___x_661_);
lean_ctor_set(v___x_664_, 1, v___x_663_);
v___x_665_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_665_, 0, v___x_664_);
lean_ctor_set_uint8(v___x_665_, sizeof(void*)*1, v___x_640_);
v___x_666_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_666_, 0, v___x_660_);
lean_ctor_set(v___x_666_, 1, v___x_665_);
v___x_667_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_667_, 0, v___x_666_);
lean_ctor_set(v___x_667_, 1, v___x_643_);
v___x_668_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_668_, 0, v___x_667_);
lean_ctor_set(v___x_668_, 1, v___x_645_);
v___x_669_ = ((lean_object*)(lp_reflect_x2dnts_NTS_instReprFact_repr___redArg___closed__19));
v___x_670_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_670_, 0, v___x_668_);
lean_ctor_set(v___x_670_, 1, v___x_669_);
v___x_671_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_671_, 0, v___x_670_);
lean_ctor_set(v___x_671_, 1, v___x_634_);
v___x_672_ = lp_reflect_x2dnts_NTS_instReprStatus_repr(v_status_633_, v___x_662_);
v___x_673_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_673_, 0, v___x_661_);
lean_ctor_set(v___x_673_, 1, v___x_672_);
v___x_674_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_674_, 0, v___x_673_);
lean_ctor_set_uint8(v___x_674_, sizeof(void*)*1, v___x_640_);
v___x_675_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_675_, 0, v___x_671_);
lean_ctor_set(v___x_675_, 1, v___x_674_);
v___x_676_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__20, &lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__20_once, _init_lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__20);
v___x_677_ = ((lean_object*)(lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__21));
v___x_678_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_678_, 0, v___x_677_);
lean_ctor_set(v___x_678_, 1, v___x_675_);
v___x_679_ = ((lean_object*)(lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__22));
v___x_680_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_680_, 0, v___x_678_);
lean_ctor_set(v___x_680_, 1, v___x_679_);
v___x_681_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_681_, 0, v___x_676_);
lean_ctor_set(v___x_681_, 1, v___x_680_);
v___x_682_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_682_, 0, v___x_681_);
lean_ctor_set_uint8(v___x_682_, sizeof(void*)*1, v___x_640_);
return v___x_682_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instReprBelief_repr(lean_object* v_x_683_, lean_object* v_prec_684_){
_start:
{
lean_object* v___x_685_; 
v___x_685_ = lp_reflect_x2dnts_NTS_instReprBelief_repr___redArg(v_x_683_);
return v___x_685_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instReprBelief_repr___boxed(lean_object* v_x_686_, lean_object* v_prec_687_){
_start:
{
lean_object* v_res_688_; 
v_res_688_ = lp_reflect_x2dnts_NTS_instReprBelief_repr(v_x_686_, v_prec_687_);
lean_dec(v_prec_687_);
return v_res_688_;
}
}
LEAN_EXPORT uint8_t lp_reflect_x2dnts_List_elem___at___00NTS_Source_allowed_spec__0(lean_object* v_a_691_, lean_object* v_x_692_){
_start:
{
if (lean_obj_tag(v_x_692_) == 0)
{
uint8_t v___x_693_; 
v___x_693_ = 0;
return v___x_693_;
}
else
{
lean_object* v_head_694_; lean_object* v_tail_695_; uint8_t v___x_696_; 
v_head_694_ = lean_ctor_get(v_x_692_, 0);
v_tail_695_ = lean_ctor_get(v_x_692_, 1);
v___x_696_ = lean_string_dec_eq(v_a_691_, v_head_694_);
if (v___x_696_ == 0)
{
v_x_692_ = v_tail_695_;
goto _start;
}
else
{
return v___x_696_;
}
}
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_List_elem___at___00NTS_Source_allowed_spec__0___boxed(lean_object* v_a_698_, lean_object* v_x_699_){
_start:
{
uint8_t v_res_700_; lean_object* v_r_701_; 
v_res_700_ = lp_reflect_x2dnts_List_elem___at___00NTS_Source_allowed_spec__0(v_a_698_, v_x_699_);
lean_dec(v_x_699_);
lean_dec_ref(v_a_698_);
v_r_701_ = lean_box(v_res_700_);
return v_r_701_;
}
}
LEAN_EXPORT uint8_t lp_reflect_x2dnts_List_any___at___00NTS_Source_allowed_spec__1(lean_object* v_entry_702_, lean_object* v_grant_703_, uint8_t v_store_704_, lean_object* v_x_705_){
_start:
{
if (lean_obj_tag(v_x_705_) == 0)
{
uint8_t v___x_706_; 
v___x_706_ = 0;
return v___x_706_;
}
else
{
lean_object* v_head_707_; lean_object* v_tail_708_; lean_object* v_id_709_; uint8_t v_store_710_; lean_object* v_entries_711_; uint8_t v_revoked_712_; uint8_t v___y_714_; uint8_t v___x_719_; 
v_head_707_ = lean_ctor_get(v_x_705_, 0);
v_tail_708_ = lean_ctor_get(v_x_705_, 1);
v_id_709_ = lean_ctor_get(v_head_707_, 0);
v_store_710_ = lean_ctor_get_uint8(v_head_707_, sizeof(void*)*2);
v_entries_711_ = lean_ctor_get(v_head_707_, 1);
v_revoked_712_ = lean_ctor_get_uint8(v_head_707_, sizeof(void*)*2 + 1);
v___x_719_ = lean_string_dec_eq(v_id_709_, v_grant_703_);
if (v___x_719_ == 0)
{
v___y_714_ = v___x_719_;
goto v___jp_713_;
}
else
{
uint8_t v___x_720_; 
v___x_720_ = lp_reflect_x2dnts_NTS_instDecidableEqStore(v_store_710_, v_store_704_);
v___y_714_ = v___x_720_;
goto v___jp_713_;
}
v___jp_713_:
{
if (v___y_714_ == 0)
{
v_x_705_ = v_tail_708_;
goto _start;
}
else
{
if (v_revoked_712_ == 0)
{
uint8_t v___x_716_; 
v___x_716_ = lp_reflect_x2dnts_List_elem___at___00NTS_Source_allowed_spec__0(v_entry_702_, v_entries_711_);
if (v___x_716_ == 0)
{
v_x_705_ = v_tail_708_;
goto _start;
}
else
{
return v___x_716_;
}
}
else
{
v_x_705_ = v_tail_708_;
goto _start;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_List_any___at___00NTS_Source_allowed_spec__1___boxed(lean_object* v_entry_721_, lean_object* v_grant_722_, lean_object* v_store_723_, lean_object* v_x_724_){
_start:
{
uint8_t v_store_boxed_725_; uint8_t v_res_726_; lean_object* v_r_727_; 
v_store_boxed_725_ = lean_unbox(v_store_723_);
v_res_726_ = lp_reflect_x2dnts_List_any___at___00NTS_Source_allowed_spec__1(v_entry_721_, v_grant_722_, v_store_boxed_725_, v_x_724_);
lean_dec(v_x_724_);
lean_dec_ref(v_grant_722_);
lean_dec_ref(v_entry_721_);
v_r_727_ = lean_box(v_res_726_);
return v_r_727_;
}
}
LEAN_EXPORT uint8_t lp_reflect_x2dnts_NTS_Source_allowed(lean_object* v_grants_728_, uint8_t v_store_729_, lean_object* v_x_730_){
_start:
{
if (lean_obj_tag(v_x_730_) == 0)
{
uint8_t v___x_731_; 
v___x_731_ = 1;
return v___x_731_;
}
else
{
lean_object* v_entry_732_; lean_object* v_grant_733_; uint8_t v___x_734_; 
v_entry_732_ = lean_ctor_get(v_x_730_, 0);
v_grant_733_ = lean_ctor_get(v_x_730_, 2);
v___x_734_ = lp_reflect_x2dnts_List_any___at___00NTS_Source_allowed_spec__1(v_entry_732_, v_grant_733_, v_store_729_, v_grants_728_);
return v___x_734_;
}
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Source_allowed___boxed(lean_object* v_grants_735_, lean_object* v_store_736_, lean_object* v_x_737_){
_start:
{
uint8_t v_store_boxed_738_; uint8_t v_res_739_; lean_object* v_r_740_; 
v_store_boxed_738_ = lean_unbox(v_store_736_);
v_res_739_ = lp_reflect_x2dnts_NTS_Source_allowed(v_grants_735_, v_store_boxed_738_, v_x_737_);
lean_dec(v_x_737_);
lean_dec(v_grants_735_);
v_r_740_ = lean_box(v_res_739_);
return v_r_740_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts___private_NTS_Person_0__NTS_instReprSource_repr_match__1_splitter___redArg(lean_object* v_x_741_, lean_object* v_h__1_742_, lean_object* v_h__2_743_){
_start:
{
if (lean_obj_tag(v_x_741_) == 0)
{
lean_object* v___x_744_; lean_object* v___x_745_; 
lean_dec(v_h__2_743_);
v___x_744_ = lean_box(0);
v___x_745_ = lean_apply_1(v_h__1_742_, v___x_744_);
return v___x_745_;
}
else
{
lean_object* v_entry_746_; lean_object* v_quote_747_; lean_object* v_grant_748_; lean_object* v___x_749_; 
lean_dec(v_h__1_742_);
v_entry_746_ = lean_ctor_get(v_x_741_, 0);
lean_inc_ref(v_entry_746_);
v_quote_747_ = lean_ctor_get(v_x_741_, 1);
lean_inc_ref(v_quote_747_);
v_grant_748_ = lean_ctor_get(v_x_741_, 2);
lean_inc_ref(v_grant_748_);
lean_dec_ref_known(v_x_741_, 3);
v___x_749_ = lean_apply_3(v_h__2_743_, v_entry_746_, v_quote_747_, v_grant_748_);
return v___x_749_;
}
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts___private_NTS_Person_0__NTS_instReprSource_repr_match__1_splitter(lean_object* v_motive_750_, lean_object* v_x_751_, lean_object* v_h__1_752_, lean_object* v_h__2_753_){
_start:
{
if (lean_obj_tag(v_x_751_) == 0)
{
lean_object* v___x_754_; lean_object* v___x_755_; 
lean_dec(v_h__2_753_);
v___x_754_ = lean_box(0);
v___x_755_ = lean_apply_1(v_h__1_752_, v___x_754_);
return v___x_755_;
}
else
{
lean_object* v_entry_756_; lean_object* v_quote_757_; lean_object* v_grant_758_; lean_object* v___x_759_; 
lean_dec(v_h__1_752_);
v_entry_756_ = lean_ctor_get(v_x_751_, 0);
lean_inc_ref(v_entry_756_);
v_quote_757_ = lean_ctor_get(v_x_751_, 1);
lean_inc_ref(v_quote_757_);
v_grant_758_ = lean_ctor_get(v_x_751_, 2);
lean_inc_ref(v_grant_758_);
lean_dec_ref_known(v_x_751_, 3);
v___x_759_ = lean_apply_3(v_h__2_753_, v_entry_756_, v_quote_757_, v_grant_758_);
return v___x_759_;
}
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Whose_ctorIdx(lean_object* v_x_760_){
_start:
{
if (lean_obj_tag(v_x_760_) == 0)
{
lean_object* v___x_761_; 
v___x_761_ = lean_unsigned_to_nat(0u);
return v___x_761_;
}
else
{
lean_object* v___x_762_; 
v___x_762_ = lean_unsigned_to_nat(1u);
return v___x_762_;
}
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Whose_ctorIdx___boxed(lean_object* v_x_763_){
_start:
{
lean_object* v_res_764_; 
v_res_764_ = lp_reflect_x2dnts_NTS_Whose_ctorIdx(v_x_763_);
lean_dec(v_x_763_);
return v_res_764_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Whose_ctorElim___redArg(lean_object* v_t_765_, lean_object* v_k_766_){
_start:
{
if (lean_obj_tag(v_t_765_) == 0)
{
return v_k_766_;
}
else
{
lean_object* v_name_767_; lean_object* v___x_768_; 
v_name_767_ = lean_ctor_get(v_t_765_, 0);
lean_inc_ref(v_name_767_);
lean_dec_ref_known(v_t_765_, 1);
v___x_768_ = lean_apply_1(v_k_766_, v_name_767_);
return v___x_768_;
}
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Whose_ctorElim(lean_object* v_motive_769_, lean_object* v_ctorIdx_770_, lean_object* v_t_771_, lean_object* v_h_772_, lean_object* v_k_773_){
_start:
{
lean_object* v___x_774_; 
v___x_774_ = lp_reflect_x2dnts_NTS_Whose_ctorElim___redArg(v_t_771_, v_k_773_);
return v___x_774_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Whose_ctorElim___boxed(lean_object* v_motive_775_, lean_object* v_ctorIdx_776_, lean_object* v_t_777_, lean_object* v_h_778_, lean_object* v_k_779_){
_start:
{
lean_object* v_res_780_; 
v_res_780_ = lp_reflect_x2dnts_NTS_Whose_ctorElim(v_motive_775_, v_ctorIdx_776_, v_t_777_, v_h_778_, v_k_779_);
lean_dec(v_ctorIdx_776_);
return v_res_780_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Whose_self_elim___redArg(lean_object* v_t_781_, lean_object* v_self_782_){
_start:
{
lean_object* v___x_783_; 
v___x_783_ = lp_reflect_x2dnts_NTS_Whose_ctorElim___redArg(v_t_781_, v_self_782_);
return v___x_783_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Whose_self_elim(lean_object* v_motive_784_, lean_object* v_t_785_, lean_object* v_h_786_, lean_object* v_self_787_){
_start:
{
lean_object* v___x_788_; 
v___x_788_ = lp_reflect_x2dnts_NTS_Whose_ctorElim___redArg(v_t_785_, v_self_787_);
return v___x_788_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Whose_other_elim___redArg(lean_object* v_t_789_, lean_object* v_other_790_){
_start:
{
lean_object* v___x_791_; 
v___x_791_ = lp_reflect_x2dnts_NTS_Whose_ctorElim___redArg(v_t_789_, v_other_790_);
return v___x_791_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Whose_other_elim(lean_object* v_motive_792_, lean_object* v_t_793_, lean_object* v_h_794_, lean_object* v_other_795_){
_start:
{
lean_object* v___x_796_; 
v___x_796_ = lp_reflect_x2dnts_NTS_Whose_ctorElim___redArg(v_t_793_, v_other_795_);
return v___x_796_;
}
}
LEAN_EXPORT uint8_t lp_reflect_x2dnts_NTS_instDecidableEqWhose_decEq(lean_object* v_x_797_, lean_object* v_x_798_){
_start:
{
if (lean_obj_tag(v_x_797_) == 0)
{
if (lean_obj_tag(v_x_798_) == 0)
{
uint8_t v___x_799_; 
v___x_799_ = 1;
return v___x_799_;
}
else
{
uint8_t v___x_800_; 
v___x_800_ = 0;
return v___x_800_;
}
}
else
{
lean_object* v_name_801_; uint8_t v___x_802_; 
v_name_801_ = lean_ctor_get(v_x_797_, 0);
v___x_802_ = 0;
if (lean_obj_tag(v_x_798_) == 0)
{
return v___x_802_;
}
else
{
lean_object* v_name_803_; uint8_t v___x_804_; 
v_name_803_ = lean_ctor_get(v_x_798_, 0);
v___x_804_ = lean_string_dec_eq(v_name_801_, v_name_803_);
if (v___x_804_ == 0)
{
return v___x_802_;
}
else
{
return v___x_804_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instDecidableEqWhose_decEq___boxed(lean_object* v_x_805_, lean_object* v_x_806_){
_start:
{
uint8_t v_res_807_; lean_object* v_r_808_; 
v_res_807_ = lp_reflect_x2dnts_NTS_instDecidableEqWhose_decEq(v_x_805_, v_x_806_);
lean_dec(v_x_806_);
lean_dec(v_x_805_);
v_r_808_ = lean_box(v_res_807_);
return v_r_808_;
}
}
LEAN_EXPORT uint8_t lp_reflect_x2dnts_NTS_instDecidableEqWhose(lean_object* v_x_809_, lean_object* v_x_810_){
_start:
{
uint8_t v___x_811_; 
v___x_811_ = lp_reflect_x2dnts_NTS_instDecidableEqWhose_decEq(v_x_809_, v_x_810_);
return v___x_811_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instDecidableEqWhose___boxed(lean_object* v_x_812_, lean_object* v_x_813_){
_start:
{
uint8_t v_res_814_; lean_object* v_r_815_; 
v_res_814_ = lp_reflect_x2dnts_NTS_instDecidableEqWhose(v_x_812_, v_x_813_);
lean_dec(v_x_813_);
lean_dec(v_x_812_);
v_r_815_ = lean_box(v_res_814_);
return v_r_815_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instReprWhose_repr(lean_object* v_x_825_, lean_object* v_prec_826_){
_start:
{
lean_object* v___y_828_; 
if (lean_obj_tag(v_x_825_) == 0)
{
lean_object* v___x_834_; uint8_t v___x_835_; 
v___x_834_ = lean_unsigned_to_nat(1024u);
v___x_835_ = lean_nat_dec_le(v___x_834_, v_prec_826_);
if (v___x_835_ == 0)
{
lean_object* v___x_836_; 
v___x_836_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprStore_repr___closed__4, &lp_reflect_x2dnts_NTS_instReprStore_repr___closed__4_once, _init_lp_reflect_x2dnts_NTS_instReprStore_repr___closed__4);
v___y_828_ = v___x_836_;
goto v___jp_827_;
}
else
{
lean_object* v___x_837_; 
v___x_837_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprStore_repr___closed__5, &lp_reflect_x2dnts_NTS_instReprStore_repr___closed__5_once, _init_lp_reflect_x2dnts_NTS_instReprStore_repr___closed__5);
v___y_828_ = v___x_837_;
goto v___jp_827_;
}
}
else
{
lean_object* v_name_838_; lean_object* v___x_840_; uint8_t v_isShared_841_; uint8_t v_isSharedCheck_858_; 
v_name_838_ = lean_ctor_get(v_x_825_, 0);
v_isSharedCheck_858_ = !lean_is_exclusive(v_x_825_);
if (v_isSharedCheck_858_ == 0)
{
v___x_840_ = v_x_825_;
v_isShared_841_ = v_isSharedCheck_858_;
goto v_resetjp_839_;
}
else
{
lean_inc(v_name_838_);
lean_dec(v_x_825_);
v___x_840_ = lean_box(0);
v_isShared_841_ = v_isSharedCheck_858_;
goto v_resetjp_839_;
}
v_resetjp_839_:
{
lean_object* v___y_843_; lean_object* v___x_854_; uint8_t v___x_855_; 
v___x_854_ = lean_unsigned_to_nat(1024u);
v___x_855_ = lean_nat_dec_le(v___x_854_, v_prec_826_);
if (v___x_855_ == 0)
{
lean_object* v___x_856_; 
v___x_856_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprStore_repr___closed__4, &lp_reflect_x2dnts_NTS_instReprStore_repr___closed__4_once, _init_lp_reflect_x2dnts_NTS_instReprStore_repr___closed__4);
v___y_843_ = v___x_856_;
goto v___jp_842_;
}
else
{
lean_object* v___x_857_; 
v___x_857_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprStore_repr___closed__5, &lp_reflect_x2dnts_NTS_instReprStore_repr___closed__5_once, _init_lp_reflect_x2dnts_NTS_instReprStore_repr___closed__5);
v___y_843_ = v___x_857_;
goto v___jp_842_;
}
v___jp_842_:
{
lean_object* v___x_844_; lean_object* v___x_845_; lean_object* v___x_847_; 
v___x_844_ = ((lean_object*)(lp_reflect_x2dnts_NTS_instReprWhose_repr___closed__4));
v___x_845_ = l_String_quote(v_name_838_);
if (v_isShared_841_ == 0)
{
lean_ctor_set_tag(v___x_840_, 3);
lean_ctor_set(v___x_840_, 0, v___x_845_);
v___x_847_ = v___x_840_;
goto v_reusejp_846_;
}
else
{
lean_object* v_reuseFailAlloc_853_; 
v_reuseFailAlloc_853_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v_reuseFailAlloc_853_, 0, v___x_845_);
v___x_847_ = v_reuseFailAlloc_853_;
goto v_reusejp_846_;
}
v_reusejp_846_:
{
lean_object* v___x_848_; lean_object* v___x_849_; uint8_t v___x_850_; lean_object* v___x_851_; lean_object* v___x_852_; 
v___x_848_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_848_, 0, v___x_844_);
lean_ctor_set(v___x_848_, 1, v___x_847_);
lean_inc(v___y_843_);
v___x_849_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_849_, 0, v___y_843_);
lean_ctor_set(v___x_849_, 1, v___x_848_);
v___x_850_ = 0;
v___x_851_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_851_, 0, v___x_849_);
lean_ctor_set_uint8(v___x_851_, sizeof(void*)*1, v___x_850_);
v___x_852_ = l_Repr_addAppParen(v___x_851_, v_prec_826_);
return v___x_852_;
}
}
}
}
v___jp_827_:
{
lean_object* v___x_829_; lean_object* v___x_830_; uint8_t v___x_831_; lean_object* v___x_832_; lean_object* v___x_833_; 
v___x_829_ = ((lean_object*)(lp_reflect_x2dnts_NTS_instReprWhose_repr___closed__1));
lean_inc(v___y_828_);
v___x_830_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_830_, 0, v___y_828_);
lean_ctor_set(v___x_830_, 1, v___x_829_);
v___x_831_ = 0;
v___x_832_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_832_, 0, v___x_830_);
lean_ctor_set_uint8(v___x_832_, sizeof(void*)*1, v___x_831_);
v___x_833_ = l_Repr_addAppParen(v___x_832_, v_prec_826_);
return v___x_833_;
}
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instReprWhose_repr___boxed(lean_object* v_x_859_, lean_object* v_prec_860_){
_start:
{
lean_object* v_res_861_; 
v_res_861_ = lp_reflect_x2dnts_NTS_instReprWhose_repr(v_x_859_, v_prec_860_);
lean_dec(v_prec_860_);
return v_res_861_;
}
}
static lean_object* _init_lp_reflect_x2dnts_Prod_repr___at___00List_repr___at___00NTS_instReprAct_repr_spec__0_spec__0___redArg___closed__3(void){
_start:
{
lean_object* v___x_869_; lean_object* v___x_870_; 
v___x_869_ = ((lean_object*)(lp_reflect_x2dnts_Prod_repr___at___00List_repr___at___00NTS_instReprAct_repr_spec__0_spec__0___redArg___closed__0));
v___x_870_ = lean_string_length(v___x_869_);
return v___x_870_;
}
}
static lean_object* _init_lp_reflect_x2dnts_Prod_repr___at___00List_repr___at___00NTS_instReprAct_repr_spec__0_spec__0___redArg___closed__4(void){
_start:
{
lean_object* v___x_871_; lean_object* v___x_872_; 
v___x_871_ = lean_obj_once(&lp_reflect_x2dnts_Prod_repr___at___00List_repr___at___00NTS_instReprAct_repr_spec__0_spec__0___redArg___closed__3, &lp_reflect_x2dnts_Prod_repr___at___00List_repr___at___00NTS_instReprAct_repr_spec__0_spec__0___redArg___closed__3_once, _init_lp_reflect_x2dnts_Prod_repr___at___00List_repr___at___00NTS_instReprAct_repr_spec__0_spec__0___redArg___closed__3);
v___x_872_ = lean_nat_to_int(v___x_871_);
return v___x_872_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_Prod_repr___at___00List_repr___at___00NTS_instReprAct_repr_spec__0_spec__0___redArg(lean_object* v_x_877_){
_start:
{
lean_object* v_fst_878_; lean_object* v_snd_879_; lean_object* v___x_881_; uint8_t v_isShared_882_; uint8_t v_isSharedCheck_904_; 
v_fst_878_ = lean_ctor_get(v_x_877_, 0);
v_snd_879_ = lean_ctor_get(v_x_877_, 1);
v_isSharedCheck_904_ = !lean_is_exclusive(v_x_877_);
if (v_isSharedCheck_904_ == 0)
{
v___x_881_ = v_x_877_;
v_isShared_882_ = v_isSharedCheck_904_;
goto v_resetjp_880_;
}
else
{
lean_inc(v_snd_879_);
lean_inc(v_fst_878_);
lean_dec(v_x_877_);
v___x_881_ = lean_box(0);
v_isShared_882_ = v_isSharedCheck_904_;
goto v_resetjp_880_;
}
v_resetjp_880_:
{
lean_object* v___x_883_; uint8_t v___x_884_; lean_object* v___x_885_; lean_object* v___x_886_; lean_object* v___x_888_; 
v___x_883_ = lean_unsigned_to_nat(0u);
v___x_884_ = lean_unbox(v_fst_878_);
lean_dec(v_fst_878_);
v___x_885_ = lp_reflect_x2dnts_NTS_instReprPrime_repr(v___x_884_, v___x_883_);
v___x_886_ = lean_box(0);
if (v_isShared_882_ == 0)
{
lean_ctor_set_tag(v___x_881_, 1);
lean_ctor_set(v___x_881_, 1, v___x_886_);
lean_ctor_set(v___x_881_, 0, v___x_885_);
v___x_888_ = v___x_881_;
goto v_reusejp_887_;
}
else
{
lean_object* v_reuseFailAlloc_903_; 
v_reuseFailAlloc_903_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_903_, 0, v___x_885_);
lean_ctor_set(v_reuseFailAlloc_903_, 1, v___x_886_);
v___x_888_ = v_reuseFailAlloc_903_;
goto v_reusejp_887_;
}
v_reusejp_887_:
{
lean_object* v___x_889_; lean_object* v___x_890_; lean_object* v___x_891_; lean_object* v___x_892_; lean_object* v___x_893_; lean_object* v___x_894_; lean_object* v___x_895_; lean_object* v___x_896_; lean_object* v___x_897_; lean_object* v___x_898_; lean_object* v___x_899_; lean_object* v___x_900_; uint8_t v___x_901_; lean_object* v___x_902_; 
v___x_889_ = l_Nat_reprFast(v_snd_879_);
v___x_890_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_890_, 0, v___x_889_);
v___x_891_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_891_, 0, v___x_890_);
lean_ctor_set(v___x_891_, 1, v___x_888_);
v___x_892_ = l_List_reverse___redArg(v___x_891_);
v___x_893_ = ((lean_object*)(lp_reflect_x2dnts_Prod_repr___at___00List_repr___at___00NTS_instReprAct_repr_spec__0_spec__0___redArg___closed__1));
v___x_894_ = l_Std_Format_joinSep___at___00Lean_Syntax_formatStxAux_spec__2(v___x_892_, v___x_893_);
v___x_895_ = lean_obj_once(&lp_reflect_x2dnts_Prod_repr___at___00List_repr___at___00NTS_instReprAct_repr_spec__0_spec__0___redArg___closed__4, &lp_reflect_x2dnts_Prod_repr___at___00List_repr___at___00NTS_instReprAct_repr_spec__0_spec__0___redArg___closed__4_once, _init_lp_reflect_x2dnts_Prod_repr___at___00List_repr___at___00NTS_instReprAct_repr_spec__0_spec__0___redArg___closed__4);
v___x_896_ = ((lean_object*)(lp_reflect_x2dnts_Prod_repr___at___00List_repr___at___00NTS_instReprAct_repr_spec__0_spec__0___redArg___closed__5));
v___x_897_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_897_, 0, v___x_896_);
lean_ctor_set(v___x_897_, 1, v___x_894_);
v___x_898_ = ((lean_object*)(lp_reflect_x2dnts_Prod_repr___at___00List_repr___at___00NTS_instReprAct_repr_spec__0_spec__0___redArg___closed__6));
v___x_899_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_899_, 0, v___x_897_);
lean_ctor_set(v___x_899_, 1, v___x_898_);
v___x_900_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_900_, 0, v___x_895_);
lean_ctor_set(v___x_900_, 1, v___x_899_);
v___x_901_ = 0;
v___x_902_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_902_, 0, v___x_900_);
lean_ctor_set_uint8(v___x_902_, sizeof(void*)*1, v___x_901_);
return v___x_902_;
}
}
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_List_foldl___at___00List_foldl___at___00Std_Format_joinSep___at___00List_repr___at___00NTS_instReprAct_repr_spec__0_spec__1_spec__2_spec__4(lean_object* v_x_905_, lean_object* v_x_906_, lean_object* v_x_907_){
_start:
{
if (lean_obj_tag(v_x_907_) == 0)
{
lean_dec(v_x_905_);
return v_x_906_;
}
else
{
lean_object* v_head_908_; lean_object* v_tail_909_; lean_object* v___x_911_; uint8_t v_isShared_912_; uint8_t v_isSharedCheck_919_; 
v_head_908_ = lean_ctor_get(v_x_907_, 0);
v_tail_909_ = lean_ctor_get(v_x_907_, 1);
v_isSharedCheck_919_ = !lean_is_exclusive(v_x_907_);
if (v_isSharedCheck_919_ == 0)
{
v___x_911_ = v_x_907_;
v_isShared_912_ = v_isSharedCheck_919_;
goto v_resetjp_910_;
}
else
{
lean_inc(v_tail_909_);
lean_inc(v_head_908_);
lean_dec(v_x_907_);
v___x_911_ = lean_box(0);
v_isShared_912_ = v_isSharedCheck_919_;
goto v_resetjp_910_;
}
v_resetjp_910_:
{
lean_object* v___x_914_; 
lean_inc(v_x_905_);
if (v_isShared_912_ == 0)
{
lean_ctor_set_tag(v___x_911_, 5);
lean_ctor_set(v___x_911_, 1, v_x_905_);
lean_ctor_set(v___x_911_, 0, v_x_906_);
v___x_914_ = v___x_911_;
goto v_reusejp_913_;
}
else
{
lean_object* v_reuseFailAlloc_918_; 
v_reuseFailAlloc_918_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v_reuseFailAlloc_918_, 0, v_x_906_);
lean_ctor_set(v_reuseFailAlloc_918_, 1, v_x_905_);
v___x_914_ = v_reuseFailAlloc_918_;
goto v_reusejp_913_;
}
v_reusejp_913_:
{
lean_object* v___x_915_; lean_object* v___x_916_; 
v___x_915_ = lp_reflect_x2dnts_Prod_repr___at___00List_repr___at___00NTS_instReprAct_repr_spec__0_spec__0___redArg(v_head_908_);
v___x_916_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_916_, 0, v___x_914_);
lean_ctor_set(v___x_916_, 1, v___x_915_);
v_x_906_ = v___x_916_;
v_x_907_ = v_tail_909_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_List_foldl___at___00Std_Format_joinSep___at___00List_repr___at___00NTS_instReprAct_repr_spec__0_spec__1_spec__2(lean_object* v_x_920_, lean_object* v_x_921_, lean_object* v_x_922_){
_start:
{
if (lean_obj_tag(v_x_922_) == 0)
{
lean_dec(v_x_920_);
return v_x_921_;
}
else
{
lean_object* v_head_923_; lean_object* v_tail_924_; lean_object* v___x_926_; uint8_t v_isShared_927_; uint8_t v_isSharedCheck_934_; 
v_head_923_ = lean_ctor_get(v_x_922_, 0);
v_tail_924_ = lean_ctor_get(v_x_922_, 1);
v_isSharedCheck_934_ = !lean_is_exclusive(v_x_922_);
if (v_isSharedCheck_934_ == 0)
{
v___x_926_ = v_x_922_;
v_isShared_927_ = v_isSharedCheck_934_;
goto v_resetjp_925_;
}
else
{
lean_inc(v_tail_924_);
lean_inc(v_head_923_);
lean_dec(v_x_922_);
v___x_926_ = lean_box(0);
v_isShared_927_ = v_isSharedCheck_934_;
goto v_resetjp_925_;
}
v_resetjp_925_:
{
lean_object* v___x_929_; 
lean_inc(v_x_920_);
if (v_isShared_927_ == 0)
{
lean_ctor_set_tag(v___x_926_, 5);
lean_ctor_set(v___x_926_, 1, v_x_920_);
lean_ctor_set(v___x_926_, 0, v_x_921_);
v___x_929_ = v___x_926_;
goto v_reusejp_928_;
}
else
{
lean_object* v_reuseFailAlloc_933_; 
v_reuseFailAlloc_933_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v_reuseFailAlloc_933_, 0, v_x_921_);
lean_ctor_set(v_reuseFailAlloc_933_, 1, v_x_920_);
v___x_929_ = v_reuseFailAlloc_933_;
goto v_reusejp_928_;
}
v_reusejp_928_:
{
lean_object* v___x_930_; lean_object* v___x_931_; lean_object* v___x_932_; 
v___x_930_ = lp_reflect_x2dnts_Prod_repr___at___00List_repr___at___00NTS_instReprAct_repr_spec__0_spec__0___redArg(v_head_923_);
v___x_931_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_931_, 0, v___x_929_);
lean_ctor_set(v___x_931_, 1, v___x_930_);
v___x_932_ = lp_reflect_x2dnts_List_foldl___at___00List_foldl___at___00Std_Format_joinSep___at___00List_repr___at___00NTS_instReprAct_repr_spec__0_spec__1_spec__2_spec__4(v_x_920_, v___x_931_, v_tail_924_);
return v___x_932_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_Std_Format_joinSep___at___00List_repr___at___00NTS_instReprAct_repr_spec__0_spec__1(lean_object* v_x_935_, lean_object* v_x_936_){
_start:
{
if (lean_obj_tag(v_x_935_) == 0)
{
lean_object* v___x_937_; 
lean_dec(v_x_936_);
v___x_937_ = lean_box(0);
return v___x_937_;
}
else
{
lean_object* v_tail_938_; 
v_tail_938_ = lean_ctor_get(v_x_935_, 1);
if (lean_obj_tag(v_tail_938_) == 0)
{
lean_object* v_head_939_; lean_object* v___x_940_; 
lean_dec(v_x_936_);
v_head_939_ = lean_ctor_get(v_x_935_, 0);
lean_inc(v_head_939_);
lean_dec_ref_known(v_x_935_, 2);
v___x_940_ = lp_reflect_x2dnts_Prod_repr___at___00List_repr___at___00NTS_instReprAct_repr_spec__0_spec__0___redArg(v_head_939_);
return v___x_940_;
}
else
{
lean_object* v_head_941_; lean_object* v___x_942_; lean_object* v___x_943_; 
lean_inc(v_tail_938_);
v_head_941_ = lean_ctor_get(v_x_935_, 0);
lean_inc(v_head_941_);
lean_dec_ref_known(v_x_935_, 2);
v___x_942_ = lp_reflect_x2dnts_Prod_repr___at___00List_repr___at___00NTS_instReprAct_repr_spec__0_spec__0___redArg(v_head_941_);
v___x_943_ = lp_reflect_x2dnts_List_foldl___at___00Std_Format_joinSep___at___00List_repr___at___00NTS_instReprAct_repr_spec__0_spec__1_spec__2(v_x_936_, v___x_942_, v_tail_938_);
return v___x_943_;
}
}
}
}
static lean_object* _init_lp_reflect_x2dnts_List_repr___at___00NTS_instReprAct_repr_spec__0___redArg___closed__4(void){
_start:
{
lean_object* v___x_949_; lean_object* v___x_950_; 
v___x_949_ = ((lean_object*)(lp_reflect_x2dnts_List_repr___at___00NTS_instReprAct_repr_spec__0___redArg___closed__2));
v___x_950_ = lean_string_length(v___x_949_);
return v___x_950_;
}
}
static lean_object* _init_lp_reflect_x2dnts_List_repr___at___00NTS_instReprAct_repr_spec__0___redArg___closed__5(void){
_start:
{
lean_object* v___x_951_; lean_object* v___x_952_; 
v___x_951_ = lean_obj_once(&lp_reflect_x2dnts_List_repr___at___00NTS_instReprAct_repr_spec__0___redArg___closed__4, &lp_reflect_x2dnts_List_repr___at___00NTS_instReprAct_repr_spec__0___redArg___closed__4_once, _init_lp_reflect_x2dnts_List_repr___at___00NTS_instReprAct_repr_spec__0___redArg___closed__4);
v___x_952_ = lean_nat_to_int(v___x_951_);
return v___x_952_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_List_repr___at___00NTS_instReprAct_repr_spec__0___redArg(lean_object* v_a_957_){
_start:
{
if (lean_obj_tag(v_a_957_) == 0)
{
lean_object* v___x_958_; 
v___x_958_ = ((lean_object*)(lp_reflect_x2dnts_List_repr___at___00NTS_instReprAct_repr_spec__0___redArg___closed__1));
return v___x_958_;
}
else
{
lean_object* v___x_959_; lean_object* v___x_960_; lean_object* v___x_961_; lean_object* v___x_962_; lean_object* v___x_963_; lean_object* v___x_964_; lean_object* v___x_965_; lean_object* v___x_966_; uint8_t v___x_967_; lean_object* v___x_968_; 
v___x_959_ = ((lean_object*)(lp_reflect_x2dnts_Prod_repr___at___00List_repr___at___00NTS_instReprAct_repr_spec__0_spec__0___redArg___closed__1));
v___x_960_ = lp_reflect_x2dnts_Std_Format_joinSep___at___00List_repr___at___00NTS_instReprAct_repr_spec__0_spec__1(v_a_957_, v___x_959_);
v___x_961_ = lean_obj_once(&lp_reflect_x2dnts_List_repr___at___00NTS_instReprAct_repr_spec__0___redArg___closed__5, &lp_reflect_x2dnts_List_repr___at___00NTS_instReprAct_repr_spec__0___redArg___closed__5_once, _init_lp_reflect_x2dnts_List_repr___at___00NTS_instReprAct_repr_spec__0___redArg___closed__5);
v___x_962_ = ((lean_object*)(lp_reflect_x2dnts_List_repr___at___00NTS_instReprAct_repr_spec__0___redArg___closed__6));
v___x_963_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_963_, 0, v___x_962_);
lean_ctor_set(v___x_963_, 1, v___x_960_);
v___x_964_ = ((lean_object*)(lp_reflect_x2dnts_List_repr___at___00NTS_instReprAct_repr_spec__0___redArg___closed__7));
v___x_965_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_965_, 0, v___x_963_);
lean_ctor_set(v___x_965_, 1, v___x_964_);
v___x_966_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_966_, 0, v___x_961_);
lean_ctor_set(v___x_966_, 1, v___x_965_);
v___x_967_ = 0;
v___x_968_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_968_, 0, v___x_966_);
lean_ctor_set_uint8(v___x_968_, sizeof(void*)*1, v___x_967_);
return v___x_968_;
}
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_Prod_repr___at___00Option_repr___at___00NTS_instReprAct_repr_spec__1_spec__3___redArg(lean_object* v_x_969_){
_start:
{
lean_object* v_fst_970_; lean_object* v_snd_971_; lean_object* v___x_973_; uint8_t v_isShared_974_; uint8_t v_isSharedCheck_995_; 
v_fst_970_ = lean_ctor_get(v_x_969_, 0);
v_snd_971_ = lean_ctor_get(v_x_969_, 1);
v_isSharedCheck_995_ = !lean_is_exclusive(v_x_969_);
if (v_isSharedCheck_995_ == 0)
{
v___x_973_ = v_x_969_;
v_isShared_974_ = v_isSharedCheck_995_;
goto v_resetjp_972_;
}
else
{
lean_inc(v_snd_971_);
lean_inc(v_fst_970_);
lean_dec(v_x_969_);
v___x_973_ = lean_box(0);
v_isShared_974_ = v_isSharedCheck_995_;
goto v_resetjp_972_;
}
v_resetjp_972_:
{
lean_object* v___x_975_; lean_object* v___x_976_; lean_object* v___x_977_; lean_object* v___x_979_; 
v___x_975_ = lean_unsigned_to_nat(0u);
v___x_976_ = lp_reflect_x2dnts_NTS_instReprWhose_repr(v_fst_970_, v___x_975_);
v___x_977_ = lean_box(0);
if (v_isShared_974_ == 0)
{
lean_ctor_set_tag(v___x_973_, 1);
lean_ctor_set(v___x_973_, 1, v___x_977_);
lean_ctor_set(v___x_973_, 0, v___x_976_);
v___x_979_ = v___x_973_;
goto v_reusejp_978_;
}
else
{
lean_object* v_reuseFailAlloc_994_; 
v_reuseFailAlloc_994_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_994_, 0, v___x_976_);
lean_ctor_set(v_reuseFailAlloc_994_, 1, v___x_977_);
v___x_979_ = v_reuseFailAlloc_994_;
goto v_reusejp_978_;
}
v_reusejp_978_:
{
uint8_t v___x_980_; lean_object* v___x_981_; lean_object* v___x_982_; lean_object* v___x_983_; lean_object* v___x_984_; lean_object* v___x_985_; lean_object* v___x_986_; lean_object* v___x_987_; lean_object* v___x_988_; lean_object* v___x_989_; lean_object* v___x_990_; lean_object* v___x_991_; uint8_t v___x_992_; lean_object* v___x_993_; 
v___x_980_ = lean_unbox(v_snd_971_);
lean_dec(v_snd_971_);
v___x_981_ = lp_reflect_x2dnts_NTS_instReprPrime_repr(v___x_980_, v___x_975_);
v___x_982_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_982_, 0, v___x_981_);
lean_ctor_set(v___x_982_, 1, v___x_979_);
v___x_983_ = l_List_reverse___redArg(v___x_982_);
v___x_984_ = ((lean_object*)(lp_reflect_x2dnts_Prod_repr___at___00List_repr___at___00NTS_instReprAct_repr_spec__0_spec__0___redArg___closed__1));
v___x_985_ = l_Std_Format_joinSep___at___00Lean_Syntax_formatStxAux_spec__2(v___x_983_, v___x_984_);
v___x_986_ = lean_obj_once(&lp_reflect_x2dnts_Prod_repr___at___00List_repr___at___00NTS_instReprAct_repr_spec__0_spec__0___redArg___closed__4, &lp_reflect_x2dnts_Prod_repr___at___00List_repr___at___00NTS_instReprAct_repr_spec__0_spec__0___redArg___closed__4_once, _init_lp_reflect_x2dnts_Prod_repr___at___00List_repr___at___00NTS_instReprAct_repr_spec__0_spec__0___redArg___closed__4);
v___x_987_ = ((lean_object*)(lp_reflect_x2dnts_Prod_repr___at___00List_repr___at___00NTS_instReprAct_repr_spec__0_spec__0___redArg___closed__5));
v___x_988_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_988_, 0, v___x_987_);
lean_ctor_set(v___x_988_, 1, v___x_985_);
v___x_989_ = ((lean_object*)(lp_reflect_x2dnts_Prod_repr___at___00List_repr___at___00NTS_instReprAct_repr_spec__0_spec__0___redArg___closed__6));
v___x_990_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_990_, 0, v___x_988_);
lean_ctor_set(v___x_990_, 1, v___x_989_);
v___x_991_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_991_, 0, v___x_986_);
lean_ctor_set(v___x_991_, 1, v___x_990_);
v___x_992_ = 0;
v___x_993_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_993_, 0, v___x_991_);
lean_ctor_set_uint8(v___x_993_, sizeof(void*)*1, v___x_992_);
return v___x_993_;
}
}
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_Option_repr___at___00NTS_instReprAct_repr_spec__1(lean_object* v_x_996_, lean_object* v_x_997_){
_start:
{
if (lean_obj_tag(v_x_996_) == 0)
{
lean_object* v___x_998_; 
v___x_998_ = ((lean_object*)(lp_reflect_x2dnts_Option_repr___at___00NTS_instReprFact_repr_spec__0___closed__1));
return v___x_998_;
}
else
{
lean_object* v_val_999_; lean_object* v___x_1000_; lean_object* v___x_1001_; lean_object* v___x_1002_; lean_object* v___x_1003_; 
v_val_999_ = lean_ctor_get(v_x_996_, 0);
lean_inc(v_val_999_);
lean_dec_ref_known(v_x_996_, 1);
v___x_1000_ = ((lean_object*)(lp_reflect_x2dnts_Option_repr___at___00NTS_instReprFact_repr_spec__0___closed__3));
v___x_1001_ = lp_reflect_x2dnts_Prod_repr___at___00Option_repr___at___00NTS_instReprAct_repr_spec__1_spec__3___redArg(v_val_999_);
v___x_1002_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1002_, 0, v___x_1000_);
lean_ctor_set(v___x_1002_, 1, v___x_1001_);
v___x_1003_ = l_Repr_addAppParen(v___x_1002_, v_x_997_);
return v___x_1003_;
}
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_Option_repr___at___00NTS_instReprAct_repr_spec__1___boxed(lean_object* v_x_1004_, lean_object* v_x_1005_){
_start:
{
lean_object* v_res_1006_; 
v_res_1006_ = lp_reflect_x2dnts_Option_repr___at___00NTS_instReprAct_repr_spec__1(v_x_1004_, v_x_1005_);
lean_dec(v_x_1005_);
return v_res_1006_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instReprAct_repr___redArg(lean_object* v_x_1019_){
_start:
{
lean_object* v_drivers_1020_; lean_object* v_target_1021_; lean_object* v___x_1023_; uint8_t v_isShared_1024_; uint8_t v_isSharedCheck_1055_; 
v_drivers_1020_ = lean_ctor_get(v_x_1019_, 0);
v_target_1021_ = lean_ctor_get(v_x_1019_, 1);
v_isSharedCheck_1055_ = !lean_is_exclusive(v_x_1019_);
if (v_isSharedCheck_1055_ == 0)
{
v___x_1023_ = v_x_1019_;
v_isShared_1024_ = v_isSharedCheck_1055_;
goto v_resetjp_1022_;
}
else
{
lean_inc(v_target_1021_);
lean_inc(v_drivers_1020_);
lean_dec(v_x_1019_);
v___x_1023_ = lean_box(0);
v_isShared_1024_ = v_isSharedCheck_1055_;
goto v_resetjp_1022_;
}
v_resetjp_1022_:
{
lean_object* v___x_1025_; lean_object* v___x_1026_; lean_object* v___x_1027_; lean_object* v___x_1028_; lean_object* v___x_1029_; lean_object* v___x_1031_; 
v___x_1025_ = ((lean_object*)(lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__5));
v___x_1026_ = ((lean_object*)(lp_reflect_x2dnts_NTS_instReprAct_repr___redArg___closed__3));
v___x_1027_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__15, &lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__15_once, _init_lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__15);
v___x_1028_ = lean_unsigned_to_nat(0u);
v___x_1029_ = lp_reflect_x2dnts_List_repr___at___00NTS_instReprAct_repr_spec__0___redArg(v_drivers_1020_);
if (v_isShared_1024_ == 0)
{
lean_ctor_set_tag(v___x_1023_, 4);
lean_ctor_set(v___x_1023_, 1, v___x_1029_);
lean_ctor_set(v___x_1023_, 0, v___x_1027_);
v___x_1031_ = v___x_1023_;
goto v_reusejp_1030_;
}
else
{
lean_object* v_reuseFailAlloc_1054_; 
v_reuseFailAlloc_1054_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1054_, 0, v___x_1027_);
lean_ctor_set(v_reuseFailAlloc_1054_, 1, v___x_1029_);
v___x_1031_ = v_reuseFailAlloc_1054_;
goto v_reusejp_1030_;
}
v_reusejp_1030_:
{
uint8_t v___x_1032_; lean_object* v___x_1033_; lean_object* v___x_1034_; lean_object* v___x_1035_; lean_object* v___x_1036_; lean_object* v___x_1037_; lean_object* v___x_1038_; lean_object* v___x_1039_; lean_object* v___x_1040_; lean_object* v___x_1041_; lean_object* v___x_1042_; lean_object* v___x_1043_; lean_object* v___x_1044_; lean_object* v___x_1045_; lean_object* v___x_1046_; lean_object* v___x_1047_; lean_object* v___x_1048_; lean_object* v___x_1049_; lean_object* v___x_1050_; lean_object* v___x_1051_; lean_object* v___x_1052_; lean_object* v___x_1053_; 
v___x_1032_ = 0;
v___x_1033_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_1033_, 0, v___x_1031_);
lean_ctor_set_uint8(v___x_1033_, sizeof(void*)*1, v___x_1032_);
v___x_1034_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1034_, 0, v___x_1026_);
lean_ctor_set(v___x_1034_, 1, v___x_1033_);
v___x_1035_ = ((lean_object*)(lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__9));
v___x_1036_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1036_, 0, v___x_1034_);
lean_ctor_set(v___x_1036_, 1, v___x_1035_);
v___x_1037_ = lean_box(1);
v___x_1038_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1038_, 0, v___x_1036_);
lean_ctor_set(v___x_1038_, 1, v___x_1037_);
v___x_1039_ = ((lean_object*)(lp_reflect_x2dnts_NTS_instReprAct_repr___redArg___closed__5));
v___x_1040_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1040_, 0, v___x_1038_);
lean_ctor_set(v___x_1040_, 1, v___x_1039_);
v___x_1041_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1041_, 0, v___x_1040_);
lean_ctor_set(v___x_1041_, 1, v___x_1025_);
v___x_1042_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprFact_repr___redArg___closed__17, &lp_reflect_x2dnts_NTS_instReprFact_repr___redArg___closed__17_once, _init_lp_reflect_x2dnts_NTS_instReprFact_repr___redArg___closed__17);
v___x_1043_ = lp_reflect_x2dnts_Option_repr___at___00NTS_instReprAct_repr_spec__1(v_target_1021_, v___x_1028_);
v___x_1044_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1044_, 0, v___x_1042_);
lean_ctor_set(v___x_1044_, 1, v___x_1043_);
v___x_1045_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_1045_, 0, v___x_1044_);
lean_ctor_set_uint8(v___x_1045_, sizeof(void*)*1, v___x_1032_);
v___x_1046_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1046_, 0, v___x_1041_);
lean_ctor_set(v___x_1046_, 1, v___x_1045_);
v___x_1047_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__20, &lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__20_once, _init_lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__20);
v___x_1048_ = ((lean_object*)(lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__21));
v___x_1049_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1049_, 0, v___x_1048_);
lean_ctor_set(v___x_1049_, 1, v___x_1046_);
v___x_1050_ = ((lean_object*)(lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__22));
v___x_1051_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1051_, 0, v___x_1049_);
lean_ctor_set(v___x_1051_, 1, v___x_1050_);
v___x_1052_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1052_, 0, v___x_1047_);
lean_ctor_set(v___x_1052_, 1, v___x_1051_);
v___x_1053_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_1053_, 0, v___x_1052_);
lean_ctor_set_uint8(v___x_1053_, sizeof(void*)*1, v___x_1032_);
return v___x_1053_;
}
}
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instReprAct_repr(lean_object* v_x_1056_, lean_object* v_prec_1057_){
_start:
{
lean_object* v___x_1058_; 
v___x_1058_ = lp_reflect_x2dnts_NTS_instReprAct_repr___redArg(v_x_1056_);
return v___x_1058_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instReprAct_repr___boxed(lean_object* v_x_1059_, lean_object* v_prec_1060_){
_start:
{
lean_object* v_res_1061_; 
v_res_1061_ = lp_reflect_x2dnts_NTS_instReprAct_repr(v_x_1059_, v_prec_1060_);
lean_dec(v_prec_1060_);
return v_res_1061_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_List_repr___at___00NTS_instReprAct_repr_spec__0(lean_object* v_a_1062_, lean_object* v_n_1063_){
_start:
{
lean_object* v___x_1064_; 
v___x_1064_ = lp_reflect_x2dnts_List_repr___at___00NTS_instReprAct_repr_spec__0___redArg(v_a_1062_);
return v___x_1064_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_List_repr___at___00NTS_instReprAct_repr_spec__0___boxed(lean_object* v_a_1065_, lean_object* v_n_1066_){
_start:
{
lean_object* v_res_1067_; 
v_res_1067_ = lp_reflect_x2dnts_List_repr___at___00NTS_instReprAct_repr_spec__0(v_a_1065_, v_n_1066_);
lean_dec(v_n_1066_);
return v_res_1067_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_Prod_repr___at___00List_repr___at___00NTS_instReprAct_repr_spec__0_spec__0(lean_object* v_x_1068_, lean_object* v_x_1069_){
_start:
{
lean_object* v___x_1070_; 
v___x_1070_ = lp_reflect_x2dnts_Prod_repr___at___00List_repr___at___00NTS_instReprAct_repr_spec__0_spec__0___redArg(v_x_1068_);
return v___x_1070_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_Prod_repr___at___00List_repr___at___00NTS_instReprAct_repr_spec__0_spec__0___boxed(lean_object* v_x_1071_, lean_object* v_x_1072_){
_start:
{
lean_object* v_res_1073_; 
v_res_1073_ = lp_reflect_x2dnts_Prod_repr___at___00List_repr___at___00NTS_instReprAct_repr_spec__0_spec__0(v_x_1071_, v_x_1072_);
lean_dec(v_x_1072_);
return v_res_1073_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_Prod_repr___at___00Option_repr___at___00NTS_instReprAct_repr_spec__1_spec__3(lean_object* v_x_1074_, lean_object* v_x_1075_){
_start:
{
lean_object* v___x_1076_; 
v___x_1076_ = lp_reflect_x2dnts_Prod_repr___at___00Option_repr___at___00NTS_instReprAct_repr_spec__1_spec__3___redArg(v_x_1074_);
return v___x_1076_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_Prod_repr___at___00Option_repr___at___00NTS_instReprAct_repr_spec__1_spec__3___boxed(lean_object* v_x_1077_, lean_object* v_x_1078_){
_start:
{
lean_object* v_res_1079_; 
v_res_1079_ = lp_reflect_x2dnts_Prod_repr___at___00Option_repr___at___00NTS_instReprAct_repr_spec__1_spec__3(v_x_1077_, v_x_1078_);
lean_dec(v_x_1078_);
return v_res_1079_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_List_mapTR_loop___at___00NTS_Act_wellFormed_spec__0(lean_object* v_a_1082_, lean_object* v_a_1083_){
_start:
{
if (lean_obj_tag(v_a_1082_) == 0)
{
lean_object* v___x_1084_; 
v___x_1084_ = l_List_reverse___redArg(v_a_1083_);
return v___x_1084_;
}
else
{
lean_object* v_head_1085_; lean_object* v_tail_1086_; lean_object* v___x_1088_; uint8_t v_isShared_1089_; uint8_t v_isSharedCheck_1095_; 
v_head_1085_ = lean_ctor_get(v_a_1082_, 0);
v_tail_1086_ = lean_ctor_get(v_a_1082_, 1);
v_isSharedCheck_1095_ = !lean_is_exclusive(v_a_1082_);
if (v_isSharedCheck_1095_ == 0)
{
v___x_1088_ = v_a_1082_;
v_isShared_1089_ = v_isSharedCheck_1095_;
goto v_resetjp_1087_;
}
else
{
lean_inc(v_tail_1086_);
lean_inc(v_head_1085_);
lean_dec(v_a_1082_);
v___x_1088_ = lean_box(0);
v_isShared_1089_ = v_isSharedCheck_1095_;
goto v_resetjp_1087_;
}
v_resetjp_1087_:
{
lean_object* v_fst_1090_; lean_object* v___x_1092_; 
v_fst_1090_ = lean_ctor_get(v_head_1085_, 0);
lean_inc(v_fst_1090_);
lean_dec(v_head_1085_);
if (v_isShared_1089_ == 0)
{
lean_ctor_set(v___x_1088_, 1, v_a_1083_);
lean_ctor_set(v___x_1088_, 0, v_fst_1090_);
v___x_1092_ = v___x_1088_;
goto v_reusejp_1091_;
}
else
{
lean_object* v_reuseFailAlloc_1094_; 
v_reuseFailAlloc_1094_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1094_, 0, v_fst_1090_);
lean_ctor_set(v_reuseFailAlloc_1094_, 1, v_a_1083_);
v___x_1092_ = v_reuseFailAlloc_1094_;
goto v_reusejp_1091_;
}
v_reusejp_1091_:
{
v_a_1082_ = v_tail_1086_;
v_a_1083_ = v___x_1092_;
goto _start;
}
}
}
}
}
LEAN_EXPORT uint8_t lp_reflect_x2dnts_List_all___at___00NTS_Act_wellFormed_spec__1(lean_object* v_x_1096_){
_start:
{
if (lean_obj_tag(v_x_1096_) == 0)
{
uint8_t v___x_1097_; 
v___x_1097_ = 1;
return v___x_1097_;
}
else
{
lean_object* v_head_1098_; lean_object* v_tail_1099_; lean_object* v_snd_1100_; lean_object* v___x_1101_; uint8_t v___x_1102_; 
v_head_1098_ = lean_ctor_get(v_x_1096_, 0);
v_tail_1099_ = lean_ctor_get(v_x_1096_, 1);
v_snd_1100_ = lean_ctor_get(v_head_1098_, 1);
v___x_1101_ = lean_unsigned_to_nat(10u);
v___x_1102_ = lean_nat_dec_le(v_snd_1100_, v___x_1101_);
if (v___x_1102_ == 0)
{
return v___x_1102_;
}
else
{
v_x_1096_ = v_tail_1099_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_List_all___at___00NTS_Act_wellFormed_spec__1___boxed(lean_object* v_x_1104_){
_start:
{
uint8_t v_res_1105_; lean_object* v_r_1106_; 
v_res_1105_ = lp_reflect_x2dnts_List_all___at___00NTS_Act_wellFormed_spec__1(v_x_1104_);
lean_dec(v_x_1104_);
v_r_1106_ = lean_box(v_res_1105_);
return v_r_1106_;
}
}
LEAN_EXPORT uint8_t lp_reflect_x2dnts_NTS_Act_wellFormed(lean_object* v_a_1107_){
_start:
{
lean_object* v_drivers_1108_; lean_object* v___x_1109_; lean_object* v___x_1110_; lean_object* v___x_1111_; uint8_t v___x_1112_; 
v_drivers_1108_ = lean_ctor_get(v_a_1107_, 0);
lean_inc_n(v_drivers_1108_, 2);
lean_dec_ref(v_a_1107_);
v___x_1109_ = lean_alloc_closure((void*)(lp_reflect_x2dnts_NTS_instDecidableEqPrime___boxed), 2, 0);
v___x_1110_ = lean_box(0);
v___x_1111_ = lp_reflect_x2dnts_List_mapTR_loop___at___00NTS_Act_wellFormed_spec__0(v_drivers_1108_, v___x_1110_);
v___x_1112_ = l_List_nodupDecidable___redArg(v___x_1109_, v___x_1111_);
if (v___x_1112_ == 0)
{
lean_dec(v_drivers_1108_);
return v___x_1112_;
}
else
{
uint8_t v___x_1113_; 
v___x_1113_ = lp_reflect_x2dnts_List_all___at___00NTS_Act_wellFormed_spec__1(v_drivers_1108_);
lean_dec(v_drivers_1108_);
return v___x_1113_;
}
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Act_wellFormed___boxed(lean_object* v_a_1114_){
_start:
{
uint8_t v_res_1115_; lean_object* v_r_1116_; 
v_res_1115_ = lp_reflect_x2dnts_NTS_Act_wellFormed(v_a_1114_);
v_r_1116_ = lean_box(v_res_1115_);
return v_r_1116_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_RelationKind_toCtorIdx(lean_object* v_x_1117_){
_start:
{
lean_object* v___x_1118_; 
v___x_1118_ = lean_unsigned_to_nat(0u);
return v___x_1118_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_RelationKind_ofNat(lean_object* v_n_1119_){
_start:
{
lean_object* v___x_1120_; 
v___x_1120_ = lean_box(0);
return v___x_1120_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_RelationKind_ofNat___boxed(lean_object* v_n_1121_){
_start:
{
lean_object* v_res_1122_; 
v_res_1122_ = lp_reflect_x2dnts_NTS_RelationKind_ofNat(v_n_1121_);
lean_dec(v_n_1121_);
return v_res_1122_;
}
}
LEAN_EXPORT uint8_t lp_reflect_x2dnts_NTS_instDecidableEqRelationKind(lean_object* v_x_1123_, lean_object* v_y_1124_){
_start:
{
uint8_t v___x_1125_; 
v___x_1125_ = 1;
return v___x_1125_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instDecidableEqRelationKind___boxed(lean_object* v_x_1126_, lean_object* v_y_1127_){
_start:
{
uint8_t v_res_1128_; lean_object* v_r_1129_; 
v_res_1128_ = lp_reflect_x2dnts_NTS_instDecidableEqRelationKind(v_x_1126_, v_y_1127_);
v_r_1129_ = lean_box(v_res_1128_);
return v_r_1129_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instReprRelationKind_repr___redArg(lean_object* v_prec_1133_){
_start:
{
lean_object* v___y_1135_; lean_object* v___x_1141_; uint8_t v___x_1142_; 
v___x_1141_ = lean_unsigned_to_nat(1024u);
v___x_1142_ = lean_nat_dec_le(v___x_1141_, v_prec_1133_);
if (v___x_1142_ == 0)
{
lean_object* v___x_1143_; 
v___x_1143_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprStore_repr___closed__4, &lp_reflect_x2dnts_NTS_instReprStore_repr___closed__4_once, _init_lp_reflect_x2dnts_NTS_instReprStore_repr___closed__4);
v___y_1135_ = v___x_1143_;
goto v___jp_1134_;
}
else
{
lean_object* v___x_1144_; 
v___x_1144_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprStore_repr___closed__5, &lp_reflect_x2dnts_NTS_instReprStore_repr___closed__5_once, _init_lp_reflect_x2dnts_NTS_instReprStore_repr___closed__5);
v___y_1135_ = v___x_1144_;
goto v___jp_1134_;
}
v___jp_1134_:
{
lean_object* v___x_1136_; lean_object* v___x_1137_; uint8_t v___x_1138_; lean_object* v___x_1139_; lean_object* v___x_1140_; 
v___x_1136_ = ((lean_object*)(lp_reflect_x2dnts_NTS_instReprRelationKind_repr___redArg___closed__1));
lean_inc(v___y_1135_);
v___x_1137_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1137_, 0, v___y_1135_);
lean_ctor_set(v___x_1137_, 1, v___x_1136_);
v___x_1138_ = 0;
v___x_1139_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_1139_, 0, v___x_1137_);
lean_ctor_set_uint8(v___x_1139_, sizeof(void*)*1, v___x_1138_);
v___x_1140_ = l_Repr_addAppParen(v___x_1139_, v_prec_1133_);
return v___x_1140_;
}
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instReprRelationKind_repr___redArg___boxed(lean_object* v_prec_1145_){
_start:
{
lean_object* v_res_1146_; 
v_res_1146_ = lp_reflect_x2dnts_NTS_instReprRelationKind_repr___redArg(v_prec_1145_);
lean_dec(v_prec_1145_);
return v_res_1146_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instReprRelationKind_repr(lean_object* v_x_1147_, lean_object* v_prec_1148_){
_start:
{
lean_object* v___x_1149_; 
v___x_1149_ = lp_reflect_x2dnts_NTS_instReprRelationKind_repr___redArg(v_prec_1148_);
return v___x_1149_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instReprRelationKind_repr___boxed(lean_object* v_x_1150_, lean_object* v_prec_1151_){
_start:
{
lean_object* v_res_1152_; 
v_res_1152_ = lp_reflect_x2dnts_NTS_instReprRelationKind_repr(v_x_1150_, v_prec_1151_);
lean_dec(v_prec_1151_);
return v_res_1152_;
}
}
static lean_object* _init_lp_reflect_x2dnts_NTS_instReprRelationship_repr___redArg___closed__6(void){
_start:
{
lean_object* v___x_1167_; lean_object* v___x_1168_; 
v___x_1167_ = lean_unsigned_to_nat(0u);
v___x_1168_ = lp_reflect_x2dnts_NTS_instReprRelationKind_repr___redArg(v___x_1167_);
return v___x_1168_;
}
}
static lean_object* _init_lp_reflect_x2dnts_NTS_instReprRelationship_repr___redArg___closed__7(void){
_start:
{
lean_object* v___x_1169_; lean_object* v___x_1170_; lean_object* v___x_1171_; 
v___x_1169_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprRelationship_repr___redArg___closed__6, &lp_reflect_x2dnts_NTS_instReprRelationship_repr___redArg___closed__6_once, _init_lp_reflect_x2dnts_NTS_instReprRelationship_repr___redArg___closed__6);
v___x_1170_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprBelief_repr___redArg___closed__4, &lp_reflect_x2dnts_NTS_instReprBelief_repr___redArg___closed__4_once, _init_lp_reflect_x2dnts_NTS_instReprBelief_repr___redArg___closed__4);
v___x_1171_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1171_, 0, v___x_1170_);
lean_ctor_set(v___x_1171_, 1, v___x_1169_);
return v___x_1171_;
}
}
static lean_object* _init_lp_reflect_x2dnts_NTS_instReprRelationship_repr___redArg___closed__8(void){
_start:
{
uint8_t v___x_1172_; lean_object* v___x_1173_; lean_object* v___x_1174_; 
v___x_1172_ = 0;
v___x_1173_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprRelationship_repr___redArg___closed__7, &lp_reflect_x2dnts_NTS_instReprRelationship_repr___redArg___closed__7_once, _init_lp_reflect_x2dnts_NTS_instReprRelationship_repr___redArg___closed__7);
v___x_1174_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_1174_, 0, v___x_1173_);
lean_ctor_set_uint8(v___x_1174_, sizeof(void*)*1, v___x_1172_);
return v___x_1174_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instReprRelationship_repr___redArg(lean_object* v_x_1175_){
_start:
{
lean_object* v_name_1176_; lean_object* v_validFrom_1177_; lean_object* v_validTo_1178_; lean_object* v_recordedAt_1179_; lean_object* v_source_1180_; lean_object* v___x_1181_; lean_object* v___x_1182_; lean_object* v___x_1183_; lean_object* v___x_1184_; lean_object* v___x_1185_; lean_object* v___x_1186_; uint8_t v___x_1187_; lean_object* v___x_1188_; lean_object* v___x_1189_; lean_object* v___x_1190_; lean_object* v___x_1191_; lean_object* v___x_1192_; lean_object* v___x_1193_; lean_object* v___x_1194_; lean_object* v___x_1195_; lean_object* v___x_1196_; lean_object* v___x_1197_; lean_object* v___x_1198_; lean_object* v___x_1199_; lean_object* v___x_1200_; lean_object* v___x_1201_; lean_object* v___x_1202_; lean_object* v___x_1203_; lean_object* v___x_1204_; lean_object* v___x_1205_; lean_object* v___x_1206_; lean_object* v___x_1207_; lean_object* v___x_1208_; lean_object* v___x_1209_; lean_object* v___x_1210_; lean_object* v___x_1211_; lean_object* v___x_1212_; lean_object* v___x_1213_; lean_object* v___x_1214_; lean_object* v___x_1215_; lean_object* v___x_1216_; lean_object* v___x_1217_; lean_object* v___x_1218_; lean_object* v___x_1219_; lean_object* v___x_1220_; lean_object* v___x_1221_; lean_object* v___x_1222_; lean_object* v___x_1223_; lean_object* v___x_1224_; lean_object* v___x_1225_; lean_object* v___x_1226_; lean_object* v___x_1227_; lean_object* v___x_1228_; lean_object* v___x_1229_; lean_object* v___x_1230_; lean_object* v___x_1231_; lean_object* v___x_1232_; lean_object* v___x_1233_; lean_object* v___x_1234_; lean_object* v___x_1235_; lean_object* v___x_1236_; lean_object* v___x_1237_; lean_object* v___x_1238_; lean_object* v___x_1239_; lean_object* v___x_1240_; lean_object* v___x_1241_; lean_object* v___x_1242_; lean_object* v___x_1243_; lean_object* v___x_1244_; lean_object* v___x_1245_; lean_object* v___x_1246_; lean_object* v___x_1247_; lean_object* v___x_1248_; 
v_name_1176_ = lean_ctor_get(v_x_1175_, 0);
lean_inc_ref(v_name_1176_);
v_validFrom_1177_ = lean_ctor_get(v_x_1175_, 2);
lean_inc_ref(v_validFrom_1177_);
v_validTo_1178_ = lean_ctor_get(v_x_1175_, 3);
lean_inc(v_validTo_1178_);
v_recordedAt_1179_ = lean_ctor_get(v_x_1175_, 4);
lean_inc_ref(v_recordedAt_1179_);
v_source_1180_ = lean_ctor_get(v_x_1175_, 5);
lean_inc(v_source_1180_);
lean_dec_ref(v_x_1175_);
v___x_1181_ = ((lean_object*)(lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__5));
v___x_1182_ = ((lean_object*)(lp_reflect_x2dnts_NTS_instReprRelationship_repr___redArg___closed__3));
v___x_1183_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprBelief_repr___redArg___closed__4, &lp_reflect_x2dnts_NTS_instReprBelief_repr___redArg___closed__4_once, _init_lp_reflect_x2dnts_NTS_instReprBelief_repr___redArg___closed__4);
v___x_1184_ = l_String_quote(v_name_1176_);
v___x_1185_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1185_, 0, v___x_1184_);
v___x_1186_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1186_, 0, v___x_1183_);
lean_ctor_set(v___x_1186_, 1, v___x_1185_);
v___x_1187_ = 0;
v___x_1188_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_1188_, 0, v___x_1186_);
lean_ctor_set_uint8(v___x_1188_, sizeof(void*)*1, v___x_1187_);
v___x_1189_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1189_, 0, v___x_1182_);
lean_ctor_set(v___x_1189_, 1, v___x_1188_);
v___x_1190_ = ((lean_object*)(lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__9));
v___x_1191_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1191_, 0, v___x_1189_);
lean_ctor_set(v___x_1191_, 1, v___x_1190_);
v___x_1192_ = lean_box(1);
v___x_1193_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1193_, 0, v___x_1191_);
lean_ctor_set(v___x_1193_, 1, v___x_1192_);
v___x_1194_ = ((lean_object*)(lp_reflect_x2dnts_NTS_instReprRelationship_repr___redArg___closed__5));
v___x_1195_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1195_, 0, v___x_1193_);
lean_ctor_set(v___x_1195_, 1, v___x_1194_);
v___x_1196_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1196_, 0, v___x_1195_);
lean_ctor_set(v___x_1196_, 1, v___x_1181_);
v___x_1197_ = lean_unsigned_to_nat(0u);
v___x_1198_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprRelationship_repr___redArg___closed__8, &lp_reflect_x2dnts_NTS_instReprRelationship_repr___redArg___closed__8_once, _init_lp_reflect_x2dnts_NTS_instReprRelationship_repr___redArg___closed__8);
v___x_1199_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1199_, 0, v___x_1196_);
lean_ctor_set(v___x_1199_, 1, v___x_1198_);
v___x_1200_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1200_, 0, v___x_1199_);
lean_ctor_set(v___x_1200_, 1, v___x_1190_);
v___x_1201_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1201_, 0, v___x_1200_);
lean_ctor_set(v___x_1201_, 1, v___x_1192_);
v___x_1202_ = ((lean_object*)(lp_reflect_x2dnts_NTS_instReprFact_repr___redArg___closed__8));
v___x_1203_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1203_, 0, v___x_1201_);
lean_ctor_set(v___x_1203_, 1, v___x_1202_);
v___x_1204_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1204_, 0, v___x_1203_);
lean_ctor_set(v___x_1204_, 1, v___x_1181_);
v___x_1205_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprFact_repr___redArg___closed__9, &lp_reflect_x2dnts_NTS_instReprFact_repr___redArg___closed__9_once, _init_lp_reflect_x2dnts_NTS_instReprFact_repr___redArg___closed__9);
v___x_1206_ = l_String_quote(v_validFrom_1177_);
v___x_1207_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1207_, 0, v___x_1206_);
v___x_1208_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1208_, 0, v___x_1205_);
lean_ctor_set(v___x_1208_, 1, v___x_1207_);
v___x_1209_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_1209_, 0, v___x_1208_);
lean_ctor_set_uint8(v___x_1209_, sizeof(void*)*1, v___x_1187_);
v___x_1210_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1210_, 0, v___x_1204_);
lean_ctor_set(v___x_1210_, 1, v___x_1209_);
v___x_1211_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1211_, 0, v___x_1210_);
lean_ctor_set(v___x_1211_, 1, v___x_1190_);
v___x_1212_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1212_, 0, v___x_1211_);
lean_ctor_set(v___x_1212_, 1, v___x_1192_);
v___x_1213_ = ((lean_object*)(lp_reflect_x2dnts_NTS_instReprFact_repr___redArg___closed__11));
v___x_1214_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1214_, 0, v___x_1212_);
lean_ctor_set(v___x_1214_, 1, v___x_1213_);
v___x_1215_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1215_, 0, v___x_1214_);
lean_ctor_set(v___x_1215_, 1, v___x_1181_);
v___x_1216_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__15, &lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__15_once, _init_lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__15);
v___x_1217_ = lp_reflect_x2dnts_Option_repr___at___00NTS_instReprFact_repr_spec__0(v_validTo_1178_, v___x_1197_);
v___x_1218_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1218_, 0, v___x_1216_);
lean_ctor_set(v___x_1218_, 1, v___x_1217_);
v___x_1219_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_1219_, 0, v___x_1218_);
lean_ctor_set_uint8(v___x_1219_, sizeof(void*)*1, v___x_1187_);
v___x_1220_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1220_, 0, v___x_1215_);
lean_ctor_set(v___x_1220_, 1, v___x_1219_);
v___x_1221_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1221_, 0, v___x_1220_);
lean_ctor_set(v___x_1221_, 1, v___x_1190_);
v___x_1222_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1222_, 0, v___x_1221_);
lean_ctor_set(v___x_1222_, 1, v___x_1192_);
v___x_1223_ = ((lean_object*)(lp_reflect_x2dnts_NTS_instReprFact_repr___redArg___closed__13));
v___x_1224_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1224_, 0, v___x_1222_);
lean_ctor_set(v___x_1224_, 1, v___x_1223_);
v___x_1225_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1225_, 0, v___x_1224_);
lean_ctor_set(v___x_1225_, 1, v___x_1181_);
v___x_1226_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprFact_repr___redArg___closed__14, &lp_reflect_x2dnts_NTS_instReprFact_repr___redArg___closed__14_once, _init_lp_reflect_x2dnts_NTS_instReprFact_repr___redArg___closed__14);
v___x_1227_ = l_String_quote(v_recordedAt_1179_);
v___x_1228_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1228_, 0, v___x_1227_);
v___x_1229_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1229_, 0, v___x_1226_);
lean_ctor_set(v___x_1229_, 1, v___x_1228_);
v___x_1230_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_1230_, 0, v___x_1229_);
lean_ctor_set_uint8(v___x_1230_, sizeof(void*)*1, v___x_1187_);
v___x_1231_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1231_, 0, v___x_1225_);
lean_ctor_set(v___x_1231_, 1, v___x_1230_);
v___x_1232_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1232_, 0, v___x_1231_);
lean_ctor_set(v___x_1232_, 1, v___x_1190_);
v___x_1233_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1233_, 0, v___x_1232_);
lean_ctor_set(v___x_1233_, 1, v___x_1192_);
v___x_1234_ = ((lean_object*)(lp_reflect_x2dnts_NTS_instReprFact_repr___redArg___closed__16));
v___x_1235_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1235_, 0, v___x_1233_);
lean_ctor_set(v___x_1235_, 1, v___x_1234_);
v___x_1236_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1236_, 0, v___x_1235_);
lean_ctor_set(v___x_1236_, 1, v___x_1181_);
v___x_1237_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprFact_repr___redArg___closed__17, &lp_reflect_x2dnts_NTS_instReprFact_repr___redArg___closed__17_once, _init_lp_reflect_x2dnts_NTS_instReprFact_repr___redArg___closed__17);
v___x_1238_ = lp_reflect_x2dnts_NTS_instReprSource_repr(v_source_1180_, v___x_1197_);
v___x_1239_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1239_, 0, v___x_1237_);
lean_ctor_set(v___x_1239_, 1, v___x_1238_);
v___x_1240_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_1240_, 0, v___x_1239_);
lean_ctor_set_uint8(v___x_1240_, sizeof(void*)*1, v___x_1187_);
v___x_1241_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1241_, 0, v___x_1236_);
lean_ctor_set(v___x_1241_, 1, v___x_1240_);
v___x_1242_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__20, &lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__20_once, _init_lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__20);
v___x_1243_ = ((lean_object*)(lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__21));
v___x_1244_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1244_, 0, v___x_1243_);
lean_ctor_set(v___x_1244_, 1, v___x_1241_);
v___x_1245_ = ((lean_object*)(lp_reflect_x2dnts_NTS_instReprGrant_repr___redArg___closed__22));
v___x_1246_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1246_, 0, v___x_1244_);
lean_ctor_set(v___x_1246_, 1, v___x_1245_);
v___x_1247_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1247_, 0, v___x_1242_);
lean_ctor_set(v___x_1247_, 1, v___x_1246_);
v___x_1248_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_1248_, 0, v___x_1247_);
lean_ctor_set_uint8(v___x_1248_, sizeof(void*)*1, v___x_1187_);
return v___x_1248_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instReprRelationship_repr(lean_object* v_x_1249_, lean_object* v_prec_1250_){
_start:
{
lean_object* v___x_1251_; 
v___x_1251_ = lp_reflect_x2dnts_NTS_instReprRelationship_repr___redArg(v_x_1249_);
return v___x_1251_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instReprRelationship_repr___boxed(lean_object* v_x_1252_, lean_object* v_prec_1253_){
_start:
{
lean_object* v_res_1254_; 
v_res_1254_ = lp_reflect_x2dnts_NTS_instReprRelationship_repr(v_x_1252_, v_prec_1253_);
lean_dec(v_prec_1253_);
return v_res_1254_;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_reflect_x2dnts_NTS_Output(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_reflect_x2dnts_NTS_Person(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
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
