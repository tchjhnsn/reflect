// Lean compiler output
// Module: NTS.State
// Imports: public import Init public meta import Init public import NTS.Output public import NTS.Person
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
lean_object* lp_reflect_x2dnts_NTS_instReprWhose_repr(lean_object*, lean_object*);
lean_object* lp_reflect_x2dnts_Option_repr___at___00NTS_instReprFact_repr_spec__0(lean_object*, lean_object*);
lean_object* lp_reflect_x2dnts_NTS_instReprScope_repr(lean_object*, lean_object*);
lean_object* lean_string_length(lean_object*);
lean_object* lp_reflect_x2dnts_NTS_instDecidableEqPrime___boxed(lean_object*, lean_object*);
lean_object* l_List_eraseDupsBy___redArg(lean_object*, lean_object*);
uint8_t lp_reflect_x2dnts_NTS_instDecidableEqPrime(uint8_t, uint8_t);
lean_object* l_Nat_reprFast(lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lp_reflect_x2dnts_NTS_Scope_path(lean_object*);
lean_object* l_List_get_x3fInternal___redArg(lean_object*, lean_object*);
uint8_t lp_reflect_x2dnts_NTS_Prime_instDecidableEqCategory(uint8_t, uint8_t);
static const lean_string_object lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "{ "};
static const lean_object* lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__0 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__0_value;
static const lean_string_object lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "clock"};
static const lean_object* lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__1 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__1_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__1_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__2 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__2_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__2_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__3 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__3_value;
static const lean_string_object lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = " := "};
static const lean_object* lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__4 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__4_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__4_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__5 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__5_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__3_value),((lean_object*)&lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__5_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__6 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__6_value;
static lean_once_cell_t lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__7;
static const lean_string_object lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ","};
static const lean_object* lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__8 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__8_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__8_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__9 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__9_value;
static const lean_string_object lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "localDate"};
static const lean_object* lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__10 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__10_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__10_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__11 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__11_value;
static lean_once_cell_t lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__12;
static const lean_string_object lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "place"};
static const lean_object* lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__13 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__13_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__13_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__14 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__14_value;
static const lean_string_object lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "bodyState"};
static const lean_object* lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__15 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__15_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__15_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__16 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__16_value;
static const lean_string_object lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = " }"};
static const lean_object* lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__17 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__17_value;
static lean_once_cell_t lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__18_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__18;
static lean_once_cell_t lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__19_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__19;
static const lean_ctor_object lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__0_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__20 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__20_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__17_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__21 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__21_value;
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instReprAnchor_repr(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instReprAnchor_repr___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_reflect_x2dnts_NTS_instReprAnchor___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_reflect_x2dnts_NTS_instReprAnchor_repr___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_reflect_x2dnts_NTS_instReprAnchor___closed__0 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprAnchor___closed__0_value;
LEAN_EXPORT const lean_object* lp_reflect_x2dnts_NTS_instReprAnchor = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprAnchor___closed__0_value;
static const lean_string_object lp_reflect_x2dnts_NTS_instReprTimeLink_repr___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "anchor"};
static const lean_object* lp_reflect_x2dnts_NTS_instReprTimeLink_repr___redArg___closed__0 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprTimeLink_repr___redArg___closed__0_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_instReprTimeLink_repr___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_reflect_x2dnts_NTS_instReprTimeLink_repr___redArg___closed__0_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_instReprTimeLink_repr___redArg___closed__1 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprTimeLink_repr___redArg___closed__1_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_instReprTimeLink_repr___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_reflect_x2dnts_NTS_instReprTimeLink_repr___redArg___closed__1_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_instReprTimeLink_repr___redArg___closed__2 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprTimeLink_repr___redArg___closed__2_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_instReprTimeLink_repr___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_reflect_x2dnts_NTS_instReprTimeLink_repr___redArg___closed__2_value),((lean_object*)&lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__5_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_instReprTimeLink_repr___redArg___closed__3 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprTimeLink_repr___redArg___closed__3_value;
static lean_once_cell_t lp_reflect_x2dnts_NTS_instReprTimeLink_repr___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_reflect_x2dnts_NTS_instReprTimeLink_repr___redArg___closed__4;
static const lean_string_object lp_reflect_x2dnts_NTS_instReprTimeLink_repr___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "felt"};
static const lean_object* lp_reflect_x2dnts_NTS_instReprTimeLink_repr___redArg___closed__5 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprTimeLink_repr___redArg___closed__5_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_instReprTimeLink_repr___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_reflect_x2dnts_NTS_instReprTimeLink_repr___redArg___closed__5_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_instReprTimeLink_repr___redArg___closed__6 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprTimeLink_repr___redArg___closed__6_value;
static lean_once_cell_t lp_reflect_x2dnts_NTS_instReprTimeLink_repr___redArg___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_reflect_x2dnts_NTS_instReprTimeLink_repr___redArg___closed__7;
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instReprTimeLink_repr___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instReprTimeLink_repr(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instReprTimeLink_repr___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_reflect_x2dnts_NTS_instReprTimeLink___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_reflect_x2dnts_NTS_instReprTimeLink_repr___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_reflect_x2dnts_NTS_instReprTimeLink___closed__0 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprTimeLink___closed__0_value;
LEAN_EXPORT const lean_object* lp_reflect_x2dnts_NTS_instReprTimeLink = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprTimeLink___closed__0_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_Prime_Category_pair___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(1) << 1) | 1))}};
static const lean_object* lp_reflect_x2dnts_NTS_Prime_Category_pair___closed__0 = (const lean_object*)&lp_reflect_x2dnts_NTS_Prime_Category_pair___closed__0_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_Prime_Category_pair___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(2) << 1) | 1)),((lean_object*)(((size_t)(11) << 1) | 1))}};
static const lean_object* lp_reflect_x2dnts_NTS_Prime_Category_pair___closed__1 = (const lean_object*)&lp_reflect_x2dnts_NTS_Prime_Category_pair___closed__1_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_Prime_Category_pair___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(3) << 1) | 1)),((lean_object*)(((size_t)(6) << 1) | 1))}};
static const lean_object* lp_reflect_x2dnts_NTS_Prime_Category_pair___closed__2 = (const lean_object*)&lp_reflect_x2dnts_NTS_Prime_Category_pair___closed__2_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_Prime_Category_pair___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(4) << 1) | 1)),((lean_object*)(((size_t)(9) << 1) | 1))}};
static const lean_object* lp_reflect_x2dnts_NTS_Prime_Category_pair___closed__3 = (const lean_object*)&lp_reflect_x2dnts_NTS_Prime_Category_pair___closed__3_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_Prime_Category_pair___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(5) << 1) | 1)),((lean_object*)(((size_t)(10) << 1) | 1))}};
static const lean_object* lp_reflect_x2dnts_NTS_Prime_Category_pair___closed__4 = (const lean_object*)&lp_reflect_x2dnts_NTS_Prime_Category_pair___closed__4_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_Prime_Category_pair___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(7) << 1) | 1)),((lean_object*)(((size_t)(8) << 1) | 1))}};
static const lean_object* lp_reflect_x2dnts_NTS_Prime_Category_pair___closed__5 = (const lean_object*)&lp_reflect_x2dnts_NTS_Prime_Category_pair___closed__5_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_Prime_Category_pair___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(12) << 1) | 1)),((lean_object*)(((size_t)(13) << 1) | 1))}};
static const lean_object* lp_reflect_x2dnts_NTS_Prime_Category_pair___closed__6 = (const lean_object*)&lp_reflect_x2dnts_NTS_Prime_Category_pair___closed__6_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_Prime_Category_pair___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(14) << 1) | 1)),((lean_object*)(((size_t)(15) << 1) | 1))}};
static const lean_object* lp_reflect_x2dnts_NTS_Prime_Category_pair___closed__7 = (const lean_object*)&lp_reflect_x2dnts_NTS_Prime_Category_pair___closed__7_value;
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Prime_Category_pair(uint8_t);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Prime_Category_pair___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Arrangement_ctorIdx(uint8_t);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Arrangement_ctorIdx___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Arrangement_toCtorIdx(uint8_t);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Arrangement_toCtorIdx___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Arrangement_ctorElim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Arrangement_ctorElim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Arrangement_ctorElim(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Arrangement_ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Arrangement_firstLeads_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Arrangement_firstLeads_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Arrangement_firstLeads_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Arrangement_firstLeads_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Arrangement_secondLeads_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Arrangement_secondLeads_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Arrangement_secondLeads_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Arrangement_secondLeads_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Arrangement_joined_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Arrangement_joined_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Arrangement_joined_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Arrangement_joined_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Arrangement_opposed_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Arrangement_opposed_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Arrangement_opposed_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Arrangement_opposed_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_reflect_x2dnts_NTS_Arrangement_ofNat(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Arrangement_ofNat___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_reflect_x2dnts_NTS_instDecidableEqArrangement(uint8_t, uint8_t);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instDecidableEqArrangement___boxed(lean_object*, lean_object*);
static const lean_string_object lp_reflect_x2dnts_NTS_instReprArrangement_repr___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 27, .m_capacity = 27, .m_length = 26, .m_data = "NTS.Arrangement.firstLeads"};
static const lean_object* lp_reflect_x2dnts_NTS_instReprArrangement_repr___closed__0 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprArrangement_repr___closed__0_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_instReprArrangement_repr___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_reflect_x2dnts_NTS_instReprArrangement_repr___closed__0_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_instReprArrangement_repr___closed__1 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprArrangement_repr___closed__1_value;
static const lean_string_object lp_reflect_x2dnts_NTS_instReprArrangement_repr___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 28, .m_capacity = 28, .m_length = 27, .m_data = "NTS.Arrangement.secondLeads"};
static const lean_object* lp_reflect_x2dnts_NTS_instReprArrangement_repr___closed__2 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprArrangement_repr___closed__2_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_instReprArrangement_repr___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_reflect_x2dnts_NTS_instReprArrangement_repr___closed__2_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_instReprArrangement_repr___closed__3 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprArrangement_repr___closed__3_value;
static const lean_string_object lp_reflect_x2dnts_NTS_instReprArrangement_repr___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 23, .m_capacity = 23, .m_length = 22, .m_data = "NTS.Arrangement.joined"};
static const lean_object* lp_reflect_x2dnts_NTS_instReprArrangement_repr___closed__4 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprArrangement_repr___closed__4_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_instReprArrangement_repr___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_reflect_x2dnts_NTS_instReprArrangement_repr___closed__4_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_instReprArrangement_repr___closed__5 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprArrangement_repr___closed__5_value;
static const lean_string_object lp_reflect_x2dnts_NTS_instReprArrangement_repr___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 24, .m_capacity = 24, .m_length = 23, .m_data = "NTS.Arrangement.opposed"};
static const lean_object* lp_reflect_x2dnts_NTS_instReprArrangement_repr___closed__6 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprArrangement_repr___closed__6_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_instReprArrangement_repr___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_reflect_x2dnts_NTS_instReprArrangement_repr___closed__6_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_instReprArrangement_repr___closed__7 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprArrangement_repr___closed__7_value;
static lean_once_cell_t lp_reflect_x2dnts_NTS_instReprArrangement_repr___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_reflect_x2dnts_NTS_instReprArrangement_repr___closed__8;
static lean_once_cell_t lp_reflect_x2dnts_NTS_instReprArrangement_repr___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_reflect_x2dnts_NTS_instReprArrangement_repr___closed__9;
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instReprArrangement_repr(uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instReprArrangement_repr___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_reflect_x2dnts_NTS_instReprArrangement___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_reflect_x2dnts_NTS_instReprArrangement_repr___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_reflect_x2dnts_NTS_instReprArrangement___closed__0 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprArrangement___closed__0_value;
LEAN_EXPORT const lean_object* lp_reflect_x2dnts_NTS_instReprArrangement = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprArrangement___closed__0_value;
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Amplitude_ctorIdx(uint8_t);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Amplitude_ctorIdx___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Amplitude_toCtorIdx(uint8_t);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Amplitude_toCtorIdx___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Amplitude_ctorElim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Amplitude_ctorElim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Amplitude_ctorElim(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Amplitude_ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Amplitude_low_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Amplitude_low_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Amplitude_low_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Amplitude_low_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Amplitude_raised_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Amplitude_raised_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Amplitude_raised_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Amplitude_raised_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Amplitude_dominant_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Amplitude_dominant_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Amplitude_dominant_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Amplitude_dominant_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Amplitude_absorbed_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Amplitude_absorbed_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Amplitude_absorbed_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Amplitude_absorbed_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_reflect_x2dnts_NTS_Amplitude_ofNat(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Amplitude_ofNat___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_reflect_x2dnts_NTS_instDecidableEqAmplitude(uint8_t, uint8_t);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instDecidableEqAmplitude___boxed(lean_object*, lean_object*);
static const lean_string_object lp_reflect_x2dnts_NTS_instReprAmplitude_repr___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "NTS.Amplitude.low"};
static const lean_object* lp_reflect_x2dnts_NTS_instReprAmplitude_repr___closed__0 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprAmplitude_repr___closed__0_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_instReprAmplitude_repr___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_reflect_x2dnts_NTS_instReprAmplitude_repr___closed__0_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_instReprAmplitude_repr___closed__1 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprAmplitude_repr___closed__1_value;
static const lean_string_object lp_reflect_x2dnts_NTS_instReprAmplitude_repr___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = "NTS.Amplitude.raised"};
static const lean_object* lp_reflect_x2dnts_NTS_instReprAmplitude_repr___closed__2 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprAmplitude_repr___closed__2_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_instReprAmplitude_repr___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_reflect_x2dnts_NTS_instReprAmplitude_repr___closed__2_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_instReprAmplitude_repr___closed__3 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprAmplitude_repr___closed__3_value;
static const lean_string_object lp_reflect_x2dnts_NTS_instReprAmplitude_repr___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 23, .m_capacity = 23, .m_length = 22, .m_data = "NTS.Amplitude.dominant"};
static const lean_object* lp_reflect_x2dnts_NTS_instReprAmplitude_repr___closed__4 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprAmplitude_repr___closed__4_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_instReprAmplitude_repr___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_reflect_x2dnts_NTS_instReprAmplitude_repr___closed__4_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_instReprAmplitude_repr___closed__5 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprAmplitude_repr___closed__5_value;
static const lean_string_object lp_reflect_x2dnts_NTS_instReprAmplitude_repr___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 23, .m_capacity = 23, .m_length = 22, .m_data = "NTS.Amplitude.absorbed"};
static const lean_object* lp_reflect_x2dnts_NTS_instReprAmplitude_repr___closed__6 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprAmplitude_repr___closed__6_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_instReprAmplitude_repr___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_reflect_x2dnts_NTS_instReprAmplitude_repr___closed__6_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_instReprAmplitude_repr___closed__7 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprAmplitude_repr___closed__7_value;
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instReprAmplitude_repr(uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instReprAmplitude_repr___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_reflect_x2dnts_NTS_instReprAmplitude___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_reflect_x2dnts_NTS_instReprAmplitude_repr___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_reflect_x2dnts_NTS_instReprAmplitude___closed__0 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprAmplitude___closed__0_value;
LEAN_EXPORT const lean_object* lp_reflect_x2dnts_NTS_instReprAmplitude = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprAmplitude___closed__0_value;
LEAN_EXPORT uint8_t lp_reflect_x2dnts_NTS_instDecidableEqRegion_decEq(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instDecidableEqRegion_decEq___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_reflect_x2dnts_NTS_instDecidableEqRegion(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instDecidableEqRegion___boxed(lean_object*, lean_object*);
static const lean_string_object lp_reflect_x2dnts_NTS_instReprRegion_repr___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "arrangement"};
static const lean_object* lp_reflect_x2dnts_NTS_instReprRegion_repr___redArg___closed__0 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprRegion_repr___redArg___closed__0_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_instReprRegion_repr___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_reflect_x2dnts_NTS_instReprRegion_repr___redArg___closed__0_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_instReprRegion_repr___redArg___closed__1 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprRegion_repr___redArg___closed__1_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_instReprRegion_repr___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_reflect_x2dnts_NTS_instReprRegion_repr___redArg___closed__1_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_instReprRegion_repr___redArg___closed__2 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprRegion_repr___redArg___closed__2_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_instReprRegion_repr___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_reflect_x2dnts_NTS_instReprRegion_repr___redArg___closed__2_value),((lean_object*)&lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__5_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_instReprRegion_repr___redArg___closed__3 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprRegion_repr___redArg___closed__3_value;
static lean_once_cell_t lp_reflect_x2dnts_NTS_instReprRegion_repr___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_reflect_x2dnts_NTS_instReprRegion_repr___redArg___closed__4;
static const lean_string_object lp_reflect_x2dnts_NTS_instReprRegion_repr___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "amplitude"};
static const lean_object* lp_reflect_x2dnts_NTS_instReprRegion_repr___redArg___closed__5 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprRegion_repr___redArg___closed__5_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_instReprRegion_repr___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_reflect_x2dnts_NTS_instReprRegion_repr___redArg___closed__5_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_instReprRegion_repr___redArg___closed__6 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprRegion_repr___redArg___closed__6_value;
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instReprRegion_repr___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instReprRegion_repr___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instReprRegion_repr(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instReprRegion_repr___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_reflect_x2dnts_NTS_instReprRegion___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_reflect_x2dnts_NTS_instReprRegion_repr___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_reflect_x2dnts_NTS_instReprRegion___closed__0 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprRegion___closed__0_value;
LEAN_EXPORT const lean_object* lp_reflect_x2dnts_NTS_instReprRegion = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprRegion___closed__0_value;
LEAN_EXPORT uint8_t lp_reflect_x2dnts_NTS_instDecidableOk(uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instDecidableOk___boxed(lean_object*, lean_object*);
static const lean_string_object lp_reflect_x2dnts_NTS_instReprSelfTerm_repr___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_reflect_x2dnts_NTS_instReprSelfTerm_repr___redArg___closed__0 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprSelfTerm_repr___redArg___closed__0_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_instReprSelfTerm_repr___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_reflect_x2dnts_NTS_instReprSelfTerm_repr___redArg___closed__0_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_instReprSelfTerm_repr___redArg___closed__1 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprSelfTerm_repr___redArg___closed__1_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_instReprSelfTerm_repr___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_reflect_x2dnts_NTS_instReprSelfTerm_repr___redArg___closed__1_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_instReprSelfTerm_repr___redArg___closed__2 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprSelfTerm_repr___redArg___closed__2_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_instReprSelfTerm_repr___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_reflect_x2dnts_NTS_instReprSelfTerm_repr___redArg___closed__2_value),((lean_object*)&lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__5_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_instReprSelfTerm_repr___redArg___closed__3 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprSelfTerm_repr___redArg___closed__3_value;
static const lean_string_object lp_reflect_x2dnts_NTS_instReprSelfTerm_repr___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "whose"};
static const lean_object* lp_reflect_x2dnts_NTS_instReprSelfTerm_repr___redArg___closed__4 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprSelfTerm_repr___redArg___closed__4_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_instReprSelfTerm_repr___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_reflect_x2dnts_NTS_instReprSelfTerm_repr___redArg___closed__4_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_instReprSelfTerm_repr___redArg___closed__5 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprSelfTerm_repr___redArg___closed__5_value;
static const lean_string_object lp_reflect_x2dnts_NTS_instReprSelfTerm_repr___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "comparedTo"};
static const lean_object* lp_reflect_x2dnts_NTS_instReprSelfTerm_repr___redArg___closed__6 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprSelfTerm_repr___redArg___closed__6_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_instReprSelfTerm_repr___redArg___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_reflect_x2dnts_NTS_instReprSelfTerm_repr___redArg___closed__6_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_instReprSelfTerm_repr___redArg___closed__7 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprSelfTerm_repr___redArg___closed__7_value;
static lean_once_cell_t lp_reflect_x2dnts_NTS_instReprSelfTerm_repr___redArg___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_reflect_x2dnts_NTS_instReprSelfTerm_repr___redArg___closed__8;
static const lean_string_object lp_reflect_x2dnts_NTS_instReprSelfTerm_repr___redArg___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "scope"};
static const lean_object* lp_reflect_x2dnts_NTS_instReprSelfTerm_repr___redArg___closed__9 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprSelfTerm_repr___redArg___closed__9_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_instReprSelfTerm_repr___redArg___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_reflect_x2dnts_NTS_instReprSelfTerm_repr___redArg___closed__9_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_instReprSelfTerm_repr___redArg___closed__10 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprSelfTerm_repr___redArg___closed__10_value;
static const lean_string_object lp_reflect_x2dnts_NTS_instReprSelfTerm_repr___redArg___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "fact"};
static const lean_object* lp_reflect_x2dnts_NTS_instReprSelfTerm_repr___redArg___closed__11 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprSelfTerm_repr___redArg___closed__11_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_instReprSelfTerm_repr___redArg___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_reflect_x2dnts_NTS_instReprSelfTerm_repr___redArg___closed__11_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_instReprSelfTerm_repr___redArg___closed__12 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprSelfTerm_repr___redArg___closed__12_value;
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instReprSelfTerm_repr___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instReprSelfTerm_repr(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instReprSelfTerm_repr___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_reflect_x2dnts_NTS_instReprSelfTerm___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_reflect_x2dnts_NTS_instReprSelfTerm_repr___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_reflect_x2dnts_NTS_instReprSelfTerm___closed__0 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprSelfTerm___closed__0_value;
LEAN_EXPORT const lean_object* lp_reflect_x2dnts_NTS_instReprSelfTerm = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprSelfTerm___closed__0_value;
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_SelfTerm_basis(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_SelfTerm_basis___boxed(lean_object*);
static const lean_string_object lp_reflect_x2dnts_NTS_tallExample___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "tall"};
static const lean_object* lp_reflect_x2dnts_NTS_tallExample___closed__0 = (const lean_object*)&lp_reflect_x2dnts_NTS_tallExample___closed__0_value;
static const lean_string_object lp_reflect_x2dnts_NTS_tallExample___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "people at work"};
static const lean_object* lp_reflect_x2dnts_NTS_tallExample___closed__1 = (const lean_object*)&lp_reflect_x2dnts_NTS_tallExample___closed__1_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_tallExample___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)&lp_reflect_x2dnts_NTS_tallExample___closed__1_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_tallExample___closed__2 = (const lean_object*)&lp_reflect_x2dnts_NTS_tallExample___closed__2_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_tallExample___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*0 + 8, .m_other = 0, .m_tag = 0}, .m_objs = {LEAN_SCALAR_PTR_LITERAL(7, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_reflect_x2dnts_NTS_tallExample___closed__3 = (const lean_object*)&lp_reflect_x2dnts_NTS_tallExample___closed__3_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_tallExample___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)&lp_reflect_x2dnts_NTS_tallExample___closed__3_value),LEAN_SCALAR_PTR_LITERAL(11, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_reflect_x2dnts_NTS_tallExample___closed__4 = (const lean_object*)&lp_reflect_x2dnts_NTS_tallExample___closed__4_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_tallExample___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)&lp_reflect_x2dnts_NTS_tallExample___closed__4_value),LEAN_SCALAR_PTR_LITERAL(9, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_reflect_x2dnts_NTS_tallExample___closed__5 = (const lean_object*)&lp_reflect_x2dnts_NTS_tallExample___closed__5_value;
static const lean_string_object lp_reflect_x2dnts_NTS_tallExample___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "height"};
static const lean_object* lp_reflect_x2dnts_NTS_tallExample___closed__6 = (const lean_object*)&lp_reflect_x2dnts_NTS_tallExample___closed__6_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_tallExample___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)&lp_reflect_x2dnts_NTS_tallExample___closed__6_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_tallExample___closed__7 = (const lean_object*)&lp_reflect_x2dnts_NTS_tallExample___closed__7_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_tallExample___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*5 + 0, .m_other = 5, .m_tag = 0}, .m_objs = {((lean_object*)&lp_reflect_x2dnts_NTS_tallExample___closed__0_value),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_reflect_x2dnts_NTS_tallExample___closed__2_value),((lean_object*)&lp_reflect_x2dnts_NTS_tallExample___closed__5_value),((lean_object*)&lp_reflect_x2dnts_NTS_tallExample___closed__7_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_tallExample___closed__8 = (const lean_object*)&lp_reflect_x2dnts_NTS_tallExample___closed__8_value;
LEAN_EXPORT const lean_object* lp_reflect_x2dnts_NTS_tallExample = (const lean_object*)&lp_reflect_x2dnts_NTS_tallExample___closed__8_value;
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Domain_ctorIdx(uint8_t);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Domain_ctorIdx___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Domain_toCtorIdx(uint8_t);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Domain_toCtorIdx___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Domain_ctorElim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Domain_ctorElim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Domain_ctorElim(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Domain_ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Domain_matterOfFact_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Domain_matterOfFact_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Domain_matterOfFact_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Domain_matterOfFact_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Domain_rightAndWrong_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Domain_rightAndWrong_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Domain_rightAndWrong_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Domain_rightAndWrong_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Domain_beauty_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Domain_beauty_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Domain_beauty_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Domain_beauty_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Domain_sacred_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Domain_sacred_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Domain_sacred_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Domain_sacred_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Domain_howIFeel_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Domain_howIFeel_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Domain_howIFeel_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Domain_howIFeel_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_reflect_x2dnts_NTS_Domain_ofNat(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Domain_ofNat___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_reflect_x2dnts_NTS_instDecidableEqDomain(uint8_t, uint8_t);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instDecidableEqDomain___boxed(lean_object*, lean_object*);
static const lean_string_object lp_reflect_x2dnts_NTS_instReprDomain_repr___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 24, .m_capacity = 24, .m_length = 23, .m_data = "NTS.Domain.matterOfFact"};
static const lean_object* lp_reflect_x2dnts_NTS_instReprDomain_repr___closed__0 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprDomain_repr___closed__0_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_instReprDomain_repr___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_reflect_x2dnts_NTS_instReprDomain_repr___closed__0_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_instReprDomain_repr___closed__1 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprDomain_repr___closed__1_value;
static const lean_string_object lp_reflect_x2dnts_NTS_instReprDomain_repr___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 25, .m_capacity = 25, .m_length = 24, .m_data = "NTS.Domain.rightAndWrong"};
static const lean_object* lp_reflect_x2dnts_NTS_instReprDomain_repr___closed__2 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprDomain_repr___closed__2_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_instReprDomain_repr___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_reflect_x2dnts_NTS_instReprDomain_repr___closed__2_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_instReprDomain_repr___closed__3 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprDomain_repr___closed__3_value;
static const lean_string_object lp_reflect_x2dnts_NTS_instReprDomain_repr___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "NTS.Domain.beauty"};
static const lean_object* lp_reflect_x2dnts_NTS_instReprDomain_repr___closed__4 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprDomain_repr___closed__4_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_instReprDomain_repr___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_reflect_x2dnts_NTS_instReprDomain_repr___closed__4_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_instReprDomain_repr___closed__5 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprDomain_repr___closed__5_value;
static const lean_string_object lp_reflect_x2dnts_NTS_instReprDomain_repr___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "NTS.Domain.sacred"};
static const lean_object* lp_reflect_x2dnts_NTS_instReprDomain_repr___closed__6 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprDomain_repr___closed__6_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_instReprDomain_repr___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_reflect_x2dnts_NTS_instReprDomain_repr___closed__6_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_instReprDomain_repr___closed__7 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprDomain_repr___closed__7_value;
static const lean_string_object lp_reflect_x2dnts_NTS_instReprDomain_repr___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "NTS.Domain.howIFeel"};
static const lean_object* lp_reflect_x2dnts_NTS_instReprDomain_repr___closed__8 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprDomain_repr___closed__8_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_instReprDomain_repr___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_reflect_x2dnts_NTS_instReprDomain_repr___closed__8_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_instReprDomain_repr___closed__9 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprDomain_repr___closed__9_value;
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instReprDomain_repr(uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instReprDomain_repr___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_reflect_x2dnts_NTS_instReprDomain___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_reflect_x2dnts_NTS_instReprDomain_repr___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_reflect_x2dnts_NTS_instReprDomain___closed__0 = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprDomain___closed__0_value;
LEAN_EXPORT const lean_object* lp_reflect_x2dnts_NTS_instReprDomain = (const lean_object*)&lp_reflect_x2dnts_NTS_instReprDomain___closed__0_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_Domain_all___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(4) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_reflect_x2dnts_NTS_Domain_all___closed__0 = (const lean_object*)&lp_reflect_x2dnts_NTS_Domain_all___closed__0_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_Domain_all___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(3) << 1) | 1)),((lean_object*)&lp_reflect_x2dnts_NTS_Domain_all___closed__0_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_Domain_all___closed__1 = (const lean_object*)&lp_reflect_x2dnts_NTS_Domain_all___closed__1_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_Domain_all___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(2) << 1) | 1)),((lean_object*)&lp_reflect_x2dnts_NTS_Domain_all___closed__1_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_Domain_all___closed__2 = (const lean_object*)&lp_reflect_x2dnts_NTS_Domain_all___closed__2_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_Domain_all___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(1) << 1) | 1)),((lean_object*)&lp_reflect_x2dnts_NTS_Domain_all___closed__2_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_Domain_all___closed__3 = (const lean_object*)&lp_reflect_x2dnts_NTS_Domain_all___closed__3_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_Domain_all___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_reflect_x2dnts_NTS_Domain_all___closed__3_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_Domain_all___closed__4 = (const lean_object*)&lp_reflect_x2dnts_NTS_Domain_all___closed__4_value;
LEAN_EXPORT const lean_object* lp_reflect_x2dnts_NTS_Domain_all = (const lean_object*)&lp_reflect_x2dnts_NTS_Domain_all___closed__4_value;
static const lean_string_object lp_reflect_x2dnts_NTS_Domain_id___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "matter_of_fact"};
static const lean_object* lp_reflect_x2dnts_NTS_Domain_id___closed__0 = (const lean_object*)&lp_reflect_x2dnts_NTS_Domain_id___closed__0_value;
static const lean_string_object lp_reflect_x2dnts_NTS_Domain_id___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "right_and_wrong"};
static const lean_object* lp_reflect_x2dnts_NTS_Domain_id___closed__1 = (const lean_object*)&lp_reflect_x2dnts_NTS_Domain_id___closed__1_value;
static const lean_string_object lp_reflect_x2dnts_NTS_Domain_id___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "beauty"};
static const lean_object* lp_reflect_x2dnts_NTS_Domain_id___closed__2 = (const lean_object*)&lp_reflect_x2dnts_NTS_Domain_id___closed__2_value;
static const lean_string_object lp_reflect_x2dnts_NTS_Domain_id___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "sacred"};
static const lean_object* lp_reflect_x2dnts_NTS_Domain_id___closed__3 = (const lean_object*)&lp_reflect_x2dnts_NTS_Domain_id___closed__3_value;
static const lean_string_object lp_reflect_x2dnts_NTS_Domain_id___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "how_i_feel"};
static const lean_object* lp_reflect_x2dnts_NTS_Domain_id___closed__4 = (const lean_object*)&lp_reflect_x2dnts_NTS_Domain_id___closed__4_value;
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Domain_id(uint8_t);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Domain_id___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_reflect_x2dnts_NTS_Domain_governor(uint8_t);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Domain_governor___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_List_eraseDups___at___00NTS_Domain_proper_spec__0(lean_object*);
static const lean_ctor_object lp_reflect_x2dnts_NTS_Domain_proper___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(15) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_reflect_x2dnts_NTS_Domain_proper___closed__0 = (const lean_object*)&lp_reflect_x2dnts_NTS_Domain_proper___closed__0_value;
static const lean_ctor_object lp_reflect_x2dnts_NTS_Domain_proper___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(14) << 1) | 1)),((lean_object*)&lp_reflect_x2dnts_NTS_Domain_proper___closed__0_value)}};
static const lean_object* lp_reflect_x2dnts_NTS_Domain_proper___closed__1 = (const lean_object*)&lp_reflect_x2dnts_NTS_Domain_proper___closed__1_value;
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Domain_proper(uint8_t);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Domain_proper___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_reflect_x2dnts_List_elem___at___00NTS_onMismatch_spec__0(uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_List_elem___at___00NTS_onMismatch_spec__0___boxed(lean_object*, lean_object*);
static const lean_string_object lp_reflect_x2dnts_NTS_onMismatch___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = "What tells you that\?"};
static const lean_object* lp_reflect_x2dnts_NTS_onMismatch___closed__0 = (const lean_object*)&lp_reflect_x2dnts_NTS_onMismatch___closed__0_value;
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_onMismatch(uint8_t, uint8_t, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_onMismatch___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts___private_NTS_State_0__NTS_instReprDomain_repr_match__1_splitter___redArg(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts___private_NTS_State_0__NTS_instReprDomain_repr_match__1_splitter___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts___private_NTS_State_0__NTS_instReprDomain_repr_match__1_splitter(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_reflect_x2dnts___private_NTS_State_0__NTS_instReprDomain_repr_match__1_splitter___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__7(void){
_start:
{
lean_object* v___x_14_; lean_object* v___x_15_; 
v___x_14_ = lean_unsigned_to_nat(9u);
v___x_15_ = lean_nat_to_int(v___x_14_);
return v___x_15_;
}
}
static lean_object* _init_lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__12(void){
_start:
{
lean_object* v___x_22_; lean_object* v___x_23_; 
v___x_22_ = lean_unsigned_to_nat(13u);
v___x_23_ = lean_nat_to_int(v___x_22_);
return v___x_23_;
}
}
static lean_object* _init_lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__18(void){
_start:
{
lean_object* v___x_31_; lean_object* v___x_32_; 
v___x_31_ = ((lean_object*)(lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__0));
v___x_32_ = lean_string_length(v___x_31_);
return v___x_32_;
}
}
static lean_object* _init_lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__19(void){
_start:
{
lean_object* v___x_33_; lean_object* v___x_34_; 
v___x_33_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__18, &lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__18_once, _init_lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__18);
v___x_34_ = lean_nat_to_int(v___x_33_);
return v___x_34_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg(lean_object* v_x_39_){
_start:
{
lean_object* v_clock_40_; lean_object* v_localDate_41_; lean_object* v_place_42_; lean_object* v_bodyState_43_; lean_object* v___x_44_; lean_object* v___x_45_; lean_object* v___x_46_; lean_object* v___x_47_; lean_object* v___x_48_; lean_object* v___x_49_; uint8_t v___x_50_; lean_object* v___x_51_; lean_object* v___x_52_; lean_object* v___x_53_; lean_object* v___x_54_; lean_object* v___x_55_; lean_object* v___x_56_; lean_object* v___x_57_; lean_object* v___x_58_; lean_object* v___x_59_; lean_object* v___x_60_; lean_object* v___x_61_; lean_object* v___x_62_; lean_object* v___x_63_; lean_object* v___x_64_; lean_object* v___x_65_; lean_object* v___x_66_; lean_object* v___x_67_; lean_object* v___x_68_; lean_object* v___x_69_; lean_object* v___x_70_; lean_object* v___x_71_; lean_object* v___x_72_; lean_object* v___x_73_; lean_object* v___x_74_; lean_object* v___x_75_; lean_object* v___x_76_; lean_object* v___x_77_; lean_object* v___x_78_; lean_object* v___x_79_; lean_object* v___x_80_; lean_object* v___x_81_; lean_object* v___x_82_; lean_object* v___x_83_; lean_object* v___x_84_; lean_object* v___x_85_; lean_object* v___x_86_; lean_object* v___x_87_; lean_object* v___x_88_; lean_object* v___x_89_; lean_object* v___x_90_; lean_object* v___x_91_; 
v_clock_40_ = lean_ctor_get(v_x_39_, 0);
lean_inc(v_clock_40_);
v_localDate_41_ = lean_ctor_get(v_x_39_, 1);
lean_inc_ref(v_localDate_41_);
v_place_42_ = lean_ctor_get(v_x_39_, 2);
lean_inc(v_place_42_);
v_bodyState_43_ = lean_ctor_get(v_x_39_, 3);
lean_inc(v_bodyState_43_);
lean_dec_ref(v_x_39_);
v___x_44_ = ((lean_object*)(lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__5));
v___x_45_ = ((lean_object*)(lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__6));
v___x_46_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__7, &lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__7_once, _init_lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__7);
v___x_47_ = l_Nat_reprFast(v_clock_40_);
v___x_48_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_48_, 0, v___x_47_);
v___x_49_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_49_, 0, v___x_46_);
lean_ctor_set(v___x_49_, 1, v___x_48_);
v___x_50_ = 0;
v___x_51_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_51_, 0, v___x_49_);
lean_ctor_set_uint8(v___x_51_, sizeof(void*)*1, v___x_50_);
v___x_52_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_52_, 0, v___x_45_);
lean_ctor_set(v___x_52_, 1, v___x_51_);
v___x_53_ = ((lean_object*)(lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__9));
v___x_54_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_54_, 0, v___x_52_);
lean_ctor_set(v___x_54_, 1, v___x_53_);
v___x_55_ = lean_box(1);
v___x_56_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_56_, 0, v___x_54_);
lean_ctor_set(v___x_56_, 1, v___x_55_);
v___x_57_ = ((lean_object*)(lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__11));
v___x_58_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_58_, 0, v___x_56_);
lean_ctor_set(v___x_58_, 1, v___x_57_);
v___x_59_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_59_, 0, v___x_58_);
lean_ctor_set(v___x_59_, 1, v___x_44_);
v___x_60_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__12, &lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__12_once, _init_lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__12);
v___x_61_ = l_String_quote(v_localDate_41_);
v___x_62_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_62_, 0, v___x_61_);
v___x_63_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_63_, 0, v___x_60_);
lean_ctor_set(v___x_63_, 1, v___x_62_);
v___x_64_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_64_, 0, v___x_63_);
lean_ctor_set_uint8(v___x_64_, sizeof(void*)*1, v___x_50_);
v___x_65_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_65_, 0, v___x_59_);
lean_ctor_set(v___x_65_, 1, v___x_64_);
v___x_66_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_66_, 0, v___x_65_);
lean_ctor_set(v___x_66_, 1, v___x_53_);
v___x_67_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_67_, 0, v___x_66_);
lean_ctor_set(v___x_67_, 1, v___x_55_);
v___x_68_ = ((lean_object*)(lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__14));
v___x_69_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_69_, 0, v___x_67_);
lean_ctor_set(v___x_69_, 1, v___x_68_);
v___x_70_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_70_, 0, v___x_69_);
lean_ctor_set(v___x_70_, 1, v___x_44_);
v___x_71_ = lean_unsigned_to_nat(0u);
v___x_72_ = lp_reflect_x2dnts_Option_repr___at___00NTS_instReprFact_repr_spec__0(v_place_42_, v___x_71_);
v___x_73_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_73_, 0, v___x_46_);
lean_ctor_set(v___x_73_, 1, v___x_72_);
v___x_74_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_74_, 0, v___x_73_);
lean_ctor_set_uint8(v___x_74_, sizeof(void*)*1, v___x_50_);
v___x_75_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_75_, 0, v___x_70_);
lean_ctor_set(v___x_75_, 1, v___x_74_);
v___x_76_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_76_, 0, v___x_75_);
lean_ctor_set(v___x_76_, 1, v___x_53_);
v___x_77_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_77_, 0, v___x_76_);
lean_ctor_set(v___x_77_, 1, v___x_55_);
v___x_78_ = ((lean_object*)(lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__16));
v___x_79_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_79_, 0, v___x_77_);
lean_ctor_set(v___x_79_, 1, v___x_78_);
v___x_80_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_80_, 0, v___x_79_);
lean_ctor_set(v___x_80_, 1, v___x_44_);
v___x_81_ = lp_reflect_x2dnts_Option_repr___at___00NTS_instReprFact_repr_spec__0(v_bodyState_43_, v___x_71_);
v___x_82_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_82_, 0, v___x_60_);
lean_ctor_set(v___x_82_, 1, v___x_81_);
v___x_83_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_83_, 0, v___x_82_);
lean_ctor_set_uint8(v___x_83_, sizeof(void*)*1, v___x_50_);
v___x_84_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_84_, 0, v___x_80_);
lean_ctor_set(v___x_84_, 1, v___x_83_);
v___x_85_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__19, &lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__19_once, _init_lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__19);
v___x_86_ = ((lean_object*)(lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__20));
v___x_87_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_87_, 0, v___x_86_);
lean_ctor_set(v___x_87_, 1, v___x_84_);
v___x_88_ = ((lean_object*)(lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__21));
v___x_89_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_89_, 0, v___x_87_);
lean_ctor_set(v___x_89_, 1, v___x_88_);
v___x_90_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_90_, 0, v___x_85_);
lean_ctor_set(v___x_90_, 1, v___x_89_);
v___x_91_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_91_, 0, v___x_90_);
lean_ctor_set_uint8(v___x_91_, sizeof(void*)*1, v___x_50_);
return v___x_91_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instReprAnchor_repr(lean_object* v_x_92_, lean_object* v_prec_93_){
_start:
{
lean_object* v___x_94_; 
v___x_94_ = lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg(v_x_92_);
return v___x_94_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instReprAnchor_repr___boxed(lean_object* v_x_95_, lean_object* v_prec_96_){
_start:
{
lean_object* v_res_97_; 
v_res_97_ = lp_reflect_x2dnts_NTS_instReprAnchor_repr(v_x_95_, v_prec_96_);
lean_dec(v_prec_96_);
return v_res_97_;
}
}
static lean_object* _init_lp_reflect_x2dnts_NTS_instReprTimeLink_repr___redArg___closed__4(void){
_start:
{
lean_object* v___x_109_; lean_object* v___x_110_; 
v___x_109_ = lean_unsigned_to_nat(10u);
v___x_110_ = lean_nat_to_int(v___x_109_);
return v___x_110_;
}
}
static lean_object* _init_lp_reflect_x2dnts_NTS_instReprTimeLink_repr___redArg___closed__7(void){
_start:
{
lean_object* v___x_114_; lean_object* v___x_115_; 
v___x_114_ = lean_unsigned_to_nat(8u);
v___x_115_ = lean_nat_to_int(v___x_114_);
return v___x_115_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instReprTimeLink_repr___redArg(lean_object* v_x_116_){
_start:
{
lean_object* v_anchor_117_; lean_object* v_felt_118_; lean_object* v___x_120_; uint8_t v_isShared_121_; uint8_t v_isSharedCheck_152_; 
v_anchor_117_ = lean_ctor_get(v_x_116_, 0);
v_felt_118_ = lean_ctor_get(v_x_116_, 1);
v_isSharedCheck_152_ = !lean_is_exclusive(v_x_116_);
if (v_isSharedCheck_152_ == 0)
{
v___x_120_ = v_x_116_;
v_isShared_121_ = v_isSharedCheck_152_;
goto v_resetjp_119_;
}
else
{
lean_inc(v_felt_118_);
lean_inc(v_anchor_117_);
lean_dec(v_x_116_);
v___x_120_ = lean_box(0);
v_isShared_121_ = v_isSharedCheck_152_;
goto v_resetjp_119_;
}
v_resetjp_119_:
{
lean_object* v___x_122_; lean_object* v___x_123_; lean_object* v___x_124_; lean_object* v___x_125_; lean_object* v___x_126_; lean_object* v___x_128_; 
v___x_122_ = ((lean_object*)(lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__5));
v___x_123_ = ((lean_object*)(lp_reflect_x2dnts_NTS_instReprTimeLink_repr___redArg___closed__3));
v___x_124_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprTimeLink_repr___redArg___closed__4, &lp_reflect_x2dnts_NTS_instReprTimeLink_repr___redArg___closed__4_once, _init_lp_reflect_x2dnts_NTS_instReprTimeLink_repr___redArg___closed__4);
v___x_125_ = lean_unsigned_to_nat(0u);
v___x_126_ = lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg(v_anchor_117_);
if (v_isShared_121_ == 0)
{
lean_ctor_set_tag(v___x_120_, 4);
lean_ctor_set(v___x_120_, 1, v___x_126_);
lean_ctor_set(v___x_120_, 0, v___x_124_);
v___x_128_ = v___x_120_;
goto v_reusejp_127_;
}
else
{
lean_object* v_reuseFailAlloc_151_; 
v_reuseFailAlloc_151_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v_reuseFailAlloc_151_, 0, v___x_124_);
lean_ctor_set(v_reuseFailAlloc_151_, 1, v___x_126_);
v___x_128_ = v_reuseFailAlloc_151_;
goto v_reusejp_127_;
}
v_reusejp_127_:
{
uint8_t v___x_129_; lean_object* v___x_130_; lean_object* v___x_131_; lean_object* v___x_132_; lean_object* v___x_133_; lean_object* v___x_134_; lean_object* v___x_135_; lean_object* v___x_136_; lean_object* v___x_137_; lean_object* v___x_138_; lean_object* v___x_139_; lean_object* v___x_140_; lean_object* v___x_141_; lean_object* v___x_142_; lean_object* v___x_143_; lean_object* v___x_144_; lean_object* v___x_145_; lean_object* v___x_146_; lean_object* v___x_147_; lean_object* v___x_148_; lean_object* v___x_149_; lean_object* v___x_150_; 
v___x_129_ = 0;
v___x_130_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_130_, 0, v___x_128_);
lean_ctor_set_uint8(v___x_130_, sizeof(void*)*1, v___x_129_);
v___x_131_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_131_, 0, v___x_123_);
lean_ctor_set(v___x_131_, 1, v___x_130_);
v___x_132_ = ((lean_object*)(lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__9));
v___x_133_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_133_, 0, v___x_131_);
lean_ctor_set(v___x_133_, 1, v___x_132_);
v___x_134_ = lean_box(1);
v___x_135_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_135_, 0, v___x_133_);
lean_ctor_set(v___x_135_, 1, v___x_134_);
v___x_136_ = ((lean_object*)(lp_reflect_x2dnts_NTS_instReprTimeLink_repr___redArg___closed__6));
v___x_137_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_137_, 0, v___x_135_);
lean_ctor_set(v___x_137_, 1, v___x_136_);
v___x_138_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_138_, 0, v___x_137_);
lean_ctor_set(v___x_138_, 1, v___x_122_);
v___x_139_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprTimeLink_repr___redArg___closed__7, &lp_reflect_x2dnts_NTS_instReprTimeLink_repr___redArg___closed__7_once, _init_lp_reflect_x2dnts_NTS_instReprTimeLink_repr___redArg___closed__7);
v___x_140_ = lp_reflect_x2dnts_Option_repr___at___00NTS_instReprFact_repr_spec__0(v_felt_118_, v___x_125_);
v___x_141_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_141_, 0, v___x_139_);
lean_ctor_set(v___x_141_, 1, v___x_140_);
v___x_142_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_142_, 0, v___x_141_);
lean_ctor_set_uint8(v___x_142_, sizeof(void*)*1, v___x_129_);
v___x_143_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_143_, 0, v___x_138_);
lean_ctor_set(v___x_143_, 1, v___x_142_);
v___x_144_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__19, &lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__19_once, _init_lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__19);
v___x_145_ = ((lean_object*)(lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__20));
v___x_146_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_146_, 0, v___x_145_);
lean_ctor_set(v___x_146_, 1, v___x_143_);
v___x_147_ = ((lean_object*)(lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__21));
v___x_148_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_148_, 0, v___x_146_);
lean_ctor_set(v___x_148_, 1, v___x_147_);
v___x_149_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_149_, 0, v___x_144_);
lean_ctor_set(v___x_149_, 1, v___x_148_);
v___x_150_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_150_, 0, v___x_149_);
lean_ctor_set_uint8(v___x_150_, sizeof(void*)*1, v___x_129_);
return v___x_150_;
}
}
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instReprTimeLink_repr(lean_object* v_x_153_, lean_object* v_prec_154_){
_start:
{
lean_object* v___x_155_; 
v___x_155_ = lp_reflect_x2dnts_NTS_instReprTimeLink_repr___redArg(v_x_153_);
return v___x_155_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instReprTimeLink_repr___boxed(lean_object* v_x_156_, lean_object* v_prec_157_){
_start:
{
lean_object* v_res_158_; 
v_res_158_ = lp_reflect_x2dnts_NTS_instReprTimeLink_repr(v_x_156_, v_prec_157_);
lean_dec(v_prec_157_);
return v_res_158_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Prime_Category_pair(uint8_t v_x_201_){
_start:
{
switch(v_x_201_)
{
case 0:
{
lean_object* v___x_202_; 
v___x_202_ = ((lean_object*)(lp_reflect_x2dnts_NTS_Prime_Category_pair___closed__0));
return v___x_202_;
}
case 1:
{
lean_object* v___x_203_; 
v___x_203_ = ((lean_object*)(lp_reflect_x2dnts_NTS_Prime_Category_pair___closed__1));
return v___x_203_;
}
case 2:
{
lean_object* v___x_204_; 
v___x_204_ = ((lean_object*)(lp_reflect_x2dnts_NTS_Prime_Category_pair___closed__2));
return v___x_204_;
}
case 3:
{
lean_object* v___x_205_; 
v___x_205_ = ((lean_object*)(lp_reflect_x2dnts_NTS_Prime_Category_pair___closed__3));
return v___x_205_;
}
case 4:
{
lean_object* v___x_206_; 
v___x_206_ = ((lean_object*)(lp_reflect_x2dnts_NTS_Prime_Category_pair___closed__4));
return v___x_206_;
}
case 5:
{
lean_object* v___x_207_; 
v___x_207_ = ((lean_object*)(lp_reflect_x2dnts_NTS_Prime_Category_pair___closed__5));
return v___x_207_;
}
case 6:
{
lean_object* v___x_208_; 
v___x_208_ = ((lean_object*)(lp_reflect_x2dnts_NTS_Prime_Category_pair___closed__6));
return v___x_208_;
}
default: 
{
lean_object* v___x_209_; 
v___x_209_ = ((lean_object*)(lp_reflect_x2dnts_NTS_Prime_Category_pair___closed__7));
return v___x_209_;
}
}
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Prime_Category_pair___boxed(lean_object* v_x_210_){
_start:
{
uint8_t v_x_140__boxed_211_; lean_object* v_res_212_; 
v_x_140__boxed_211_ = lean_unbox(v_x_210_);
v_res_212_ = lp_reflect_x2dnts_NTS_Prime_Category_pair(v_x_140__boxed_211_);
return v_res_212_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Arrangement_ctorIdx(uint8_t v_x_213_){
_start:
{
switch(v_x_213_)
{
case 0:
{
lean_object* v___x_214_; 
v___x_214_ = lean_unsigned_to_nat(0u);
return v___x_214_;
}
case 1:
{
lean_object* v___x_215_; 
v___x_215_ = lean_unsigned_to_nat(1u);
return v___x_215_;
}
case 2:
{
lean_object* v___x_216_; 
v___x_216_ = lean_unsigned_to_nat(2u);
return v___x_216_;
}
default: 
{
lean_object* v___x_217_; 
v___x_217_ = lean_unsigned_to_nat(3u);
return v___x_217_;
}
}
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Arrangement_ctorIdx___boxed(lean_object* v_x_218_){
_start:
{
uint8_t v_x_boxed_219_; lean_object* v_res_220_; 
v_x_boxed_219_ = lean_unbox(v_x_218_);
v_res_220_ = lp_reflect_x2dnts_NTS_Arrangement_ctorIdx(v_x_boxed_219_);
return v_res_220_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Arrangement_toCtorIdx(uint8_t v_x_221_){
_start:
{
lean_object* v___x_222_; 
v___x_222_ = lp_reflect_x2dnts_NTS_Arrangement_ctorIdx(v_x_221_);
return v___x_222_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Arrangement_toCtorIdx___boxed(lean_object* v_x_223_){
_start:
{
uint8_t v_x_4__boxed_224_; lean_object* v_res_225_; 
v_x_4__boxed_224_ = lean_unbox(v_x_223_);
v_res_225_ = lp_reflect_x2dnts_NTS_Arrangement_toCtorIdx(v_x_4__boxed_224_);
return v_res_225_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Arrangement_ctorElim___redArg(lean_object* v_k_226_){
_start:
{
lean_inc(v_k_226_);
return v_k_226_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Arrangement_ctorElim___redArg___boxed(lean_object* v_k_227_){
_start:
{
lean_object* v_res_228_; 
v_res_228_ = lp_reflect_x2dnts_NTS_Arrangement_ctorElim___redArg(v_k_227_);
lean_dec(v_k_227_);
return v_res_228_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Arrangement_ctorElim(lean_object* v_motive_229_, lean_object* v_ctorIdx_230_, uint8_t v_t_231_, lean_object* v_h_232_, lean_object* v_k_233_){
_start:
{
lean_inc(v_k_233_);
return v_k_233_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Arrangement_ctorElim___boxed(lean_object* v_motive_234_, lean_object* v_ctorIdx_235_, lean_object* v_t_236_, lean_object* v_h_237_, lean_object* v_k_238_){
_start:
{
uint8_t v_t_boxed_239_; lean_object* v_res_240_; 
v_t_boxed_239_ = lean_unbox(v_t_236_);
v_res_240_ = lp_reflect_x2dnts_NTS_Arrangement_ctorElim(v_motive_234_, v_ctorIdx_235_, v_t_boxed_239_, v_h_237_, v_k_238_);
lean_dec(v_k_238_);
lean_dec(v_ctorIdx_235_);
return v_res_240_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Arrangement_firstLeads_elim___redArg(lean_object* v_firstLeads_241_){
_start:
{
lean_inc(v_firstLeads_241_);
return v_firstLeads_241_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Arrangement_firstLeads_elim___redArg___boxed(lean_object* v_firstLeads_242_){
_start:
{
lean_object* v_res_243_; 
v_res_243_ = lp_reflect_x2dnts_NTS_Arrangement_firstLeads_elim___redArg(v_firstLeads_242_);
lean_dec(v_firstLeads_242_);
return v_res_243_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Arrangement_firstLeads_elim(lean_object* v_motive_244_, uint8_t v_t_245_, lean_object* v_h_246_, lean_object* v_firstLeads_247_){
_start:
{
lean_inc(v_firstLeads_247_);
return v_firstLeads_247_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Arrangement_firstLeads_elim___boxed(lean_object* v_motive_248_, lean_object* v_t_249_, lean_object* v_h_250_, lean_object* v_firstLeads_251_){
_start:
{
uint8_t v_t_boxed_252_; lean_object* v_res_253_; 
v_t_boxed_252_ = lean_unbox(v_t_249_);
v_res_253_ = lp_reflect_x2dnts_NTS_Arrangement_firstLeads_elim(v_motive_248_, v_t_boxed_252_, v_h_250_, v_firstLeads_251_);
lean_dec(v_firstLeads_251_);
return v_res_253_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Arrangement_secondLeads_elim___redArg(lean_object* v_secondLeads_254_){
_start:
{
lean_inc(v_secondLeads_254_);
return v_secondLeads_254_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Arrangement_secondLeads_elim___redArg___boxed(lean_object* v_secondLeads_255_){
_start:
{
lean_object* v_res_256_; 
v_res_256_ = lp_reflect_x2dnts_NTS_Arrangement_secondLeads_elim___redArg(v_secondLeads_255_);
lean_dec(v_secondLeads_255_);
return v_res_256_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Arrangement_secondLeads_elim(lean_object* v_motive_257_, uint8_t v_t_258_, lean_object* v_h_259_, lean_object* v_secondLeads_260_){
_start:
{
lean_inc(v_secondLeads_260_);
return v_secondLeads_260_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Arrangement_secondLeads_elim___boxed(lean_object* v_motive_261_, lean_object* v_t_262_, lean_object* v_h_263_, lean_object* v_secondLeads_264_){
_start:
{
uint8_t v_t_boxed_265_; lean_object* v_res_266_; 
v_t_boxed_265_ = lean_unbox(v_t_262_);
v_res_266_ = lp_reflect_x2dnts_NTS_Arrangement_secondLeads_elim(v_motive_261_, v_t_boxed_265_, v_h_263_, v_secondLeads_264_);
lean_dec(v_secondLeads_264_);
return v_res_266_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Arrangement_joined_elim___redArg(lean_object* v_joined_267_){
_start:
{
lean_inc(v_joined_267_);
return v_joined_267_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Arrangement_joined_elim___redArg___boxed(lean_object* v_joined_268_){
_start:
{
lean_object* v_res_269_; 
v_res_269_ = lp_reflect_x2dnts_NTS_Arrangement_joined_elim___redArg(v_joined_268_);
lean_dec(v_joined_268_);
return v_res_269_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Arrangement_joined_elim(lean_object* v_motive_270_, uint8_t v_t_271_, lean_object* v_h_272_, lean_object* v_joined_273_){
_start:
{
lean_inc(v_joined_273_);
return v_joined_273_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Arrangement_joined_elim___boxed(lean_object* v_motive_274_, lean_object* v_t_275_, lean_object* v_h_276_, lean_object* v_joined_277_){
_start:
{
uint8_t v_t_boxed_278_; lean_object* v_res_279_; 
v_t_boxed_278_ = lean_unbox(v_t_275_);
v_res_279_ = lp_reflect_x2dnts_NTS_Arrangement_joined_elim(v_motive_274_, v_t_boxed_278_, v_h_276_, v_joined_277_);
lean_dec(v_joined_277_);
return v_res_279_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Arrangement_opposed_elim___redArg(lean_object* v_opposed_280_){
_start:
{
lean_inc(v_opposed_280_);
return v_opposed_280_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Arrangement_opposed_elim___redArg___boxed(lean_object* v_opposed_281_){
_start:
{
lean_object* v_res_282_; 
v_res_282_ = lp_reflect_x2dnts_NTS_Arrangement_opposed_elim___redArg(v_opposed_281_);
lean_dec(v_opposed_281_);
return v_res_282_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Arrangement_opposed_elim(lean_object* v_motive_283_, uint8_t v_t_284_, lean_object* v_h_285_, lean_object* v_opposed_286_){
_start:
{
lean_inc(v_opposed_286_);
return v_opposed_286_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Arrangement_opposed_elim___boxed(lean_object* v_motive_287_, lean_object* v_t_288_, lean_object* v_h_289_, lean_object* v_opposed_290_){
_start:
{
uint8_t v_t_boxed_291_; lean_object* v_res_292_; 
v_t_boxed_291_ = lean_unbox(v_t_288_);
v_res_292_ = lp_reflect_x2dnts_NTS_Arrangement_opposed_elim(v_motive_287_, v_t_boxed_291_, v_h_289_, v_opposed_290_);
lean_dec(v_opposed_290_);
return v_res_292_;
}
}
LEAN_EXPORT uint8_t lp_reflect_x2dnts_NTS_Arrangement_ofNat(lean_object* v_n_293_){
_start:
{
lean_object* v___x_294_; uint8_t v___x_295_; 
v___x_294_ = lean_unsigned_to_nat(1u);
v___x_295_ = lean_nat_dec_le(v_n_293_, v___x_294_);
if (v___x_295_ == 0)
{
lean_object* v___x_296_; uint8_t v___x_297_; 
v___x_296_ = lean_unsigned_to_nat(2u);
v___x_297_ = lean_nat_dec_le(v_n_293_, v___x_296_);
if (v___x_297_ == 0)
{
uint8_t v___x_298_; 
v___x_298_ = 3;
return v___x_298_;
}
else
{
uint8_t v___x_299_; 
v___x_299_ = 2;
return v___x_299_;
}
}
else
{
lean_object* v___x_300_; uint8_t v___x_301_; 
v___x_300_ = lean_unsigned_to_nat(0u);
v___x_301_ = lean_nat_dec_le(v_n_293_, v___x_300_);
if (v___x_301_ == 0)
{
uint8_t v___x_302_; 
v___x_302_ = 1;
return v___x_302_;
}
else
{
uint8_t v___x_303_; 
v___x_303_ = 0;
return v___x_303_;
}
}
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Arrangement_ofNat___boxed(lean_object* v_n_304_){
_start:
{
uint8_t v_res_305_; lean_object* v_r_306_; 
v_res_305_ = lp_reflect_x2dnts_NTS_Arrangement_ofNat(v_n_304_);
lean_dec(v_n_304_);
v_r_306_ = lean_box(v_res_305_);
return v_r_306_;
}
}
LEAN_EXPORT uint8_t lp_reflect_x2dnts_NTS_instDecidableEqArrangement(uint8_t v_x_307_, uint8_t v_y_308_){
_start:
{
lean_object* v___x_309_; lean_object* v___x_310_; uint8_t v___x_311_; 
v___x_309_ = lp_reflect_x2dnts_NTS_Arrangement_ctorIdx(v_x_307_);
v___x_310_ = lp_reflect_x2dnts_NTS_Arrangement_ctorIdx(v_y_308_);
v___x_311_ = lean_nat_dec_eq(v___x_309_, v___x_310_);
lean_dec(v___x_310_);
lean_dec(v___x_309_);
return v___x_311_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instDecidableEqArrangement___boxed(lean_object* v_x_312_, lean_object* v_y_313_){
_start:
{
uint8_t v_x_13__boxed_314_; uint8_t v_y_14__boxed_315_; uint8_t v_res_316_; lean_object* v_r_317_; 
v_x_13__boxed_314_ = lean_unbox(v_x_312_);
v_y_14__boxed_315_ = lean_unbox(v_y_313_);
v_res_316_ = lp_reflect_x2dnts_NTS_instDecidableEqArrangement(v_x_13__boxed_314_, v_y_14__boxed_315_);
v_r_317_ = lean_box(v_res_316_);
return v_r_317_;
}
}
static lean_object* _init_lp_reflect_x2dnts_NTS_instReprArrangement_repr___closed__8(void){
_start:
{
lean_object* v___x_330_; lean_object* v___x_331_; 
v___x_330_ = lean_unsigned_to_nat(2u);
v___x_331_ = lean_nat_to_int(v___x_330_);
return v___x_331_;
}
}
static lean_object* _init_lp_reflect_x2dnts_NTS_instReprArrangement_repr___closed__9(void){
_start:
{
lean_object* v___x_332_; lean_object* v___x_333_; 
v___x_332_ = lean_unsigned_to_nat(1u);
v___x_333_ = lean_nat_to_int(v___x_332_);
return v___x_333_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instReprArrangement_repr(uint8_t v_x_334_, lean_object* v_prec_335_){
_start:
{
lean_object* v___y_337_; lean_object* v___y_344_; lean_object* v___y_351_; lean_object* v___y_358_; 
switch(v_x_334_)
{
case 0:
{
lean_object* v___x_364_; uint8_t v___x_365_; 
v___x_364_ = lean_unsigned_to_nat(1024u);
v___x_365_ = lean_nat_dec_le(v___x_364_, v_prec_335_);
if (v___x_365_ == 0)
{
lean_object* v___x_366_; 
v___x_366_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprArrangement_repr___closed__8, &lp_reflect_x2dnts_NTS_instReprArrangement_repr___closed__8_once, _init_lp_reflect_x2dnts_NTS_instReprArrangement_repr___closed__8);
v___y_337_ = v___x_366_;
goto v___jp_336_;
}
else
{
lean_object* v___x_367_; 
v___x_367_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprArrangement_repr___closed__9, &lp_reflect_x2dnts_NTS_instReprArrangement_repr___closed__9_once, _init_lp_reflect_x2dnts_NTS_instReprArrangement_repr___closed__9);
v___y_337_ = v___x_367_;
goto v___jp_336_;
}
}
case 1:
{
lean_object* v___x_368_; uint8_t v___x_369_; 
v___x_368_ = lean_unsigned_to_nat(1024u);
v___x_369_ = lean_nat_dec_le(v___x_368_, v_prec_335_);
if (v___x_369_ == 0)
{
lean_object* v___x_370_; 
v___x_370_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprArrangement_repr___closed__8, &lp_reflect_x2dnts_NTS_instReprArrangement_repr___closed__8_once, _init_lp_reflect_x2dnts_NTS_instReprArrangement_repr___closed__8);
v___y_344_ = v___x_370_;
goto v___jp_343_;
}
else
{
lean_object* v___x_371_; 
v___x_371_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprArrangement_repr___closed__9, &lp_reflect_x2dnts_NTS_instReprArrangement_repr___closed__9_once, _init_lp_reflect_x2dnts_NTS_instReprArrangement_repr___closed__9);
v___y_344_ = v___x_371_;
goto v___jp_343_;
}
}
case 2:
{
lean_object* v___x_372_; uint8_t v___x_373_; 
v___x_372_ = lean_unsigned_to_nat(1024u);
v___x_373_ = lean_nat_dec_le(v___x_372_, v_prec_335_);
if (v___x_373_ == 0)
{
lean_object* v___x_374_; 
v___x_374_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprArrangement_repr___closed__8, &lp_reflect_x2dnts_NTS_instReprArrangement_repr___closed__8_once, _init_lp_reflect_x2dnts_NTS_instReprArrangement_repr___closed__8);
v___y_351_ = v___x_374_;
goto v___jp_350_;
}
else
{
lean_object* v___x_375_; 
v___x_375_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprArrangement_repr___closed__9, &lp_reflect_x2dnts_NTS_instReprArrangement_repr___closed__9_once, _init_lp_reflect_x2dnts_NTS_instReprArrangement_repr___closed__9);
v___y_351_ = v___x_375_;
goto v___jp_350_;
}
}
default: 
{
lean_object* v___x_376_; uint8_t v___x_377_; 
v___x_376_ = lean_unsigned_to_nat(1024u);
v___x_377_ = lean_nat_dec_le(v___x_376_, v_prec_335_);
if (v___x_377_ == 0)
{
lean_object* v___x_378_; 
v___x_378_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprArrangement_repr___closed__8, &lp_reflect_x2dnts_NTS_instReprArrangement_repr___closed__8_once, _init_lp_reflect_x2dnts_NTS_instReprArrangement_repr___closed__8);
v___y_358_ = v___x_378_;
goto v___jp_357_;
}
else
{
lean_object* v___x_379_; 
v___x_379_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprArrangement_repr___closed__9, &lp_reflect_x2dnts_NTS_instReprArrangement_repr___closed__9_once, _init_lp_reflect_x2dnts_NTS_instReprArrangement_repr___closed__9);
v___y_358_ = v___x_379_;
goto v___jp_357_;
}
}
}
v___jp_336_:
{
lean_object* v___x_338_; lean_object* v___x_339_; uint8_t v___x_340_; lean_object* v___x_341_; lean_object* v___x_342_; 
v___x_338_ = ((lean_object*)(lp_reflect_x2dnts_NTS_instReprArrangement_repr___closed__1));
lean_inc(v___y_337_);
v___x_339_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_339_, 0, v___y_337_);
lean_ctor_set(v___x_339_, 1, v___x_338_);
v___x_340_ = 0;
v___x_341_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_341_, 0, v___x_339_);
lean_ctor_set_uint8(v___x_341_, sizeof(void*)*1, v___x_340_);
v___x_342_ = l_Repr_addAppParen(v___x_341_, v_prec_335_);
return v___x_342_;
}
v___jp_343_:
{
lean_object* v___x_345_; lean_object* v___x_346_; uint8_t v___x_347_; lean_object* v___x_348_; lean_object* v___x_349_; 
v___x_345_ = ((lean_object*)(lp_reflect_x2dnts_NTS_instReprArrangement_repr___closed__3));
lean_inc(v___y_344_);
v___x_346_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_346_, 0, v___y_344_);
lean_ctor_set(v___x_346_, 1, v___x_345_);
v___x_347_ = 0;
v___x_348_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_348_, 0, v___x_346_);
lean_ctor_set_uint8(v___x_348_, sizeof(void*)*1, v___x_347_);
v___x_349_ = l_Repr_addAppParen(v___x_348_, v_prec_335_);
return v___x_349_;
}
v___jp_350_:
{
lean_object* v___x_352_; lean_object* v___x_353_; uint8_t v___x_354_; lean_object* v___x_355_; lean_object* v___x_356_; 
v___x_352_ = ((lean_object*)(lp_reflect_x2dnts_NTS_instReprArrangement_repr___closed__5));
lean_inc(v___y_351_);
v___x_353_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_353_, 0, v___y_351_);
lean_ctor_set(v___x_353_, 1, v___x_352_);
v___x_354_ = 0;
v___x_355_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_355_, 0, v___x_353_);
lean_ctor_set_uint8(v___x_355_, sizeof(void*)*1, v___x_354_);
v___x_356_ = l_Repr_addAppParen(v___x_355_, v_prec_335_);
return v___x_356_;
}
v___jp_357_:
{
lean_object* v___x_359_; lean_object* v___x_360_; uint8_t v___x_361_; lean_object* v___x_362_; lean_object* v___x_363_; 
v___x_359_ = ((lean_object*)(lp_reflect_x2dnts_NTS_instReprArrangement_repr___closed__7));
lean_inc(v___y_358_);
v___x_360_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_360_, 0, v___y_358_);
lean_ctor_set(v___x_360_, 1, v___x_359_);
v___x_361_ = 0;
v___x_362_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_362_, 0, v___x_360_);
lean_ctor_set_uint8(v___x_362_, sizeof(void*)*1, v___x_361_);
v___x_363_ = l_Repr_addAppParen(v___x_362_, v_prec_335_);
return v___x_363_;
}
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instReprArrangement_repr___boxed(lean_object* v_x_380_, lean_object* v_prec_381_){
_start:
{
uint8_t v_x_233__boxed_382_; lean_object* v_res_383_; 
v_x_233__boxed_382_ = lean_unbox(v_x_380_);
v_res_383_ = lp_reflect_x2dnts_NTS_instReprArrangement_repr(v_x_233__boxed_382_, v_prec_381_);
lean_dec(v_prec_381_);
return v_res_383_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Amplitude_ctorIdx(uint8_t v_x_386_){
_start:
{
switch(v_x_386_)
{
case 0:
{
lean_object* v___x_387_; 
v___x_387_ = lean_unsigned_to_nat(0u);
return v___x_387_;
}
case 1:
{
lean_object* v___x_388_; 
v___x_388_ = lean_unsigned_to_nat(1u);
return v___x_388_;
}
case 2:
{
lean_object* v___x_389_; 
v___x_389_ = lean_unsigned_to_nat(2u);
return v___x_389_;
}
default: 
{
lean_object* v___x_390_; 
v___x_390_ = lean_unsigned_to_nat(3u);
return v___x_390_;
}
}
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Amplitude_ctorIdx___boxed(lean_object* v_x_391_){
_start:
{
uint8_t v_x_boxed_392_; lean_object* v_res_393_; 
v_x_boxed_392_ = lean_unbox(v_x_391_);
v_res_393_ = lp_reflect_x2dnts_NTS_Amplitude_ctorIdx(v_x_boxed_392_);
return v_res_393_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Amplitude_toCtorIdx(uint8_t v_x_394_){
_start:
{
lean_object* v___x_395_; 
v___x_395_ = lp_reflect_x2dnts_NTS_Amplitude_ctorIdx(v_x_394_);
return v___x_395_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Amplitude_toCtorIdx___boxed(lean_object* v_x_396_){
_start:
{
uint8_t v_x_4__boxed_397_; lean_object* v_res_398_; 
v_x_4__boxed_397_ = lean_unbox(v_x_396_);
v_res_398_ = lp_reflect_x2dnts_NTS_Amplitude_toCtorIdx(v_x_4__boxed_397_);
return v_res_398_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Amplitude_ctorElim___redArg(lean_object* v_k_399_){
_start:
{
lean_inc(v_k_399_);
return v_k_399_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Amplitude_ctorElim___redArg___boxed(lean_object* v_k_400_){
_start:
{
lean_object* v_res_401_; 
v_res_401_ = lp_reflect_x2dnts_NTS_Amplitude_ctorElim___redArg(v_k_400_);
lean_dec(v_k_400_);
return v_res_401_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Amplitude_ctorElim(lean_object* v_motive_402_, lean_object* v_ctorIdx_403_, uint8_t v_t_404_, lean_object* v_h_405_, lean_object* v_k_406_){
_start:
{
lean_inc(v_k_406_);
return v_k_406_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Amplitude_ctorElim___boxed(lean_object* v_motive_407_, lean_object* v_ctorIdx_408_, lean_object* v_t_409_, lean_object* v_h_410_, lean_object* v_k_411_){
_start:
{
uint8_t v_t_boxed_412_; lean_object* v_res_413_; 
v_t_boxed_412_ = lean_unbox(v_t_409_);
v_res_413_ = lp_reflect_x2dnts_NTS_Amplitude_ctorElim(v_motive_407_, v_ctorIdx_408_, v_t_boxed_412_, v_h_410_, v_k_411_);
lean_dec(v_k_411_);
lean_dec(v_ctorIdx_408_);
return v_res_413_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Amplitude_low_elim___redArg(lean_object* v_low_414_){
_start:
{
lean_inc(v_low_414_);
return v_low_414_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Amplitude_low_elim___redArg___boxed(lean_object* v_low_415_){
_start:
{
lean_object* v_res_416_; 
v_res_416_ = lp_reflect_x2dnts_NTS_Amplitude_low_elim___redArg(v_low_415_);
lean_dec(v_low_415_);
return v_res_416_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Amplitude_low_elim(lean_object* v_motive_417_, uint8_t v_t_418_, lean_object* v_h_419_, lean_object* v_low_420_){
_start:
{
lean_inc(v_low_420_);
return v_low_420_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Amplitude_low_elim___boxed(lean_object* v_motive_421_, lean_object* v_t_422_, lean_object* v_h_423_, lean_object* v_low_424_){
_start:
{
uint8_t v_t_boxed_425_; lean_object* v_res_426_; 
v_t_boxed_425_ = lean_unbox(v_t_422_);
v_res_426_ = lp_reflect_x2dnts_NTS_Amplitude_low_elim(v_motive_421_, v_t_boxed_425_, v_h_423_, v_low_424_);
lean_dec(v_low_424_);
return v_res_426_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Amplitude_raised_elim___redArg(lean_object* v_raised_427_){
_start:
{
lean_inc(v_raised_427_);
return v_raised_427_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Amplitude_raised_elim___redArg___boxed(lean_object* v_raised_428_){
_start:
{
lean_object* v_res_429_; 
v_res_429_ = lp_reflect_x2dnts_NTS_Amplitude_raised_elim___redArg(v_raised_428_);
lean_dec(v_raised_428_);
return v_res_429_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Amplitude_raised_elim(lean_object* v_motive_430_, uint8_t v_t_431_, lean_object* v_h_432_, lean_object* v_raised_433_){
_start:
{
lean_inc(v_raised_433_);
return v_raised_433_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Amplitude_raised_elim___boxed(lean_object* v_motive_434_, lean_object* v_t_435_, lean_object* v_h_436_, lean_object* v_raised_437_){
_start:
{
uint8_t v_t_boxed_438_; lean_object* v_res_439_; 
v_t_boxed_438_ = lean_unbox(v_t_435_);
v_res_439_ = lp_reflect_x2dnts_NTS_Amplitude_raised_elim(v_motive_434_, v_t_boxed_438_, v_h_436_, v_raised_437_);
lean_dec(v_raised_437_);
return v_res_439_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Amplitude_dominant_elim___redArg(lean_object* v_dominant_440_){
_start:
{
lean_inc(v_dominant_440_);
return v_dominant_440_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Amplitude_dominant_elim___redArg___boxed(lean_object* v_dominant_441_){
_start:
{
lean_object* v_res_442_; 
v_res_442_ = lp_reflect_x2dnts_NTS_Amplitude_dominant_elim___redArg(v_dominant_441_);
lean_dec(v_dominant_441_);
return v_res_442_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Amplitude_dominant_elim(lean_object* v_motive_443_, uint8_t v_t_444_, lean_object* v_h_445_, lean_object* v_dominant_446_){
_start:
{
lean_inc(v_dominant_446_);
return v_dominant_446_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Amplitude_dominant_elim___boxed(lean_object* v_motive_447_, lean_object* v_t_448_, lean_object* v_h_449_, lean_object* v_dominant_450_){
_start:
{
uint8_t v_t_boxed_451_; lean_object* v_res_452_; 
v_t_boxed_451_ = lean_unbox(v_t_448_);
v_res_452_ = lp_reflect_x2dnts_NTS_Amplitude_dominant_elim(v_motive_447_, v_t_boxed_451_, v_h_449_, v_dominant_450_);
lean_dec(v_dominant_450_);
return v_res_452_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Amplitude_absorbed_elim___redArg(lean_object* v_absorbed_453_){
_start:
{
lean_inc(v_absorbed_453_);
return v_absorbed_453_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Amplitude_absorbed_elim___redArg___boxed(lean_object* v_absorbed_454_){
_start:
{
lean_object* v_res_455_; 
v_res_455_ = lp_reflect_x2dnts_NTS_Amplitude_absorbed_elim___redArg(v_absorbed_454_);
lean_dec(v_absorbed_454_);
return v_res_455_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Amplitude_absorbed_elim(lean_object* v_motive_456_, uint8_t v_t_457_, lean_object* v_h_458_, lean_object* v_absorbed_459_){
_start:
{
lean_inc(v_absorbed_459_);
return v_absorbed_459_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Amplitude_absorbed_elim___boxed(lean_object* v_motive_460_, lean_object* v_t_461_, lean_object* v_h_462_, lean_object* v_absorbed_463_){
_start:
{
uint8_t v_t_boxed_464_; lean_object* v_res_465_; 
v_t_boxed_464_ = lean_unbox(v_t_461_);
v_res_465_ = lp_reflect_x2dnts_NTS_Amplitude_absorbed_elim(v_motive_460_, v_t_boxed_464_, v_h_462_, v_absorbed_463_);
lean_dec(v_absorbed_463_);
return v_res_465_;
}
}
LEAN_EXPORT uint8_t lp_reflect_x2dnts_NTS_Amplitude_ofNat(lean_object* v_n_466_){
_start:
{
lean_object* v___x_467_; uint8_t v___x_468_; 
v___x_467_ = lean_unsigned_to_nat(1u);
v___x_468_ = lean_nat_dec_le(v_n_466_, v___x_467_);
if (v___x_468_ == 0)
{
lean_object* v___x_469_; uint8_t v___x_470_; 
v___x_469_ = lean_unsigned_to_nat(2u);
v___x_470_ = lean_nat_dec_le(v_n_466_, v___x_469_);
if (v___x_470_ == 0)
{
uint8_t v___x_471_; 
v___x_471_ = 3;
return v___x_471_;
}
else
{
uint8_t v___x_472_; 
v___x_472_ = 2;
return v___x_472_;
}
}
else
{
lean_object* v___x_473_; uint8_t v___x_474_; 
v___x_473_ = lean_unsigned_to_nat(0u);
v___x_474_ = lean_nat_dec_le(v_n_466_, v___x_473_);
if (v___x_474_ == 0)
{
uint8_t v___x_475_; 
v___x_475_ = 1;
return v___x_475_;
}
else
{
uint8_t v___x_476_; 
v___x_476_ = 0;
return v___x_476_;
}
}
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Amplitude_ofNat___boxed(lean_object* v_n_477_){
_start:
{
uint8_t v_res_478_; lean_object* v_r_479_; 
v_res_478_ = lp_reflect_x2dnts_NTS_Amplitude_ofNat(v_n_477_);
lean_dec(v_n_477_);
v_r_479_ = lean_box(v_res_478_);
return v_r_479_;
}
}
LEAN_EXPORT uint8_t lp_reflect_x2dnts_NTS_instDecidableEqAmplitude(uint8_t v_x_480_, uint8_t v_y_481_){
_start:
{
lean_object* v___x_482_; lean_object* v___x_483_; uint8_t v___x_484_; 
v___x_482_ = lp_reflect_x2dnts_NTS_Amplitude_ctorIdx(v_x_480_);
v___x_483_ = lp_reflect_x2dnts_NTS_Amplitude_ctorIdx(v_y_481_);
v___x_484_ = lean_nat_dec_eq(v___x_482_, v___x_483_);
lean_dec(v___x_483_);
lean_dec(v___x_482_);
return v___x_484_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instDecidableEqAmplitude___boxed(lean_object* v_x_485_, lean_object* v_y_486_){
_start:
{
uint8_t v_x_13__boxed_487_; uint8_t v_y_14__boxed_488_; uint8_t v_res_489_; lean_object* v_r_490_; 
v_x_13__boxed_487_ = lean_unbox(v_x_485_);
v_y_14__boxed_488_ = lean_unbox(v_y_486_);
v_res_489_ = lp_reflect_x2dnts_NTS_instDecidableEqAmplitude(v_x_13__boxed_487_, v_y_14__boxed_488_);
v_r_490_ = lean_box(v_res_489_);
return v_r_490_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instReprAmplitude_repr(uint8_t v_x_503_, lean_object* v_prec_504_){
_start:
{
lean_object* v___y_506_; lean_object* v___y_513_; lean_object* v___y_520_; lean_object* v___y_527_; 
switch(v_x_503_)
{
case 0:
{
lean_object* v___x_533_; uint8_t v___x_534_; 
v___x_533_ = lean_unsigned_to_nat(1024u);
v___x_534_ = lean_nat_dec_le(v___x_533_, v_prec_504_);
if (v___x_534_ == 0)
{
lean_object* v___x_535_; 
v___x_535_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprArrangement_repr___closed__8, &lp_reflect_x2dnts_NTS_instReprArrangement_repr___closed__8_once, _init_lp_reflect_x2dnts_NTS_instReprArrangement_repr___closed__8);
v___y_506_ = v___x_535_;
goto v___jp_505_;
}
else
{
lean_object* v___x_536_; 
v___x_536_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprArrangement_repr___closed__9, &lp_reflect_x2dnts_NTS_instReprArrangement_repr___closed__9_once, _init_lp_reflect_x2dnts_NTS_instReprArrangement_repr___closed__9);
v___y_506_ = v___x_536_;
goto v___jp_505_;
}
}
case 1:
{
lean_object* v___x_537_; uint8_t v___x_538_; 
v___x_537_ = lean_unsigned_to_nat(1024u);
v___x_538_ = lean_nat_dec_le(v___x_537_, v_prec_504_);
if (v___x_538_ == 0)
{
lean_object* v___x_539_; 
v___x_539_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprArrangement_repr___closed__8, &lp_reflect_x2dnts_NTS_instReprArrangement_repr___closed__8_once, _init_lp_reflect_x2dnts_NTS_instReprArrangement_repr___closed__8);
v___y_513_ = v___x_539_;
goto v___jp_512_;
}
else
{
lean_object* v___x_540_; 
v___x_540_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprArrangement_repr___closed__9, &lp_reflect_x2dnts_NTS_instReprArrangement_repr___closed__9_once, _init_lp_reflect_x2dnts_NTS_instReprArrangement_repr___closed__9);
v___y_513_ = v___x_540_;
goto v___jp_512_;
}
}
case 2:
{
lean_object* v___x_541_; uint8_t v___x_542_; 
v___x_541_ = lean_unsigned_to_nat(1024u);
v___x_542_ = lean_nat_dec_le(v___x_541_, v_prec_504_);
if (v___x_542_ == 0)
{
lean_object* v___x_543_; 
v___x_543_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprArrangement_repr___closed__8, &lp_reflect_x2dnts_NTS_instReprArrangement_repr___closed__8_once, _init_lp_reflect_x2dnts_NTS_instReprArrangement_repr___closed__8);
v___y_520_ = v___x_543_;
goto v___jp_519_;
}
else
{
lean_object* v___x_544_; 
v___x_544_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprArrangement_repr___closed__9, &lp_reflect_x2dnts_NTS_instReprArrangement_repr___closed__9_once, _init_lp_reflect_x2dnts_NTS_instReprArrangement_repr___closed__9);
v___y_520_ = v___x_544_;
goto v___jp_519_;
}
}
default: 
{
lean_object* v___x_545_; uint8_t v___x_546_; 
v___x_545_ = lean_unsigned_to_nat(1024u);
v___x_546_ = lean_nat_dec_le(v___x_545_, v_prec_504_);
if (v___x_546_ == 0)
{
lean_object* v___x_547_; 
v___x_547_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprArrangement_repr___closed__8, &lp_reflect_x2dnts_NTS_instReprArrangement_repr___closed__8_once, _init_lp_reflect_x2dnts_NTS_instReprArrangement_repr___closed__8);
v___y_527_ = v___x_547_;
goto v___jp_526_;
}
else
{
lean_object* v___x_548_; 
v___x_548_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprArrangement_repr___closed__9, &lp_reflect_x2dnts_NTS_instReprArrangement_repr___closed__9_once, _init_lp_reflect_x2dnts_NTS_instReprArrangement_repr___closed__9);
v___y_527_ = v___x_548_;
goto v___jp_526_;
}
}
}
v___jp_505_:
{
lean_object* v___x_507_; lean_object* v___x_508_; uint8_t v___x_509_; lean_object* v___x_510_; lean_object* v___x_511_; 
v___x_507_ = ((lean_object*)(lp_reflect_x2dnts_NTS_instReprAmplitude_repr___closed__1));
lean_inc(v___y_506_);
v___x_508_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_508_, 0, v___y_506_);
lean_ctor_set(v___x_508_, 1, v___x_507_);
v___x_509_ = 0;
v___x_510_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_510_, 0, v___x_508_);
lean_ctor_set_uint8(v___x_510_, sizeof(void*)*1, v___x_509_);
v___x_511_ = l_Repr_addAppParen(v___x_510_, v_prec_504_);
return v___x_511_;
}
v___jp_512_:
{
lean_object* v___x_514_; lean_object* v___x_515_; uint8_t v___x_516_; lean_object* v___x_517_; lean_object* v___x_518_; 
v___x_514_ = ((lean_object*)(lp_reflect_x2dnts_NTS_instReprAmplitude_repr___closed__3));
lean_inc(v___y_513_);
v___x_515_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_515_, 0, v___y_513_);
lean_ctor_set(v___x_515_, 1, v___x_514_);
v___x_516_ = 0;
v___x_517_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_517_, 0, v___x_515_);
lean_ctor_set_uint8(v___x_517_, sizeof(void*)*1, v___x_516_);
v___x_518_ = l_Repr_addAppParen(v___x_517_, v_prec_504_);
return v___x_518_;
}
v___jp_519_:
{
lean_object* v___x_521_; lean_object* v___x_522_; uint8_t v___x_523_; lean_object* v___x_524_; lean_object* v___x_525_; 
v___x_521_ = ((lean_object*)(lp_reflect_x2dnts_NTS_instReprAmplitude_repr___closed__5));
lean_inc(v___y_520_);
v___x_522_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_522_, 0, v___y_520_);
lean_ctor_set(v___x_522_, 1, v___x_521_);
v___x_523_ = 0;
v___x_524_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_524_, 0, v___x_522_);
lean_ctor_set_uint8(v___x_524_, sizeof(void*)*1, v___x_523_);
v___x_525_ = l_Repr_addAppParen(v___x_524_, v_prec_504_);
return v___x_525_;
}
v___jp_526_:
{
lean_object* v___x_528_; lean_object* v___x_529_; uint8_t v___x_530_; lean_object* v___x_531_; lean_object* v___x_532_; 
v___x_528_ = ((lean_object*)(lp_reflect_x2dnts_NTS_instReprAmplitude_repr___closed__7));
lean_inc(v___y_527_);
v___x_529_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_529_, 0, v___y_527_);
lean_ctor_set(v___x_529_, 1, v___x_528_);
v___x_530_ = 0;
v___x_531_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_531_, 0, v___x_529_);
lean_ctor_set_uint8(v___x_531_, sizeof(void*)*1, v___x_530_);
v___x_532_ = l_Repr_addAppParen(v___x_531_, v_prec_504_);
return v___x_532_;
}
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instReprAmplitude_repr___boxed(lean_object* v_x_549_, lean_object* v_prec_550_){
_start:
{
uint8_t v_x_229__boxed_551_; lean_object* v_res_552_; 
v_x_229__boxed_551_ = lean_unbox(v_x_549_);
v_res_552_ = lp_reflect_x2dnts_NTS_instReprAmplitude_repr(v_x_229__boxed_551_, v_prec_550_);
lean_dec(v_prec_550_);
return v_res_552_;
}
}
LEAN_EXPORT uint8_t lp_reflect_x2dnts_NTS_instDecidableEqRegion_decEq(lean_object* v_x_555_, lean_object* v_x_556_){
_start:
{
uint8_t v_arrangement_557_; uint8_t v_amplitude_558_; uint8_t v_arrangement_559_; uint8_t v_amplitude_560_; uint8_t v___x_561_; 
v_arrangement_557_ = lean_ctor_get_uint8(v_x_555_, 0);
v_amplitude_558_ = lean_ctor_get_uint8(v_x_555_, 1);
v_arrangement_559_ = lean_ctor_get_uint8(v_x_556_, 0);
v_amplitude_560_ = lean_ctor_get_uint8(v_x_556_, 1);
v___x_561_ = lp_reflect_x2dnts_NTS_instDecidableEqArrangement(v_arrangement_557_, v_arrangement_559_);
if (v___x_561_ == 0)
{
return v___x_561_;
}
else
{
uint8_t v___x_562_; 
v___x_562_ = lp_reflect_x2dnts_NTS_instDecidableEqAmplitude(v_amplitude_558_, v_amplitude_560_);
return v___x_562_;
}
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instDecidableEqRegion_decEq___boxed(lean_object* v_x_563_, lean_object* v_x_564_){
_start:
{
uint8_t v_res_565_; lean_object* v_r_566_; 
v_res_565_ = lp_reflect_x2dnts_NTS_instDecidableEqRegion_decEq(v_x_563_, v_x_564_);
lean_dec_ref(v_x_564_);
lean_dec_ref(v_x_563_);
v_r_566_ = lean_box(v_res_565_);
return v_r_566_;
}
}
LEAN_EXPORT uint8_t lp_reflect_x2dnts_NTS_instDecidableEqRegion(lean_object* v_x_567_, lean_object* v_x_568_){
_start:
{
uint8_t v___x_569_; 
v___x_569_ = lp_reflect_x2dnts_NTS_instDecidableEqRegion_decEq(v_x_567_, v_x_568_);
return v___x_569_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instDecidableEqRegion___boxed(lean_object* v_x_570_, lean_object* v_x_571_){
_start:
{
uint8_t v_res_572_; lean_object* v_r_573_; 
v_res_572_ = lp_reflect_x2dnts_NTS_instDecidableEqRegion(v_x_570_, v_x_571_);
lean_dec_ref(v_x_571_);
lean_dec_ref(v_x_570_);
v_r_573_ = lean_box(v_res_572_);
return v_r_573_;
}
}
static lean_object* _init_lp_reflect_x2dnts_NTS_instReprRegion_repr___redArg___closed__4(void){
_start:
{
lean_object* v___x_583_; lean_object* v___x_584_; 
v___x_583_ = lean_unsigned_to_nat(15u);
v___x_584_ = lean_nat_to_int(v___x_583_);
return v___x_584_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instReprRegion_repr___redArg(lean_object* v_x_588_){
_start:
{
uint8_t v_arrangement_589_; uint8_t v_amplitude_590_; lean_object* v___x_591_; lean_object* v___x_592_; lean_object* v___x_593_; lean_object* v___x_594_; lean_object* v___x_595_; lean_object* v___x_596_; uint8_t v___x_597_; lean_object* v___x_598_; lean_object* v___x_599_; lean_object* v___x_600_; lean_object* v___x_601_; lean_object* v___x_602_; lean_object* v___x_603_; lean_object* v___x_604_; lean_object* v___x_605_; lean_object* v___x_606_; lean_object* v___x_607_; lean_object* v___x_608_; lean_object* v___x_609_; lean_object* v___x_610_; lean_object* v___x_611_; lean_object* v___x_612_; lean_object* v___x_613_; lean_object* v___x_614_; lean_object* v___x_615_; lean_object* v___x_616_; lean_object* v___x_617_; lean_object* v___x_618_; 
v_arrangement_589_ = lean_ctor_get_uint8(v_x_588_, 0);
v_amplitude_590_ = lean_ctor_get_uint8(v_x_588_, 1);
v___x_591_ = ((lean_object*)(lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__5));
v___x_592_ = ((lean_object*)(lp_reflect_x2dnts_NTS_instReprRegion_repr___redArg___closed__3));
v___x_593_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprRegion_repr___redArg___closed__4, &lp_reflect_x2dnts_NTS_instReprRegion_repr___redArg___closed__4_once, _init_lp_reflect_x2dnts_NTS_instReprRegion_repr___redArg___closed__4);
v___x_594_ = lean_unsigned_to_nat(0u);
v___x_595_ = lp_reflect_x2dnts_NTS_instReprArrangement_repr(v_arrangement_589_, v___x_594_);
v___x_596_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_596_, 0, v___x_593_);
lean_ctor_set(v___x_596_, 1, v___x_595_);
v___x_597_ = 0;
v___x_598_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_598_, 0, v___x_596_);
lean_ctor_set_uint8(v___x_598_, sizeof(void*)*1, v___x_597_);
v___x_599_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_599_, 0, v___x_592_);
lean_ctor_set(v___x_599_, 1, v___x_598_);
v___x_600_ = ((lean_object*)(lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__9));
v___x_601_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_601_, 0, v___x_599_);
lean_ctor_set(v___x_601_, 1, v___x_600_);
v___x_602_ = lean_box(1);
v___x_603_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_603_, 0, v___x_601_);
lean_ctor_set(v___x_603_, 1, v___x_602_);
v___x_604_ = ((lean_object*)(lp_reflect_x2dnts_NTS_instReprRegion_repr___redArg___closed__6));
v___x_605_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_605_, 0, v___x_603_);
lean_ctor_set(v___x_605_, 1, v___x_604_);
v___x_606_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_606_, 0, v___x_605_);
lean_ctor_set(v___x_606_, 1, v___x_591_);
v___x_607_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__12, &lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__12_once, _init_lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__12);
v___x_608_ = lp_reflect_x2dnts_NTS_instReprAmplitude_repr(v_amplitude_590_, v___x_594_);
v___x_609_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_609_, 0, v___x_607_);
lean_ctor_set(v___x_609_, 1, v___x_608_);
v___x_610_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_610_, 0, v___x_609_);
lean_ctor_set_uint8(v___x_610_, sizeof(void*)*1, v___x_597_);
v___x_611_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_611_, 0, v___x_606_);
lean_ctor_set(v___x_611_, 1, v___x_610_);
v___x_612_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__19, &lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__19_once, _init_lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__19);
v___x_613_ = ((lean_object*)(lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__20));
v___x_614_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_614_, 0, v___x_613_);
lean_ctor_set(v___x_614_, 1, v___x_611_);
v___x_615_ = ((lean_object*)(lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__21));
v___x_616_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_616_, 0, v___x_614_);
lean_ctor_set(v___x_616_, 1, v___x_615_);
v___x_617_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_617_, 0, v___x_612_);
lean_ctor_set(v___x_617_, 1, v___x_616_);
v___x_618_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_618_, 0, v___x_617_);
lean_ctor_set_uint8(v___x_618_, sizeof(void*)*1, v___x_597_);
return v___x_618_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instReprRegion_repr___redArg___boxed(lean_object* v_x_619_){
_start:
{
lean_object* v_res_620_; 
v_res_620_ = lp_reflect_x2dnts_NTS_instReprRegion_repr___redArg(v_x_619_);
lean_dec_ref(v_x_619_);
return v_res_620_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instReprRegion_repr(lean_object* v_x_621_, lean_object* v_prec_622_){
_start:
{
lean_object* v___x_623_; 
v___x_623_ = lp_reflect_x2dnts_NTS_instReprRegion_repr___redArg(v_x_621_);
return v___x_623_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instReprRegion_repr___boxed(lean_object* v_x_624_, lean_object* v_prec_625_){
_start:
{
lean_object* v_res_626_; 
v_res_626_ = lp_reflect_x2dnts_NTS_instReprRegion_repr(v_x_624_, v_prec_625_);
lean_dec(v_prec_625_);
lean_dec_ref(v_x_624_);
return v_res_626_;
}
}
LEAN_EXPORT uint8_t lp_reflect_x2dnts_NTS_instDecidableOk(uint8_t v_c_629_, lean_object* v_r_630_){
_start:
{
uint8_t v_arrangement_631_; uint8_t v_amplitude_632_; uint8_t v___x_633_; uint8_t v___x_634_; 
v_arrangement_631_ = lean_ctor_get_uint8(v_r_630_, 0);
v_amplitude_632_ = lean_ctor_get_uint8(v_r_630_, 1);
v___x_633_ = 3;
v___x_634_ = lp_reflect_x2dnts_NTS_instDecidableEqAmplitude(v_amplitude_632_, v___x_633_);
if (v___x_634_ == 0)
{
uint8_t v___x_635_; 
v___x_635_ = 1;
return v___x_635_;
}
else
{
uint8_t v___x_636_; uint8_t v___x_637_; 
v___x_636_ = 7;
v___x_637_ = lp_reflect_x2dnts_NTS_Prime_instDecidableEqCategory(v_c_629_, v___x_636_);
if (v___x_637_ == 0)
{
return v___x_637_;
}
else
{
uint8_t v___x_638_; uint8_t v___x_639_; 
v___x_638_ = 1;
v___x_639_ = lp_reflect_x2dnts_NTS_instDecidableEqArrangement(v_arrangement_631_, v___x_638_);
return v___x_639_;
}
}
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instDecidableOk___boxed(lean_object* v_c_640_, lean_object* v_r_641_){
_start:
{
uint8_t v_c_boxed_642_; uint8_t v_res_643_; lean_object* v_r_644_; 
v_c_boxed_642_ = lean_unbox(v_c_640_);
v_res_643_ = lp_reflect_x2dnts_NTS_instDecidableOk(v_c_boxed_642_, v_r_641_);
lean_dec_ref(v_r_641_);
v_r_644_ = lean_box(v_res_643_);
return v_r_644_;
}
}
static lean_object* _init_lp_reflect_x2dnts_NTS_instReprSelfTerm_repr___redArg___closed__8(void){
_start:
{
lean_object* v___x_660_; lean_object* v___x_661_; 
v___x_660_ = lean_unsigned_to_nat(14u);
v___x_661_ = lean_nat_to_int(v___x_660_);
return v___x_661_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instReprSelfTerm_repr___redArg(lean_object* v_x_668_){
_start:
{
lean_object* v_term_669_; lean_object* v_whose_670_; lean_object* v_comparedTo_671_; lean_object* v_scope_672_; lean_object* v_fact_673_; lean_object* v___x_674_; lean_object* v___x_675_; lean_object* v___x_676_; lean_object* v___x_677_; lean_object* v___x_678_; lean_object* v___x_679_; uint8_t v___x_680_; lean_object* v___x_681_; lean_object* v___x_682_; lean_object* v___x_683_; lean_object* v___x_684_; lean_object* v___x_685_; lean_object* v___x_686_; lean_object* v___x_687_; lean_object* v___x_688_; lean_object* v___x_689_; lean_object* v___x_690_; lean_object* v___x_691_; lean_object* v___x_692_; lean_object* v___x_693_; lean_object* v___x_694_; lean_object* v___x_695_; lean_object* v___x_696_; lean_object* v___x_697_; lean_object* v___x_698_; lean_object* v___x_699_; lean_object* v___x_700_; lean_object* v___x_701_; lean_object* v___x_702_; lean_object* v___x_703_; lean_object* v___x_704_; lean_object* v___x_705_; lean_object* v___x_706_; lean_object* v___x_707_; lean_object* v___x_708_; lean_object* v___x_709_; lean_object* v___x_710_; lean_object* v___x_711_; lean_object* v___x_712_; lean_object* v___x_713_; lean_object* v___x_714_; lean_object* v___x_715_; lean_object* v___x_716_; lean_object* v___x_717_; lean_object* v___x_718_; lean_object* v___x_719_; lean_object* v___x_720_; lean_object* v___x_721_; lean_object* v___x_722_; lean_object* v___x_723_; lean_object* v___x_724_; lean_object* v___x_725_; lean_object* v___x_726_; lean_object* v___x_727_; lean_object* v___x_728_; lean_object* v___x_729_; lean_object* v___x_730_; 
v_term_669_ = lean_ctor_get(v_x_668_, 0);
lean_inc_ref(v_term_669_);
v_whose_670_ = lean_ctor_get(v_x_668_, 1);
lean_inc(v_whose_670_);
v_comparedTo_671_ = lean_ctor_get(v_x_668_, 2);
lean_inc(v_comparedTo_671_);
v_scope_672_ = lean_ctor_get(v_x_668_, 3);
lean_inc_ref(v_scope_672_);
v_fact_673_ = lean_ctor_get(v_x_668_, 4);
lean_inc(v_fact_673_);
lean_dec_ref(v_x_668_);
v___x_674_ = ((lean_object*)(lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__5));
v___x_675_ = ((lean_object*)(lp_reflect_x2dnts_NTS_instReprSelfTerm_repr___redArg___closed__3));
v___x_676_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprTimeLink_repr___redArg___closed__7, &lp_reflect_x2dnts_NTS_instReprTimeLink_repr___redArg___closed__7_once, _init_lp_reflect_x2dnts_NTS_instReprTimeLink_repr___redArg___closed__7);
v___x_677_ = l_String_quote(v_term_669_);
v___x_678_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_678_, 0, v___x_677_);
v___x_679_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_679_, 0, v___x_676_);
lean_ctor_set(v___x_679_, 1, v___x_678_);
v___x_680_ = 0;
v___x_681_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_681_, 0, v___x_679_);
lean_ctor_set_uint8(v___x_681_, sizeof(void*)*1, v___x_680_);
v___x_682_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_682_, 0, v___x_675_);
lean_ctor_set(v___x_682_, 1, v___x_681_);
v___x_683_ = ((lean_object*)(lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__9));
v___x_684_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_684_, 0, v___x_682_);
lean_ctor_set(v___x_684_, 1, v___x_683_);
v___x_685_ = lean_box(1);
v___x_686_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_686_, 0, v___x_684_);
lean_ctor_set(v___x_686_, 1, v___x_685_);
v___x_687_ = ((lean_object*)(lp_reflect_x2dnts_NTS_instReprSelfTerm_repr___redArg___closed__5));
v___x_688_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_688_, 0, v___x_686_);
lean_ctor_set(v___x_688_, 1, v___x_687_);
v___x_689_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_689_, 0, v___x_688_);
lean_ctor_set(v___x_689_, 1, v___x_674_);
v___x_690_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__7, &lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__7_once, _init_lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__7);
v___x_691_ = lean_unsigned_to_nat(0u);
v___x_692_ = lp_reflect_x2dnts_NTS_instReprWhose_repr(v_whose_670_, v___x_691_);
v___x_693_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_693_, 0, v___x_690_);
lean_ctor_set(v___x_693_, 1, v___x_692_);
v___x_694_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_694_, 0, v___x_693_);
lean_ctor_set_uint8(v___x_694_, sizeof(void*)*1, v___x_680_);
v___x_695_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_695_, 0, v___x_689_);
lean_ctor_set(v___x_695_, 1, v___x_694_);
v___x_696_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_696_, 0, v___x_695_);
lean_ctor_set(v___x_696_, 1, v___x_683_);
v___x_697_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_697_, 0, v___x_696_);
lean_ctor_set(v___x_697_, 1, v___x_685_);
v___x_698_ = ((lean_object*)(lp_reflect_x2dnts_NTS_instReprSelfTerm_repr___redArg___closed__7));
v___x_699_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_699_, 0, v___x_697_);
lean_ctor_set(v___x_699_, 1, v___x_698_);
v___x_700_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_700_, 0, v___x_699_);
lean_ctor_set(v___x_700_, 1, v___x_674_);
v___x_701_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprSelfTerm_repr___redArg___closed__8, &lp_reflect_x2dnts_NTS_instReprSelfTerm_repr___redArg___closed__8_once, _init_lp_reflect_x2dnts_NTS_instReprSelfTerm_repr___redArg___closed__8);
v___x_702_ = lp_reflect_x2dnts_Option_repr___at___00NTS_instReprFact_repr_spec__0(v_comparedTo_671_, v___x_691_);
v___x_703_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_703_, 0, v___x_701_);
lean_ctor_set(v___x_703_, 1, v___x_702_);
v___x_704_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_704_, 0, v___x_703_);
lean_ctor_set_uint8(v___x_704_, sizeof(void*)*1, v___x_680_);
v___x_705_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_705_, 0, v___x_700_);
lean_ctor_set(v___x_705_, 1, v___x_704_);
v___x_706_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_706_, 0, v___x_705_);
lean_ctor_set(v___x_706_, 1, v___x_683_);
v___x_707_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_707_, 0, v___x_706_);
lean_ctor_set(v___x_707_, 1, v___x_685_);
v___x_708_ = ((lean_object*)(lp_reflect_x2dnts_NTS_instReprSelfTerm_repr___redArg___closed__10));
v___x_709_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_709_, 0, v___x_707_);
lean_ctor_set(v___x_709_, 1, v___x_708_);
v___x_710_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_710_, 0, v___x_709_);
lean_ctor_set(v___x_710_, 1, v___x_674_);
v___x_711_ = lp_reflect_x2dnts_NTS_instReprScope_repr(v_scope_672_, v___x_691_);
v___x_712_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_712_, 0, v___x_690_);
lean_ctor_set(v___x_712_, 1, v___x_711_);
v___x_713_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_713_, 0, v___x_712_);
lean_ctor_set_uint8(v___x_713_, sizeof(void*)*1, v___x_680_);
v___x_714_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_714_, 0, v___x_710_);
lean_ctor_set(v___x_714_, 1, v___x_713_);
v___x_715_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_715_, 0, v___x_714_);
lean_ctor_set(v___x_715_, 1, v___x_683_);
v___x_716_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_716_, 0, v___x_715_);
lean_ctor_set(v___x_716_, 1, v___x_685_);
v___x_717_ = ((lean_object*)(lp_reflect_x2dnts_NTS_instReprSelfTerm_repr___redArg___closed__12));
v___x_718_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_718_, 0, v___x_716_);
lean_ctor_set(v___x_718_, 1, v___x_717_);
v___x_719_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_719_, 0, v___x_718_);
lean_ctor_set(v___x_719_, 1, v___x_674_);
v___x_720_ = lp_reflect_x2dnts_Option_repr___at___00NTS_instReprFact_repr_spec__0(v_fact_673_, v___x_691_);
v___x_721_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_721_, 0, v___x_676_);
lean_ctor_set(v___x_721_, 1, v___x_720_);
v___x_722_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_722_, 0, v___x_721_);
lean_ctor_set_uint8(v___x_722_, sizeof(void*)*1, v___x_680_);
v___x_723_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_723_, 0, v___x_719_);
lean_ctor_set(v___x_723_, 1, v___x_722_);
v___x_724_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__19, &lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__19_once, _init_lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__19);
v___x_725_ = ((lean_object*)(lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__20));
v___x_726_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_726_, 0, v___x_725_);
lean_ctor_set(v___x_726_, 1, v___x_723_);
v___x_727_ = ((lean_object*)(lp_reflect_x2dnts_NTS_instReprAnchor_repr___redArg___closed__21));
v___x_728_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_728_, 0, v___x_726_);
lean_ctor_set(v___x_728_, 1, v___x_727_);
v___x_729_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_729_, 0, v___x_724_);
lean_ctor_set(v___x_729_, 1, v___x_728_);
v___x_730_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_730_, 0, v___x_729_);
lean_ctor_set_uint8(v___x_730_, sizeof(void*)*1, v___x_680_);
return v___x_730_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instReprSelfTerm_repr(lean_object* v_x_731_, lean_object* v_prec_732_){
_start:
{
lean_object* v___x_733_; 
v___x_733_ = lp_reflect_x2dnts_NTS_instReprSelfTerm_repr___redArg(v_x_731_);
return v___x_733_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instReprSelfTerm_repr___boxed(lean_object* v_x_734_, lean_object* v_prec_735_){
_start:
{
lean_object* v_res_736_; 
v_res_736_ = lp_reflect_x2dnts_NTS_instReprSelfTerm_repr(v_x_734_, v_prec_735_);
lean_dec(v_prec_735_);
return v_res_736_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_SelfTerm_basis(lean_object* v_t_739_){
_start:
{
lean_object* v_scope_740_; lean_object* v___x_741_; lean_object* v___x_742_; lean_object* v___x_743_; 
v_scope_740_ = lean_ctor_get(v_t_739_, 3);
v___x_741_ = lp_reflect_x2dnts_NTS_Scope_path(v_scope_740_);
v___x_742_ = lean_unsigned_to_nat(1u);
v___x_743_ = l_List_get_x3fInternal___redArg(v___x_741_, v___x_742_);
lean_dec(v___x_741_);
return v___x_743_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_SelfTerm_basis___boxed(lean_object* v_t_744_){
_start:
{
lean_object* v_res_745_; 
v_res_745_ = lp_reflect_x2dnts_NTS_SelfTerm_basis(v_t_744_);
lean_dec_ref(v_t_744_);
return v_res_745_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Domain_ctorIdx(uint8_t v_x_768_){
_start:
{
switch(v_x_768_)
{
case 0:
{
lean_object* v___x_769_; 
v___x_769_ = lean_unsigned_to_nat(0u);
return v___x_769_;
}
case 1:
{
lean_object* v___x_770_; 
v___x_770_ = lean_unsigned_to_nat(1u);
return v___x_770_;
}
case 2:
{
lean_object* v___x_771_; 
v___x_771_ = lean_unsigned_to_nat(2u);
return v___x_771_;
}
case 3:
{
lean_object* v___x_772_; 
v___x_772_ = lean_unsigned_to_nat(3u);
return v___x_772_;
}
default: 
{
lean_object* v___x_773_; 
v___x_773_ = lean_unsigned_to_nat(4u);
return v___x_773_;
}
}
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Domain_ctorIdx___boxed(lean_object* v_x_774_){
_start:
{
uint8_t v_x_boxed_775_; lean_object* v_res_776_; 
v_x_boxed_775_ = lean_unbox(v_x_774_);
v_res_776_ = lp_reflect_x2dnts_NTS_Domain_ctorIdx(v_x_boxed_775_);
return v_res_776_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Domain_toCtorIdx(uint8_t v_x_777_){
_start:
{
lean_object* v___x_778_; 
v___x_778_ = lp_reflect_x2dnts_NTS_Domain_ctorIdx(v_x_777_);
return v___x_778_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Domain_toCtorIdx___boxed(lean_object* v_x_779_){
_start:
{
uint8_t v_x_4__boxed_780_; lean_object* v_res_781_; 
v_x_4__boxed_780_ = lean_unbox(v_x_779_);
v_res_781_ = lp_reflect_x2dnts_NTS_Domain_toCtorIdx(v_x_4__boxed_780_);
return v_res_781_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Domain_ctorElim___redArg(lean_object* v_k_782_){
_start:
{
lean_inc(v_k_782_);
return v_k_782_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Domain_ctorElim___redArg___boxed(lean_object* v_k_783_){
_start:
{
lean_object* v_res_784_; 
v_res_784_ = lp_reflect_x2dnts_NTS_Domain_ctorElim___redArg(v_k_783_);
lean_dec(v_k_783_);
return v_res_784_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Domain_ctorElim(lean_object* v_motive_785_, lean_object* v_ctorIdx_786_, uint8_t v_t_787_, lean_object* v_h_788_, lean_object* v_k_789_){
_start:
{
lean_inc(v_k_789_);
return v_k_789_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Domain_ctorElim___boxed(lean_object* v_motive_790_, lean_object* v_ctorIdx_791_, lean_object* v_t_792_, lean_object* v_h_793_, lean_object* v_k_794_){
_start:
{
uint8_t v_t_boxed_795_; lean_object* v_res_796_; 
v_t_boxed_795_ = lean_unbox(v_t_792_);
v_res_796_ = lp_reflect_x2dnts_NTS_Domain_ctorElim(v_motive_790_, v_ctorIdx_791_, v_t_boxed_795_, v_h_793_, v_k_794_);
lean_dec(v_k_794_);
lean_dec(v_ctorIdx_791_);
return v_res_796_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Domain_matterOfFact_elim___redArg(lean_object* v_matterOfFact_797_){
_start:
{
lean_inc(v_matterOfFact_797_);
return v_matterOfFact_797_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Domain_matterOfFact_elim___redArg___boxed(lean_object* v_matterOfFact_798_){
_start:
{
lean_object* v_res_799_; 
v_res_799_ = lp_reflect_x2dnts_NTS_Domain_matterOfFact_elim___redArg(v_matterOfFact_798_);
lean_dec(v_matterOfFact_798_);
return v_res_799_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Domain_matterOfFact_elim(lean_object* v_motive_800_, uint8_t v_t_801_, lean_object* v_h_802_, lean_object* v_matterOfFact_803_){
_start:
{
lean_inc(v_matterOfFact_803_);
return v_matterOfFact_803_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Domain_matterOfFact_elim___boxed(lean_object* v_motive_804_, lean_object* v_t_805_, lean_object* v_h_806_, lean_object* v_matterOfFact_807_){
_start:
{
uint8_t v_t_boxed_808_; lean_object* v_res_809_; 
v_t_boxed_808_ = lean_unbox(v_t_805_);
v_res_809_ = lp_reflect_x2dnts_NTS_Domain_matterOfFact_elim(v_motive_804_, v_t_boxed_808_, v_h_806_, v_matterOfFact_807_);
lean_dec(v_matterOfFact_807_);
return v_res_809_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Domain_rightAndWrong_elim___redArg(lean_object* v_rightAndWrong_810_){
_start:
{
lean_inc(v_rightAndWrong_810_);
return v_rightAndWrong_810_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Domain_rightAndWrong_elim___redArg___boxed(lean_object* v_rightAndWrong_811_){
_start:
{
lean_object* v_res_812_; 
v_res_812_ = lp_reflect_x2dnts_NTS_Domain_rightAndWrong_elim___redArg(v_rightAndWrong_811_);
lean_dec(v_rightAndWrong_811_);
return v_res_812_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Domain_rightAndWrong_elim(lean_object* v_motive_813_, uint8_t v_t_814_, lean_object* v_h_815_, lean_object* v_rightAndWrong_816_){
_start:
{
lean_inc(v_rightAndWrong_816_);
return v_rightAndWrong_816_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Domain_rightAndWrong_elim___boxed(lean_object* v_motive_817_, lean_object* v_t_818_, lean_object* v_h_819_, lean_object* v_rightAndWrong_820_){
_start:
{
uint8_t v_t_boxed_821_; lean_object* v_res_822_; 
v_t_boxed_821_ = lean_unbox(v_t_818_);
v_res_822_ = lp_reflect_x2dnts_NTS_Domain_rightAndWrong_elim(v_motive_817_, v_t_boxed_821_, v_h_819_, v_rightAndWrong_820_);
lean_dec(v_rightAndWrong_820_);
return v_res_822_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Domain_beauty_elim___redArg(lean_object* v_beauty_823_){
_start:
{
lean_inc(v_beauty_823_);
return v_beauty_823_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Domain_beauty_elim___redArg___boxed(lean_object* v_beauty_824_){
_start:
{
lean_object* v_res_825_; 
v_res_825_ = lp_reflect_x2dnts_NTS_Domain_beauty_elim___redArg(v_beauty_824_);
lean_dec(v_beauty_824_);
return v_res_825_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Domain_beauty_elim(lean_object* v_motive_826_, uint8_t v_t_827_, lean_object* v_h_828_, lean_object* v_beauty_829_){
_start:
{
lean_inc(v_beauty_829_);
return v_beauty_829_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Domain_beauty_elim___boxed(lean_object* v_motive_830_, lean_object* v_t_831_, lean_object* v_h_832_, lean_object* v_beauty_833_){
_start:
{
uint8_t v_t_boxed_834_; lean_object* v_res_835_; 
v_t_boxed_834_ = lean_unbox(v_t_831_);
v_res_835_ = lp_reflect_x2dnts_NTS_Domain_beauty_elim(v_motive_830_, v_t_boxed_834_, v_h_832_, v_beauty_833_);
lean_dec(v_beauty_833_);
return v_res_835_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Domain_sacred_elim___redArg(lean_object* v_sacred_836_){
_start:
{
lean_inc(v_sacred_836_);
return v_sacred_836_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Domain_sacred_elim___redArg___boxed(lean_object* v_sacred_837_){
_start:
{
lean_object* v_res_838_; 
v_res_838_ = lp_reflect_x2dnts_NTS_Domain_sacred_elim___redArg(v_sacred_837_);
lean_dec(v_sacred_837_);
return v_res_838_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Domain_sacred_elim(lean_object* v_motive_839_, uint8_t v_t_840_, lean_object* v_h_841_, lean_object* v_sacred_842_){
_start:
{
lean_inc(v_sacred_842_);
return v_sacred_842_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Domain_sacred_elim___boxed(lean_object* v_motive_843_, lean_object* v_t_844_, lean_object* v_h_845_, lean_object* v_sacred_846_){
_start:
{
uint8_t v_t_boxed_847_; lean_object* v_res_848_; 
v_t_boxed_847_ = lean_unbox(v_t_844_);
v_res_848_ = lp_reflect_x2dnts_NTS_Domain_sacred_elim(v_motive_843_, v_t_boxed_847_, v_h_845_, v_sacred_846_);
lean_dec(v_sacred_846_);
return v_res_848_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Domain_howIFeel_elim___redArg(lean_object* v_howIFeel_849_){
_start:
{
lean_inc(v_howIFeel_849_);
return v_howIFeel_849_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Domain_howIFeel_elim___redArg___boxed(lean_object* v_howIFeel_850_){
_start:
{
lean_object* v_res_851_; 
v_res_851_ = lp_reflect_x2dnts_NTS_Domain_howIFeel_elim___redArg(v_howIFeel_850_);
lean_dec(v_howIFeel_850_);
return v_res_851_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Domain_howIFeel_elim(lean_object* v_motive_852_, uint8_t v_t_853_, lean_object* v_h_854_, lean_object* v_howIFeel_855_){
_start:
{
lean_inc(v_howIFeel_855_);
return v_howIFeel_855_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Domain_howIFeel_elim___boxed(lean_object* v_motive_856_, lean_object* v_t_857_, lean_object* v_h_858_, lean_object* v_howIFeel_859_){
_start:
{
uint8_t v_t_boxed_860_; lean_object* v_res_861_; 
v_t_boxed_860_ = lean_unbox(v_t_857_);
v_res_861_ = lp_reflect_x2dnts_NTS_Domain_howIFeel_elim(v_motive_856_, v_t_boxed_860_, v_h_858_, v_howIFeel_859_);
lean_dec(v_howIFeel_859_);
return v_res_861_;
}
}
LEAN_EXPORT uint8_t lp_reflect_x2dnts_NTS_Domain_ofNat(lean_object* v_n_862_){
_start:
{
lean_object* v___x_863_; uint8_t v___x_864_; 
v___x_863_ = lean_unsigned_to_nat(1u);
v___x_864_ = lean_nat_dec_le(v_n_862_, v___x_863_);
if (v___x_864_ == 0)
{
lean_object* v___x_865_; uint8_t v___x_866_; 
v___x_865_ = lean_unsigned_to_nat(2u);
v___x_866_ = lean_nat_dec_le(v_n_862_, v___x_865_);
if (v___x_866_ == 0)
{
lean_object* v___x_867_; uint8_t v___x_868_; 
v___x_867_ = lean_unsigned_to_nat(3u);
v___x_868_ = lean_nat_dec_le(v_n_862_, v___x_867_);
if (v___x_868_ == 0)
{
uint8_t v___x_869_; 
v___x_869_ = 4;
return v___x_869_;
}
else
{
uint8_t v___x_870_; 
v___x_870_ = 3;
return v___x_870_;
}
}
else
{
uint8_t v___x_871_; 
v___x_871_ = 2;
return v___x_871_;
}
}
else
{
lean_object* v___x_872_; uint8_t v___x_873_; 
v___x_872_ = lean_unsigned_to_nat(0u);
v___x_873_ = lean_nat_dec_le(v_n_862_, v___x_872_);
if (v___x_873_ == 0)
{
uint8_t v___x_874_; 
v___x_874_ = 1;
return v___x_874_;
}
else
{
uint8_t v___x_875_; 
v___x_875_ = 0;
return v___x_875_;
}
}
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Domain_ofNat___boxed(lean_object* v_n_876_){
_start:
{
uint8_t v_res_877_; lean_object* v_r_878_; 
v_res_877_ = lp_reflect_x2dnts_NTS_Domain_ofNat(v_n_876_);
lean_dec(v_n_876_);
v_r_878_ = lean_box(v_res_877_);
return v_r_878_;
}
}
LEAN_EXPORT uint8_t lp_reflect_x2dnts_NTS_instDecidableEqDomain(uint8_t v_x_879_, uint8_t v_y_880_){
_start:
{
lean_object* v___x_881_; lean_object* v___x_882_; uint8_t v___x_883_; 
v___x_881_ = lp_reflect_x2dnts_NTS_Domain_ctorIdx(v_x_879_);
v___x_882_ = lp_reflect_x2dnts_NTS_Domain_ctorIdx(v_y_880_);
v___x_883_ = lean_nat_dec_eq(v___x_881_, v___x_882_);
lean_dec(v___x_882_);
lean_dec(v___x_881_);
return v___x_883_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instDecidableEqDomain___boxed(lean_object* v_x_884_, lean_object* v_y_885_){
_start:
{
uint8_t v_x_13__boxed_886_; uint8_t v_y_14__boxed_887_; uint8_t v_res_888_; lean_object* v_r_889_; 
v_x_13__boxed_886_ = lean_unbox(v_x_884_);
v_y_14__boxed_887_ = lean_unbox(v_y_885_);
v_res_888_ = lp_reflect_x2dnts_NTS_instDecidableEqDomain(v_x_13__boxed_886_, v_y_14__boxed_887_);
v_r_889_ = lean_box(v_res_888_);
return v_r_889_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instReprDomain_repr(uint8_t v_x_905_, lean_object* v_prec_906_){
_start:
{
lean_object* v___y_908_; lean_object* v___y_915_; lean_object* v___y_922_; lean_object* v___y_929_; lean_object* v___y_936_; 
switch(v_x_905_)
{
case 0:
{
lean_object* v___x_942_; uint8_t v___x_943_; 
v___x_942_ = lean_unsigned_to_nat(1024u);
v___x_943_ = lean_nat_dec_le(v___x_942_, v_prec_906_);
if (v___x_943_ == 0)
{
lean_object* v___x_944_; 
v___x_944_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprArrangement_repr___closed__8, &lp_reflect_x2dnts_NTS_instReprArrangement_repr___closed__8_once, _init_lp_reflect_x2dnts_NTS_instReprArrangement_repr___closed__8);
v___y_908_ = v___x_944_;
goto v___jp_907_;
}
else
{
lean_object* v___x_945_; 
v___x_945_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprArrangement_repr___closed__9, &lp_reflect_x2dnts_NTS_instReprArrangement_repr___closed__9_once, _init_lp_reflect_x2dnts_NTS_instReprArrangement_repr___closed__9);
v___y_908_ = v___x_945_;
goto v___jp_907_;
}
}
case 1:
{
lean_object* v___x_946_; uint8_t v___x_947_; 
v___x_946_ = lean_unsigned_to_nat(1024u);
v___x_947_ = lean_nat_dec_le(v___x_946_, v_prec_906_);
if (v___x_947_ == 0)
{
lean_object* v___x_948_; 
v___x_948_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprArrangement_repr___closed__8, &lp_reflect_x2dnts_NTS_instReprArrangement_repr___closed__8_once, _init_lp_reflect_x2dnts_NTS_instReprArrangement_repr___closed__8);
v___y_915_ = v___x_948_;
goto v___jp_914_;
}
else
{
lean_object* v___x_949_; 
v___x_949_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprArrangement_repr___closed__9, &lp_reflect_x2dnts_NTS_instReprArrangement_repr___closed__9_once, _init_lp_reflect_x2dnts_NTS_instReprArrangement_repr___closed__9);
v___y_915_ = v___x_949_;
goto v___jp_914_;
}
}
case 2:
{
lean_object* v___x_950_; uint8_t v___x_951_; 
v___x_950_ = lean_unsigned_to_nat(1024u);
v___x_951_ = lean_nat_dec_le(v___x_950_, v_prec_906_);
if (v___x_951_ == 0)
{
lean_object* v___x_952_; 
v___x_952_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprArrangement_repr___closed__8, &lp_reflect_x2dnts_NTS_instReprArrangement_repr___closed__8_once, _init_lp_reflect_x2dnts_NTS_instReprArrangement_repr___closed__8);
v___y_922_ = v___x_952_;
goto v___jp_921_;
}
else
{
lean_object* v___x_953_; 
v___x_953_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprArrangement_repr___closed__9, &lp_reflect_x2dnts_NTS_instReprArrangement_repr___closed__9_once, _init_lp_reflect_x2dnts_NTS_instReprArrangement_repr___closed__9);
v___y_922_ = v___x_953_;
goto v___jp_921_;
}
}
case 3:
{
lean_object* v___x_954_; uint8_t v___x_955_; 
v___x_954_ = lean_unsigned_to_nat(1024u);
v___x_955_ = lean_nat_dec_le(v___x_954_, v_prec_906_);
if (v___x_955_ == 0)
{
lean_object* v___x_956_; 
v___x_956_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprArrangement_repr___closed__8, &lp_reflect_x2dnts_NTS_instReprArrangement_repr___closed__8_once, _init_lp_reflect_x2dnts_NTS_instReprArrangement_repr___closed__8);
v___y_929_ = v___x_956_;
goto v___jp_928_;
}
else
{
lean_object* v___x_957_; 
v___x_957_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprArrangement_repr___closed__9, &lp_reflect_x2dnts_NTS_instReprArrangement_repr___closed__9_once, _init_lp_reflect_x2dnts_NTS_instReprArrangement_repr___closed__9);
v___y_929_ = v___x_957_;
goto v___jp_928_;
}
}
default: 
{
lean_object* v___x_958_; uint8_t v___x_959_; 
v___x_958_ = lean_unsigned_to_nat(1024u);
v___x_959_ = lean_nat_dec_le(v___x_958_, v_prec_906_);
if (v___x_959_ == 0)
{
lean_object* v___x_960_; 
v___x_960_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprArrangement_repr___closed__8, &lp_reflect_x2dnts_NTS_instReprArrangement_repr___closed__8_once, _init_lp_reflect_x2dnts_NTS_instReprArrangement_repr___closed__8);
v___y_936_ = v___x_960_;
goto v___jp_935_;
}
else
{
lean_object* v___x_961_; 
v___x_961_ = lean_obj_once(&lp_reflect_x2dnts_NTS_instReprArrangement_repr___closed__9, &lp_reflect_x2dnts_NTS_instReprArrangement_repr___closed__9_once, _init_lp_reflect_x2dnts_NTS_instReprArrangement_repr___closed__9);
v___y_936_ = v___x_961_;
goto v___jp_935_;
}
}
}
v___jp_907_:
{
lean_object* v___x_909_; lean_object* v___x_910_; uint8_t v___x_911_; lean_object* v___x_912_; lean_object* v___x_913_; 
v___x_909_ = ((lean_object*)(lp_reflect_x2dnts_NTS_instReprDomain_repr___closed__1));
lean_inc(v___y_908_);
v___x_910_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_910_, 0, v___y_908_);
lean_ctor_set(v___x_910_, 1, v___x_909_);
v___x_911_ = 0;
v___x_912_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_912_, 0, v___x_910_);
lean_ctor_set_uint8(v___x_912_, sizeof(void*)*1, v___x_911_);
v___x_913_ = l_Repr_addAppParen(v___x_912_, v_prec_906_);
return v___x_913_;
}
v___jp_914_:
{
lean_object* v___x_916_; lean_object* v___x_917_; uint8_t v___x_918_; lean_object* v___x_919_; lean_object* v___x_920_; 
v___x_916_ = ((lean_object*)(lp_reflect_x2dnts_NTS_instReprDomain_repr___closed__3));
lean_inc(v___y_915_);
v___x_917_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_917_, 0, v___y_915_);
lean_ctor_set(v___x_917_, 1, v___x_916_);
v___x_918_ = 0;
v___x_919_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_919_, 0, v___x_917_);
lean_ctor_set_uint8(v___x_919_, sizeof(void*)*1, v___x_918_);
v___x_920_ = l_Repr_addAppParen(v___x_919_, v_prec_906_);
return v___x_920_;
}
v___jp_921_:
{
lean_object* v___x_923_; lean_object* v___x_924_; uint8_t v___x_925_; lean_object* v___x_926_; lean_object* v___x_927_; 
v___x_923_ = ((lean_object*)(lp_reflect_x2dnts_NTS_instReprDomain_repr___closed__5));
lean_inc(v___y_922_);
v___x_924_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_924_, 0, v___y_922_);
lean_ctor_set(v___x_924_, 1, v___x_923_);
v___x_925_ = 0;
v___x_926_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_926_, 0, v___x_924_);
lean_ctor_set_uint8(v___x_926_, sizeof(void*)*1, v___x_925_);
v___x_927_ = l_Repr_addAppParen(v___x_926_, v_prec_906_);
return v___x_927_;
}
v___jp_928_:
{
lean_object* v___x_930_; lean_object* v___x_931_; uint8_t v___x_932_; lean_object* v___x_933_; lean_object* v___x_934_; 
v___x_930_ = ((lean_object*)(lp_reflect_x2dnts_NTS_instReprDomain_repr___closed__7));
lean_inc(v___y_929_);
v___x_931_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_931_, 0, v___y_929_);
lean_ctor_set(v___x_931_, 1, v___x_930_);
v___x_932_ = 0;
v___x_933_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_933_, 0, v___x_931_);
lean_ctor_set_uint8(v___x_933_, sizeof(void*)*1, v___x_932_);
v___x_934_ = l_Repr_addAppParen(v___x_933_, v_prec_906_);
return v___x_934_;
}
v___jp_935_:
{
lean_object* v___x_937_; lean_object* v___x_938_; uint8_t v___x_939_; lean_object* v___x_940_; lean_object* v___x_941_; 
v___x_937_ = ((lean_object*)(lp_reflect_x2dnts_NTS_instReprDomain_repr___closed__9));
lean_inc(v___y_936_);
v___x_938_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_938_, 0, v___y_936_);
lean_ctor_set(v___x_938_, 1, v___x_937_);
v___x_939_ = 0;
v___x_940_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_940_, 0, v___x_938_);
lean_ctor_set_uint8(v___x_940_, sizeof(void*)*1, v___x_939_);
v___x_941_ = l_Repr_addAppParen(v___x_940_, v_prec_906_);
return v___x_941_;
}
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_instReprDomain_repr___boxed(lean_object* v_x_962_, lean_object* v_prec_963_){
_start:
{
uint8_t v_x_285__boxed_964_; lean_object* v_res_965_; 
v_x_285__boxed_964_ = lean_unbox(v_x_962_);
v_res_965_ = lp_reflect_x2dnts_NTS_instReprDomain_repr(v_x_285__boxed_964_, v_prec_963_);
lean_dec(v_prec_963_);
return v_res_965_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Domain_id(uint8_t v_x_994_){
_start:
{
switch(v_x_994_)
{
case 0:
{
lean_object* v___x_995_; 
v___x_995_ = ((lean_object*)(lp_reflect_x2dnts_NTS_Domain_id___closed__0));
return v___x_995_;
}
case 1:
{
lean_object* v___x_996_; 
v___x_996_ = ((lean_object*)(lp_reflect_x2dnts_NTS_Domain_id___closed__1));
return v___x_996_;
}
case 2:
{
lean_object* v___x_997_; 
v___x_997_ = ((lean_object*)(lp_reflect_x2dnts_NTS_Domain_id___closed__2));
return v___x_997_;
}
case 3:
{
lean_object* v___x_998_; 
v___x_998_ = ((lean_object*)(lp_reflect_x2dnts_NTS_Domain_id___closed__3));
return v___x_998_;
}
default: 
{
lean_object* v___x_999_; 
v___x_999_ = ((lean_object*)(lp_reflect_x2dnts_NTS_Domain_id___closed__4));
return v___x_999_;
}
}
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Domain_id___boxed(lean_object* v_x_1000_){
_start:
{
uint8_t v_x_49__boxed_1001_; lean_object* v_res_1002_; 
v_x_49__boxed_1001_ = lean_unbox(v_x_1000_);
v_res_1002_ = lp_reflect_x2dnts_NTS_Domain_id(v_x_49__boxed_1001_);
return v_res_1002_;
}
}
LEAN_EXPORT uint8_t lp_reflect_x2dnts_NTS_Domain_governor(uint8_t v_x_1003_){
_start:
{
switch(v_x_1003_)
{
case 0:
{
uint8_t v___x_1004_; 
v___x_1004_ = 9;
return v___x_1004_;
}
case 1:
{
uint8_t v___x_1005_; 
v___x_1005_ = 4;
return v___x_1005_;
}
case 2:
{
uint8_t v___x_1006_; 
v___x_1006_ = 14;
return v___x_1006_;
}
case 3:
{
uint8_t v___x_1007_; 
v___x_1007_ = 15;
return v___x_1007_;
}
default: 
{
uint8_t v___x_1008_; 
v___x_1008_ = 3;
return v___x_1008_;
}
}
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Domain_governor___boxed(lean_object* v_x_1009_){
_start:
{
uint8_t v_x_39__boxed_1010_; uint8_t v_res_1011_; lean_object* v_r_1012_; 
v_x_39__boxed_1010_ = lean_unbox(v_x_1009_);
v_res_1011_ = lp_reflect_x2dnts_NTS_Domain_governor(v_x_39__boxed_1010_);
v_r_1012_ = lean_box(v_res_1011_);
return v_r_1012_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_List_eraseDups___at___00NTS_Domain_proper_spec__0(lean_object* v_as_1013_){
_start:
{
lean_object* v___f_1014_; lean_object* v___x_1015_; 
v___f_1014_ = lean_alloc_closure((void*)(lp_reflect_x2dnts_NTS_instDecidableEqPrime___boxed), 2, 0);
v___x_1015_ = l_List_eraseDupsBy___redArg(v___f_1014_, v_as_1013_);
return v___x_1015_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Domain_proper(uint8_t v_d_1024_){
_start:
{
uint8_t v___x_1025_; lean_object* v___x_1026_; lean_object* v___x_1027_; lean_object* v___x_1028_; lean_object* v___x_1029_; 
v___x_1025_ = lp_reflect_x2dnts_NTS_Domain_governor(v_d_1024_);
v___x_1026_ = ((lean_object*)(lp_reflect_x2dnts_NTS_Domain_proper___closed__1));
v___x_1027_ = lean_box(v___x_1025_);
v___x_1028_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1028_, 0, v___x_1027_);
lean_ctor_set(v___x_1028_, 1, v___x_1026_);
v___x_1029_ = lp_reflect_x2dnts_List_eraseDups___at___00NTS_Domain_proper_spec__0(v___x_1028_);
return v___x_1029_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_Domain_proper___boxed(lean_object* v_d_1030_){
_start:
{
uint8_t v_d_boxed_1031_; lean_object* v_res_1032_; 
v_d_boxed_1031_ = lean_unbox(v_d_1030_);
v_res_1032_ = lp_reflect_x2dnts_NTS_Domain_proper(v_d_boxed_1031_);
return v_res_1032_;
}
}
LEAN_EXPORT uint8_t lp_reflect_x2dnts_List_elem___at___00NTS_onMismatch_spec__0(uint8_t v_a_1033_, lean_object* v_x_1034_){
_start:
{
if (lean_obj_tag(v_x_1034_) == 0)
{
uint8_t v___x_1035_; 
v___x_1035_ = 0;
return v___x_1035_;
}
else
{
lean_object* v_head_1036_; lean_object* v_tail_1037_; uint8_t v___x_1038_; uint8_t v___x_1039_; 
v_head_1036_ = lean_ctor_get(v_x_1034_, 0);
v_tail_1037_ = lean_ctor_get(v_x_1034_, 1);
v___x_1038_ = lean_unbox(v_head_1036_);
v___x_1039_ = lp_reflect_x2dnts_NTS_instDecidableEqPrime(v_a_1033_, v___x_1038_);
if (v___x_1039_ == 0)
{
v_x_1034_ = v_tail_1037_;
goto _start;
}
else
{
return v___x_1039_;
}
}
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_List_elem___at___00NTS_onMismatch_spec__0___boxed(lean_object* v_a_1041_, lean_object* v_x_1042_){
_start:
{
uint8_t v_a_boxed_1043_; uint8_t v_res_1044_; lean_object* v_r_1045_; 
v_a_boxed_1043_ = lean_unbox(v_a_1041_);
v_res_1044_ = lp_reflect_x2dnts_List_elem___at___00NTS_onMismatch_spec__0(v_a_boxed_1043_, v_x_1042_);
lean_dec(v_x_1042_);
v_r_1045_ = lean_box(v_res_1044_);
return v_r_1045_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_onMismatch(uint8_t v_asked_1047_, uint8_t v_d_1048_, uint8_t v_basis_1049_, lean_object* v_s_1050_){
_start:
{
if (v_asked_1047_ == 0)
{
lean_object* v___x_1051_; 
lean_dec_ref(v_s_1050_);
v___x_1051_ = lean_box(0);
return v___x_1051_;
}
else
{
lean_object* v___x_1052_; uint8_t v___x_1053_; 
v___x_1052_ = lp_reflect_x2dnts_NTS_Domain_proper(v_d_1048_);
v___x_1053_ = lp_reflect_x2dnts_List_elem___at___00NTS_onMismatch_spec__0(v_basis_1049_, v___x_1052_);
lean_dec(v___x_1052_);
if (v___x_1053_ == 0)
{
lean_object* v___x_1054_; lean_object* v___x_1055_; lean_object* v___x_1056_; lean_object* v___x_1057_; lean_object* v___x_1058_; 
v___x_1054_ = lean_alloc_ctor(0, 0, 1);
lean_ctor_set_uint8(v___x_1054_, 0, v_basis_1049_);
v___x_1055_ = ((lean_object*)(lp_reflect_x2dnts_NTS_onMismatch___closed__0));
v___x_1056_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1056_, 0, v_s_1050_);
lean_ctor_set(v___x_1056_, 1, v___x_1054_);
lean_ctor_set(v___x_1056_, 2, v___x_1055_);
v___x_1057_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1057_, 0, v___x_1056_);
v___x_1058_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1058_, 0, v___x_1057_);
return v___x_1058_;
}
else
{
lean_object* v___x_1059_; 
lean_dec_ref(v_s_1050_);
v___x_1059_ = lean_box(0);
return v___x_1059_;
}
}
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts_NTS_onMismatch___boxed(lean_object* v_asked_1060_, lean_object* v_d_1061_, lean_object* v_basis_1062_, lean_object* v_s_1063_){
_start:
{
uint8_t v_asked_boxed_1064_; uint8_t v_d_boxed_1065_; uint8_t v_basis_boxed_1066_; lean_object* v_res_1067_; 
v_asked_boxed_1064_ = lean_unbox(v_asked_1060_);
v_d_boxed_1065_ = lean_unbox(v_d_1061_);
v_basis_boxed_1066_ = lean_unbox(v_basis_1062_);
v_res_1067_ = lp_reflect_x2dnts_NTS_onMismatch(v_asked_boxed_1064_, v_d_boxed_1065_, v_basis_boxed_1066_, v_s_1063_);
return v_res_1067_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts___private_NTS_State_0__NTS_instReprDomain_repr_match__1_splitter___redArg(uint8_t v_x_1068_, lean_object* v_h__1_1069_, lean_object* v_h__2_1070_, lean_object* v_h__3_1071_, lean_object* v_h__4_1072_, lean_object* v_h__5_1073_){
_start:
{
switch(v_x_1068_)
{
case 0:
{
lean_object* v___x_1074_; lean_object* v___x_1075_; 
lean_dec(v_h__5_1073_);
lean_dec(v_h__4_1072_);
lean_dec(v_h__3_1071_);
lean_dec(v_h__2_1070_);
v___x_1074_ = lean_box(0);
v___x_1075_ = lean_apply_1(v_h__1_1069_, v___x_1074_);
return v___x_1075_;
}
case 1:
{
lean_object* v___x_1076_; lean_object* v___x_1077_; 
lean_dec(v_h__5_1073_);
lean_dec(v_h__4_1072_);
lean_dec(v_h__3_1071_);
lean_dec(v_h__1_1069_);
v___x_1076_ = lean_box(0);
v___x_1077_ = lean_apply_1(v_h__2_1070_, v___x_1076_);
return v___x_1077_;
}
case 2:
{
lean_object* v___x_1078_; lean_object* v___x_1079_; 
lean_dec(v_h__5_1073_);
lean_dec(v_h__4_1072_);
lean_dec(v_h__2_1070_);
lean_dec(v_h__1_1069_);
v___x_1078_ = lean_box(0);
v___x_1079_ = lean_apply_1(v_h__3_1071_, v___x_1078_);
return v___x_1079_;
}
case 3:
{
lean_object* v___x_1080_; lean_object* v___x_1081_; 
lean_dec(v_h__5_1073_);
lean_dec(v_h__3_1071_);
lean_dec(v_h__2_1070_);
lean_dec(v_h__1_1069_);
v___x_1080_ = lean_box(0);
v___x_1081_ = lean_apply_1(v_h__4_1072_, v___x_1080_);
return v___x_1081_;
}
default: 
{
lean_object* v___x_1082_; lean_object* v___x_1083_; 
lean_dec(v_h__4_1072_);
lean_dec(v_h__3_1071_);
lean_dec(v_h__2_1070_);
lean_dec(v_h__1_1069_);
v___x_1082_ = lean_box(0);
v___x_1083_ = lean_apply_1(v_h__5_1073_, v___x_1082_);
return v___x_1083_;
}
}
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts___private_NTS_State_0__NTS_instReprDomain_repr_match__1_splitter___redArg___boxed(lean_object* v_x_1084_, lean_object* v_h__1_1085_, lean_object* v_h__2_1086_, lean_object* v_h__3_1087_, lean_object* v_h__4_1088_, lean_object* v_h__5_1089_){
_start:
{
uint8_t v_x_51__boxed_1090_; lean_object* v_res_1091_; 
v_x_51__boxed_1090_ = lean_unbox(v_x_1084_);
v_res_1091_ = lp_reflect_x2dnts___private_NTS_State_0__NTS_instReprDomain_repr_match__1_splitter___redArg(v_x_51__boxed_1090_, v_h__1_1085_, v_h__2_1086_, v_h__3_1087_, v_h__4_1088_, v_h__5_1089_);
return v_res_1091_;
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts___private_NTS_State_0__NTS_instReprDomain_repr_match__1_splitter(lean_object* v_motive_1092_, uint8_t v_x_1093_, lean_object* v_h__1_1094_, lean_object* v_h__2_1095_, lean_object* v_h__3_1096_, lean_object* v_h__4_1097_, lean_object* v_h__5_1098_){
_start:
{
switch(v_x_1093_)
{
case 0:
{
lean_object* v___x_1099_; lean_object* v___x_1100_; 
lean_dec(v_h__5_1098_);
lean_dec(v_h__4_1097_);
lean_dec(v_h__3_1096_);
lean_dec(v_h__2_1095_);
v___x_1099_ = lean_box(0);
v___x_1100_ = lean_apply_1(v_h__1_1094_, v___x_1099_);
return v___x_1100_;
}
case 1:
{
lean_object* v___x_1101_; lean_object* v___x_1102_; 
lean_dec(v_h__5_1098_);
lean_dec(v_h__4_1097_);
lean_dec(v_h__3_1096_);
lean_dec(v_h__1_1094_);
v___x_1101_ = lean_box(0);
v___x_1102_ = lean_apply_1(v_h__2_1095_, v___x_1101_);
return v___x_1102_;
}
case 2:
{
lean_object* v___x_1103_; lean_object* v___x_1104_; 
lean_dec(v_h__5_1098_);
lean_dec(v_h__4_1097_);
lean_dec(v_h__2_1095_);
lean_dec(v_h__1_1094_);
v___x_1103_ = lean_box(0);
v___x_1104_ = lean_apply_1(v_h__3_1096_, v___x_1103_);
return v___x_1104_;
}
case 3:
{
lean_object* v___x_1105_; lean_object* v___x_1106_; 
lean_dec(v_h__5_1098_);
lean_dec(v_h__3_1096_);
lean_dec(v_h__2_1095_);
lean_dec(v_h__1_1094_);
v___x_1105_ = lean_box(0);
v___x_1106_ = lean_apply_1(v_h__4_1097_, v___x_1105_);
return v___x_1106_;
}
default: 
{
lean_object* v___x_1107_; lean_object* v___x_1108_; 
lean_dec(v_h__4_1097_);
lean_dec(v_h__3_1096_);
lean_dec(v_h__2_1095_);
lean_dec(v_h__1_1094_);
v___x_1107_ = lean_box(0);
v___x_1108_ = lean_apply_1(v_h__5_1098_, v___x_1107_);
return v___x_1108_;
}
}
}
}
LEAN_EXPORT lean_object* lp_reflect_x2dnts___private_NTS_State_0__NTS_instReprDomain_repr_match__1_splitter___boxed(lean_object* v_motive_1109_, lean_object* v_x_1110_, lean_object* v_h__1_1111_, lean_object* v_h__2_1112_, lean_object* v_h__3_1113_, lean_object* v_h__4_1114_, lean_object* v_h__5_1115_){
_start:
{
uint8_t v_x_74__boxed_1116_; lean_object* v_res_1117_; 
v_x_74__boxed_1116_ = lean_unbox(v_x_1110_);
v_res_1117_ = lp_reflect_x2dnts___private_NTS_State_0__NTS_instReprDomain_repr_match__1_splitter(v_motive_1109_, v_x_74__boxed_1116_, v_h__1_1111_, v_h__2_1112_, v_h__3_1113_, v_h__4_1114_, v_h__5_1115_);
return v_res_1117_;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_reflect_x2dnts_NTS_Output(uint8_t builtin);
lean_object* initialize_reflect_x2dnts_NTS_Person(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_reflect_x2dnts_NTS_State(uint8_t builtin) {
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
res = initialize_reflect_x2dnts_NTS_Person(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
