// Lean compiler output
// Module: NTS
// Imports: public import Init public meta import Init public import NTS.Elements public import NTS.Scope public import NTS.Output public import NTS.Frames public import NTS.Person public import NTS.Onboarding public import NTS.State
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
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_reflect_x2dnts_NTS_Elements(uint8_t builtin);
lean_object* initialize_reflect_x2dnts_NTS_Scope(uint8_t builtin);
lean_object* initialize_reflect_x2dnts_NTS_Output(uint8_t builtin);
lean_object* initialize_reflect_x2dnts_NTS_Frames(uint8_t builtin);
lean_object* initialize_reflect_x2dnts_NTS_Person(uint8_t builtin);
lean_object* initialize_reflect_x2dnts_NTS_Onboarding(uint8_t builtin);
lean_object* initialize_reflect_x2dnts_NTS_State(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_reflect_x2dnts_NTS(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_reflect_x2dnts_NTS_Elements(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_reflect_x2dnts_NTS_Scope(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_reflect_x2dnts_NTS_Output(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_reflect_x2dnts_NTS_Frames(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_reflect_x2dnts_NTS_Person(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_reflect_x2dnts_NTS_Onboarding(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_reflect_x2dnts_NTS_State(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
