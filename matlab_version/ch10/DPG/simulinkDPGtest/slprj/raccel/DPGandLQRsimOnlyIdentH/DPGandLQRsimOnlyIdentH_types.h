#ifndef DPGandLQRsimOnlyIdentH_types_h_
#define DPGandLQRsimOnlyIdentH_types_h_
#include "rtwtypes.h"
#ifndef DEFINED_TYPEDEF_FOR_Critic_params_init_
#define DEFINED_TYPEDEF_FOR_Critic_params_init_
typedef struct { real_T W1_c [ 30 ] ; real_T b1_c [ 10 ] ; real_T W2_c [ 100
] ; real_T b2_c [ 10 ] ; real_T W3_c [ 10 ] ; real_T b3_c ; }
Critic_params_init ;
#endif
#ifndef SS_UINT64
#define SS_UINT64 19
#endif
#ifndef SS_INT64
#define SS_INT64 20
#endif
typedef struct P_ P ;
#endif
