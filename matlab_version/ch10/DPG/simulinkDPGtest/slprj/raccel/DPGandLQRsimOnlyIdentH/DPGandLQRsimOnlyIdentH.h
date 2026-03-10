#ifndef DPGandLQRsimOnlyIdentH_h_
#define DPGandLQRsimOnlyIdentH_h_
#ifndef DPGandLQRsimOnlyIdentH_COMMON_INCLUDES_
#define DPGandLQRsimOnlyIdentH_COMMON_INCLUDES_
#include <stdlib.h>
#include "sl_AsyncioQueue/AsyncioQueueCAPI.h"
#include "rtwtypes.h"
#include "sigstream_rtw.h"
#include "simtarget/slSimTgtSigstreamRTW.h"
#include "simtarget/slSimTgtSlioCoreRTW.h"
#include "simtarget/slSimTgtSlioClientsRTW.h"
#include "simtarget/slSimTgtSlioSdiRTW.h"
#include "simstruc.h"
#include "fixedpoint.h"
#include "raccel.h"
#include "slsv_diagnostic_codegen_c_api.h"
#include "rt_logging_simtarget.h"
#include "rt_nonfinite.h"
#include "math.h"
#include "dt_info.h"
#include "ext_work.h"
#endif
#include "DPGandLQRsimOnlyIdentH_types.h"
#include <stddef.h>
#include "rtw_modelmap_simtarget.h"
#include "rt_defines.h"
#include <string.h>
#define MODEL_NAME DPGandLQRsimOnlyIdentH
#define NSAMPLE_TIMES (3) 
#define NINPUTS (0)       
#define NOUTPUTS (0)     
#define NBLOCKIO (17) 
#define NUM_ZC_EVENTS (0) 
#ifndef NCSTATES
#define NCSTATES (0)   
#elif NCSTATES != 0
#error Invalid specification of NCSTATES defined in compiler command
#endif
#ifndef rtmGetDataMapInfo
#define rtmGetDataMapInfo(rtm) (*rt_dataMapInfoPtr)
#endif
#ifndef rtmSetDataMapInfo
#define rtmSetDataMapInfo(rtm, val) (rt_dataMapInfoPtr = &val)
#endif
#ifndef IN_RACCEL_MAIN
#endif
typedef struct { real_T k2qoybmvyf [ 2 ] ; real_T gyeu202tnh ; real_T
a02qbwyxe5 ; real_T d0ken2h5az ; real_T iqc5no5xul [ 2 ] ; real_T o4x0bb4qep
[ 2 ] ; real_T osgh5xcie0 ; real_T fhsdyf2cqn [ 2 ] ; real_T iphp0mrtyf [ 9 ]
; real_T c214bb40o0 [ 2 ] ; real_T nrt0orp0ia ; real_T jljgpt1uuw ; real_T
kdu2kwuya5 [ 9 ] ; real_T mynalkxgfq ; } B ; typedef struct {
Critic_params_init goakn2ro5e ; real_T cwmkszu2vs [ 2 ] ; real_T i5u1fdujnc [
2 ] ; real_T ahaxjwkaux ; real_T dxdkwpugr2 [ 2 ] ; real_T fze0mnrtem [ 9 ] ;
struct { void * LoggedData ; } hw3gvza0vb ; struct { void * LoggedData [ 2 ]
; } dqtjfkuaog ; struct { void * LoggedData [ 2 ] ; } fhhie5ozyo ; struct {
void * LoggedData ; } cxrc1d0et0 ; struct { void * AQHandles ; } jtk0xk12y4 ;
struct { void * AQHandles ; } ievcol2m0f ; int32_T apvjyrk1kq ; int32_T
plmgzaywij ; uint32_T ps4bbnys1x ; uint32_T g40vxebjkh ; uint32_T hiuf1kla5p
[ 2 ] ; uint32_T e4fpa5ayvw ; uint32_T fd0joxmdqa ; uint32_T edos2yz5ud [ 2 ]
; uint32_T lou2quwhf5 [ 625 ] ; boolean_T kpge4pibdv ; boolean_T nabcrgmpdz ;
boolean_T hl4scohicx ; boolean_T kqolj3mtcj ; boolean_T nylg1rx1au ;
boolean_T pr0bsqeaxc ; boolean_T erqmr50ezx ; boolean_T muztya0zrn ;
boolean_T jcvyspp2y2 ; boolean_T c0ex1etr1z ; boolean_T lx4zkua40z ; } DW ;
typedef struct { rtwCAPI_ModelMappingInfo mmi ; } DataMapInfo ; struct P_ {
Critic_params_init critic_params_init ; real_T A [ 4 ] ; real_T B [ 2 ] ;
real_T C [ 4 ] ; real_T H_critic [ 9 ] ; real_T Herror [ 9 ] ; real_T K_lqr [
2 ] ; real_T BandLimitedWhiteNoise_seed ; real_T
DelayOneStep_InitialCondition ; real_T WhiteNoise_Mean ; real_T
WhiteNoise_StdDev ; real_T Output_Gain ; real_T SineWave_Amp ; real_T
SineWave_Bias ; real_T SineWave_Freq ; real_T SineWave_Phase ; real_T
SineWave1_Amp ; real_T SineWave1_Bias ; real_T SineWave1_Freq ; real_T
SineWave1_Phase ; real_T SineWave2_Amp ; real_T SineWave2_Bias ; real_T
SineWave2_Freq ; real_T SineWave2_Phase ; real_T SineWave3_Amp ; real_T
SineWave3_Bias ; real_T SineWave3_Freq ; real_T SineWave3_Phase ; real_T
SineWave4_Amp ; real_T SineWave4_Bias ; real_T SineWave4_Freq ; real_T
SineWave4_Phase ; real_T DiscreteStateSpace_D [ 2 ] ; real_T
DiscreteStateSpace_InitialCondition ; } ; extern const char_T *
RT_MEMORY_ALLOCATION_ERROR ; extern B rtB ; extern DW rtDW ; extern P rtP ;
extern mxArray * mr_DPGandLQRsimOnlyIdentH_GetDWork ( ) ; extern void
mr_DPGandLQRsimOnlyIdentH_SetDWork ( const mxArray * ssDW ) ; extern mxArray
* mr_DPGandLQRsimOnlyIdentH_GetSimStateDisallowedBlocks ( ) ; extern const
rtwCAPI_ModelMappingStaticInfo * DPGandLQRsimOnlyIdentH_GetCAPIStaticMap ( void
) ; extern SimStruct * const rtS ; extern DataMapInfo * rt_dataMapInfoPtr ;
extern rtwCAPI_ModelMappingInfo * rt_modelMapInfoPtr ; void MdlOutputs ( int_T
tid ) ; void MdlOutputsParameterSampleTime ( int_T tid ) ; void MdlUpdate ( int_T tid ) ; void MdlTerminate ( void ) ; void MdlInitializeSizes ( void ) ; void MdlInitializeSampleTimes ( void ) ; SimStruct * raccel_register_model ( ssExecutionInfo * executionInfo ) ;
#endif
