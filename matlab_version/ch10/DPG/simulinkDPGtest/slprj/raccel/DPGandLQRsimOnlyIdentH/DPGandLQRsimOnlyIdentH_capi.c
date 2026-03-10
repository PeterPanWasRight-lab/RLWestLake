#include "rtw_capi.h"
#ifdef HOST_CAPI_BUILD
#include "DPGandLQRsimOnlyIdentH_capi_host.h"
#define sizeof(...) ((size_t)(0xFFFF))
#undef rt_offsetof
#define rt_offsetof(s,el) ((uint16_T)(0xFFFF))
#define TARGET_CONST
#define TARGET_STRING(s) (s)
#ifndef SS_UINT64
#define SS_UINT64 19
#endif
#ifndef SS_INT64
#define SS_INT64 20
#endif
#else
#include "builtin_typeid_types.h"
#include "DPGandLQRsimOnlyIdentH.h"
#include "DPGandLQRsimOnlyIdentH_capi.h"
#include "DPGandLQRsimOnlyIdentH_private.h"
#ifdef LIGHT_WEIGHT_CAPI
#define TARGET_CONST
#define TARGET_STRING(s)               ((NULL))
#else
#define TARGET_CONST                   const
#define TARGET_STRING(s)               (s)
#endif
#endif
static const rtwCAPI_Signals rtBlockSignals [ ] = { { 0 , 1 , TARGET_STRING ( "DPGandLQRsimOnlyIdentH/Critic的参数辨识" ) , TARGET_STRING ( "" ) , 0 , 0 , 0 , 0 , 0 } , { 1 , 1 , TARGET_STRING ( "DPGandLQRsimOnlyIdentH/Critic的参数辨识" ) , TARGET_STRING ( "" ) , 1 , 0 , 1 , 0 , 0 } , { 2 , 2 , TARGET_STRING ( "DPGandLQRsimOnlyIdentH/Critic的参数辨识1" ) , TARGET_STRING ( "" ) , 1 , 0 , 1 , 0 , 0 } , { 3 , 2 , TARGET_STRING ( "DPGandLQRsimOnlyIdentH/Critic的参数辨识1" ) , TARGET_STRING ( "" ) , 3 , 0 , 1 , 0 , 0 } , { 4 , 0 , TARGET_STRING ( "DPGandLQRsimOnlyIdentH/Discrete State-Space" ) , TARGET_STRING ( "" ) , 0 , 0 , 2 , 0 , 1 } , { 5 , 0 , TARGET_STRING ( "DPGandLQRsimOnlyIdentH/Gain2" ) , TARGET_STRING ( "" ) , 0 , 0 , 2 , 0 , 1 } , { 6 , 0 , TARGET_STRING ( "DPGandLQRsimOnlyIdentH/LQR反馈" ) , TARGET_STRING ( "" ) , 0 , 0 , 1 , 0 , 1 } , { 7 , 0 , TARGET_STRING ( "DPGandLQRsimOnlyIdentH/Transpose4" ) , TARGET_STRING ( "" ) , 0 , 0 , 3 , 0 , 2 } , { 8 , 0 , TARGET_STRING ( "DPGandLQRsimOnlyIdentH/Transpose5" ) , TARGET_STRING ( "" ) , 0 , 0 , 0 , 0 , 0 } , { 9 , 0 , TARGET_STRING ( "DPGandLQRsimOnlyIdentH/Sum" ) , TARGET_STRING ( "" ) , 0 , 0 , 2 , 0 , 0 } , { 10 , 0 , TARGET_STRING ( "DPGandLQRsimOnlyIdentH/Sum2" ) , TARGET_STRING ( "" ) , 0 , 0 , 1 , 0 , 1 } , { 11 , 0 , TARGET_STRING ( "DPGandLQRsimOnlyIdentH/Sum3" ) , TARGET_STRING ( "" ) , 0 , 0 , 1 , 0 , 0 } , { 12 , 0 , TARGET_STRING ( "DPGandLQRsimOnlyIdentH/Delay One Step" ) , TARGET_STRING ( "" ) , 0 , 0 , 2 , 0 , 1 } , { 13 , 0 , TARGET_STRING ( "DPGandLQRsimOnlyIdentH/Band-Limited White Noise/Output" ) , TARGET_STRING ( "" ) , 0 , 0 , 1 , 0 , 1 } , { 0 , 0 , ( NULL ) , ( NULL ) , 0 , 0 , 0 , 0 , 0 } } ; static const rtwCAPI_BlockParameters rtBlockParameters [ ] = { { 14 , TARGET_STRING ( "DPGandLQRsimOnlyIdentH/Band-Limited White Noise" ) , TARGET_STRING ( "seed" ) , 0 , 1 , 0 } , { 15 , TARGET_STRING ( "DPGandLQRsimOnlyIdentH/Discrete State-Space" ) , TARGET_STRING ( "D" ) , 0 , 2 , 0 } , { 16 , TARGET_STRING ( "DPGandLQRsimOnlyIdentH/Discrete State-Space" ) , TARGET_STRING ( "InitialCondition" ) , 0 , 1 , 0 } , { 17 , TARGET_STRING ( "DPGandLQRsimOnlyIdentH/Sine Wave" ) , TARGET_STRING ( "Amplitude" ) , 0 , 1 , 0 } , { 18 , TARGET_STRING ( "DPGandLQRsimOnlyIdentH/Sine Wave" ) , TARGET_STRING ( "Bias" ) , 0 , 1 , 0 } , { 19 , TARGET_STRING ( "DPGandLQRsimOnlyIdentH/Sine Wave" ) , TARGET_STRING ( "Frequency" ) , 0 , 1 , 0 } , { 20 , TARGET_STRING ( "DPGandLQRsimOnlyIdentH/Sine Wave" ) , TARGET_STRING ( "Phase" ) , 0 , 1 , 0 } , { 21 , TARGET_STRING ( "DPGandLQRsimOnlyIdentH/Sine Wave1" ) , TARGET_STRING ( "Amplitude" ) , 0 , 1 , 0 } , { 22 , TARGET_STRING ( "DPGandLQRsimOnlyIdentH/Sine Wave1" ) , TARGET_STRING ( "Bias" ) , 0 , 1 , 0 } , { 23 , TARGET_STRING ( "DPGandLQRsimOnlyIdentH/Sine Wave1" ) , TARGET_STRING ( "Frequency" ) , 0 , 1 , 0 } , { 24 , TARGET_STRING ( "DPGandLQRsimOnlyIdentH/Sine Wave1" ) , TARGET_STRING ( "Phase" ) , 0 , 1 , 0 } , { 25 , TARGET_STRING ( "DPGandLQRsimOnlyIdentH/Sine Wave2" ) , TARGET_STRING ( "Amplitude" ) , 0 , 1 , 0 } , { 26 , TARGET_STRING ( "DPGandLQRsimOnlyIdentH/Sine Wave2" ) , TARGET_STRING ( "Bias" ) , 0 , 1 , 0 } , { 27 , TARGET_STRING ( "DPGandLQRsimOnlyIdentH/Sine Wave2" ) , TARGET_STRING ( "Frequency" ) , 0 , 1 , 0 } , { 28 , TARGET_STRING ( "DPGandLQRsimOnlyIdentH/Sine Wave2" ) , TARGET_STRING ( "Phase" ) , 0 , 1 , 0 } , { 29 , TARGET_STRING ( "DPGandLQRsimOnlyIdentH/Sine Wave3" ) , TARGET_STRING ( "Amplitude" ) , 0 , 1 , 0 } , { 30 , TARGET_STRING ( "DPGandLQRsimOnlyIdentH/Sine Wave3" ) , TARGET_STRING ( "Bias" ) , 0 , 1 , 0 } , { 31 , TARGET_STRING ( "DPGandLQRsimOnlyIdentH/Sine Wave3" ) , TARGET_STRING ( "Frequency" ) , 0 , 1 , 0 } , { 32 , TARGET_STRING ( "DPGandLQRsimOnlyIdentH/Sine Wave3" ) , TARGET_STRING ( "Phase" ) , 0 , 1 , 0 } , { 33 , TARGET_STRING ( "DPGandLQRsimOnlyIdentH/Sine Wave4" ) , TARGET_STRING ( "Amplitude" ) , 0 , 1 , 0 } , { 34 , TARGET_STRING ( "DPGandLQRsimOnlyIdentH/Sine Wave4" ) , TARGET_STRING ( "Bias" ) , 0 , 1 , 0 } , { 35 , TARGET_STRING ( "DPGandLQRsimOnlyIdentH/Sine Wave4" ) , TARGET_STRING ( "Frequency" ) , 0 , 1 , 0 } , { 36 , TARGET_STRING ( "DPGandLQRsimOnlyIdentH/Sine Wave4" ) , TARGET_STRING ( "Phase" ) , 0 , 1 , 0 } , { 37 , TARGET_STRING ( "DPGandLQRsimOnlyIdentH/Delay One Step" ) , TARGET_STRING ( "InitialCondition" ) , 0 , 1 , 0 } , { 38 , TARGET_STRING ( "DPGandLQRsimOnlyIdentH/Band-Limited White Noise/Output" ) , TARGET_STRING ( "Gain" ) , 0 , 1 , 0 } , { 39 , TARGET_STRING ( "DPGandLQRsimOnlyIdentH/Band-Limited White Noise/White Noise" ) , TARGET_STRING ( "Mean" ) , 0 , 1 , 0 } , { 40 , TARGET_STRING ( "DPGandLQRsimOnlyIdentH/Band-Limited White Noise/White Noise" ) , TARGET_STRING ( "StdDev" ) , 0 , 1 , 0 } , { 0 , ( NULL ) , ( NULL ) , 0 , 0 , 0 } } ; static int_T rt_LoggedStateIdxList [ ] = { - 1 } ; static const rtwCAPI_Signals rtRootInputs [ ] = { { 0 , 0 , ( NULL ) , ( NULL ) , 0 , 0 , 0 , 0 , 0 } } ; static const rtwCAPI_Signals rtRootOutputs [ ] = { { 0 , 0 , ( NULL ) , ( NULL ) , 0 , 0 , 0 , 0 , 0 } } ; static const rtwCAPI_ModelParameters rtModelParameters [ ] = { { 41 , TARGET_STRING ( "critic_params_init" ) , 1 , 1 , 0 } , { 42 , TARGET_STRING ( "A" ) , 0 , 8 , 0 } , { 43 , TARGET_STRING ( "B" ) , 0 , 2 , 0 } , { 44 , TARGET_STRING ( "C" ) , 0 , 8 , 0 } , { 45 , TARGET_STRING ( "H_critic" ) , 0 , 0 , 0 } , { 46 , TARGET_STRING ( "Herror" ) , 0 , 0 , 0 } , { 47 , TARGET_STRING ( "K_lqr" ) , 0 , 9 , 0 } , { 0 , ( NULL ) , 0 , 0 , 0 } } ;
#ifndef HOST_CAPI_BUILD
static void * rtDataAddrMap [ ] = { & rtB . kdu2kwuya5 [ 0 ] , & rtB .
mynalkxgfq , & rtB . nrt0orp0ia , & rtB . jljgpt1uuw , & rtB . fhsdyf2cqn [ 0
] , & rtB . iqc5no5xul [ 0 ] , & rtB . gyeu202tnh , & rtB . c214bb40o0 [ 0 ]
, & rtB . iphp0mrtyf [ 0 ] , & rtB . o4x0bb4qep [ 0 ] , & rtB . osgh5xcie0 ,
& rtB . d0ken2h5az , & rtB . k2qoybmvyf [ 0 ] , & rtB . a02qbwyxe5 , & rtP .
BandLimitedWhiteNoise_seed , & rtP . DiscreteStateSpace_D [ 0 ] , & rtP .
DiscreteStateSpace_InitialCondition , & rtP . SineWave_Amp , & rtP .
SineWave_Bias , & rtP . SineWave_Freq , & rtP . SineWave_Phase , & rtP .
SineWave1_Amp , & rtP . SineWave1_Bias , & rtP . SineWave1_Freq , & rtP .
SineWave1_Phase , & rtP . SineWave2_Amp , & rtP . SineWave2_Bias , & rtP .
SineWave2_Freq , & rtP . SineWave2_Phase , & rtP . SineWave3_Amp , & rtP .
SineWave3_Bias , & rtP . SineWave3_Freq , & rtP . SineWave3_Phase , & rtP .
SineWave4_Amp , & rtP . SineWave4_Bias , & rtP . SineWave4_Freq , & rtP .
SineWave4_Phase , & rtP . DelayOneStep_InitialCondition , & rtP . Output_Gain
, & rtP . WhiteNoise_Mean , & rtP . WhiteNoise_StdDev , & rtP .
critic_params_init , & rtP . A [ 0 ] , & rtP . B [ 0 ] , & rtP . C [ 0 ] , &
rtP . H_critic [ 0 ] , & rtP . Herror [ 0 ] , & rtP . K_lqr [ 0 ] , } ;
static int32_T * rtVarDimsAddrMap [ ] = { ( NULL ) } ;
#endif
static TARGET_CONST rtwCAPI_DataTypeMap rtDataTypeMap [ ] = { { "double" ,
"real_T" , 0 , 0 , sizeof ( real_T ) , ( uint8_T ) SS_DOUBLE , 0 , 0 , 0 } ,
{ "struct" , "Critic_params_init" , 6 , 1 , sizeof ( Critic_params_init ) , ( uint8_T ) SS_STRUCT , 0 , 0 , 0 } } ;
#ifdef HOST_CAPI_BUILD
#undef sizeof
#endif
static TARGET_CONST rtwCAPI_ElementMap rtElementMap [ ] = { { ( NULL ) , 0 ,
0 , 0 , 0 } , { "W1_c" , rt_offsetof ( Critic_params_init , W1_c ) , 0 , 4 ,
0 } , { "b1_c" , rt_offsetof ( Critic_params_init , b1_c ) , 0 , 5 , 0 } , {
"W2_c" , rt_offsetof ( Critic_params_init , W2_c ) , 0 , 6 , 0 } , { "b2_c" ,
rt_offsetof ( Critic_params_init , b2_c ) , 0 , 5 , 0 } , { "W3_c" ,
rt_offsetof ( Critic_params_init , W3_c ) , 0 , 7 , 0 } , { "b3_c" ,
rt_offsetof ( Critic_params_init , b3_c ) , 0 , 1 , 0 } } ; static const
rtwCAPI_DimensionMap rtDimensionMap [ ] = { { rtwCAPI_MATRIX_COL_MAJOR , 0 ,
2 , 0 } , { rtwCAPI_SCALAR , 2 , 2 , 0 } , { rtwCAPI_VECTOR , 4 , 2 , 0 } , {
rtwCAPI_MATRIX_COL_MAJOR , 6 , 2 , 0 } , { rtwCAPI_MATRIX_COL_MAJOR , 8 , 2 ,
0 } , { rtwCAPI_MATRIX_COL_MAJOR , 10 , 2 , 0 } , { rtwCAPI_MATRIX_COL_MAJOR
, 12 , 2 , 0 } , { rtwCAPI_MATRIX_COL_MAJOR , 14 , 2 , 0 } , {
rtwCAPI_MATRIX_COL_MAJOR , 16 , 2 , 0 } , { rtwCAPI_VECTOR , 6 , 2 , 0 } } ;
static const uint_T rtDimensionArray [ ] = { 3 , 3 , 1 , 1 , 2 , 1 , 1 , 2 ,
10 , 3 , 10 , 1 , 10 , 10 , 1 , 10 , 2 , 2 } ; static const real_T
rtcapiStoredFloats [ ] = { 0.0 , 0.01 } ; static const rtwCAPI_FixPtMap
rtFixPtMap [ ] = { { ( NULL ) , ( NULL ) , rtwCAPI_FIX_RESERVED , 0 , 0 , ( boolean_T ) 0 } , } ; static const rtwCAPI_SampleTimeMap rtSampleTimeMap [ ] = { { ( const void * ) & rtcapiStoredFloats [ 0 ] , ( const void * ) & rtcapiStoredFloats [ 0 ] , ( int8_T ) 0 , ( uint8_T ) 0 } , { ( const void * ) & rtcapiStoredFloats [ 1 ] , ( const void * ) & rtcapiStoredFloats [ 0 ] , ( int8_T ) 1 , ( uint8_T ) 0 } , { ( NULL ) , ( NULL ) , 2 , 0 } } ; static rtwCAPI_ModelMappingStaticInfo mmiStatic = { { rtBlockSignals , 14 , rtRootInputs , 0 , rtRootOutputs , 0 } , { rtBlockParameters , 27 , rtModelParameters , 7 } , { ( NULL ) , 0 } , { rtDataTypeMap , rtDimensionMap , rtFixPtMap , rtElementMap , rtSampleTimeMap , rtDimensionArray } , "float" , { 1290642410U , 1489452638U , 1477234421U , 454574533U } , ( NULL ) , 0 , ( boolean_T ) 0 , rt_LoggedStateIdxList } ; const rtwCAPI_ModelMappingStaticInfo * DPGandLQRsimOnlyIdentH_GetCAPIStaticMap ( void ) { return & mmiStatic ; }
#ifndef HOST_CAPI_BUILD
void DPGandLQRsimOnlyIdentH_InitializeDataMapInfo ( void ) {
rtwCAPI_SetVersion ( ( * rt_dataMapInfoPtr ) . mmi , 1 ) ;
rtwCAPI_SetStaticMap ( ( * rt_dataMapInfoPtr ) . mmi , & mmiStatic ) ;
rtwCAPI_SetLoggingStaticMap ( ( * rt_dataMapInfoPtr ) . mmi , ( NULL ) ) ;
rtwCAPI_SetDataAddressMap ( ( * rt_dataMapInfoPtr ) . mmi , rtDataAddrMap ) ;
rtwCAPI_SetVarDimsAddressMap ( ( * rt_dataMapInfoPtr ) . mmi ,
rtVarDimsAddrMap ) ; rtwCAPI_SetInstanceLoggingInfo ( ( * rt_dataMapInfoPtr )
. mmi , ( NULL ) ) ; rtwCAPI_SetChildMMIArray ( ( * rt_dataMapInfoPtr ) . mmi
, ( NULL ) ) ; rtwCAPI_SetChildMMIArrayLen ( ( * rt_dataMapInfoPtr ) . mmi ,
0 ) ; }
#else
#ifdef __cplusplus
extern "C" {
#endif
void DPGandLQRsimOnlyIdentH_host_InitializeDataMapInfo ( DPGandLQRsimOnlyIdentH_host_DataMapInfo_T * dataMap , const char * path ) { rtwCAPI_SetVersion ( dataMap -> mmi , 1 ) ; rtwCAPI_SetStaticMap ( dataMap -> mmi , & mmiStatic ) ; rtwCAPI_SetDataAddressMap ( dataMap -> mmi , ( NULL ) ) ; rtwCAPI_SetVarDimsAddressMap ( dataMap -> mmi , ( NULL ) ) ; rtwCAPI_SetPath ( dataMap -> mmi , path ) ; rtwCAPI_SetFullPath ( dataMap -> mmi , ( NULL ) ) ; rtwCAPI_SetChildMMIArray ( dataMap -> mmi , ( NULL ) ) ; rtwCAPI_SetChildMMIArrayLen ( dataMap -> mmi , 0 ) ; }
#ifdef __cplusplus
}
#endif
#endif
