#include "DPGandLQRsimOnlyIdentH_capi_host.h"
static DPGandLQRsimOnlyIdentH_host_DataMapInfo_T root;
static int initialized = 0;
__declspec( dllexport ) rtwCAPI_ModelMappingInfo *getRootMappingInfo()
{
    if (initialized == 0) {
        initialized = 1;
        DPGandLQRsimOnlyIdentH_host_InitializeDataMapInfo(&(root), "DPGandLQRsimOnlyIdentH");
    }
    return &root.mmi;
}

rtwCAPI_ModelMappingInfo *mexFunction(){return(getRootMappingInfo());}
