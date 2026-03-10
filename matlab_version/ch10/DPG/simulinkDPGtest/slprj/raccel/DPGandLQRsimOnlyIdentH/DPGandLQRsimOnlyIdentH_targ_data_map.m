    function targMap = targDataMap(),

    ;%***********************
    ;% Create Parameter Map *
    ;%***********************
    
        nTotData      = 0; %add to this count as we go
        nTotSects     = 2;
        sectIdxOffset = 0;

        ;%
        ;% Define dummy sections & preallocate arrays
        ;%
        dumSection.nData = -1;
        dumSection.data  = [];

        dumData.logicalSrcIdx = -1;
        dumData.dtTransOffset = -1;

        ;%
        ;% Init/prealloc paramMap
        ;%
        paramMap.nSections           = nTotSects;
        paramMap.sectIdxOffset       = sectIdxOffset;
            paramMap.sections(nTotSects) = dumSection; %prealloc
        paramMap.nTotData            = -1;

        ;%
        ;% Auto data (rtP)
        ;%
            section.nData     = 1;
            section.data(1)  = dumData; %prealloc

                    ;% rtP.critic_params_init
                    section.data(1).logicalSrcIdx = 0;
                    section.data(1).dtTransOffset = 0;

            nTotData = nTotData + section.nData;
            paramMap.sections(1) = section;
            clear section

            section.nData     = 33;
            section.data(33)  = dumData; %prealloc

                    ;% rtP.A
                    section.data(1).logicalSrcIdx = 1;
                    section.data(1).dtTransOffset = 0;

                    ;% rtP.B
                    section.data(2).logicalSrcIdx = 2;
                    section.data(2).dtTransOffset = 4;

                    ;% rtP.C
                    section.data(3).logicalSrcIdx = 3;
                    section.data(3).dtTransOffset = 6;

                    ;% rtP.H_critic
                    section.data(4).logicalSrcIdx = 4;
                    section.data(4).dtTransOffset = 10;

                    ;% rtP.Herror
                    section.data(5).logicalSrcIdx = 5;
                    section.data(5).dtTransOffset = 19;

                    ;% rtP.K_lqr
                    section.data(6).logicalSrcIdx = 6;
                    section.data(6).dtTransOffset = 28;

                    ;% rtP.BandLimitedWhiteNoise_seed
                    section.data(7).logicalSrcIdx = 7;
                    section.data(7).dtTransOffset = 30;

                    ;% rtP.DelayOneStep_InitialCondition
                    section.data(8).logicalSrcIdx = 8;
                    section.data(8).dtTransOffset = 31;

                    ;% rtP.WhiteNoise_Mean
                    section.data(9).logicalSrcIdx = 9;
                    section.data(9).dtTransOffset = 32;

                    ;% rtP.WhiteNoise_StdDev
                    section.data(10).logicalSrcIdx = 10;
                    section.data(10).dtTransOffset = 33;

                    ;% rtP.Output_Gain
                    section.data(11).logicalSrcIdx = 11;
                    section.data(11).dtTransOffset = 34;

                    ;% rtP.SineWave_Amp
                    section.data(12).logicalSrcIdx = 12;
                    section.data(12).dtTransOffset = 35;

                    ;% rtP.SineWave_Bias
                    section.data(13).logicalSrcIdx = 13;
                    section.data(13).dtTransOffset = 36;

                    ;% rtP.SineWave_Freq
                    section.data(14).logicalSrcIdx = 14;
                    section.data(14).dtTransOffset = 37;

                    ;% rtP.SineWave_Phase
                    section.data(15).logicalSrcIdx = 15;
                    section.data(15).dtTransOffset = 38;

                    ;% rtP.SineWave1_Amp
                    section.data(16).logicalSrcIdx = 16;
                    section.data(16).dtTransOffset = 39;

                    ;% rtP.SineWave1_Bias
                    section.data(17).logicalSrcIdx = 17;
                    section.data(17).dtTransOffset = 40;

                    ;% rtP.SineWave1_Freq
                    section.data(18).logicalSrcIdx = 18;
                    section.data(18).dtTransOffset = 41;

                    ;% rtP.SineWave1_Phase
                    section.data(19).logicalSrcIdx = 19;
                    section.data(19).dtTransOffset = 42;

                    ;% rtP.SineWave2_Amp
                    section.data(20).logicalSrcIdx = 20;
                    section.data(20).dtTransOffset = 43;

                    ;% rtP.SineWave2_Bias
                    section.data(21).logicalSrcIdx = 21;
                    section.data(21).dtTransOffset = 44;

                    ;% rtP.SineWave2_Freq
                    section.data(22).logicalSrcIdx = 22;
                    section.data(22).dtTransOffset = 45;

                    ;% rtP.SineWave2_Phase
                    section.data(23).logicalSrcIdx = 23;
                    section.data(23).dtTransOffset = 46;

                    ;% rtP.SineWave3_Amp
                    section.data(24).logicalSrcIdx = 24;
                    section.data(24).dtTransOffset = 47;

                    ;% rtP.SineWave3_Bias
                    section.data(25).logicalSrcIdx = 25;
                    section.data(25).dtTransOffset = 48;

                    ;% rtP.SineWave3_Freq
                    section.data(26).logicalSrcIdx = 26;
                    section.data(26).dtTransOffset = 49;

                    ;% rtP.SineWave3_Phase
                    section.data(27).logicalSrcIdx = 27;
                    section.data(27).dtTransOffset = 50;

                    ;% rtP.SineWave4_Amp
                    section.data(28).logicalSrcIdx = 28;
                    section.data(28).dtTransOffset = 51;

                    ;% rtP.SineWave4_Bias
                    section.data(29).logicalSrcIdx = 29;
                    section.data(29).dtTransOffset = 52;

                    ;% rtP.SineWave4_Freq
                    section.data(30).logicalSrcIdx = 30;
                    section.data(30).dtTransOffset = 53;

                    ;% rtP.SineWave4_Phase
                    section.data(31).logicalSrcIdx = 31;
                    section.data(31).dtTransOffset = 54;

                    ;% rtP.DiscreteStateSpace_D
                    section.data(32).logicalSrcIdx = 32;
                    section.data(32).dtTransOffset = 55;

                    ;% rtP.DiscreteStateSpace_InitialCondition
                    section.data(33).logicalSrcIdx = 33;
                    section.data(33).dtTransOffset = 57;

            nTotData = nTotData + section.nData;
            paramMap.sections(2) = section;
            clear section


            ;%
            ;% Non-auto Data (parameter)
            ;%


        ;%
        ;% Add final counts to struct.
        ;%
        paramMap.nTotData = nTotData;



    ;%**************************
    ;% Create Block Output Map *
    ;%**************************
    
        nTotData      = 0; %add to this count as we go
        nTotSects     = 1;
        sectIdxOffset = 0;

        ;%
        ;% Define dummy sections & preallocate arrays
        ;%
        dumSection.nData = -1;
        dumSection.data  = [];

        dumData.logicalSrcIdx = -1;
        dumData.dtTransOffset = -1;

        ;%
        ;% Init/prealloc sigMap
        ;%
        sigMap.nSections           = nTotSects;
        sigMap.sectIdxOffset       = sectIdxOffset;
            sigMap.sections(nTotSects) = dumSection; %prealloc
        sigMap.nTotData            = -1;

        ;%
        ;% Auto data (rtB)
        ;%
            section.nData     = 14;
            section.data(14)  = dumData; %prealloc

                    ;% rtB.k2qoybmvyf
                    section.data(1).logicalSrcIdx = 1;
                    section.data(1).dtTransOffset = 0;

                    ;% rtB.gyeu202tnh
                    section.data(2).logicalSrcIdx = 2;
                    section.data(2).dtTransOffset = 2;

                    ;% rtB.a02qbwyxe5
                    section.data(3).logicalSrcIdx = 3;
                    section.data(3).dtTransOffset = 3;

                    ;% rtB.d0ken2h5az
                    section.data(4).logicalSrcIdx = 4;
                    section.data(4).dtTransOffset = 4;

                    ;% rtB.iqc5no5xul
                    section.data(5).logicalSrcIdx = 5;
                    section.data(5).dtTransOffset = 5;

                    ;% rtB.o4x0bb4qep
                    section.data(6).logicalSrcIdx = 6;
                    section.data(6).dtTransOffset = 7;

                    ;% rtB.osgh5xcie0
                    section.data(7).logicalSrcIdx = 7;
                    section.data(7).dtTransOffset = 9;

                    ;% rtB.fhsdyf2cqn
                    section.data(8).logicalSrcIdx = 9;
                    section.data(8).dtTransOffset = 10;

                    ;% rtB.iphp0mrtyf
                    section.data(9).logicalSrcIdx = 10;
                    section.data(9).dtTransOffset = 12;

                    ;% rtB.c214bb40o0
                    section.data(10).logicalSrcIdx = 11;
                    section.data(10).dtTransOffset = 21;

                    ;% rtB.nrt0orp0ia
                    section.data(11).logicalSrcIdx = 12;
                    section.data(11).dtTransOffset = 23;

                    ;% rtB.jljgpt1uuw
                    section.data(12).logicalSrcIdx = 14;
                    section.data(12).dtTransOffset = 24;

                    ;% rtB.kdu2kwuya5
                    section.data(13).logicalSrcIdx = 15;
                    section.data(13).dtTransOffset = 25;

                    ;% rtB.mynalkxgfq
                    section.data(14).logicalSrcIdx = 16;
                    section.data(14).dtTransOffset = 34;

            nTotData = nTotData + section.nData;
            sigMap.sections(1) = section;
            clear section


            ;%
            ;% Non-auto Data (signal)
            ;%


        ;%
        ;% Add final counts to struct.
        ;%
        sigMap.nTotData = nTotData;



    ;%*******************
    ;% Create DWork Map *
    ;%*******************
    
        nTotData      = 0; %add to this count as we go
        nTotSects     = 6;
        sectIdxOffset = 1;

        ;%
        ;% Define dummy sections & preallocate arrays
        ;%
        dumSection.nData = -1;
        dumSection.data  = [];

        dumData.logicalSrcIdx = -1;
        dumData.dtTransOffset = -1;

        ;%
        ;% Init/prealloc dworkMap
        ;%
        dworkMap.nSections           = nTotSects;
        dworkMap.sectIdxOffset       = sectIdxOffset;
            dworkMap.sections(nTotSects) = dumSection; %prealloc
        dworkMap.nTotData            = -1;

        ;%
        ;% Auto data (rtDW)
        ;%
            section.nData     = 1;
            section.data(1)  = dumData; %prealloc

                    ;% rtDW.goakn2ro5e
                    section.data(1).logicalSrcIdx = 0;
                    section.data(1).dtTransOffset = 0;

            nTotData = nTotData + section.nData;
            dworkMap.sections(1) = section;
            clear section

            section.nData     = 5;
            section.data(5)  = dumData; %prealloc

                    ;% rtDW.cwmkszu2vs
                    section.data(1).logicalSrcIdx = 1;
                    section.data(1).dtTransOffset = 0;

                    ;% rtDW.i5u1fdujnc
                    section.data(2).logicalSrcIdx = 2;
                    section.data(2).dtTransOffset = 2;

                    ;% rtDW.ahaxjwkaux
                    section.data(3).logicalSrcIdx = 3;
                    section.data(3).dtTransOffset = 4;

                    ;% rtDW.dxdkwpugr2
                    section.data(4).logicalSrcIdx = 4;
                    section.data(4).dtTransOffset = 5;

                    ;% rtDW.fze0mnrtem
                    section.data(5).logicalSrcIdx = 5;
                    section.data(5).dtTransOffset = 7;

            nTotData = nTotData + section.nData;
            dworkMap.sections(2) = section;
            clear section

            section.nData     = 6;
            section.data(6)  = dumData; %prealloc

                    ;% rtDW.hw3gvza0vb.LoggedData
                    section.data(1).logicalSrcIdx = 6;
                    section.data(1).dtTransOffset = 0;

                    ;% rtDW.dqtjfkuaog.LoggedData
                    section.data(2).logicalSrcIdx = 7;
                    section.data(2).dtTransOffset = 1;

                    ;% rtDW.fhhie5ozyo.LoggedData
                    section.data(3).logicalSrcIdx = 8;
                    section.data(3).dtTransOffset = 3;

                    ;% rtDW.cxrc1d0et0.LoggedData
                    section.data(4).logicalSrcIdx = 9;
                    section.data(4).dtTransOffset = 5;

                    ;% rtDW.jtk0xk12y4.AQHandles
                    section.data(5).logicalSrcIdx = 10;
                    section.data(5).dtTransOffset = 6;

                    ;% rtDW.ievcol2m0f.AQHandles
                    section.data(6).logicalSrcIdx = 11;
                    section.data(6).dtTransOffset = 7;

            nTotData = nTotData + section.nData;
            dworkMap.sections(3) = section;
            clear section

            section.nData     = 2;
            section.data(2)  = dumData; %prealloc

                    ;% rtDW.apvjyrk1kq
                    section.data(1).logicalSrcIdx = 12;
                    section.data(1).dtTransOffset = 0;

                    ;% rtDW.plmgzaywij
                    section.data(2).logicalSrcIdx = 13;
                    section.data(2).dtTransOffset = 1;

            nTotData = nTotData + section.nData;
            dworkMap.sections(4) = section;
            clear section

            section.nData     = 7;
            section.data(7)  = dumData; %prealloc

                    ;% rtDW.ps4bbnys1x
                    section.data(1).logicalSrcIdx = 14;
                    section.data(1).dtTransOffset = 0;

                    ;% rtDW.g40vxebjkh
                    section.data(2).logicalSrcIdx = 15;
                    section.data(2).dtTransOffset = 1;

                    ;% rtDW.hiuf1kla5p
                    section.data(3).logicalSrcIdx = 16;
                    section.data(3).dtTransOffset = 2;

                    ;% rtDW.e4fpa5ayvw
                    section.data(4).logicalSrcIdx = 17;
                    section.data(4).dtTransOffset = 4;

                    ;% rtDW.fd0joxmdqa
                    section.data(5).logicalSrcIdx = 18;
                    section.data(5).dtTransOffset = 5;

                    ;% rtDW.edos2yz5ud
                    section.data(6).logicalSrcIdx = 19;
                    section.data(6).dtTransOffset = 6;

                    ;% rtDW.lou2quwhf5
                    section.data(7).logicalSrcIdx = 20;
                    section.data(7).dtTransOffset = 8;

            nTotData = nTotData + section.nData;
            dworkMap.sections(5) = section;
            clear section

            section.nData     = 11;
            section.data(11)  = dumData; %prealloc

                    ;% rtDW.kpge4pibdv
                    section.data(1).logicalSrcIdx = 21;
                    section.data(1).dtTransOffset = 0;

                    ;% rtDW.nabcrgmpdz
                    section.data(2).logicalSrcIdx = 22;
                    section.data(2).dtTransOffset = 1;

                    ;% rtDW.hl4scohicx
                    section.data(3).logicalSrcIdx = 23;
                    section.data(3).dtTransOffset = 2;

                    ;% rtDW.kqolj3mtcj
                    section.data(4).logicalSrcIdx = 24;
                    section.data(4).dtTransOffset = 3;

                    ;% rtDW.nylg1rx1au
                    section.data(5).logicalSrcIdx = 25;
                    section.data(5).dtTransOffset = 4;

                    ;% rtDW.pr0bsqeaxc
                    section.data(6).logicalSrcIdx = 26;
                    section.data(6).dtTransOffset = 5;

                    ;% rtDW.erqmr50ezx
                    section.data(7).logicalSrcIdx = 27;
                    section.data(7).dtTransOffset = 6;

                    ;% rtDW.muztya0zrn
                    section.data(8).logicalSrcIdx = 28;
                    section.data(8).dtTransOffset = 7;

                    ;% rtDW.jcvyspp2y2
                    section.data(9).logicalSrcIdx = 29;
                    section.data(9).dtTransOffset = 8;

                    ;% rtDW.c0ex1etr1z
                    section.data(10).logicalSrcIdx = 30;
                    section.data(10).dtTransOffset = 9;

                    ;% rtDW.lx4zkua40z
                    section.data(11).logicalSrcIdx = 31;
                    section.data(11).dtTransOffset = 10;

            nTotData = nTotData + section.nData;
            dworkMap.sections(6) = section;
            clear section


            ;%
            ;% Non-auto Data (dwork)
            ;%


        ;%
        ;% Add final counts to struct.
        ;%
        dworkMap.nTotData = nTotData;



    ;%
    ;% Add individual maps to base struct.
    ;%

    targMap.paramMap  = paramMap;
    targMap.signalMap = sigMap;
    targMap.dworkMap  = dworkMap;

    ;%
    ;% Add checksums to base struct.
    ;%


    targMap.checksum0 = 1290642410;
    targMap.checksum1 = 1489452638;
    targMap.checksum2 = 1477234421;
    targMap.checksum3 = 454574533;

