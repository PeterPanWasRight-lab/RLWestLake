classdef MyGridWorld < rl.env.MATLABEnvironment
    %MYGRIDWORLD: Template for defining custom environment in MATLAB.    
    
    %% Properties (set properties' attributes accordingly)
    properties
       Gridsize = "[5,5]"
       CurrentState = "[1,1]"
       BlockState = "[3,5],[4,5],[5,3]"
       TerminalState = "[4,2]"
       currentstatenum = []
       terminalstatenum = []
       blocknum = []
       gridmatrix = []
       action = []
       State = []

    end
    
    properties(Access = protected)
        % Initialize internal flag to indicate episode termination
        IsDone = false        
    end

    %% Necessary Methods
    methods              
        % Contructor method creates an instance of the environment
        % Change class name and constructor name accordingly
        function this = MyGridWorld()
            % Initialize Observation settings
            ObservationInfo = rlNumericSpec(1);
            ObservationInfo.Name = 'Grid World';
                        
            % Initialize Action settings   
            ActionInfo = rlFiniteSetSpec([-7 -1 1 7]);
            ActionInfo.Name = 'Grid Action';
            
            % The following line implements built-in functions of RL env
            this = this@rl.env.MATLABEnvironment(ObservationInfo,ActionInfo);
        end
        
        % Apply system dynamics and simulates the environment with the 
        % given action for one step.
        function [Observation,Reward,IsDone,Info] = step(this,Action)
            Info = [];
            this.action = Action;
            this.currentstatenum = judgestate(this);
            Reward = this.gridmatrix(this.currentstatenum);
            Observation = this.currentstatenum;
            if Reward==10
                IsDone = 1;
            else
                IsDone = 0;
            end            
            this.State = Observation;
        end
        
        % Reset environment to initial state and output initial observation
        function InitialObservation = reset(this)
           Grid_size = str2num(this.Gridsize);

           % currentstatenum
           tmp = str2num(this.CurrentState);
           tmp = tmp + 1;
           this.currentstatenum = (tmp(2)-1)*(Grid_size(1)+2)+tmp(1);

           % terminalstatenum
           tmp = str2num(this.TerminalState);
           tmp = tmp + 1;
           this.terminalstatenum = (tmp(2)-1)*(Grid_size(1)+2)+tmp(1);

           % blocknum
           tmp1 = 1:1:Grid_size(1)+2;
           tmp2 = 1:Grid_size(1)+2:(Grid_size(2)+2-1)*(Grid_size(1)+2)+1;
           tmp3 = Grid_size(1)+2:Grid_size(1)+2:(Grid_size(1)+2)*(Grid_size(2)+2);
           tmp4 = (Grid_size(2)+2-1)*(Grid_size(1)+2)+1:1:(Grid_size(1)+2)*(Grid_size(2)+2);
           tmp5 = str2num(this.BlockState);
           tmp5 = tmp5 + 1;
           num = length(tmp5)/2;
           tmp6 =zeros(1,num);
           for i = 1:num
               tmp6(i) = (tmp5(2*i)-1)*(Grid_size(1)+2)+tmp5(2*i-1);
           end
           combined = [tmp1,tmp2,tmp3,tmp4,tmp6];
           this.blocknum = sort(unique(combined));

           this.gridmatrix = ones(Grid_size(1)+2,Grid_size(2)+2);
           this.gridmatrix = -this.gridmatrix;
           this.gridmatrix(this.blocknum) = -10;
           this.gridmatrix(this.terminalstatenum) = 10;

           InitialObservation = this.currentstatenum;
        end
    end
    %% Optional Methods (set methods' attributes accordingly)
    methods
        function State = judgestate(this)
            if this.gridmatrix(this.currentstatenum+this.action)==-10
                State = this.currentstatenum;
            else
                State = this.currentstatenum + this.action;
            end
        end
       
    end
    
end
