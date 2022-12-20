% This function is a simple 2-dimensional moving average interpolator called by the IMATI function.

function [out]=mov_avrg(inp, Mask)
samp_inp=inp.*Mask;
[as1,as2]=size(inp);
weight_window=(1/9)*[1 1 1; 1 1 1;1 1 1];
out=inp;
for i1=3:as1-2
    for j1=3:as2-2
        if Mask(i1,j1)==0
            inp_window=[samp_inp(i1-1,j1-1) samp_inp(i1-1,j1) samp_inp(i1-1,j1+1);samp_inp(i1,j1-1) samp_inp(i1,j1) samp_inp(i1,j1+1);samp_inp(i1+1,j1-1) samp_inp(i1+1,j1) samp_inp(i1+1,j1+1)];
            out(i1,j1)=sum(sum(inp_window.*weight_window));
        end
    end
end
       
