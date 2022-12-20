% This function is a simple 2-dimensional moving average interpolator called by the csim_alm_inpaint_2D function.

function [out]=my_mov_avrg(inp, Mask)
% samp_inp=inp.*Mask;
n = 3;
h=(1/n^2)*ones(n);
Mask_log = logical(Mask);
out=imfilter(inp,h);
out(Mask_log) = inp(Mask_log);

end
       
