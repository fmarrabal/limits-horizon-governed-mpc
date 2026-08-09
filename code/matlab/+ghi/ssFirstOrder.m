function [Ad, Bd, Cd, Dd] = ssFirstOrder(tau, dt)
%SSFIRSTORDER  a_{k+1} = a_k + (dt/tau)(ad - a_k).
if nargin < 2, dt = 1.0; end
g  = dt / tau;
Ad = 1 - g;  Bd = g;  Cd = 1;  Dd = 0;
end
