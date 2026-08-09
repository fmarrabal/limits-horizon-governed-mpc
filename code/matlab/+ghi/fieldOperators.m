function [K, C, G] = fieldOperators(L, A, p)
%FIELDOPERATORS Operadores del campo homeostatico sobre el grafo.
%
%   K = w0^2 I + c^2 L      rigidez: reaccion + difusion espacial
%   C = 2 zeta w0 I + D L   disipacion: uniforme + estructural
%   G = b A + beta A^3      antisimetricos: adveccion + dispersion tipo KdV
%
%   REGLA DE COLOCACION (clasica, NO es un resultado propio): G va sobre la
%   VELOCIDAD (giroscopico, seguro por Kelvin-Tait-Chetaev porque u'^T G u' = 0)
%   y nunca sobre la POSICION (circulatorio, produce flutter). Ver Kirillov,
%   Doklady Mathematics 76(2):780-785 (2007) y Udwadia, ASME J. Appl. Mech.
%   86(2):021002 (2019).
M = size(L, 1);
I = eye(M);
K = p.w0^2 * I + p.c^2 * L;
C = 2 * p.zeta * p.w0 * I + p.D * L;
G = p.b * A + p.beta * (A * A * A);
end
