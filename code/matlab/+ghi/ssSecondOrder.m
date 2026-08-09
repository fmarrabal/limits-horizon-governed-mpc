function [Ad, Bd, Cd, Dd] = ssSecondOrder(w0, zeta, dt)
%SSSECONDORDER Espacio de estados EXACTO del Verlet de velocidad implementado.
%
%   Estado (a, v), entrada ad, salida a. Se obtiene por linealidad evaluando el
%   paso en los vectores base, asi que por construccion coincide con el codigo
%   que corre en el lazo (ghi.Order2.step) y no con una idealizacion continua.
if nargin < 3, dt = 1.0; end
paso = @(a, v, ad) local_step(a, v, ad, w0, zeta, dt);
[a1, v1] = paso(1, 0, 0);
[a2, v2] = paso(0, 1, 0);
[a3, v3] = paso(0, 0, 1);
Ad = [a1 a2; v1 v2];
Bd = [a3; v3];
Cd = [1 0];
Dd = 0;
end

function [a2, v2] = local_step(a, v, ad, w0, zeta, dt)
acc = -2 * zeta * w0 * v - w0^2 * (a - ad);
a2  = a + dt * v + 0.5 * dt^2 * acc;
vh  = v + 0.5 * dt * acc;
v2  = vh + 0.5 * dt * (-2 * zeta * w0 * vh - w0^2 * (a2 - ad));
end
