function [tau, g] = matchFirstOrder(w0, zeta, dt, wMatch)
%MATCHFIRSTORDER  tau tal que |H1| = |H2| a la frecuencia de igualacion.
%
%   AMBOS en tiempo DISCRETO. Igualar sobre las transferencias continuas es un
%   error cuando w0*dt no es << 1: en el banco w0*dt = 0.9 y el desajuste real
%   que eso introducia era de 1.69x.
%
%   Sin igualacion la comparacion 1er vs 2o orden no dice nada: siempre se
%   puede hacer un filtro mas lento. Igualados en atenuacion a Nyquist, la
%   comparacion pasa a ser de FASE y de ganancia en la banda util.
if nargin < 3 || isempty(dt),     dt = 1.0;        end
if nargin < 4 || isempty(wMatch), wMatch = pi/dt;  end

[A2, B2, C2, D2] = ghi.ssSecondOrder(w0, zeta, dt);
g = ghi.magAt(A2, B2, C2, D2, wMatch, dt);
assert(g > 0 && g < 1, 'atenuacion fuera de rango: g = %g', g);

z = exp(1i * wMatch * dt);
if abs(z + 1) < 1e-12          % Nyquist exacto: forma cerrada
    tau = dt * (1/g + 1) / 2;
else                            % biseccion generica
    lo = dt * 0.5 + 1e-9;  hi = 1e6;
    for it = 1:200
        mid = 0.5 * (lo + hi);
        [A1, B1, C1, D1] = ghi.ssFirstOrder(mid, dt);
        if ghi.magAt(A1, B1, C1, D1, wMatch, dt) > g
            lo = mid;
        else
            hi = mid;
        end
    end
    tau = 0.5 * (lo + hi);
end
end
