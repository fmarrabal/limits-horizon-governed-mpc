function p = phaseAt(Ad, Bd, Cd, Dd, w, dt)
%PHASEAT Fase de la respuesta en frecuencia del sistema DISCRETO.
if nargin < 6, dt = 1.0; end
z = exp(1i * w * dt);
H = Cd * ((z * eye(size(Ad, 1)) - Ad) \ Bd) + Dd;
p = angle(H);
end
