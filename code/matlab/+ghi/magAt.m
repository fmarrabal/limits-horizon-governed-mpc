function m = magAt(Ad, Bd, Cd, Dd, w, dt)
%MAGAT Modulo de la respuesta en frecuencia del sistema DISCRETO, en z=exp(jwdt).
if nargin < 6, dt = 1.0; end
z = exp(1i * w * dt);
H = Cd * ((z * eye(size(Ad, 1)) - Ad) \ Bd) + Dd;
m = abs(H);
end
