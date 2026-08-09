function r = rhoG(A, b, beta)
%RHOG Radio espectral de G = bA + beta A^3.
%   A antisimetrica tiene autovalores i*mu_k, luego A^3 los tiene -i*mu_k^3 y
%   G los tiene i*(b mu_k - beta mu_k^3).
mu = abs(imag(eig(A)));
r  = max(abs(b * mu - beta * mu.^3));
end
