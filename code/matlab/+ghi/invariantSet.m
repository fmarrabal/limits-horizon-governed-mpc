function [H, k, it] = invariantSet(plant, Kf, maxIter, tol)
%INVARIANTSET Conjunto maximo positivamente invariante (Gilbert & Tan, 1991).
%
%   Devuelve Omega = {x : H x <= k}, el mayor conjunto contenido en
%   {|x| <= xmax, |Kf x| <= umax} e invariante bajo x+ = (A + B Kf) x.
%
%   O_0 = X ;  O_{j+1} = O_j ^ Acl^{-1} O_j ;  parar cuando anadir Acl^{j+1}
%   no recorta nada.

if nargin < 3 || isempty(maxIter), maxIter = 60;  end
if nargin < 4 || isempty(tol),     tol     = 1e-9; end

n   = plant.n;
Acl = plant.A + plant.B * Kf;
H0  = [eye(n); -eye(n); Kf; -Kf];
k0  = [plant.xmax(:); plant.xmax(:); plant.umax(:); plant.umax(:)];
H = H0;  k = k0;
Apow = eye(n);
opts = optimoptions('linprog', 'Display', 'none');

for it = 1:maxIter
    Apow = Acl * Apow;
    Hn = H0 * Apow;  kn = k0;
    redundant = true;
    for j = 1:size(Hn, 1)
        [~, fval, flag] = linprog(-Hn(j, :)', H, k, [], [], [], [], opts);
        if flag ~= 1 || (-fval) > kn(j) + tol
            redundant = false;
            break
        end
    end
    if redundant
        [H, k] = ghi.removeRedundant(H, k);
        return
    end
    H = [H; Hn];  k = [k; kn];   %#ok<AGROW>
end
error('ghi:invariantSet', 'Gilbert-Tan no convergio en %d iteraciones', maxIter);
end
