function [Hr, kr] = removeRedundant(H, k, tol)
%REMOVEREDUNDANT Elimina desigualdades redundantes resolviendo un LP por fila.
%   Se replica exactamente el criterio de la version Python para que el numero
%   y el orden de las filas de Omega coincidan entre ambas implementaciones.
if nargin < 3, tol = 1e-9; end
nr = size(H, 1);
keep = false(1, nr);
opts = optimoptions('linprog', 'Display', 'none');
for j = 1:nr
    idx = [];
    for i = 1:nr
        if i ~= j && ((keep(i)) || i > j)
            idx(end+1) = i; %#ok<AGROW>
        end
    end
    if isempty(idx)
        keep(j) = true;
        continue
    end
    [~, fval, flag] = linprog(-H(j, :)', H(idx, :), k(idx), [], [], [], [], opts);
    if flag ~= 1
        keep(j) = true;
    elseif (-fval) > k(j) + tol
        keep(j) = true;
    end
end
if ~any(keep), keep = true(1, nr); end
Hr = H(keep, :);
kr = k(keep);
end
