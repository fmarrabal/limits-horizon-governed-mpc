function adm = admissibleSet(M, x0, Ja, grid, tol)
%ADMISSIBLESET  A(x, J_a) = {alpha en Delta : V*(x,alpha) <= J_a}.
%
%   Se BARRE una malla en vez de proyectar. La razon no es pereza: V*(x,.) es
%   CONCAVA en alpha, asi que su subnivel NO es convexo en general y una
%   proyeccion convexa ordinaria no vale. Es tambien la razon por la que
%   Bemporad & Munoz de la Pena resuelven la seleccion de peso por enumeracion.
if nargin < 4 || isempty(grid), grid = linspace(0, 1, 21); end
if nargin < 5 || isempty(tol),  tol  = 1e-7; end

V = inf(1, numel(grid));
for j = 1:numel(grid)
    a = grid(j);
    s = ghi.solveMOMPC(M, x0, [1 - a; a]);
    if ~isempty(s), V(j) = s.V; end
end
adm = struct('grid', grid, 'V', V, 'feasible', V <= Ja + tol, 'Ja', Ja);
end
