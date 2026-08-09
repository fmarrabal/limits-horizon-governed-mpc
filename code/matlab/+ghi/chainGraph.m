function [L, A] = chainGraph(M)
%CHAINGRAPH Cadena dirigida 1 -> 2 -> ... -> M.
%
%   Analogo de un lazo solar, una linea de proceso o una hilera de modulos de
%   invernadero. La adveccion A apunta en el sentido del flujo del proceso, que
%   en una planta es un DATO FISICO, no una convencion -- a diferencia del
%   grafo de modulos de un transformer, donde la direccion es arbitraria.
assert(M >= 2, 'hacen falta al menos 2 nodos');
Adj = zeros(M);
A   = zeros(M);
for i = 1:M-1
    Adj(i, i+1) = 1;  Adj(i+1, i) = 1;
    A(i, i+1)   = 1;  A(i+1, i)   = -1;
end
L = diag(sum(Adj, 2)) - Adj;
end
