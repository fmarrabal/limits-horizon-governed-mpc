function J = costs(prob, x0, U)
%COSTS Vector de costes J(U,x0) sobre una secuencia de control DADA.
%
%   U es (m x N). Es la funcion que produce J_a a partir de la secuencia
%   desplazada, asi que debe ser exacta y coincidir con la del QP. Por eso
%   J nunca se lee del epigrafo del QP: siempre se recalcula aqui.

N = prob.N;
x = x0(:);
J = zeros(numel(prob.obj), 1);
for k = 1:N
    u = U(:, k);
    for i = 1:numel(prob.obj)
        J(i) = J(i) + ghi.stageCost(prob.obj(i), x, u);
    end
    x = prob.plant.A * x + prob.plant.B * u;
end
for i = 1:numel(prob.obj)
    J(i) = J(i) + ghi.terminalCost(prob.obj(i), x);
end
end
