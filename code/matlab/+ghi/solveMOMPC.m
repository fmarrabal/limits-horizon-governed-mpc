function sol = solveMOMPC(M, x0, alpha)
%SOLVEMOMPC Resuelve el QP condensado con quadprog y devuelve la solucion.
%
%   J NUNCA se lee del epigrafo. Si alpha_i = 0, las variables de epigrafo del
%   objetivo i no aparecen en la funcion objetivo y el solver las deja en
%   cualquier punto factible por encima del valor verdadero. Se recalcula
%   siempre con ghi.costs, que es exacta y es la MISMA funcion que produce J_a.
%
%   sol = [] si el QP es infactible.

x0 = x0(:);  alpha = alpha(:);
qp = M.build(x0, alpha);

opts = optimoptions('quadprog', 'Display', 'none', ...
                    'Algorithm', 'interior-point-convex', ...
                    'OptimalityTolerance', 1e-12, ...
                    'ConstraintTolerance', 1e-12, ...
                    'StepTolerance', 1e-14, 'MaxIterations', 500);
% quadprog minimiza 1/2 z'Hz + f'z  s.a.  Az <= b : misma convencion.
[z, ~, flag] = quadprog(qp.G, qp.g, qp.Ain, qp.bin, [], [], [], [], [], opts);

if flag <= 0 || isempty(z)
    sol = [];
    return
end

U = reshape(z(1:M.nU), M.m, M.N);
X = reshape(M.Sx * x0 + M.Su * z(1:M.nU), M.n, M.N + 1);
J = ghi.costs(M.prob, x0, U);
sol = struct('U', U, 'X', X, 'J', J, 'V', alpha' * J, 'alpha', alpha, 'z', z);
end
