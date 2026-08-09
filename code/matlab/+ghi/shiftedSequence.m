function Us = shiftedSequence(M, sol)
%SHIFTEDSEQUENCE  U_s = [u_1*, ..., u_{N-1}*, Kf x_N*].
%   Es la cola desplazada que produce J_a y con ella el certificado heredado.
xN = sol.X(:, M.N + 1);
Us = [sol.U(:, 2:end), M.term.Kf * xN];
umax = M.prob.plant.umax(:);
Us = max(min(Us, repmat(umax, 1, M.N)), repmat(-umax, 1, M.N));
end
