function [prob, term] = designTerminal(prob, weights)
%DESIGNTERMINAL Ingredientes terminales CORRECTOS: (Kf, P_i, Omega).
%
%   POR QUE EXISTE ESTA FUNCION
%   Los ingredientes heredados del TFM NO satisfacen la desigualdad terminal.
%   Con los pesos del banco de conflicto:
%       eig(S_0) = [-11.04, +0.02]   eig(S_1) = [-3.38, +9.03]
%   violada para AMBOS objetivos; y la "region terminal" H x_N <= 10 es una
%   CAJA, no un conjunto invariante. Sin eso, la restriccion V* <= J_a no
%   certifica decrecimiento de Lyapunov.
%
%   CONSTRUCCION
%     Kf : LQR de la mezcla uniforme. COMUN a todos los objetivos -- la
%          secuencia desplazada debe ser factible para todos a la vez, asi que
%          la ley terminal no puede depender de i.
%     P_i: ecuacion de Lyapunov discreta  P = Acl' P Acl + Q + Kf' R Kf,
%          que da S_i = 0 EXACTAMENTE (la version mas apretada posible).
%     Om : conjunto maximo positivamente invariante (Gilbert-Tan 1991).

if nargin < 2 || isempty(weights)
    weights = ones(1, numel(prob.obj)) / numel(prob.obj);
end

A = prob.plant.A;  B = prob.plant.B;
n = prob.plant.n;  m = prob.plant.m;

% --- ley auxiliar comun ------------------------------------------------
Q = zeros(n); R = zeros(m);
for i = 1:numel(prob.obj)
    ob = prob.obj(i);
    if strcmp(ob.kind, 'quad')
        Q = Q + weights(i) * ob.Q;
        R = R + weights(i) * ob.R;
    else   % la cota cuadratica basta para fijar Kf
        Q = Q + weights(i) * (ob.Q' * ob.Q);
        R = R + weights(i) * (ob.R' * ob.R);
    end
end
Q = Q + 1e-9 * eye(n);
R = R + 1e-9 * eye(m);
K  = dlqr(A, B, Q, R);       % convenio de MATLAB: u = -K x
Kf = -K;                     % convenio del paquete: u = Kf x
Acl = A + B * Kf;

% --- costes terminales por Lyapunov ------------------------------------
P = cell(1, numel(prob.obj));
for i = 1:numel(prob.obj)
    ob = prob.obj(i);
    if strcmp(ob.kind, 'quad')
        W = ob.Q + Kf' * ob.R * Kf;
        % dlyap(a,q) resuelve  a X a' - X + q = 0; con a = Acl' se obtiene
        % Acl' P Acl - P + W = 0, que es lo que hace falta.
        Pi = dlyap(Acl', W);
        Pi = 0.5 * (Pi + Pi');
    else
        if isempty(ob.P)
            error('ghi:designTerminal', ...
                ['el objetivo "%s" es de norma infinito y no trae P; la ' ...
                 'construccion por Lyapunov no aplica, hay que darlo a mano'], ob.name);
        end
        Pi = ob.P;
    end
    P{i} = Pi;
    prob.obj(i).P = Pi;
end

% --- region terminal ----------------------------------------------------
[H, k, nit] = ghi.invariantSet(prob.plant, Kf);

term = struct('Kf', Kf, 'P', {P}, 'H', H, 'k', k, 'n_iter', nit);
end
