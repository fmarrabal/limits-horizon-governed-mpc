classdef MOMPC < handle
%MOMPC MPC multiobjetivo escalarizado, en forma de QP denso explicito.
%
%       U*(x,alpha) = argmin_U  sum_i alpha_i J_i(U,x)
%       s.a.  x_{k+1} = A x_k + B u_k, |x| <= xmax, |u| <= umax, x_N en Omega
%
%   El problema se CONDENSA a la forma estandar
%       min_z  1/2 z' G z + g' z    s.a.  Ain z <= bin
%   y esas cuatro matrices son la interfaz. La version Python construye
%   exactamente las mismas, en el MISMO ORDEN DE FILAS, de modo que la
%   validacion cruzada entre lenguajes es una comparacion elemento a elemento
%   y no una comparacion de resultados finales que podria esconder dos
%   formulaciones distintas que casualmente se parecen.
%
%   ORDEN DE LAS FILAS DE Ain (debe coincidir con Python):
%     1) caja de estado, k = 1..N, primero +Suk y luego -Suk
%     2) caja de entrada, k = 0..N-1, primero + y luego -
%     3) region terminal
%     4) epigrafo de los objetivos de norma infinito
%
%   Ain NO depende de x0: solo el lado derecho es afin,  bin = b_const + B_x x0.

    properties
        prob, term
        Sx, Su
        N, n, m, nU, nz
        epi          % indices de epigrafo por objetivo ([] si es cuadratico)
        Ain, b_const, B_x
        Gi, Fi, Ci   % bloques de coste por objetivo
        kindLin      % true si el objetivo es lineal en el epigrafo
    end

    methods
        function o = MOMPC(prob, term)
            o.prob = prob;  o.term = term;
            o.N = prob.N;  o.n = prob.plant.n;  o.m = prob.plant.m;
            o.nU = o.m * o.N;
            [o.Sx, o.Su] = ghi.predictionMatrices(prob.plant, o.N);

            % indices de epigrafo
            o.epi = cell(1, numel(prob.obj));
            cursor = o.nU;
            for i = 1:numel(prob.obj)
                if strcmp(prob.obj(i).kind, 'inf')
                    w = 2 * o.N + 1;                 % s_k (N), r_k (N), p (1)
                    o.epi{i} = cursor + (1:w);
                    cursor = cursor + w;
                else
                    o.epi{i} = [];
                end
            end
            o.nz = cursor;
            o.buildCostBlocks();
            o.buildConstraintData();
        end

        function buildCostBlocks(o)
            N = o.N; n = o.n; nz = o.nz; nU = o.nU;
            o.Gi = cell(1, numel(o.prob.obj));
            o.Fi = cell(1, numel(o.prob.obj));
            o.Ci = cell(1, numel(o.prob.obj));
            o.kindLin = false(1, numel(o.prob.obj));
            for i = 1:numel(o.prob.obj)
                ob = o.prob.obj(i);
                G = zeros(nz); F = zeros(nz, n); C = zeros(n);
                if strcmp(ob.kind, 'quad')
                    Qbar = zeros(n * (N + 1));
                    for k = 0:N-1
                        Qbar(k*n+(1:n), k*n+(1:n)) = ob.Q;
                    end
                    Qbar(N*n+(1:n), N*n+(1:n)) = ob.P;
                    Rbar = kron(eye(N), ob.R);
                    Hu = o.Su' * Qbar * o.Su + Rbar;
                    G(1:nU, 1:nU) = 2 * (0.5 * (Hu + Hu'));
                    F(1:nU, :)    = 2 * (o.Su' * Qbar * o.Sx);
                    C             = o.Sx' * Qbar * o.Sx;
                else
                    lin = zeros(nz, 1);
                    lin(o.epi{i}) = 1;
                    F = lin;                 % termino lineal, no depende de x0
                    o.kindLin(i) = true;
                end
                o.Gi{i} = G;  o.Fi{i} = F;  o.Ci{i} = C;
            end
        end

        function [G, g] = costData(o, x0, alpha)
            x0 = x0(:);
            G = zeros(o.nz);  g = zeros(o.nz, 1);
            for i = 1:numel(o.prob.obj)
                if o.kindLin(i)
                    gi = o.Fi{i};
                else
                    gi = o.Fi{i} * x0;
                end
                G = G + alpha(i) * o.Gi{i};
                g = g + alpha(i) * gi;
            end
            G = 0.5 * (G + G');
        end

        function buildConstraintData(o)
            N = o.N; n = o.n; m = o.m; nz = o.nz; nU = o.nU;
            p = o.prob.plant;
            rows = {}; bc = {}; bx = {};

            % 1) caja de estado, k = 1..N
            for k = 1:N
                Sxk = o.Sx(k*n+(1:n), :);
                Suk = o.Su(k*n+(1:n), :);
                for s = [1 -1]
                    A = zeros(n, nz);  A(:, 1:nU) = s * Suk;
                    rows{end+1} = A;            %#ok<AGROW>
                    bc{end+1}   = p.xmax(:);    %#ok<AGROW>
                    bx{end+1}   = -s * Sxk;     %#ok<AGROW>
                end
            end
            % 2) caja de entrada, k = 0..N-1
            for k = 0:N-1
                Sel = zeros(m, nU);  Sel(:, k*m+(1:m)) = eye(m);
                for s = [1 -1]
                    A = zeros(m, nz);  A(:, 1:nU) = s * Sel;
                    rows{end+1} = A;            %#ok<AGROW>
                    bc{end+1}   = p.umax(:);    %#ok<AGROW>
                    bx{end+1}   = zeros(m, n);  %#ok<AGROW>
                end
            end
            % 3) region terminal
            SxN = o.Sx(N*n+(1:n), :);
            SuN = o.Su(N*n+(1:n), :);
            A = zeros(size(o.term.H, 1), nz);
            A(:, 1:nU) = o.term.H * SuN;
            rows{end+1} = A;
            bc{end+1}   = o.term.k(:);
            bx{end+1}   = -o.term.H * SxN;
            % 4) epigrafo
            [Ae, Bxe, b0e] = o.epigraphRows();
            if ~isempty(Ae)
                rows{end+1} = Ae;  bc{end+1} = b0e;  bx{end+1} = Bxe;
            end

            o.Ain     = vertcat(rows{:});
            o.b_const = vertcat(bc{:});
            o.B_x     = vertcat(bx{:});
        end

        function [A, Bx, b0] = epigraphRows(o)
            N = o.N; n = o.n; m = o.m; nz = o.nz; nU = o.nU;
            rows = {}; bxs = {}; b0s = {};
            for i = 1:numel(o.prob.obj)
                ob = o.prob.obj(i);
                if ~strcmp(ob.kind, 'inf'), continue, end
                base = o.epi{i}(1) - 1;
                nq = size(ob.Q, 1);  nr = size(ob.R, 1);  np_ = size(ob.P, 1);
                for k = 0:N-1
                    Sxk = o.Sx(k*n+(1:n), :);
                    Suk = o.Su(k*n+(1:n), :);
                    for s = [1 -1]
                        A = zeros(nq, nz);  A(:, 1:nU) = s * (ob.Q * Suk);
                        A(:, base + k + 1) = -1;
                        rows{end+1} = A; b0s{end+1} = zeros(nq,1); bxs{end+1} = -s*(ob.Q*Sxk); %#ok<AGROW>
                    end
                    Sel = zeros(m, nU);  Sel(:, k*m+(1:m)) = eye(m);
                    for s = [1 -1]
                        A = zeros(nr, nz);  A(:, 1:nU) = s * (ob.R * Sel);
                        A(:, base + N + k + 1) = -1;
                        rows{end+1} = A; b0s{end+1} = zeros(nr,1); bxs{end+1} = zeros(nr,n); %#ok<AGROW>
                    end
                end
                SxN = o.Sx(N*n+(1:n), :);
                SuN = o.Su(N*n+(1:n), :);
                for s = [1 -1]
                    A = zeros(np_, nz);  A(:, 1:nU) = s * (ob.P * SuN);
                    A(:, base + 2*N + 1) = -1;
                    rows{end+1} = A; b0s{end+1} = zeros(np_,1); bxs{end+1} = -s*(ob.P*SxN); %#ok<AGROW>
                end
            end
            if isempty(rows)
                A = zeros(0, nz);  Bx = zeros(0, n);  b0 = zeros(0, 1);
            else
                A = vertcat(rows{:});  Bx = vertcat(bxs{:});  b0 = vertcat(b0s{:});
            end
        end

        function b = rhs(o, x0)
            b = o.b_const + o.B_x * x0(:);
        end

        function qp = build(o, x0, alpha)
            [G, g] = o.costData(x0, alpha);
            qp = struct('G', G, 'g', g, 'Ain', o.Ain, 'bin', o.rhs(x0), ...
                        'nU', o.nU, 'nz', o.nz, 'N', o.N, 'm', o.m);
        end
    end
end
