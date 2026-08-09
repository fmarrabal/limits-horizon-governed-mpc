function out = collocationStudy(M, p, betaMax, n)
%COLLOCATIONSTUDY Umbral real de flutter frente a la cota de Merkin.
%
%   La cota rho(G) < 2 zeta w0^2 esta deducida para el caso degenerado de
%   Merkin (frecuencias iguales). Aqui se comprueba si sigue valiendo sobre el
%   grafo de una PLANTA, donde K y C dejan de ser multiplos de la identidad.
%
%   Resultado medido: es EXACTO en el caso degenerado y CONSERVADOR en cuanto
%   la planta aporta estructura -- tanto D*L como c^2*L aumentan el margen
%   real. Es decir, se puede usar como cota SEGURA de diseno, que es lo que un
%   ingeniero de control necesita.
if nargin < 1 || isempty(M), M = 6; end
if nargin < 4 || isempty(n), n = 2001; end

[L, A] = ghi.chainGraph(M);
p0 = p;  p0.beta = 0;
[K, C, ~] = ghi.fieldOperators(L, A, p0);
mu  = abs(imag(eig(A)));
thr = 2 * p.zeta * p.w0^2;

% beta al que rho(G) alcanza el umbral
bb = linspace(0, 5, 200001);
rr = arrayfun(@(x) ghi.rhoG(A, p.b, x), bb);
idx = find(rr >= thr, 1, 'first');
if isempty(idx), betaPred = Inf; else, betaPred = bb(idx); end

if nargin < 3 || isempty(betaMax)
    if isfinite(betaPred), betaMax = max(2 * betaPred, 0.5); else, betaMax = 1.0; end
end
betas = linspace(0, betaMax, n);
A3 = A * A * A;
gyro = zeros(1, n);  circ = zeros(1, n);
for j = 1:n
    G = p.b * A + betas(j) * A3;
    gyro(j) = max(real(eig(ghi.waveState(K, C, G, 'gyroscopic'))));
    circ(j) = max(real(eig(ghi.waveState(K, C, G, 'circulatory'))));
end
idx = find(circ > 1e-9, 1, 'first');
if isempty(idx), betaObs = Inf; else, betaObs = betas(idx); end

out = struct('M', M, 'mu_max', max(mu), 'rho_A3', max(mu)^3, ...
             'umbral', thr, 'beta_pred', betaPred, 'beta_obs', betaObs, ...
             'ratio', betaObs / betaPred, ...
             'gyro_max_re', max(gyro), 'gyro_estable', max(gyro) < 0, ...
             'betas', betas, 'gyro', gyro, 'circ', circ);
end
