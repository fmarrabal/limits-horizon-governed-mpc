function out = run_pilot(nSeeds, w0, zeta, theta, gridPoints)
%RUN_PILOT Banco principal del GHI en MATLAB.
%
%   Los numeros de referencia del proyecto salen de la implementacion Python
%   (mas rapida: usa OSQP con setup cacheado). El papel de esta version es
%   confirmar que la MISMA formulacion, resuelta con quadprog, da el mismo
%   resultado cualitativo y los mismos signos en los contrastes.
%
%   Uso:  out = run_pilot();          % 10 semillas
%         out = run_pilot(5);         % mas rapido

if nargin < 1 || isempty(nSeeds),     nSeeds = 10;   end
if nargin < 2 || isempty(w0),         w0 = 0.9;      end
if nargin < 3 || isempty(zeta),       zeta = 0.5;    end
if nargin < 4 || isempty(theta),      theta = 0.5;   end
if nargin < 5 || isempty(gridPoints), gridPoints = 21; end

here = fileparts(mfilename('fullpath'));
addpath(fileparts(here));
t0 = tic;

prob = ghi.problemConflict();
[prob, term] = ghi.designTerminal(prob);
M = ghi.MOMPC(prob, term);
[tau, g] = ghi.matchFirstOrder(w0, zeta, 1.0);

fprintf('==============================================================================\n');
fprintf('BANCO PRINCIPAL DEL GHI -- MATLAB\n');
fprintf('==============================================================================\n');
fprintf('planta      : %s, N = %d\n', prob.name, prob.N);
fprintf('terminal    : Kf = [%.4f %.4f], Omega con %d filas\n', ...
        term.Kf(1), term.Kf(2), size(term.H,1));
fprintf('igualacion  : w0=%.2f zeta=%.2f -> |H| en Nyquist = %.6f, tau = %.4f\n', ...
        w0, zeta, g, tau);
fprintf('              (EN DISCRETO; igualar en continuo daba un desajuste de 1.69x)\n');
fprintf('anti-windup : theta = %.2f\n', theta);
fprintf('semillas    : %d\n\n', nSeeds);

make = {@() ghi.Order0(), @() ghi.Order1(tau), ...
        @() ghi.Order2(w0, zeta, theta), @() ghi.Frozen(0.5)};
rname = {'orden-0','orden-1','orden-2','congelado'};

sc = ghi.defaultScenarios();
snames = fieldnames(sc);
out = struct();

for si = 1:numel(snames)
    s = sc.(snames{si});
    fprintf('### %s\n', s.name);
    fprintf('  %-12s%18s%17s%10s\n', 'regulador', 'err_demanda', 'coste', 'bloqueos');
    for ri = 1:numel(make)
        E = []; C = []; B = [];
        for seed = 0:nSeeds-1
            tr = ghi.closedLoop(M, make{ri}(), s, seed, gridPoints);
            if isempty(tr), continue, end
            k = (s.warmup+1):numel(tr.alpha);
            E(end+1) = sqrt(mean((tr.alpha(k) - tr.clean(k)).^2)); %#ok<AGROW>
            C(end+1) = sum(tr.stage(k));                            %#ok<AGROW>
            B(end+1) = sum(tr.blocked(k));                          %#ok<AGROW>
        end
        out.(snames{si}).(matlab.lang.makeValidName(rname{ri})) = ...
            struct('err', E, 'coste', C, 'bloq', B);
        fprintf('  %-12s%9.4f+-%-7.4f%10.2f+-%-5.2f%10.1f\n', ...
                rname{ri}, mean(E), std(E), mean(C), std(C), mean(B));
    end
    fprintf('\n');
end

fprintf('==============================================================================\n');
fprintf('CONTRASTES PAREADOS POR SEMILLA (delta > 0 => gana el primero)\n');
cmp = {'orden_2','orden_1'; 'orden_2','orden_0'; 'orden_1','orden_0'};
for mi = 1:2
    if mi == 1, met = 'err'; lab = 'err_demanda'; else, met = 'coste'; lab = 'coste'; end
    fprintf('\n--- %s\n', lab);
    for si = 1:numel(snames)
        for ci = 1:size(cmp,1)
            a = out.(snames{si}).(cmp{ci,1}).(met);
            b = out.(snames{si}).(cmp{ci,2}).(met);
            n = min(numel(a), numel(b));
            if n < 2, continue, end
            d = b(1:n) - a(1:n);
            [~, p, ~, st] = ttest(d);
            star = ''; if p < 0.05, star = ' *'; end
            fprintf('  %-18s%s vs %-10s delta=%+9.4f t=%+7.2f p=%.4f%s\n', ...
                    snames{si}, cmp{ci,1}, cmp{ci,2}, mean(d), st.tstat, p, star);
        end
    end
end
fprintf('\ntiempo: %.1fs\n', toc(t0));
end
