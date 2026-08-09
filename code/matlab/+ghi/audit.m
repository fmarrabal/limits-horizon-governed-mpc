function checks = audit(verbose)
%AUDIT Suite de auditoria: cada hallazgo adversarial, ejecutable.
%
%   La regla es simple: todo defecto que se encontro una vez queda convertido
%   en una asercion que falla si vuelve. Ninguna comprobacion se apoya en la
%   memoria de nadie.
%
%     A1  los ingredientes terminales satisfacen S_i >= 0
%     A2  Omega es positivamente invariante bajo Acl
%     A3  Omega respeta las restricciones de estado y de entrada terminal
%     A4  V*(x,.) es CONCAVA en alpha        (correccion a Bemporad 2009)
%     A5  el conjunto admisible ES no convexo para un rango de J_a
%     A6  el radio certificado es valido
%     A7  la igualacion 1er/2o orden es exacta EN DISCRETO
%     A8  el anti-windup NO se dispara si no hay bloqueo
%     A9  la sobreoscilacion ejecutada == la del sistema DISCRETO
%     A10 la colocacion giroscopica es estable en toda la caja
%     A11 el umbral de Merkin es exacto en el degenerado y conservador fuera
%     A12 el ingrediente terminal HEREDADO DEL TFM falla (regresion)

if nargin < 1, verbose = true; end
checks = {};

    function add(code, title, ok, detail)
        checks{end+1} = struct('code', code, 'title', title, 'ok', ok, ...
                               'detail', detail); %#ok<AGROW>
    end

prob = ghi.problemConflict();
[prob, term] = ghi.designTerminal(prob);
M = ghi.MOMPC(prob, term);
plant = prob.plant;
Acl = plant.A + plant.B * term.Kf;

% ---------------------------------------------------------------- A1
det = ''; ok1 = true;
for i = 1:numel(prob.obj)
    S = ghi.terminalSlack(plant, term.Kf, prob.obj(i), term.P{i});
    e = min(eig(S));
    ok1 = ok1 && (e >= -1e-8);
    det = [det sprintf('%s: eig_min=%.3e; ', prob.obj(i).name, e)]; %#ok<AGROW>
end
add('A1', 'desigualdad terminal S_i >= 0 por objetivo', ok1, det);

% ------------------------------------------------------------- A2, A3
rng(0, 'twister');
nS = 400; worstInv = -Inf; worstCon = -Inf; nOk = 0;
lo = zeros(plant.n,1); hi = zeros(plant.n,1);
opts = optimoptions('linprog','Display','none');
for j = 1:plant.n
    e = zeros(plant.n,1); e(j) = 1;
    [~,f1,fl1] = linprog(e,  term.H, term.k, [],[],[],[], opts);
    [~,f2,fl2] = linprog(-e, term.H, term.k, [],[],[],[], opts);
    if fl1==1, lo(j)=f1; else, lo(j)=-1; end
    if fl2==1, hi(j)=-f2; else, hi(j)=1;  end
end
for s = 1:nS
    x = lo + (hi - lo) .* rand(plant.n,1);
    if any(term.H * x > term.k + 1e-12), continue, end
    nOk = nOk + 1;
    worstInv = max(worstInv, max(term.H * (Acl * x) - term.k));
    worstCon = max(worstCon, max([abs(x) - plant.xmax; ...
                                  abs(term.Kf * x) - plant.umax]));
end
add('A2', 'Omega positivamente invariante bajo Acl', worstInv <= 1e-8, ...
    sprintf('peor violacion = %.3e (%d muestras)', worstInv, nOk));
add('A3', 'Omega respeta |x|<=xmax y |Kf x|<=umax', worstCon <= 1e-8, ...
    sprintf('peor violacion = %.3e', worstCon));

% ---------------------------------------------------------------- A4
rng(1, 'twister');
pairs = [0 1; 0.1 0.9; 0.2 0.8; 0.3 0.7; 0.25 0.75];
nT = 0; vConc = 0; vConv = 0; dsum = 0;
nx = 0;
while nx < 40
    x = -7 + 14 * rand(plant.n, 1);
    if isempty(ghi.solveMOMPC(M, x, [0.5; 0.5])), continue, end
    nx = nx + 1;
    for pr = 1:size(pairs,1)
        a1 = pairs(pr,1); a2 = pairs(pr,2); am = 0.5*(a1+a2);
        s1 = ghi.solveMOMPC(M, x, [1-a1; a1]);
        s2 = ghi.solveMOMPC(M, x, [1-a2; a2]);
        sm = ghi.solveMOMPC(M, x, [1-am; am]);
        if isempty(s1)||isempty(s2)||isempty(sm), continue, end
        nT = nT + 1;
        d = sm.V - 0.5*(s1.V + s2.V);
        tol = 1e-8 * max([abs(s1.V) abs(s2.V) abs(sm.V) 1]);
        dsum = dsum + d;
        if d < -tol, vConc = vConc + 1; end
        if d >  tol, vConv = vConv + 1; end
    end
end
add('A4', 'V*(x,.) es CONCAVA en alpha (correccion a Bemporad 2009)', ...
    vConc == 0 && vConv > 0, ...
    sprintf('%d tests | violaciones de CONCAVIDAD = %d | de CONVEXIDAD = %d | d medio = %+.4e', ...
            nT, vConc, vConv, dsum / max(nT,1)));

% ---------------------------------------------------------------- A5
rng(2, 'twister');
grid = linspace(0, 1, 41);
totNiv = 0; totNC = 0; picoInt = 0; nEst = 0;
for s = 1:8
    x = -6 + 12 * rand(plant.n, 1);
    if isempty(ghi.solveMOMPC(M, x, [0.5; 0.5])), continue, end
    V = inf(1, numel(grid));
    for j = 1:numel(grid)
        sj = ghi.solveMOMPC(M, x, [1-grid(j); grid(j)]);
        if ~isempty(sj), V(j) = sj.V; end
    end
    fin = isfinite(V);
    if sum(fin) < 3, continue, end
    nEst = nEst + 1;
    [~, jp] = max(V .* fin - 1e30*(~fin));
    picoInt = picoInt + (jp > 1 && jp < numel(grid));
    niv = linspace(min(V(fin)), max(V(fin)), 27); niv = niv(2:end-1);
    totNiv = totNiv + numel(niv);
    for Ja = niv
        f = V <= Ja + 1e-9;
        if sum(f) < 2, continue, end
        i1 = find(f, 1, 'first'); i2 = find(f, 1, 'last');
        if any(~f(i1:i2)), totNC = totNC + 1; end
    end
end
add('A5', 'el conjunto admisible ES no convexo para un rango de J_a', totNC > 0, ...
    sprintf('%d estados, %d niveles | NO convexo en %d (%.1f%%) | pico interior en %d/%d', ...
            nEst, totNiv, totNC, 100*totNC/max(totNiv,1), picoInt, nEst));

% ---------------------------------------------------------------- A6
rng(3, 'twister');
worstEx = -Inf; nTests = 0; nEst6 = 0;
for s = 1:8
    x = -5 + 10 * rand(plant.n, 1);
    sol = ghi.solveMOMPC(M, x, [0.5; 0.5]);
    if isempty(sol), continue, end
    Us = ghi.shiftedSequence(M, sol);
    Ja = [0.5 0.5] * ghi.costs(prob, x, Us);
    r = ghi.certifiedRadius(sol.J, sol.V, Ja);
    al = [0.5; 0.5]; hit = false;
    for k = 1:8
        d = randn(2,1); d = d - mean(d); nd = norm(d);
        if nd < 1e-12, continue, end
        d = d / nd;
        neg = d < -1e-15;
        if any(neg), tmax = min(-al(neg) ./ d(neg)); else, tmax = Inf; end
        tlim = min([r tmax]);
        if ~isfinite(tlim) || tlim <= 1e-12, continue, end
        for frac = [0.5 0.9 0.999]
            ap = max(al + frac * tlim * d, 0); ap = ap / sum(ap);
            s2 = ghi.solveMOMPC(M, x, ap);
            if isempty(s2), continue, end
            nTests = nTests + 1; hit = true;
            worstEx = max(worstEx, s2.V - Ja);
        end
    end
    nEst6 = nEst6 + hit;
end
add('A6', 'radio certificado valido: ||da||<=r => alpha admisible', ...
    nTests > 0 && worstEx <= 1e-6, ...
    sprintf('%d estados, %d pruebas | peor exceso sobre J_a = %.3e', nEst6, nTests, worstEx));

% ---------------------------------------------------------------- A7
[tau, g2] = ghi.matchFirstOrder(0.9, 0.5, 1.0);
[A1_,B1_,C1_,D1_] = ghi.ssFirstOrder(tau);
g1 = ghi.magAt(A1_,B1_,C1_,D1_, pi, 1.0);
add('A7', 'igualacion 1er/2o orden exacta EN DISCRETO', abs(g1-g2) < 1e-12, ...
    sprintf('tau=%.4f |H1|=%.9f |H2|=%.9f err=%.2e', tau, g1, g2, abs(g1-g2)));

% ---------------------------------------------------------------- A8
regs = {ghi.Order1(5.0), ghi.Order2(0.9,0.5,0.5), ghi.Order2(0.9,0.5,0.0)};
nms  = {'orden-1','orden-2 th=0.5','orden-2 th=0.0'};
det8 = ''; ok8 = true;
for i = 1:numel(regs)
    r = regs{i}; r.reset(); fired = 0;
    for k = 1:200
        r.step(rand(), 1.0);
        if isprop(r,'v'), before = [r.a r.v]; else, before = r.a; end
        r.sync(rand(), false);
        if isprop(r,'v'), after = [r.a r.v]; else, after = r.a; end
        if any(before ~= after), fired = fired + 1; end
    end
    ok8 = ok8 && (fired == 0);
    det8 = [det8 sprintf('%s: %.3f | ', nms{i}, fired/200)]; %#ok<AGROW>
end
add('A8', 'el anti-windup NO se dispara si no hay bloqueo', ok8, det8);

% ---------------------------------------------------------------- A9
det9 = ''; ok9 = true;
for zeta = [0.3 0.5 0.7]
    r = ghi.Order2(0.9, zeta, 0.5); r.reset(0); pk = -Inf;
    for k = 1:400
        a = r.step(1.0, 1.0); r.sync(a, false); pk = max(pk, a);
    end
    exec = max(0, pk - 1);
    [Ad,Bd,Cd,Dd] = ghi.ssSecondOrder(0.9, zeta);
    xs = [0;0]; pk2 = -Inf;
    for k = 1:400
        xs = Ad*xs + Bd; pk2 = max(pk2, Cd*xs + Dd);
    end
    disc = max(0, pk2 - 1);
    ok9 = ok9 && abs(exec - disc) < 1e-9;
    det9 = [det9 sprintf('z=%.1f: %.4f%% vs %.4f%% | ', zeta, 100*exec, 100*disc)]; %#ok<AGROW>
end
add('A9', 'sobreoscilacion ejecutada == la del DISCRETO', ok9, det9);

% ------------------------------------------------------------- A10, A11
pd = struct('w0',1,'zeta',0.15,'c',0,'D',0,'b',0,'beta',0);
st = ghi.collocationStudy(6, pd);
add('A11', 'umbral de Merkin EXACTO con K y C uniformes', abs(st.ratio-1) < 0.02, ...
    sprintf('beta*_pred=%.5f beta*_obs=%.5f ratio=%.4f', st.beta_pred, st.beta_obs, st.ratio));

cons = true; det11 = '';
for cd = [0.35 0; 0 0.05; 0.35 0.05; 0.6 0]'
    pp = struct('w0',1,'zeta',0.15,'c',cd(1),'D',cd(2),'b',0,'beta',0);
    rr = ghi.collocationStudy(6, pp);
    cons = cons && (rr.ratio >= 1 - 1e-6);
    det11 = [det11 sprintf('c=%.2f,D=%.2f: %.3f | ', cd(1), cd(2), rr.ratio)]; %#ok<AGROW>
end
add('A11b','con estructura de planta el umbral es CONSERVADOR', cons, det11);

gyroOk = true; worstG = -Inf;
for c = [0 0.35 0.6]
    for D = [0 0.05 0.2]
        for b = [0 0.4]
            pp = struct('w0',1,'zeta',0.15,'c',c,'D',D,'b',b,'beta',0);
            rr = ghi.collocationStudy(6, pp, [], 201);
            worstG = max(worstG, rr.gyro_max_re);
            gyroOk = gyroOk && rr.gyro_estable;
        end
    end
end
add('A10', 'colocacion GIROSCOPICA estable en toda la caja', gyroOk, ...
    sprintf('peor max Re(lambda) = %+.3e (debe ser < 0)', worstG));

% ---------------------------------------------------------------- A12
KfTFM = [-0.5 -1.4];
Ptfm  = {diag([0.05 0.05]), diag([20 8])};
e12 = zeros(1,2);
for i = 1:2
    S = ghi.terminalSlack(plant, KfTFM, prob.obj(i), Ptfm{i});
    e12(i) = min(eig(S));
end
add('A12', ['los ingredientes del TFM VIOLAN la desigualdad terminal ' ...
            '(regresion: debe seguir fallando)'], any(e12 < -1e-9), ...
    sprintf('eig_min(S_0)=%+.4f | eig_min(S_1)=%+.4f', e12(1), e12(2)));

% ---------------------------------------------------------------- salida
if verbose
    fprintf('==============================================================================\n');
    fprintf('SUITE DE AUDITORIA -- MATLAB  (cada hallazgo adversarial, ejecutable)\n');
    fprintf('==============================================================================\n');
    nb = 0;
    for i = 1:numel(checks)
        c = checks{i};
        if c.ok, mark = 'OK  '; else, mark = 'FALLA'; nb = nb + 1; end
        fprintf('[%s] %-5s %s\n', mark, c.code, c.title);
        if ~isempty(c.detail), fprintf('         %s\n', c.detail); end
    end
    fprintf('==============================================================================\n');
    if nb == 0
        fprintf('%d/%d comprobaciones OK\n', numel(checks), numel(checks));
    else
        fprintf('%d/%d OK  --  %d FALLAN\n', numel(checks)-nb, numel(checks), nb);
    end
end
end
