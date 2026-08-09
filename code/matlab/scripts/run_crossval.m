function run_crossval(refPath)
%RUN_CROSSVAL Valida la implementacion MATLAB contra la referencia Python.
%
%   No basta con que las dos den resultados "parecidos": eso lo consigue
%   cualquier par de programas que resuelvan problemas distintos pero
%   similares. Lo que se compara aqui es la FORMULACION:
%
%     1. ingredientes terminales (Kf, P_i, Omega)      -- construccion
%     2. matrices del QP (G, g, Ain, bin) elemento a elemento
%     3. solucion del QP, costes J y secuencia desplazada
%     4. respuesta en z de los reguladores y tau de igualacion
%     5. operadores del campo y umbral de flutter
%
%   Si (1) y (2) coinciden a 1e-12 las dos implementaciones son literalmente el
%   mismo problema y solo puede diferir el solver.
%
%   Uso:  run_crossval           (busca ../../crossval/python_reference.json)
%         run_crossval(ruta)

here = fileparts(mfilename('fullpath'));
addpath(fileparts(here));                       % para ver el paquete +ghi
if nargin < 1 || isempty(refPath)
    refPath = fullfile(fileparts(fileparts(here)), 'crossval', 'python_reference.json');
end
fprintf('referencia: %s\n', refPath);
ref = jsondecode(fileread(refPath));

% Dos tolerancias, y la distincion es la clave de esta validacion:
%
%   TOL_EXACT  para la CONSTRUCCION (matrices, operadores, ingredientes). Ahi
%              las dos implementaciones deben coincidir a precision de maquina,
%              porque hacen literalmente la misma aritmetica.
%   TOL_SOLVE  RELATIVA, para lo que sale de un SOLVER. CLARABEL (Python) y
%              quadprog (MATLAB) son algoritmos distintos: coinciden hasta su
%              tolerancia, no mas. Y como J es cuadratica en U, un error de
%              1e-6 en U se propaga a ~1e-4 en J. Exigir tolerancia ABSOLUTA
%              ahi seria exigir que dos solvers distintos den bit a bit lo
%              mismo, que no es lo que se quiere demostrar.
%   Y una tercera, TOL_OPT, para el VALOR OPTIMO del QP. Ese si debe ser muy
%   tirante aunque las soluciones difieran: en un optimo el gradiente proyectado
%   se anula, asi que un error dz en la solucion mueve el objetivo en O(dz^2).
%   Comparar los valores optimos es por tanto una prueba MAS FUERTE que
%   comparar las soluciones, y es la que zanja si ambos solvers han encontrado
%   el mismo minimo o dos minimos distintos.
TOL_EXACT = 1e-10;
%   Medido: la solucion concuerda a 4.8e-7 y el valor optimo a 4.9e-10, es
%   decir MIL VECES mejor. Eso es exactamente el O(dz^2) predicho y es la
%   evidencia de que ambos solvers cayeron en el mismo minimo, no en dos.
TOL_SOLVE = 1e-6;       % RELATIVA, sobre la solucion
TOL_OPT   = 1e-9;       % RELATIVA, sobre el valor optimo
report = {};
allok = true;

    function ok = chk(code, title, err, tol, extra)
        if nargin < 5, extra = ''; end
        ok = err <= tol;
        allok = allok && ok;
        if ok, mark = 'OK  '; else, mark = 'FALLA'; end
        fprintf('[%s] %-6s %-52s err=%.3e (tol %.0e) %s\n', mark, code, title, err, tol, extra);
        report{end+1} = struct('code', code, 'ok', ok, 'err', err); %#ok<AGROW>
    end

fprintf('\n============================================================================\n');
fprintf('VALIDACION CRUZADA MATLAB <-> PYTHON\n');
fprintf('============================================================================\n');

% ---------------------------------------------------------------- 1. terminal
prob = ghi.problemConflict();
[prob, term] = ghi.designTerminal(prob);

chk('X1', 'ley terminal Kf', maxabs(term.Kf - ref.terminal.Kf(:)'), TOL_EXACT);
for i = 1:numel(term.P)
    chk(sprintf('X2.%d', i), sprintf('coste terminal P_%d', i), ...
        maxabs(term.P{i} - squeeze(ref.terminal.P(i, :, :))), TOL_EXACT);
end
% Omega: mismo numero de filas y mismo contenido (salvo orden)
sameRows = size(term.H, 1) == size(ref.terminal.H, 1);
if sameRows
    e = maxabs(sortrows([term.H term.k]) - sortrows([ref.terminal.H ref.terminal.k]));
else
    e = Inf;
end
chk('X3', 'region terminal Omega (H, k)', e, TOL_EXACT, ...
    sprintf('%d filas', size(term.H, 1)));

% ------------------------------------------------------------------ 2. QP
M = ghi.MOMPC(prob, term);
chk('X4', 'matriz de prediccion Sx', maxabs(M.Sx - ref.prediction.Sx), TOL_EXACT);
chk('X5', 'matriz de prediccion Su', maxabs(M.Su - ref.prediction.Su), TOL_EXACT);
chk('X6', 'restricciones Ain (orden de filas incluido)', ...
    maxabs(M.Ain - ref.constraints.Ain), TOL_EXACT, ...
    sprintf('%dx%d', size(M.Ain, 1), size(M.Ain, 2)));
chk('X7', 'lado derecho b_const', maxabs(M.b_const - ref.constraints.b_const(:)), TOL_EXACT);
chk('X8', 'lado derecho B_x', maxabs(M.B_x - ref.constraints.B_x), TOL_EXACT);

nC = numel(ref.cases);
eG = 0; eg = 0; eb = 0; eU = 0; eJ = 0; eV = 0; eS = 0; eO = 0; eF = 0; nSolved = 0;
for c = 1:nC
    cs = ref.cases(c);
    x0 = cs.x0(:);  al = cs.alpha(:);
    qp = M.build(x0, al);
    eG = max(eG, maxabs(qp.G   - squeeze(cs.G)));
    eg = max(eg, maxabs(qp.g   - cs.g(:)));
    eb = max(eb, maxabs(qp.bin - cs.bin(:)));
    sol = ghi.solveMOMPC(M, x0, al);
    if isempty(sol), continue, end
    nSolved = nSolved + 1;
    eU = max(eU, maxrel(sol.U(:), cs.U(:)));
    eJ = max(eJ, maxrel(sol.J(:), cs.J(:)));
    eV = max(eV, maxrel(sol.V, cs.V));
    Us = ghi.shiftedSequence(M, sol);
    eS = max(eS, maxrel(ghi.costs(prob, x0, Us), cs.J_shift(:)));

    % valor optimo del QP evaluado en AMBAS soluciones sobre las MISMAS
    % matrices: si los dos solvers hallaron el mismo minimo, coincide a O(dz^2)
    if M.nz == M.nU
        zPy = cs.U(:);  zMl = sol.z;
        fPy = 0.5 * zPy' * qp.G * zPy + qp.g' * zPy;
        fMl = 0.5 * zMl' * qp.G * zMl + qp.g' * zMl;
        eO  = max(eO, maxrel(fMl, fPy));
        % y la solucion de MATLAB debe ser factible para el bin de Python
        eF = max(eF, max(qp.Ain * zMl - cs.bin(:)));
    end
end
chk('X9',  sprintf('matriz G del QP (%d casos)', nC), eG, TOL_EXACT);
chk('X10', 'vector g del QP',                        eg, TOL_EXACT);
chk('X11', 'lado derecho bin del QP',                eb, TOL_EXACT);
chk('X12', sprintf('solucion U* -- REL (%d resueltos)', nSolved), eU, TOL_SOLVE);
chk('X13', 'vector de costes J -- REL',              eJ, TOL_SOLVE);
chk('X14', 'valor escalarizado V* -- REL',           eV, TOL_SOLVE);
chk('X15', 'costes de la secuencia desplazada -- REL', eS, TOL_SOLVE);
chk('X15b', 'VALOR OPTIMO del QP -- REL (prueba fuerte)', eO, TOL_OPT);
chk('X15c', 'la solucion MATLAB es factible para el bin de Python', max(eF, 0), 1e-8);

% ------------------------------------------------------------ 3. reguladores
eT = 0; eM = 0; eP = 0; eO = 0;
for r = 1:numel(ref.regulators.matching)
    rr = ref.regulators.matching(r);
    [tau, g] = ghi.matchFirstOrder(rr.w0, rr.zeta, 1.0);
    eT = max(eT, abs(tau - rr.tau));
    eT = max(eT, abs(g - rr.g_nyquist));
    [A2, B2, C2, D2] = ghi.ssSecondOrder(rr.w0, rr.zeta);
    [A1, B1, C1, D1] = ghi.ssFirstOrder(tau);
    eM = max(eM, maxabs(A2 - squeeze(rr.A2)));
    for k = 1:numel(rr.w)
        eM = max(eM, abs(ghi.magAt(A2, B2, C2, D2, rr.w(k)) - rr.mag2(k)));
        eM = max(eM, abs(ghi.magAt(A1, B1, C1, D1, rr.w(k)) - rr.mag1(k)));
        eP = max(eP, abs(ghi.phaseAt(A2, B2, C2, D2, rr.w(k)) - rr.pha2(k)));
        eP = max(eP, abs(ghi.phaseAt(A1, B1, C1, D1, rr.w(k)) - rr.pha1(k)));
    end
    reg = ghi.Order2(rr.w0, rr.zeta, 0.5);
    eO = max(eO, abs(overshoot(reg) - rr.overshoot_exec));
end
chk('X16', 'tau de igualacion EN DISCRETO',  eT, TOL_EXACT);
chk('X17', 'modulos |H1|, |H2|',             eM, TOL_EXACT);
chk('X18', 'fases arg H1, arg H2',           eP, TOL_EXACT);
chk('X19', 'sobreoscilacion EJECUTADA',      eO, TOL_EXACT);

% ----------------------------------------------------------------- 4. campo
[L, A] = ghi.chainGraph(ref.field.M);
p = struct('w0', ref.field.params.w0, 'zeta', ref.field.params.zeta, ...
           'c', ref.field.params.c, 'D', ref.field.params.D, ...
           'b', ref.field.params.b, 'beta', ref.field.params.beta);
[K, C, G] = ghi.fieldOperators(L, A, p);
chk('X20', 'laplaciano L del grafo', maxabs(L - ref.field.L), TOL_EXACT);
chk('X21', 'adveccion A del grafo',  maxabs(A - ref.field.A), TOL_EXACT);
chk('X22', 'operadores K, C, G',     max([maxabs(K - ref.field.K), ...
                                          maxabs(C - ref.field.C), ...
                                          maxabs(G - ref.field.G)]), TOL_EXACT);
chk('X23', 'radio espectral rho(G)', abs(ghi.rhoG(A, p.b, p.beta) - ref.field.rho_G), TOL_EXACT);
ag = max(real(eig(ghi.waveState(K, C, G, 'gyroscopic'))));
ac = max(real(eig(ghi.waveState(K, C, G, 'circulatory'))));
chk('X24', 'abscisa espectral (giroscopica)',  abs(ag - ref.field.abscissa_gyro), 1e-9);
chk('X25', 'abscisa espectral (circulatoria)', abs(ac - ref.field.abscissa_circ), 1e-9);

pd = struct('w0', 1.0, 'zeta', 0.15, 'c', 0.0, 'D', 0.0, 'b', 0.0, 'beta', 0.0);
st = ghi.collocationStudy(6, pd);
md = ref.field.merkin_degenerado;
chk('X26', 'umbral de flutter (caso degenerado)', ...
    max(abs(st.beta_pred - md.beta_pred), abs(st.beta_obs - md.beta_obs)), 1e-9, ...
    sprintf('ratio obs/pred = %.4f', st.ratio));

% ------------------------------------------------------------------ resumen
nb = sum(cellfun(@(r) ~r.ok, report));
fprintf('============================================================================\n');
if nb == 0
    fprintf('%d/%d comprobaciones OK  --  las dos implementaciones coinciden\n', ...
            numel(report), numel(report));
else
    fprintf('%d/%d OK  --  %d FALLAN\n', numel(report) - nb, numel(report), nb);
end
fprintf('============================================================================\n');
if ~allok, error('ghi:crossval', 'la validacion cruzada ha fallado'); end
end

% -------------------------------------------------------------------------
function e = maxabs(D)
if isempty(D), e = 0; else, e = max(abs(D(:))); end
end

function e = maxrel(a, b)
%MAXREL Error relativo maximo, con escala robusta cerca de cero.
a = a(:); b = b(:);
if isempty(a), e = 0; return, end
den = max(max(abs(a), abs(b)), 1);     % evita dividir por ~0
e = max(abs(a - b) ./ den);
end

function ov = overshoot(reg)
%OVERSHOOT Sobreoscilacion al escalon TAL COMO SE EJECUTA.
%   Se resetea a 0 a proposito: con el arranque por defecto (a=0.5) el escalon
%   tendria amplitud 0.5 y la sobreoscilacion relativa saldria justo la MITAD.
reg.reset(0);
pk = -Inf;
for k = 1:400
    a = reg.step(1.0, 1.0);
    reg.sync(a, false);          % sin bloqueo: sync NO debe hacer nada
    pk = max(pk, a);
end
ov = max(0, pk - 1);
end
