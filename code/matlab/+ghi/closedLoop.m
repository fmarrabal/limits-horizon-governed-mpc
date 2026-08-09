function tr = closedLoop(M, reg, sc, seed, gridPoints)
%CLOSEDLOOP Una corrida completa del lazo GHI + MPC + planta.
%
%   1. llega la demanda limpia a_d(t) y su version con ruido
%   2. el regulador PROPONE a_req
%   3. se forma J_a con la secuencia desplazada y el peso ANTERIOR
%   4. se evalua el conjunto admisible (se barre: no es convexo)
%   5. se aplica el admisible mas cercano y se marca si hubo bloqueo
%   6. se avisa al regulador SOLO si hubo bloqueo
%   7. se resuelve el MPC con el peso aplicado y se ejecuta u_0
%   8. la planta avanza con desajuste, excitacion persistente y ruido
%
%   El coste de evaluacion se pondera con la DEMANDA LIMPIA, no con el peso
%   aplicado: si no, cada regulador se puntuaria con su propio criterio.
%
%   tr = [] si el MPC se vuelve infactible.

if nargin < 4 || isempty(seed), seed = 0; end
if nargin < 5 || isempty(gridPoints), gridPoints = 21; end

rng(seed, 'twister');
grid  = linspace(0, 1, gridPoints);
step  = grid(2) - grid(1);
prob  = M.prob;
plant = prob.plant;
Areal = plant.A;
if isfield(sc, 'mismatch') && ~isempty(sc.mismatch)
    Areal = Areal + sc.mismatch;
end

x = sc.x0(:);
reg.reset();
aPrev = 0.5;
sol = ghi.solveMOMPC(M, x, [1 - aPrev; aPrev]);
if isempty(sol), tr = []; return, end

T = sc.T;
alpha = zeros(T,1); clean = zeros(T,1); req = zeros(T,1);
xs = zeros(T, plant.n); us = zeros(T,1); stage = zeros(T,1);
blocked = false(T,1); radius = zeros(T,1);

for t = 0:T-1
    adc = min(max(sc.demand(t), 0), 1);
    adr = min(max(adc + sc.noise_std * randn(), 0), 1);
    aReq = min(max(reg.step(adr, 1.0), 0), 1);

    Us = ghi.shiftedSequence(M, sol);
    Js = ghi.costs(prob, x, Us);
    Ja = [1 - aPrev, aPrev] * Js;

    adm = ghi.admissibleSet(M, x, Ja, grid);
    if any(adm.feasible)
        cand = grid(adm.feasible);
        [d, j] = min(abs(cand - aReq));
        aApp = cand(j);
        blk = d > 0.51 * step;
    else
        [~, j] = min(adm.V);
        aApp = grid(j);
        blk = true;
    end
    reg.sync(aApp, blk);

    sol = ghi.solveMOMPC(M, x, [1 - aApp; aApp]);
    if isempty(sol), tr = []; return, end
    u = sol.U(1, 1);
    r = ghi.certifiedRadius(sol.J, sol.V, Ja);

    c0 = ghi.stageCost(prob.obj(1), x, u);
    c1 = ghi.stageCost(prob.obj(2), x, u);

    k = t + 1;
    alpha(k) = aApp; clean(k) = adc; req(k) = aReq;
    xs(k, :) = x'; us(k) = u; blocked(k) = blk; radius(k) = r;
    stage(k) = (1 - adc) * c0 + adc * c1;

    wd = zeros(plant.n, 1);
    wd(end) = sc.dist_amp * sin(2 * pi * t / sc.dist_period);
    wn = zeros(plant.n, 1);
    if sc.dist_std > 0, wn = sc.dist_std * randn(plant.n, 1); end
    x = min(max(Areal * x + plant.B * u + wd + wn, -plant.xmax), plant.xmax);
    aPrev = aApp;
end

tr = struct('alpha', alpha, 'clean', clean, 'req', req, 'x', xs, 'u', us, ...
            'stage', stage, 'blocked', blocked, 'radius', radius);
end
