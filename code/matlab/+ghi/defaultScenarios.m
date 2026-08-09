function sc = defaultScenarios()
%DEFAULTSCENARIOS Los cuatro escenarios del protocolo.
%
%   A  demanda cuasi-estatica  -> se ESPERA que el 2o orden pierda
%   B  demanda periodica
%   C  fuera del sobre de diseno (desajuste + perturbacion)
%   D  fuera del sobre, fuerte
base = struct('mismatch', [], 'dist_std', 0, 'noise_std', 0.15, ...
              'dist_amp', 0.6, 'dist_period', 17, 'x0', [5; 5], ...
              'T', 60, 'warmup', 10);

sc.A_cuasiestatico = base;
sc.A_cuasiestatico.name   = 'A_cuasiestatico';
sc.A_cuasiestatico.demand = @(t) 0.75;

sc.B_periodico = base;
sc.B_periodico.name   = 'B_periodico';
sc.B_periodico.demand = @(t) 0.5 + 0.35 * sin(2 * pi * t / 20);

sc.C_OOD = base;
sc.C_OOD.name     = 'C_OOD';
sc.C_OOD.demand   = @(t) 0.5 + 0.35 * sin(2 * pi * t / 20);
sc.C_OOD.mismatch = [0 0.15; 0 0.10];
sc.C_OOD.dist_std = 0.08;

sc.D_OOD_fuerte = base;
sc.D_OOD_fuerte.name     = 'D_OOD_fuerte';
sc.D_OOD_fuerte.demand   = @(t) 0.5 + 0.35 * sin(2 * pi * t / 13);
sc.D_OOD_fuerte.mismatch = [0 0.25; 0.05 0.18];
sc.D_OOD_fuerte.dist_std = 0.15;
end
