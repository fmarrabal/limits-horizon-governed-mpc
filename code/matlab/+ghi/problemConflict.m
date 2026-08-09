function prob = problemConflict()
%PROBLEMCONFLICT Banco principal: dos objetivos que SI compiten.
%
%   J0 = economia     (penaliza casi solo el esfuerzo de control)
%   J1 = prestaciones (penaliza casi solo la desviacion de estado)
%
%   En el banco del TFM los dos objetivos casi no compiten -- el coste en lazo
%   cerrado sale identico para cualquier regulador de alpha -- asi que no puede
%   discriminar entre reguladores. Este si.
%
%   Los costes terminales P_i NO se fijan aqui: los calcula ghi.designTerminal
%   por ecuacion de Lyapunov, de modo que S_i = 0 exactamente.
%
%   Ver tambien GHI.PROBLEMTFM, GHI.DESIGNTERMINAL.

prob.name = 'conflicto';
prob.N    = 5;

prob.plant.A    = [1 1; 0 1];
prob.plant.B    = [0.5; 1];
prob.plant.xmax = [10; 10];
prob.plant.umax = 10;
prob.plant.n    = 2;
prob.plant.m    = 1;

prob.obj(1) = struct('name','economia',    'kind','quad', ...
                     'Q', diag([0.01 0.01]), 'R', 5.0,  'P', []);
prob.obj(2) = struct('name','prestaciones','kind','quad', ...
                     'Q', diag([5 1]),       'R', 0.01, 'P', []);
end
