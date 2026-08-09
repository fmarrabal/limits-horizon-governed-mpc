function prob = problemTFM()
%PROBLEMTFM Reproduccion literal del ejemplo del TFM 2012.
%
%   J0 con normas infinito y J1 cuadratica, con los P que trae el TFM.
%   ATENCION: esos P VIOLAN la desigualdad terminal. Se dejan tal cual a
%   proposito; ghi.auditTerminal lo documenta y ghi.audit lo comprueba como
%   regresion (comprobacion A12: debe seguir fallando).

prob.name = 'TFM-2012';
prob.N    = 5;

prob.plant.A    = [1 1; 0 1];
prob.plant.B    = [0.5; 1];
prob.plant.xmax = [10; 10];
prob.plant.umax = 10;
prob.plant.n    = 2;
prob.plant.m    = 1;

prob.obj(1) = struct('name','J0-inf (TFM)', 'kind','inf', ...
                     'Q', diag([0.1 1]), 'R', 0.2, ...
                     'P', [0.5649 0.4054; 0.4054 1.6027]);
prob.obj(2) = struct('name','J1-quad (TFM)','kind','quad', ...
                     'Q', diag([1 0.1]), 'R', 0.1, ...
                     'P', [9.6085 1.1401; -0.2965 9.4107]);
end
