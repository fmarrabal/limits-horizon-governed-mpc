function r = certifiedRadius(J, V, Ja)
%CERTIFIEDRADIUS  r = (J_a - V) / ||J - mean(J)||_2.
%
%   Por concavidad, J(U*(x,alpha),x) es un SUPERGRADIENTE de V*(x,.) en alpha
%   (teorema de la envolvente), luego
%       V*(x,alpha') <= V*(x,alpha) + J'(alpha' - alpha).
%   Como alpha y alpha' viven en el simplex, 1'(alpha'-alpha) = 0 y J puede
%   sustituirse por su version CENTRADA sin cambiar el producto. Cauchy-Schwarz
%   da entonces la condicion suficiente ||alpha'-alpha|| <= r.
%
%   La observacion fina es el CENTRADO: lo que limita el movimiento del peso es
%   la DISPERSION de los costes entre objetivos, no su magnitud.
J = J(:);
Jc = J - mean(J);
nrm = norm(Jc);
slack = Ja - V;
if slack < 0
    r = 0;
elseif nrm < 1e-14
    r = Inf;                 % costes degenerados: el peso no cambia nada
else
    r = slack / nrm;
end
end
