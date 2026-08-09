function S = terminalSlack(plant, Kf, ob, P)
%TERMINALSLACK  S = P - Acl' P Acl - Q - Kf' R Kf.  Debe ser semidefinida positiva.
Acl = plant.A + plant.B * Kf;
S = P - Acl' * P * Acl - ob.Q - Kf' * ob.R * Kf;
S = 0.5 * (S + S');
end
