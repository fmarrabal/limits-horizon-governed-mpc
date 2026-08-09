function c = stageCost(ob, x, u)
%STAGECOST Coste de etapa de un objetivo.
%   kind='quad' -> x'Qx + u'Ru
%   kind='inf'  -> ||Qx||_inf + ||Ru||_inf
x = x(:); u = u(:);
if strcmp(ob.kind, 'quad')
    c = x' * ob.Q * x + u' * ob.R * u;
else
    c = max(abs(ob.Q * x)) + max(abs(ob.R * u));
end
end
