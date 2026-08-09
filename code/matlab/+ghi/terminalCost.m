function c = terminalCost(ob, x)
%TERMINALCOST Coste terminal de un objetivo.
x = x(:);
if isempty(ob.P)
    c = 0;
elseif strcmp(ob.kind, 'quad')
    c = x' * ob.P * x;
else
    c = max(abs(ob.P * x));
end
end
