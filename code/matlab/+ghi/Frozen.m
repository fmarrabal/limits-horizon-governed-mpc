classdef Frozen < handle
%FROZEN Peso constante. Control nulo: aisla el mero hecho de tener parametros.
    properties
        a
    end
    properties (Constant)
        name = 'congelado'
    end
    methods
        function o = Frozen(a)
            if nargin < 1 || isempty(a), a = 0.5; end
            o.a = a;
        end
        function out = step(o, ~, ~), out = o.a; end
        function sync(~, ~, ~), end
        function reset(~, ~), end
    end
end
