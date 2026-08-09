classdef Order1 < handle
%ORDER1 Filtro de primer orden. Solo puede retrasarse: nunca adelanta fase.
%   tau se elige con ghi.matchFirstOrder, que iguala la atenuacion a Nyquist
%   EN TIEMPO DISCRETO contra el regulador de segundo orden.
    properties
        tau, a, a0
    end
    properties (Constant)
        name = 'orden-1'
    end
    methods
        function o = Order1(tau, a0)
            if nargin < 2 || isempty(a0), a0 = 0.5; end
            assert(tau > 0, 'tau debe ser positivo');
            o.tau = tau;  o.a = a0;  o.a0 = a0;
        end
        function a = step(o, ad, dt)
            if nargin < 3, dt = 1.0; end
            o.a = o.a + dt / o.tau * (ad - o.a);
            a = o.a;
        end
        function sync(o, aApplied, blocked)
            if blocked, o.a = aApplied; end
        end
        function reset(o, a0)
            if nargin < 2 || isempty(a0), a0 = o.a0; end
            o.a = a0;
        end
    end
end
