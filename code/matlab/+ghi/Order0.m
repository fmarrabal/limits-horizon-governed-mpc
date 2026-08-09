classdef Order0 < handle
%ORDER0 Sin memoria: a = a_d. Es el alpha_d del TFM.
    properties
        a, a0
    end
    properties (Constant)
        name = 'orden-0'
    end
    methods
        function o = Order0(a0)
            if nargin < 1 || isempty(a0), a0 = 0.5; end
            o.a = a0;  o.a0 = a0;
        end
        function a = step(o, ad, ~)
            o.a = ad;  a = o.a;
        end
        function sync(o, aApplied, ~)
            o.a = aApplied;
        end
        function reset(o, a0)
            if nargin < 2 || isempty(a0), a0 = o.a0; end
            o.a = a0;
        end
    end
end
