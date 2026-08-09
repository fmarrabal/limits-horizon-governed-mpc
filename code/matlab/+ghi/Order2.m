classdef Order2 < handle
%ORDER2 Campo homeostatico de segundo orden (rama de onda del HBP, un nodo).
%
%       a'' + 2 zeta w0 a' + w0^2 (a - a_d) = 0
%
%   integrado con Verlet de velocidad.
%
%   theta es la ganancia de anti-windup: la fraccion de velocidad que se
%   CONSERVA cuando el filtro de seguridad recorta.
%       theta = 1 -> arrastre (posicion sincronizada, velocidad intacta)
%       theta = 0 -> reset ingenuo (mata toda la velocidad)
%
%   OJO -- FALLO QUE ESTA CLASE EXISTE PARA NO REPETIR: sync SOLO debe actuar
%   si hubo bloqueo DE VERDAD. En la version anterior se comparaba el valor
%   aplicado (que cae en una malla) con el estado continuo del campo, asi que
%   la condicion se cumplia en cada muestra: 200/200 pasos con cero bloqueos
%   inducidos, y el zeta EFECTIVO pasaba de 0.5 a ~0.868.

    properties
        w0, zeta, theta, a, v, a0
    end
    properties (Constant)
        name = 'orden-2'
    end

    methods
        function o = Order2(w0, zeta, theta, a0)
            if nargin < 3 || isempty(theta), theta = 0.5; end
            if nargin < 4 || isempty(a0),    a0    = 0.5; end
            assert(w0 > 0 && zeta >= 0, 'w0 > 0 y zeta >= 0');
            assert(theta >= 0 && theta <= 1, 'theta en [0,1]');
            o.w0 = w0; o.zeta = zeta; o.theta = theta;
            o.a0 = a0; o.a = a0; o.v = 0;
        end

        function acc = accel(o, a, v, ad)
            acc = -2 * o.zeta * o.w0 * v - o.w0^2 * (a - ad);
        end

        function a = step(o, ad, dt)
            if nargin < 3, dt = 1.0; end
            acc  = o.accel(o.a, o.v, ad);
            o.a  = o.a + dt * o.v + 0.5 * dt^2 * acc;
            vh   = o.v + 0.5 * dt * acc;
            o.v  = vh + 0.5 * dt * o.accel(o.a, vh, ad);
            a = o.a;
        end

        function sync(o, aApplied, blocked)
            if blocked                      % SOLO al bloquear de verdad
                o.v = o.v * o.theta;
                o.a = aApplied;
            end
        end

        function reset(o, a0)
            if nargin < 2 || isempty(a0), a0 = o.a0; end
            o.a = a0;  o.v = 0;
        end
    end
end
