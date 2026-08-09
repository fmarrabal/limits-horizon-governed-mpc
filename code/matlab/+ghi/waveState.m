function W = waveState(K, C, G, placement)
%WAVESTATE Matriz de estado segun donde se coloque el operador antisimetrico.
%   'gyroscopic'  -> G sobre la velocidad: SEGURO
%   'circulatory' -> G sobre la posicion : FLUTTER por encima de un umbral
M = size(K, 1);
Z = zeros(M);  I = eye(M);
switch placement
    case 'gyroscopic'
        W = [Z I; -K -(C + G)];
    case 'circulatory'
        W = [Z I; -(K + G) -C];
    otherwise
        error('ghi:waveState', 'placement desconocido: %s', placement);
end
end
