function run_audit()
%RUN_AUDIT Lanza la suite de auditoria MATLAB.
%   Debe dar los mismos veredictos que `python -m ghi.audit`.
here = fileparts(mfilename('fullpath'));
addpath(fileparts(here));
checks = ghi.audit(true);
nb = sum(cellfun(@(c) ~c.ok, checks));
if nb > 0
    error('ghi:audit', '%d comprobaciones han fallado', nb);
end
end
