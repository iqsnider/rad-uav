our IMU outputs,

$ [underline(alpha)]^B = [underline(a)_B^E]^B - [underline(g)]^B $
$ [underline(gamma)]^B = [underline(omega)_B^E]^B $

then, the convenient coordination for our dynamics,

$ [D^E underline(s)_(B T)]^E = overline([T])^(B E) [D^E underline(s)_(B T)]^B = overline([T])^(B E) [underline(v)_B^E]^B $


$ [D^E D^E underline(s)_(B T)]^B = [D^E underline(v)^E_B]^B = [underline(a)_B^E]^B &= -[underline(Omega)^(B E)]^B [D^E underline(s)_(B T)]^B + [T]^(B E)[underline(g)]^E \ &= -[underline(Omega)^(B E)]^B [underline(v)_(B)^E]^B + [T]^(B E)[underline(g)]^E $


$ frac("d","d" t) mat(phi.alt;theta;psi) = underline(F)(phi.alt, theta)[underline(omega)^(B E)]^B $

$ frac("d","d" t) mat(p;q;r) = [underline(frac("d","d" t) omega^(B E))]^B  = ([underline(J)_B^B]^B)^(-1)([underline(n)_B]^B - [underline(Omega)^(B E)]^B [underline(J)_B^B]^B [underline(omega)^(B E)]^B) $

from 3.20,

$ [frac("d","d" t) T]^(E B) = [T]^(E B) [underline(Omega)^(B E)]^B $

where,

$ [underline(Omega)^(B E)]^B = [underline(K)(underline(omega)^(B E))]^B = mat(0, -r, q; r, 0, -p; -q, p, 0) $


and $underline(K)$ is the skew-symmetric operator, and,

$ [underline(omega)^(B E)]^B = mat(p;q;r) $

and $[T]^(E B)$ is the transpose of the euler transformation matrix $[T]^(B E)$, which is expressed as,

$ [T]^(B E) = mat(cos psi cos theta, sin psi cos theta, -sin theta;
cos psi sin theta sin phi.alt - sin psi cos phi,  sin psi sin theta sin phi.alt+cos psi cos phi.alt, cos theta sin phi.alt;
cos psi sin theta cos phi.alt + sin psi sin phi.alt, sin psi sin theta cos phi.alt - cos psi sin phi.alt, cos theta cos phi.alt) $

then show,

$ frac("d","d" t) mat(phi.alt; theta; psi) = mat(1, sin phi.alt tan theta, cos phi.alt tan theta;
0, cos phi.alt, -sin phi.alt;
0, frac(sin phi.alt,cos theta), frac(cos phi.alt, cos theta)  ) [underline(omega)^(B E)]^B $
