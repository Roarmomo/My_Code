load("ABSZ");
load("CPSZ");
load("FNSZ");
load("GNSZ");
load("MYSZ");
load("SPSZ");
load("TCSZ");
load("TNSZ");
[A]=splitData(ABSZ);
[B]=splitData(CPSZ);
[C]=splitData(FNSZ);
[D]=splitData(GNSZ);
[E]=splitData(MYSZ);
[F]=splitData(SPSZ);
[G]=splitData(TCSZ);
[H]=splitData(TNSZ);
DATA=[A(1:19,1) B(11:29,1)  C(1:19,1)  D(1:19,1) E(1:19,1) F(1:19,1)  G(1:19,4)   H(1:19,1)]
bar3(DATA)