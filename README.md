# TLFCC
Target Label-Free Confidence Calibration Under Label Shift

| Variables | Meaning |
|-|-|
|Rs|$P(Y=0)$|User-defined|
|Rt|$Q(Y=0)$|User-defined|
|alpha1,beta1|Beta distribution parameters $\alpha_{1},\beta_{1}$ to generate confidence score $\hat S\|Y=0$|
|alpha2,beta2|Beta distribution parameters $\alpha_{2},\beta_{2}$ to generate confidence score $\hat S\|Y=1$|
|g1|The preset true calibration curve of the first class, i.e., $P(H = 1\|\hat S,Y = 0)$|
|g2|The preset true calibration curve of the first class, i.e., $P(H = 1\|\hat S,Y = 1)$|
