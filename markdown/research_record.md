## Point-source model 
$$
s_j(\textbf{x}) = q_je^{i\omega_jt}
$$
$$
\sum s_j(\textbf{x}) \approx \left( \sum q_j\right)e^{i\bar \omega t}
$$

## Multipole model
$$
\begin{aligned}
   s_j(\textbf{x}) &=  p_j(\textbf{x})  q_je^{i\omega_jt} \\
                    &\approx \left(\sum_{n,m}h^2_n(kr)Y^m_n(\theta,\phi) c_n^m \right ) q_je^{i\omega_jt} \\
                    

\end{aligned}
$$
$$
\begin{aligned}
  \sum s_j(\textbf{x}) &\approx   \sum \left( \left(\sum_{n,m}h^2_n(kr)Y^m_n(\theta,\phi) c_{n,j}^m \right ) q_j \right ) e^{i\omega_jt}  \\
                        &\approx \sum_{n,m}h^2_n(\bar kr)Y^m_n(\theta,\phi) \left( \sum c_{n,j}^m q_j \right ) e^{i\bar \omega t} \\
                        &= \sum_{n,m}h^2_n(\bar kr)Y^m_n(\theta,\phi) C_{n}^m e^{i\bar \omega t}
\end{aligned}
$$

