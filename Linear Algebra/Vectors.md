* The dot product
  * $\mathbf{x}\bullet\mathbf{y}=x_1y_1+...+x_ny_n$
  * Find the angle
    * $\cos (\alpha)=\frac {\mathbf{x}\bullet\mathbf{y}}{\Vert\mathbf{x}\Vert \Vert\mathbf{y}\Vert}$
  * The vector component
    * $\mathbf{p}=\text{proj}_{\mathbf{w}}(\mathbf{v})=\left(\frac{\mathbf{w}\bullet\mathbf{v}}{\mathbf{w}\bullet\mathbf{w}}\right)\ \mathbf{w}$
  * The scalar component
    * $\text{comp}_{\mathbf{w}}(\mathbf{v}) = \frac{\mathbf{w}\bullet\mathbf{v}}{\Vert\mathbf{w}\Vert}$
* The Cross Product
  * $\mathbf{u}\times\mathbf{v}=\begin{bmatrix}u_1\\u_2\\u_3\end{bmatrix}\times\begin{bmatrix}v_1\\v_2\\v_3\end{bmatrix}=\begin{bmatrix}u_2v_3-u_3v_2\\u_3v_1-u_1v_3\\u_1v_2-u_2v_1\end{bmatrix}$
  * Rules of calculation
    * $\mathbf{a}\times (\mathbf{b}+\mathbf{c}) = \mathbf{a}\times\mathbf{b}+\mathbf{a}\times\mathbf{c}$
    * $(c\mathbf{a})\times \mathbf{b} = \mathbf{a}\times(c\mathbf{b}) = c(\mathbf{a}\times\mathbf{b})$
    * $\mathbf{a}\times \mathbf{b} = -\mathbf{b}\times\mathbf{a}$
    * $(\mathbf{a}\times\mathbf{b})\times\mathbf{c} = \mathbf{a}\times(\mathbf{b}\times\mathbf{c}) + \mathbf{b}\times(\mathbf{c}\times\mathbf{a})$
    * if $\mathbf{a}$ and $\mathbf{b}$ lie along the same line
      * $\mathbf{a}\times \mathbf{b} = \mathbf{0}$
  * Applications 
    * $\Vert\mathbf{a}\times\mathbf{b}\Vert^2 = \Vert\mathbf{a}\Vert^2\Vert\mathbf{b}\Vert^2 - (\mathbf{a}\bullet\mathbf{b})^2$
    * $\text{Area} = \Vert\mathbf{a}\times\mathbf{b}\Vert =\Vert\mathbf{a}\Vert\ \Vert\mathbf{b}\Vert\sin(\alpha)$
    * $\text{Volume} = \Vert\mathbf{a}\times\mathbf{b}\Vert \big|\Vert\mathbf{c}\Vert\cos(\theta)\big| = \big|\ \Vert\mathbf{a}\times\mathbf{b}\Vert\ \Vert \mathbf{c}\Vert\cos(\theta)\ \big|=\big| (\mathbf{a}\times\mathbf{b})\bullet\mathbf{c}\big|$
