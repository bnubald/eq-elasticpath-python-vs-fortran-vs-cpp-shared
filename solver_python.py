"""Code by Ashley Scillitoe"""
import numpy as np

def _elastic_net_cd_py(x,A,b,lamda,alpha, max_iter,tol,verbose):
    """
    Private method to perform coordinate descent (with elastic net soft  thresholding) for a given lambda and alpha value.
    Following section 2.6 of [1], the algo does one complete pass over the features, and then for following iterations it only loops over the active set (non-zero coefficients). See commit 2b0af9f58fa5ff1876f76f7aedeaf2a0d7d252c8 for a more simple (but considerably slower for large p) algo.
    """
    # TODO - covariance updates (see 2.2 of [1]) could provide further speed up...

    # Preliminaries
    b = b.reshape(-1)
    A2 = np.sum(A**2, axis=0)
    dx_tol = tol
    n,p = A.shape

    finish  = False
    success = False
    attempt = 0
    while not success:
        attempt += 1
        if (attempt > 2):
            print('Non-zero coefficients still changing after two cycles, breaking...')
            break

        for n_iter in range(max_iter):
            x_max = 0.0
            dx_max = 0.0

            # Residual
            r = b - A@x

            active_set = set(np.argwhere(x).flatten())
            if n_iter == 0 or finish: #First iter or after convergence, loop through entire set
                loop_set = set(range(p))
            elif n_iter == 1: # Now only loop through active set (i.e. non-zero coeffs)
                loop_set = active_set

            for j in loop_set:
                r = r + A[:,j]*x[j]
                rho = A[:,j]@r/(A2[j] + lamda*(1-alpha))
                x_prev = x[j]
                if j != 0: # TODO - check p0 is still at index 0 when more than parameter
                    x[j] = _soft_threshold(rho,lamda*alpha)

                    # Update changes in coeffs
                    d_x    = abs(x[j] - x_prev)
                    dx_max = max(dx_max,d_x)
                    x_max  = max(x_max,abs(x[j]))
                else:
                    x[j] = rho

                r = r - A[:,j]*x[j]

            # Convergence check - early stop if converged
            if n_iter == max_iter-1:
                conv_msg = 'Max iterations reached without convergence'
                finish = True
            if x_max == 0.0:  # if all coeff's zero
                conv_msg = 'Convergence after %d iterations, x_max=0' %n_iter
                finish = True
            elif dx_max/x_max < dx_tol: # biggest coord update of this iteration smaller than tolerance
                conv_msg = 'Convergence after %d iterations, d_x: %.2e, tol: %.2e' %(n_iter, dx_max/x_max,dx_tol)
                finish = True
            # TODO - add further duality gap check from
            #http://proceedings.mlr.press/v37/fercoq15-supp.pdf
            #l1_reg = lamda * alpha * n # For use w/ duality gap calc.
            #l2_reg = lamda * (1.0 - alpha) * n
            #gap = tol + 1

            # Check final complete cycle doesn't add to active set, if it does complete entire process (this is rare!)
            if finish:
                final_active_set = set(np.argwhere(x).flatten())
                if len(final_active_set-active_set) == 0:
                    if verbose: print(conv_msg)
                    success = True
                else:
                    if verbose: print('Final cycle added non-zero coefficients, restarting coordinate descent')
                break
    return x

def _soft_threshold(rho,lamda):
    '''Soft thresholding operator for 1D LASSO in elastic net coordinate descent algoritm'''
    if rho < -lamda:
        return (rho + lamda)
    elif rho > lamda:
        return (rho - lamda)
    else:
        return 0.0
