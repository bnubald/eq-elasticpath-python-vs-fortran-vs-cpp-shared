!-------------------------------------------------------------------------
! Subroutine:  elastic_net_cd_for
!
! Purpose:     Coordinate descent for LASSO/Ridge
!              regression
!
! Programmer:  Bryn Noel Ubald
!
! Notes:       - The input x data is replaced rather than returning vector
!              - Blas is called explicitly
!              - This subroutine is multi-threaded
!
!-------------------------------------------------------------------------
subroutine elastic_net_cd_for(x, A, b, lambda, alpha, max_iter, dx_tol, verbose, xn, An, Am, bn)
    use omp_lib
    implicit none

    INTEGER xn, xm, An, Am, bn, bm
    DOUBLE PRECISION, INTENT(IN) :: A(An, Am), b(bn), lambda, dx_tol
    DOUBLE PRECISION, INTENT(INOUT) :: x(xn)
    INTEGER, INTENT(IN) :: alpha, max_iter
    LOGICAL, INTENT(IN) :: verbose

    DOUBLE PRECISION :: A2(Am), r(An)
    INTEGER n, p, attempt, lint
    LOGICAL finish, success, equal

    Integer :: i, j, l, n_iter
    DOUBLE PRECISION :: x_max, dx_max, rho, x_prev, d_x, tmp

    Integer, Dimension( : ), Allocatable :: active_set, loop_set, final_active_set, indx

    CHARACTER(256) :: conv_msg

    DOUBLE PRECISION, External :: ddot

    n = An
    p = Am

    A2 = SUM(A**2, 1) + lambda*(1-alpha)
    indx = [ ( i, i = 1, xn ) ]

    finish  = .False.
    success = .False.
    attempt = 0

    do while ( .NOT. success )
        attempt = attempt + 1
        if ( attempt > 2 ) then
          print *, "Non-zero coefficients still changing after two cycles, breaking..."
          exit
        end if

        do n_iter = 1, max_iter
            x_max = 0.0D0
            dx_max = 0.0D0

            r = b
            call DGEMV('N',n,p,-1.0D0,A,n,x,1,1.0D0,r,1)

            active_set = pack(indx, x/=0)

            if ( n_iter == 1 .or. finish) then
              loop_set = indx
            else if (n_iter == 2) then
              loop_set = active_set
            end if

            do l = 1, size(loop_set)
                j = loop_set(l)

                x_prev = x(j)

                if ( x(j) /= 0.0D0 ) then
                    call daxpy(n, x(j), A(:, j), 1, r, 1)
                end if

                rho = ddot(n, A(:, j), 1, r, 1)
                rho = rho/A2(j)

                if (j /= 1) then
                    if (rho < -lambda) then
                        x(j) = rho + lambda
                    else if (rho > lambda) then
                        x(j) = rho - lambda
                    else
                        x(j) = 0.0D0
                    end if

                    d_x = ABS(x(j) - x_prev)
                    dx_max = MAX(dx_max, d_x)
                    x_max = MAX(x_max, abs(x(j)))
                else
                    x(j) = rho
                end if

                if ( x(j) /= 0.0D0 ) then
                  call daxpy(n, -x(j), A(:,j), 1, r, 1)
                end if

            end do

            if (x_max == 0.0D0) then
                write(conv_msg, "(A,I0,A)") 'Fortran: Convergence after ' , n_iter , ' iterations, x_max=0'
                finish = .True.
            else if (n_iter == max_iter) then
                conv_msg = 'Fortran: Max iterations reached without convergence'
                finish = .True.
            else if (dx_max/x_max < dx_tol) then
                write(conv_msg, "(A,I10,A,E10.2,A,E10.2)") 'Fortran: Convergence after ', n_iter, ' iterations, d_x: ', &
                        & dx_max/x_max, ', tol: ', dx_tol
                finish = .True.
            end if

            if (finish) then
                final_active_set = Pack([(i, i=1, size(x))], x/=0)

                equal = size(active_set) == size(final_active_set)
                if ( equal ) then
                    do i = 1, size(active_set)
                        equal = active_set(i) == final_active_set(i)
                        if ( .not. equal ) then
                            exit
                        end if
                    enddo
                else
                    do i = 1, size(final_active_set)
                        equal = any(final_active_set(i) == active_set)
                        if ( .not. equal ) then
                            exit
                        end if
                    end do
                endif
                if (equal) then
                    if ( verbose ) then
                      print *, conv_msg
                    end if
                    success = .True.
                else
                    if (verbose) then
                        print *, 'Final cycle added non-zero coefficients, restarting coordinate descent'
                    end if
                end if
                exit
            end if

        end do

    end do

    return
end subroutine


!-------------------------------------------------------------------------
! Subroutine:  elastic_net_cd_purefor
!
! Purpose:     Coordinate descent for LASSO/Ridge
!              regression
!
! Programmer:  Bryn Noel Ubald
!
! Notes:       - The input x data is replaced rather than returning vector
!              - Written using pure fortran - no external call to BLAS
!
!-------------------------------------------------------------------------
! Pure fortran - no external blas calls
subroutine elastic_net_cd_purefor(x, A, b, lambda, alpha, max_iter, dx_tol, verbose, xn, An, Am, bn)
    use omp_lib
    implicit none

    INTEGER xn, xm, An, Am, bn, bm
    DOUBLE PRECISION, INTENT(IN) :: A(An, Am), b(bn), lambda, dx_tol
    DOUBLE PRECISION, INTENT(INOUT) :: x(xn)
    INTEGER, INTENT(IN) :: alpha, max_iter
    LOGICAL, INTENT(IN) :: verbose

    DOUBLE PRECISION :: A2(Am), r(An)
    INTEGER n, p, attempt, lint
    LOGICAL finish, success, equal

    Integer :: i, j, l, n_iter
    DOUBLE PRECISION :: x_max, dx_max, rho, x_prev, d_x, tmp

    Integer, Dimension( : ), Allocatable :: active_set, loop_set, final_active_set, indx

    CHARACTER(256) :: conv_msg

    n = An
    p = Am

    A2 = SUM(A**2, 1) + lambda*(1-alpha)
    indx = [ ( i, i = 1, xn ) ]

    finish  = .False.
    success = .False.
    attempt = 0

    do while ( .NOT. success )
        attempt = attempt + 1
        if ( attempt > 2 ) then
          print *, "Non-zero coefficients still changing after two cycles, breaking..."
          exit
        end if

        do n_iter = 1, max_iter
            x_max = 0.0D0
            dx_max = 0.0D0

            r = b - MATMUL(A, x)

            active_set = pack(indx, x/=0)

            if ( n_iter == 1 .or. finish) then
              loop_set = indx
            else if (n_iter == 2) then
              loop_set = active_set
            end if

            do l = 1, size(loop_set)
                j = loop_set(l)

                x_prev = x(j)

                if ( x(j) /= 0.0D0 ) then
                    r = r + A(:, j) * x(j)
                end if

                rho = DOT_PRODUCT(A(:, j), r)/A2(j)

                if (j /= 1) then
                    if (rho < -lambda) then
                        x(j) = rho + lambda
                    else if (rho > lambda) then
                        x(j) = rho - lambda
                    else
                        x(j) = 0.0D0
                    end if
                else
                    x(j) = rho
                end if

                if ( x(j) /= 0.0D0 ) then
                  r = r - A(:,j)*x(j)
                end if

                if (j /= 1) then
                    d_x = ABS(x(j) - x_prev)
                    dx_max = MAX(dx_max, d_x)
                    x_max = MAX(x_max, abs(x(j)))
                end if

            end do

            if (x_max == 0.0D0) then
                write(conv_msg, "(A,I0,A)") 'Fortran: Convergence after ' , n_iter , ' iterations, x_max=0'
                finish = .True.
            else if (n_iter == max_iter) then
                conv_msg = 'Fortran: Max iterations reached without convergence'
                finish = .True.
            else if (dx_max/x_max < dx_tol) then
                write(conv_msg, "(A,I10,A,E10.2,A,E10.2)") 'Fortran: Convergence after ', n_iter, ' iterations, d_x: ', &
                        & dx_max/x_max, ', tol: ', dx_tol
                finish = .True.
            end if

            if (finish) then
                final_active_set = Pack([(i, i=1, size(x))], x/=0)

                !Set difference logic
                !Check if all in final_active_set was already in active_set
                equal = size(active_set) == size(final_active_set)
                if ( equal ) then
                    do i = 1, size(active_set)
                        equal = active_set(i) == final_active_set(i)
                        if ( .not. equal ) then
                            exit
                        end if
                    enddo
                else
                    do i = 1, size(final_active_set)
                        equal = any(final_active_set(i) == active_set)
                        if ( .not. equal ) then
                            exit
                        end if
                    end do
                endif

                if (equal) then
                    if ( verbose ) then
                      print *, conv_msg
                    end if
                    success = .True.
                else
                    if (verbose) then
                        print *, 'Final cycle added non-zero coefficients, restarting coordinate descent'
                    end if
                end if
                exit
            end if

        end do

    end do

    return
end subroutine
