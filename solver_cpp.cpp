#include<iostream>
#include<set>
#include <functional>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#include <Eigen/LU>

namespace py = pybind11;
using namespace std;
using namespace Eigen;
using MatrixXdRef = Eigen::Ref<MatrixXd>;
using VectorXdRef = Eigen::Ref<VectorXd>;

// https://stackoverflow.com/questions/61872816/eigen-get-array-of-indexes-where-values-in-vector-are-true-no-unnecessary-for-l
template<typename Func>
struct lambda_as_visitor_wrapper : Func {
    lambda_as_visitor_wrapper(const Func& f) : Func(f) {}
    template<typename S,typename I>
    void init(const S& v, I i, I j) { return Func::operator()(v,i,j); }
};

template<typename Mat, typename Func>
void visit_lambda(const Mat& m, const Func& f)
{
    lambda_as_visitor_wrapper<Func> visitor(f);
    m.visit(visitor);
}

inline double _soft_threshold( double rho, double lambda ) {

    if ( rho < -lambda ) {
        return rho + lambda;
    }
    else if ( rho > lambda ) {
        return rho - lambda;
    }
    else {
        return 0.0;
    }

}

/*-------------------------------------------------------------------------
  * Function:    elastic_net_cd_cpp
  *
  * Purpose:     Classes to help with coordinate descent for LASSO/Ridge
  *              regression
  *
  * Return:      void
  *
  * Programmer:  Bryn Noel Ubald
  *
  * Notes:       The input x data is replaced rather than returning vector
  *
  *-------------------------------------------------------------------------
  */
void elastic_net_cd_cpp( VectorXdRef x, MatrixXdRef A, VectorXdRef b, double lambda, double alpha = 1.0, int max_iter = 100, double tol = 1E-6, bool verbose=0 ){

    double dx_tol = tol;
    int n = A.rows();
    int p = A.cols();

    bool finish = false;
    bool success = false;
    int attempt = 0;

    int th = 0;
    double rho, x_max, d_x, dx_max, x_prev;

    using MySet = std::set<int>;
    MySet loop_set, active_set, final_active_set, indx;
    VectorXd r(n);

    VectorXd A2 = A.array().square().colwise().sum() + lambda*(1-alpha);

    for( int i = 0; i < p; ++i ) {
        indx.insert( i );
    }

    while (success == false) {
        ++attempt;
        if (attempt > 2){
            std::cout << "Non-zero coefficients still changing after two cycles, breaking...\n";
            break;
        }

        std::stringstream conv_msg;
        for ( int n_iter = 0; n_iter < max_iter; ++n_iter ){
            x_max = 0.0;
            dx_max = 0.0;

            // Residual
            r.noalias() = b - A*x;

            active_set.clear();
            visit_lambda(x,
                [&active_set, th](double v, int i, int j) {
                    if (v != 0) { active_set.insert(i); }
                });

            if ( n_iter == 0 ) {
                loop_set = indx;
            }
            else if ( finish == true ) {
                loop_set = indx;
            }
            else if ( n_iter == 1 ) {
                loop_set = active_set;
            }

            for ( auto j : loop_set ){

                x_prev = x(j);

                if ( x(j) != 0.0 ) {
                    r = r + A.col(j)*x(j);
                }

                rho = (A.col(j).array()*r.array()).sum()/A2(j);

                if ( j != 0 ) {
                    x(j) = _soft_threshold( rho, lambda );

                    d_x = std::abs(x(j) - x_prev);
                    dx_max = std::max(dx_max, d_x);
                    x_max = std::max(x_max, std::abs(x(j)));
                }
                else{
                    x(j) = rho;
                }

                if ( x(j) != 0.0 ) {
                    r = r - A.col(j)*x(j);
                }

            }

            if ( x_max == 0.0 ) {
                conv_msg << "Convergence after " << n_iter << " iterations, x_max=0\n";
                finish = true;
            }
            else if ( n_iter == max_iter - 1 ) {
                conv_msg << "Max iterations reached without convergence\n";
                finish = true;
            }
            else if ( dx_max/x_max < dx_tol ) {
                conv_msg << "Convergence after " << n_iter << " iterations, d_x: " << dx_max/x_max << ", tol: " << dx_tol << '\n';
                finish = true;
            }

            if ( finish == true ) {
                final_active_set.clear();
                visit_lambda(x,
                    [&final_active_set, th](double v, int i, int j) {
                        if (v != 0) { final_active_set.insert(i); }
                    });

                std::vector<int> diff;

                std::set_difference(final_active_set.begin(), final_active_set.end(), active_set.begin(),
                                    active_set.end(), std::inserter(diff, diff.begin()));

                if ( diff.size() == 0 ) {
                    if (verbose == true) { std::cout << conv_msg.str(); };
                    success = true;
                }
                else {
                    if (verbose == true) { std::cout << "Final cycle added non-zero coefficients, restarting coordinate descent\n"; }
                }
                break;
            }

        }

    }

}


/*-------------------------------------------------------------------------
  * Function:    elastic_net_cd_cpp_nosets
  *
  * Purpose:     Classes to help with coordinate descent for LASSO/Ridge
  *              regression - without using STL sets to build active sets
  *
  * Return:      void
  *
  * Programmer:  Bryn Noel Ubald
  *
  * Notes:       The input x data is replaced rather than returning vector
  *
  *-------------------------------------------------------------------------
  */
void elastic_net_cd_cpp_nosets( VectorXdRef x, MatrixXdRef A, VectorXdRef b, double lambda, double alpha = 1.0, int max_iter = 100, double tol = 1E-6, bool verbose=0 ){

    double dx_tol = tol;
    int n = A.rows();
    int p = A.cols();

    bool finish = false;
    bool success = false;
    bool equal = false;
    int attempt = 0;

    int th = 0, j;
    double rho, x_max, d_x, dx_max, x_prev;

    VectorXd r(n);
    std::vector<int> indx(p), active_set, loop_set, final_active_set;

    VectorXd A2 = A.array().square().colwise().sum() + lambda*(1-alpha);

    for( int i = 0; i < p; ++i ) {
        indx[i] = i;
    }

    while (success == false) {
        ++attempt;
        if (attempt > 2){
            std::cout << "Non-zero coefficients still changing after two cycles, breaking...\n";
            break;
        }

        std::stringstream conv_msg;
        for ( int n_iter = 0; n_iter < max_iter; ++n_iter ){
            x_max = 0.0;
            dx_max = 0.0;

            // Residual
            r.noalias() = b - A*x;

            active_set.clear();
            visit_lambda(x,
                [&active_set, th](double v, int i, int j) {
                    if (v != 0) { active_set.push_back(i); }
                });

            if ( n_iter == 0 || finish == true) {
                loop_set = indx;
            }
            else if ( n_iter == 1 ) {
                loop_set = active_set;
            }

            for ( size_t l = 0; l < loop_set.size(); ++l ){
                j = loop_set[l];

                x_prev = x(j);

                if ( x(j) != 0.0 ) {
                    r = r + A.col(j)*x(j);
                }

                rho = A.col(j).transpose()*r;
                rho = rho/A2(j);

                if ( j != 0 ) {
                    x(j) = _soft_threshold( rho, lambda );

                    d_x = std::abs(x(j) - x_prev);
                    dx_max = std::max(dx_max, d_x);
                    x_max = std::max(x_max, std::abs(x(j)));
                }
                else{
                    x(j) = rho;
                }

                if ( x(j) != 0.0 ) {
                    r = r - A.col(j)*x(j);
                }

            }

            if ( x_max == 0.0 ) {
                conv_msg << "Convergence after " << n_iter << " iterations, x_max=0\n";
                finish = true;
            }
            else if ( n_iter == max_iter - 1 ) {
                conv_msg << "Max iterations reached without convergence\n";
                finish = true;
            }
            else if ( dx_max/x_max < dx_tol ) {
                conv_msg << "Convergence after " << n_iter << " iterations, d_x: " << dx_max/x_max << ", tol: " << dx_tol << '\n';
                finish = true;
            }

            if ( finish == true ) {
                final_active_set.clear();
                visit_lambda(x,
                    [&final_active_set, th](double v, int i, int j) {
                        if (v != 0) { final_active_set.push_back(i); }
                    });

                equal = active_set.size() == final_active_set.size();
                if (equal) {
                    for ( size_t i = 0; i < active_set.size(); ++i ) {
                        equal = active_set[i] == final_active_set[i];
                    }
                }
                else {
                    std::vector<int>::iterator it;
                    for ( size_t i = 0; i < final_active_set.size(); ++i ) {
                        it = std::find(active_set.begin(), active_set.end(), final_active_set[i]);
                        if ( it == active_set.end() ) {
                            equal = false;
                            break;
                        }
                    }
                }

                if ( equal ) {
                    if (verbose == true) { std::cout << conv_msg.str(); };
                    success = true;
                }
                else {
                    if (verbose == true) { std::cout << "Final cycle added non-zero coefficients, restarting coordinate descent\n"; }
                }
                break;
            }

        }

    }

}

// ----------------
// Python interface
// ----------------

namespace py = pybind11;

PYBIND11_MODULE(solver_cpp,m)
{
  m.doc() = "Coordinate Descent for Elastic Path";

  m.def("elastic_net_cd_cpp", &elastic_net_cd_cpp, "Perform coordinate descent",
        py::arg("x").noconvert() = NULL,
        py::arg("A").noconvert() = NULL,
        py::arg("b").noconvert() = NULL,
        py::arg("lambda") = NULL,
        py::arg("alpha") = 1.0,
        py::arg("max_iter") = 100,
        py::arg("tol") = 1E-6,
        py::arg("verbose") = NULL
      );

    m.def("elastic_net_cd_cpp_nosets", &elastic_net_cd_cpp_nosets, "Perform coordinate descent",
          py::arg("x").noconvert() = NULL,
          py::arg("A").noconvert() = NULL,
          py::arg("b").noconvert() = NULL,
          py::arg("lambda") = NULL,
          py::arg("alpha") = 1.0,
          py::arg("max_iter") = 100,
          py::arg("tol") = 1E-6,
          py::arg("verbose") = NULL
        );
}
