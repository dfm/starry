#include <boost/math/quadrature/gauss.hpp>
#include <boost/math/quadrature/gauss_kronrod.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <stdlib.h>
#include <iostream>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <tuple>
#include "utils.h"
#include "limbdark.h"
#include "tables.h"

using namespace boost::math::quadrature;
using namespace utils;
using std::abs;
namespace py = pybind11;

template <class T>
class TransitInfo {

public:

    limbdark::GreensLimbDark<T> L;
    Vector<T> agol_c;
    T agol_norm;
    size_t n_eval;

    explicit TransitInfo(const int lmax, const Vector<double>& u) : L(lmax), n_eval(0) {
        // Convert to Agol basis
        Vector<T> u_(lmax + 1);
        u_(0) = -1.0;
        u_.segment(1, lmax) = u.template cast<T>();
        agol_c = limbdark::computeC(u_);
        agol_norm = limbdark::normC(agol_c);
    }

};

template <class T>
inline T flux(const T& b, const T& r, TransitInfo<T>& I) {
    I.n_eval += 1;
    if (b < 1 + r) {
        I.L.compute(b, r);
        return I.L.S.dot(I.agol_c) * I.agol_norm;
    } else {
        return 1.0;
    }
}

template <class T>
T dfluxdb(int n, const T& b, const T& r, TransitInfo<T>& I) {
    Vector<T> a;
    T eps;
    if (n == 2) {
        a.resize(3);
        a << 1, -2, 1;
        eps = pow(mach_eps<T>(), 1. / 3.);
    } else if (n == 4){
        a.resize(5);
        a << 1, -4, 6, -4, 1;
        eps = pow(mach_eps<T>(), 1. / 6.);
    } else if (n == 6) {
        a.resize(7);
        a << 1, -6, 15, -20, 15, -6, 1;
        eps = pow(mach_eps<T>(), 1. / 9.);
    } else if (n == 8) {
       a.resize(9);
       a << 1, -8, 28, -56, 70, -56, 28, -8, 1;
       eps = pow(mach_eps<T>(), 1. / 12.);
    } else {
        throw std::invalid_argument( "Invalid order." );
    }
    T res = 0;
    int imax = (a.rows() - 1) / 2;
    for (int i = -imax, j = 0; i <= imax; ++i, ++j ) {
        res += a(j) * flux(abs(b + i * eps), r, I);
    }
    return res / (pow(eps, n));
}

template <class T, unsigned Points>
T fluenceQuad(const T& expo, const T& b, const T& r, TransitInfo<T>& I, unsigned max_depth, const T& tol) {
    auto f = [&r, &I](const T& b_) { return flux(abs(b_), r, I); };
    return gauss_kronrod<T, Points>::integrate(f, b - 0.5 * expo, b + 0.5 * expo, max_depth, tol) / expo;
}

template <class T>
T fint(const int order, const T& t1, const T& t2, const T& b, const T& r, TransitInfo<T>& I) {
    T tavg = 0.5 * abs(t1 + t2);
    T tdif = (t2 - t1);
    T f = flux(tavg, r, I);
    if (order > 0) {
        f += (1 / 24.) * pow(tdif, 2) * dfluxdb(2, tavg, r, I);
        if (order > 2) {
            f += (1 / 1920.) * pow(tdif, 4) * dfluxdb(4, tavg, r, I);
            if (order > 4) {
                f += (1 / 322560.) * pow(tdif, 6) * dfluxdb(6, tavg, r, I);
                if (order > 6) {
                    f += (1 / 92897280.) * pow(tdif, 8) * dfluxdb(8, tavg, r, I);
                }
            }
        }
    }
    return f * tdif;
}


template <class T>
T fluenceTaylor(const int order, const T& expo, const T& b, const T& r, TransitInfo<T>& I) {

    // All possible limits of integration
    T e = 0.5 * expo;
    T P = 1 - r,
      Q = 1 + r,
      A = P - e,
      B = P + e,
      C = Q - e,
      D = Q + e,
      E = 2 * b - P,
      F = 2 * b - Q,
      Z = 0.0;
    std::vector<T> all_limits {A, B, C, D, E, F, P, Q, Z};

    // Identify and sort the relevant ones
    std::vector<T> limits;
    limits.push_back(b - e);
    for (auto lim : all_limits) {
        if ((lim > b - e) && (lim < b + e))
            limits.push_back(lim);
    }
    limits.push_back(b + e);
    std::sort(limits.begin() + 1, limits.end() - 1);

    // Compute the integrals
    T f = 0;
    T dt;
    for (size_t i = 0; i < limits.size() - 1; ++i) {
        f += fint(order, limits[i], limits[i + 1], b, r, I);
    }

    return f / expo;
}

Vector<double> computeFlux(const Vector<double>& b, const double& r, const Vector<double>& u) {

    using T = Multi;
    int npts = b.rows();
    int lmax = u.rows();
    Vector<T> f(npts);
    TransitInfo<T> I(lmax, u);
    TransitInfo<double> dblI(lmax, u);

    // Run!
    T bi;
    T r_ = T(r);
    for (int i = 0; i < npts; ++i) {
        bi = T(abs(b(i)));
        f(i) = flux(bi, r_, I);
    }

    return f.template cast<double>();

}

Vector<double> computeTaylorFluence(const Vector<double>& b, const double& r, const Vector<double>& u, const double& expo, const int order) {

    using T = Multi;
    int npts = b.rows();
    int lmax = u.rows();
    Vector<T> f(npts);
    TransitInfo<T> I(lmax, u);

    // Run!
    T bi;
    T r_ = T(r);
    T expo_ = T(expo);
    for (int i = 0; i < npts; ++i) {
        bi = T(abs(b(i)));
        f(i) = fluenceTaylor(order, expo_, bi, r_, I);
    }

    return f.template cast<double>();

}

template <typename T, unsigned Points>
std::tuple<Vector<double>, long int> computeExactFluence(const Vector<double>& b, const double& r, const Vector<double>& u, const double& expo,
    unsigned max_depth, const double& tol) {

    int npts = b.rows();
    int lmax = u.rows();
    Vector<T> f(npts);
    TransitInfo<T> I(lmax, u);

    // Run!
    T bi;
    T r_ = T(r);
    T expo_ = T(expo);
    for (int i = 0; i < npts; ++i) {
        bi = T(abs(b(i)));
        f(i) = fluenceQuad<T, Points>(expo_, bi, r_, I, max_depth, tol);
    }

    return std::make_tuple(f.template cast<double>(), I.n_eval);

}

std::tuple<Vector<double>, long int> computeLeftRiemannFluence(const Vector<double>& b, const double& r, const Vector<double>& u, const double& expo, const int ndiv) {

    using T = double;
    int npts = b.rows();
    int lmax = u.rows();
    Vector<T> f(npts);
    TransitInfo<T> I(lmax, u);

    // Run!
    T bi;
    T r_ = T(r);
    T expo_ = T(expo);
    T db = expo_ / ndiv;
    for (int i = 0; i < npts; ++i) {
        bi = T(abs(b(i)));
        f(i) = 0.0;
        T b0 = bi - 0.5 * expo_;
        for (int j = 0; j < ndiv; ++j) {
            f(i) += flux(abs(b0 + db * j), r_, I);
        }
        f(i) /= ndiv;
    }

    return std::make_tuple(f.template cast<double>(), I.n_eval);

}

std::tuple<Vector<double>, long int> computeBatmanFluence(const Vector<double>& b, const double& r, const Vector<double>& u, const double& expo, const int ndiv) {

    using T = double;
    int npts = b.rows();
    int lmax = u.rows();
    Vector<T> f(npts);
    TransitInfo<T> I(lmax, u);

    // Run!
    T bi;
    T r_ = T(r);
    T expo_ = T(expo);
    T db = expo_ / ndiv;
    for (int i = 0; i < npts; ++i) {
        bi = T(abs(b(i)));
        f(i) = 0.0;
        T b0 = bi - 0.5 * expo_;
        for (int j = 0; j <= ndiv; ++j) {
            f(i) += flux(abs(b0 + db * j), r_, I);
        }
        f(i) /= (ndiv + 1);
    }

    return std::make_tuple(f.template cast<double>(), I.n_eval);

}

std::tuple<Vector<double>, long int> computeRiemannFluence(const Vector<double>& b, const double& r, const Vector<double>& u, const double& expo, const int ndiv) {

    using T = double;
    int npts = b.rows();
    int lmax = u.rows();
    Vector<T> f(npts);
    TransitInfo<T> I(lmax, u);

    // Run!
    T bi;
    T r_ = T(r);
    T expo_ = T(expo);
    T db = expo_ / (ndiv + 1);
    for (int i = 0; i < npts; ++i) {
        bi = T(abs(b(i)));
        f(i) = 0.0;
        T b0 = bi - 0.5 * expo_ + db;
        for (int j = 0; j < ndiv; ++j) {
            f(i) += flux(abs(b0 + db * j), r_, I);
        }
        f(i) /= ndiv;
    }

    return std::make_tuple(f.template cast<double>(), I.n_eval);

}

template <typename T>
std::tuple<Vector<double>, long int> computeTrapezoidFluence(const Vector<double>& b, const double& r, const Vector<double>& u, const double& expo, const int ndiv) {

    int npts = b.rows();
    int lmax = u.rows();
    Vector<T> f(npts);
    TransitInfo<T> I(lmax, u);

    // Run!
    T bi;
    T r_ = T(r);
    T expo_ = T(expo);
    T db = expo_ / ndiv;
    T f_prev, f_next;
    for (int i = 0; i < npts; ++i) {
        bi = T(abs(b(i)));
        f(i) = 0.0;
        T b0 = bi - 0.5 * expo_;
        f_prev = flux(abs(b0), r_, I);
        for (int j = 0; j < ndiv; ++j) {
            f_next = flux(abs(b0 + db * (j + 1)), r_, I);
            f(i) += 0.5 * (f_prev + f_next);
            f_prev = f_next;
        }
        f(i) /= ndiv;
    }

    return std::make_tuple(f.template cast<double>(), I.n_eval);

}

std::tuple<Vector<double>, long int> computeSimpsonFluence(const Vector<double>& b, const double& r, const Vector<double>& u, const double& expo, int ndiv) {

    using T = double;
    int npts = b.rows();
    int lmax = u.rows();
    Vector<T> f(npts);
    TransitInfo<T> I(lmax, u);

    if (ndiv % 2 != 0) ndiv += 1;

    // Run!
    T bi;
    T r_ = T(r);
    T expo_ = T(expo);
    T db = expo_ / ndiv;
    for (int i = 0; i < npts; ++i) {
        bi = T(abs(b(i)));
        f(i) = 0.0;
        T b0 = bi - 0.5 * expo_;
        for (int j = 0; j <= ndiv; ++j) {
            T f0 = flux(abs(b0 + db * j), r_, I);
            if (j == 0 || j == ndiv)
                f(i) += f0;
            else
                f(i) += (2 + 2 * (j % 2)) * f0;
            // f(i) += 0.5 * (flux(abs(b0 + db * j), r_, I) + flux(abs(b0 + db * (j + 1)), r_, I));
        }
        f(i) /= 3 * ndiv;
    }

    return std::make_tuple(f.template cast<double>(), I.n_eval);

}



template <typename T>
inline int get_integration_limits(const T& expo, const T& b, const T& r, std::vector<T>& limits) {
    // All possible limits of integration
    T e = 0.5 * expo;
    T P = 1 - r,
      Q = 1 + r,
      A = P - e,
      B = P + e,
      C = Q - e,
      D = Q + e,
      E = 2 * b - P,
      F = 2 * b - Q;
    std::vector<T> all_limits {A, B, C, D};  //, E, F, P, Q};

    // Identify and sort the relevant ones
    limits.resize(6);
    limits[0] = b - e;
    int n = 1;
    for (auto lim : all_limits) {
        if ((lim > b - e) && (lim < b + e)) {
            limits[n++] = lim;
        }
    }
    limits[n++] = b + e;
    std::sort(limits.begin() + 1, limits.begin() + n - 1);

    return n;
}

template <typename Stencil>
std::tuple<Vector<double>, long int> compute_fluence (const Vector<double>& b, const double& r, const Vector<double>& u, const double& expo, int ndiv) {

  using T = double;

  std::vector<T> limits(6);

  int npts = b.rows();
  int lmax = u.rows();
  Vector<T> f(npts);
  TransitInfo<T> I(lmax, u);

  // Run!
  int n_limits, ndiv_inner, ndiv_count;
  T bi, db, f_prev, b_val, f_val;
  T r_ = T(r);
  T expo_ = T(expo);

  for (int i = 0; i < npts; ++i) {

      bi = T(b(i));
      n_limits = get_integration_limits(expo_, bi, r_, limits);

      f_val = T(0.0);
      b_val = limits[0];
      f_prev = flux<T>(abs(b_val), r_, I);

      ndiv_count = 0;
      for (int j = 0; j < n_limits-1; ++j) {
        db = limits[j+1] - limits[j];
        if (db > 0.0) {
          if (j == n_limits - 2) {
            ndiv_inner = ndiv - ndiv_count;
          } else {
            ndiv_inner = (int)round(ndiv * expo_ / db);
          }
          ndiv_inner = std::max(Stencil::min_div, ndiv_inner);
          ndiv_count += ndiv_inner;

          f_prev = Stencil::template integrate<T>(limits[j], limits[j+1], f_prev, ndiv_inner, r_, I, f_val);
        }
      }

      f(i) = f_val / expo_;
  }

  return std::make_tuple(f, I.n_eval);
}

template <typename Stencil>
std::tuple<Vector<double>, long int> compute_fluence_tol (const Vector<double>& b, const double& r, const Vector<double>& u, const double& expo, const double& tol) {

  using T = double;

  std::vector<T> limits(8);

  int npts = b.rows();
  int lmax = u.rows();
  Vector<T> f(npts);
  TransitInfo<T> I(lmax, u);

  // Run!
  T bi, x1, x2, y1, y2;
  T r_ = T(r);
  T expo_ = T(expo);

  for (int i = 0; i < npts; ++i) {
      bi = T(b(i));

      x1 = bi - 0.5*expo_;
      x2 = bi + 0.5*expo_;
      y1 = flux<T>(abs(x1), r_, I);
      y2 = flux<T>(abs(x2), r_, I);

      f(i) = Stencil::template integrate<T>(x1, x2, y1, y2, tol, r_, I) / expo_;
  }

  return std::make_tuple(f, I.n_eval);
}


struct TrapezoidStencil {
  const static int min_div = 2;

  template <typename T>
  static T integrate(const T& start, const T& end, const T& start_val, int ndiv, const T& r, TransitInfo<T>& info, T& f) {
    T value = T(start_val);
    T delta = (end - start) / ndiv;
    T x = start + delta;

    f += 0.5 * value * delta;
    for (int i = 1; i < ndiv; ++i) {
      value = flux<T>(abs(x), r, info);
      f += value * delta;
      x += delta;
    }

    value = flux<T>(abs(x), r, info);
    f += 0.5 * value * delta;

    return value;
  }

  template <typename T>
  static T integrate(const T& x1, const T& x2, const T& y1, const T& y2, const T& tol, const T& r, TransitInfo<T>& info) {
    T pred = 0.5 * (y1 + y2);
    T x0 = 0.5 * (x1 + x2);
    T y0 = flux<T>(abs(x0), r, info);
    if (abs(pred - y0) < tol) {
      return 0.5 * (0.5 * (y1 + y2) + y0) * (x2 - x1);
    }
    return integrate(x1, x0, y1, y0, tol, r, info) + integrate(x0, x2, y0, y2, tol, r, info);
  }

};

struct SimpsonStencil {
  const static int min_div = 3;

  template <typename T>
  static T integrate(const T& start, const T& end, const T& start_val, int ndiv, const T& r, TransitInfo<T>& info, T& f) {
    // ndiv must be odd
    ndiv += ndiv % 2;

    T value = T(start_val);
    T delta = (end - start) / ndiv;
    T factor = delta / 3;
    T x = start + delta;

    f += value * factor;
    for (int i = 1; i < ndiv; ++i) {
      value = (2 + 2 * (i % 2)) * flux<T>(abs(x), r, info);
      f += value * factor;
      x += delta;
    }

    value = flux<T>(abs(x), r, info);
    f += value * factor;

    return value;
  }

  template <typename T>
  static T integrate_inner(const T& x0, const T& dx, const T& ym, const T& y0, const T& yp, const T& tol, const T& r, TransitInfo<T>& info, int depth) {
    // Left
    T x_m = x0 - 0.5*dx;
    T val_m = flux<T>(abs(x_m), r, info);

    // Integral without the middle point
    T pred_m = dx * (2*y0/3 + 5*ym/12 - yp/12);

    // Integral including the middle point
    T int_m = 0.5 * dx * (4*val_m + ym + y0) / 3;

    if (abs(pred_m - int_m) > tol) {
      int_m = integrate_inner(x_m, 0.5*dx, ym, val_m, y0, tol, r, info, depth+1);
    }

    // Right
    T x_p = x0 + 0.5*dx;
    T val_p = flux<T>(abs(x_p), r, info);

    // Integral without the middle point
    T pred_p = dx * (2*y0/3 - ym/12 + 5*yp/12);

    // Integral including the middle point
    T int_p = 0.5 * dx * (4*val_p + yp + y0) / 3;

    if (abs(pred_p - int_p) > tol) {
      int_p = integrate_inner(x_p, 0.5*dx, y0, val_p, yp, tol, r, info, depth+1);
    }

    return int_m + int_p;
  }

  template <typename T>
  static T integrate(const T& x1, const T& x2, const T& y1, const T& y2, const T& tol, const T& r, TransitInfo<T>& info) {
    T dx = 0.5 * (x2 - x1);
    T x0 = 0.5 * (x1 + x2);
    T y0 = flux<T>(abs(x0), r, info);
    return integrate_inner(x0, dx, y1, y0, y2, tol, r, info, 0);
  }
};


PYBIND11_MODULE(fluence, m) {

    m.def("flux", &computeFlux);

    m.def("taylor_fluence", &computeTaylorFluence);

    m.def("exact_fluence", &computeExactFluence<Multi, 15>);

    m.def("gauss_fluence", &computeExactFluence<double, 3>);

    m.def("left_riemann_fluence", &computeLeftRiemannFluence);

    m.def("batman_fluence", &computeBatmanFluence);

    m.def("riemann_fluence", &computeRiemannFluence);

    m.def("trapezoid_fluence", &computeTrapezoidFluence<double>);

    m.def("trapezoid_fluence_precise", &computeTrapezoidFluence<Multi>);

    m.def("simpson_fluence", &computeSimpsonFluence);

    m.def("trapezoid_fluence2", &compute_fluence<TrapezoidStencil>);
    m.def("simpson_fluence2", &compute_fluence<SimpsonStencil>);

    m.def("trapezoid_fluence3", &compute_fluence_tol<TrapezoidStencil>);
    m.def("simpson_fluence3", &compute_fluence_tol<SimpsonStencil>);

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif

}
