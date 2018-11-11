#section support_code_apply

int APPLY_SPECIFIC(limbdark_rev)(
    PyArrayObject* input0,   // c
    PyArrayObject* input1,   // b
    PyArrayObject* input2,   // r
    PyArrayObject* input3,   // bf
    PyArrayObject** output0, // bc
    PyArrayObject** output1, // bb
    PyArrayObject** output2) // br
{
  npy_intp Nc, Nb, Nr, Nf;
  int success = get_size(input0, &Nc);
  success += get_size(input1, &Nb);
  success += get_size(input2, &Nr);
  success += get_size(input3, &Nf);
  if (success) return 1;
  if (Nb != Nr || Nb != Nf) {
    PyErr_Format(PyExc_ValueError, "dimension mismatch");
    return 1;
  }

  success += allocate_output(PyArray_NDIM(input0), PyArray_DIMS(input0), TYPENUM_OUTPUT_0, output0);
  success += allocate_output(PyArray_NDIM(input1), PyArray_DIMS(input1), TYPENUM_OUTPUT_1, output1);
  success += allocate_output(PyArray_NDIM(input2), PyArray_DIMS(input2), TYPENUM_OUTPUT_2, output2);
  if (success) {
    Py_XDECREF(*output0);
    Py_XDECREF(*output1);
    Py_XDECREF(*output2);
    return 1;
  }

  DTYPE_INPUT_0* c   = (DTYPE_INPUT_0*) PyArray_DATA(input0);
  DTYPE_INPUT_1* b   = (DTYPE_INPUT_1*) PyArray_DATA(input1);
  DTYPE_INPUT_2* r   = (DTYPE_INPUT_2*) PyArray_DATA(input2);
  DTYPE_INPUT_3* bf  = (DTYPE_INPUT_3*) PyArray_DATA(input3);
  DTYPE_OUTPUT_0* bc = (DTYPE_OUTPUT_0*)PyArray_DATA(*output0);
  DTYPE_OUTPUT_1* bb = (DTYPE_OUTPUT_1*)PyArray_DATA(*output1);
  DTYPE_OUTPUT_2* br = (DTYPE_OUTPUT_2*)PyArray_DATA(*output2);

  Eigen::Map<Eigen::Matrix<DTYPE_OUTPUT_0, Eigen::Dynamic, 1>> bc_vec(bc, Nc);
  bc_vec.setZero();

  Eigen::Map<Eigen::Matrix<DTYPE_INPUT_0, Eigen::Dynamic, 1>> cvec(c, Nc);
  starry::limbdark::GreensLimbDark<DTYPE_OUTPUT_0> L(Nc-1);

  for (npy_intp i = 0; i < Nb; ++i) {
    auto b_ = std::abs(b[i]);
    auto r_ = std::abs(r[i]);
    if (b_ > 1 + r_) {
      bb[i] = 0;
      br[i] = 0;
    } else {
      L.compute(b_, r_, true);

      //auto resn = L.S.dot(agol_c);
      //for (npy_intp j = 0; j < Nu+1; ++j) {
      //  dfdc(j) = L.S(j) * agol_c(j);
      //}
      ////dfdc = agol_norm * L.S.transpose().array() * agol_c.array();
      //dfdc(0) -= resn * M_PI;
      //dfdc(1) -= resn * 2*M_PI/3;
      //bu_vec += bf[i] * dcdu * dfdc;

      bc_vec += bf[i] * L.S.transpose();

      bb[i] = sgn(b[i]) * bf[i] * L.dSdb.dot(cvec);
      br[i] = sgn(r[i]) * bf[i] * L.dSdr.dot(cvec);
    }
  }

  return 0;
}
