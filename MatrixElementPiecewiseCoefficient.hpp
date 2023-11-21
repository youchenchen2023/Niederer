#pragma once

#include "mfem.hpp"
#include <unordered_map>
#include <memory>

const mfem::DenseMatrix quat2rot(const mfem::Vector& q);

class MatrixElementPiecewiseCoefficient : public mfem::MatrixCoefficient
{
public:
  MatrixElementPiecewiseCoefficient() : mfem::MatrixCoefficient(3) {}
  //“instantiate mfem::MatrixCoefficient with argument 3”
  MatrixElementPiecewiseCoefficient(std::shared_ptr<mfem::ParGridFunction> x,std::shared_ptr<mfem::ParGridFunction> y , std::shared_ptr<mfem::ParGridFunction> z)
  : mfem::MatrixCoefficient(3), p_gf_(x), p_gs_(y), p_gt_(z) {}
  //override Eval methods
  /** @brief Evaluate the matrix coefficient in the element described by @a T
      at the point @a ip, storing the result in @a K. */
  void Eval(mfem::DenseMatrix &K, mfem::ElementTransformation& T, const mfem::IntegrationPoint &ip);
  std::shared_ptr<mfem::ParGridFunction> p_gf_;
  std::shared_ptr<mfem::ParGridFunction> p_gs_;
  std::shared_ptr<mfem::ParGridFunction> p_gt_;
  std::unordered_map<int,mfem::Vector> heartConductivities_;
};
