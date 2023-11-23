#include "MatrixElementPiecewiseCoefficient.hpp"
#include <cassert>


void MatrixElementPiecewiseCoefficient::Eval(
   mfem::DenseMatrix &K,
   mfem::ElementTransformation& T,
   const mfem::IntegrationPoint &ip)
{
   std::unordered_map<int,mfem::Vector>::iterator iter = heartConductivities_.find(T.Attribute); // Myocyte type
   if (iter != heartConductivities_.end()) {
      mfem::DenseMatrix Q(3);
      mfem::Vector direction(3);
      p_gf_->GetVectorValue(T.ElementNo, ip, direction); // direction is the direction of fiber
      Q.SetCol(0, direction);
      p_gf_->GetVectorValue(T.ElementNo, ip, direction); // direction is the direction of sheet
      Q.SetCol(1, direction);
      p_gf_->GetVectorValue(T.ElementNo, ip, direction); // direction is the direction of trans
      Q.SetCol(2, direction);
      MultADAt(Q,iter->second,K);//K = Q*diag(iter->second)Q^T
   }
}

