#ifndef LAYER_AFFINE_H_
#define LAYER_AFFINE_H_
#include <Eigen/Core>
#include "layer.h"

using namespace Eigen;
using namespace std;

class Layer_Affine:public Layer{
  public:

    Layer_Affine(double learning_rate=0.001);
    Layer_Affine(const map<string,MatrixXd> &mat_map,const map<string,VectorXd> &mat_vec,double learning_rate=0.001);
    MatrixXd forward(const MatrixXd &X) override;
    MatrixXd backward(const MatrixXd &dout) override;
    
  private:
    void check_params(void);
    MatrixXd X;
};

#endif


