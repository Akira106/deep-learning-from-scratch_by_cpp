#ifndef LAST_LAYER_H_
#define LAST_LAYER_H_
#include <Eigen/Core>
#include <map>
#include <string>
using namespace Eigen;
using namespace std;
class Last_Layer{

  public:
    Last_Layer(void);
    virtual double forward(const MatrixXd &X, const MatrixXd &t) = 0;
    virtual MatrixXd backward(const MatrixXd &t) = 0;
    virtual ~Last_Layer()=default;

    MatrixXd y;
    double loss;

};

#endif
