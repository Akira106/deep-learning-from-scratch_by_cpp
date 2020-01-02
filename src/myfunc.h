#ifndef MYFUNC_H_
#define MYFUNC_H_

#include <Eigen/Core>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <iostream>

#define FLERR __FILE__,__LINE__
using namespace std;
using namespace Eigen;

namespace myfunc{

    class Myclock{
      public:
        void start(void);
        void stop(string msg);
      private:
        time_t s,e;

    };

    vector<string> split(const string& input, char delimiter);
    VectorXi bool2index(Vector<bool,Dynamic> bvec);
    vector<vector<int> > random_choice(int train_size,int batch_size);

    template<class T> T max(const T &obj1, const T &obj2){
        if(obj1>=obj2){
            return obj1;
        }else{
            return obj2;
        }
    }

    template<class T> T min(const T &obj1, const T &obj2){
        if(obj1<=obj2){
            return obj1;
        }else{
            return obj2;
        }
    }

    template<class T> T str2num(const string &src){
        T dst;
        stringstream(src) >> dst;
        return dst;
    }

    template<class T> string num2str(const T &src){
        stringstream ss;
        ss<<src;
        string dst = ss.str();
        return dst;
    }

    template<class T> void show_mat(const vector<vector<T> > mat){
        for(int i=0;i<mat.size();i++){
            for(int j=0;j<mat[i].size();j++){
                cout <<mat[i][j]<<" ";
            }
            cout<<endl;
        }
    }

    template<class T>
    Matrix<T,Dynamic,Dynamic>SetValueFromBool(Matrix<T,Dynamic,Dynamic> A, const Matrix<bool,Dynamic,Dynamic>&Bl,T value){
        for(int j=0;j<Bl.cols();j++){
            for(int i=0;i<Bl.rows();i++){
                if(Bl(i,j)==1){
                    A(i,j)=value;
                }
            }
        }
        return A;
    }

    template<class T>
    Vector<T,Dynamic>SetValueFromBool(Vector<T,Dynamic> A, const Vector<bool,Dynamic>&Bl,T value){
        for(int i=0;i<Bl.size();i++){
            if(Bl(i)==1){
                A(i)=value;
            }
        }

        return A;
    }

    template<class T>
    void ChangeValueFromBool(Matrix<T,Dynamic,Dynamic> &A, const Matrix<bool,Dynamic,Dynamic>&Bl,T value){
        for(int j=0;j<Bl.cols();j++){
            for(int i=0;i<Bl.rows();i++){
                if(Bl(i,j)==1){
                    A(i,j)=value;
                }
            }
        }
    }

    template<class T>
    void ChangeValueFromBool(Vector<T,Dynamic> &A, const Vector<bool,Dynamic>&Bl,T value){
        for(int i=0;i<Bl.size();i++){
            if(Bl(i)==1){
                A(i)=value;
            }
        }
    }
    
    template<class T>
    int argmax_vec(const Vector<T,Dynamic> &X){
        int size=X.size();
        T vmax=X.coeff(0);
        int index=0;
        for(int i=1;i<size;i++){
            if(X.coeff(i)>vmax){
                vmax=X.coeff(i);
                index=i;
            }
        }
        return index;
    }

    template<class T>
    VectorXi argmax(const Matrix<T,Dynamic,Dynamic> &X){
        int nrows=X.rows();
        VectorXi index(nrows);
        for(int i=0;i<nrows;i++){
            index.coeffRef(i) = argmax_vec<T>(X.row(i));
        }
        return index;
    }


    void error(string filename, int line, string message);
    void just_call_msg(string filename, int line, string message);
}
#endif
