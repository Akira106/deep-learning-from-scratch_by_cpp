#include <time.h>
#include <random>
#include <algorithm>
#include "myfunc.h"

using namespace std;
using namespace Eigen;

namespace myfunc{
    void Myclock::start(void){
        this->s=clock();
    }
    void Myclock::stop(string msg){
        this->e=clock();
        cout<<msg<<" "<<(this->e - this->s)/CLOCKS_PER_SEC<<"[sec]"<<endl;
    }

    vector<string> split(const string& input, char delimiter)
    {
        stringstream stream(input);
        string field;
        vector<string> result;
        while (getline(stream, field, delimiter)) {
            result.push_back(field);
        }
        return result;
    }

    VectorXi bool2index(Vector<bool,Dynamic> bvec){

        vector<int> index;
        for(int i=0;i<bvec.size();i++){
            if(bvec[i]==1){
                index.push_back(i);
            }
        }

        return Map<VectorXi>(&index[0],index.size());
    }
    void error(string filename, int line, string message){
        cout <<"ERROR:"<<message<<"("<<filename<<":"<<line<<")"<<endl;
        exit(1);
    }
    void just_call_msg(string filename, int line, string message){
        cout <<message<<"("<<filename<<":"<<line<<")"<<endl;
    }

    vector<vector<int> > random_choice(int train_size,int batch_size){

        vector<int> index(train_size);
        for(int i=0;i<train_size;i++){
            index[i]=i;
        }
        random_device seed_gen;
        default_random_engine engine(seed_gen());
        shuffle(index.begin(),index.end(),engine);

        int ret_col=myfunc::min(train_size,batch_size);
        int ret_row=train_size/ret_col;
        vector<vector<int> >ret_index(ret_row,vector<int>(ret_col,0));
        for(int i=0;i<ret_row;i++){
            for(int j=0;j<ret_col;j++){
                ret_index[i][j]=index[i*ret_col+j];
            }
        }
        return ret_index;
    }
}

