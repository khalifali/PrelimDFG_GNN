// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "pair.h"
namespace LAMMPS_NS {
class PairLamfoamVdw : public Pair {
public:
    explicit PairLamfoamVdw(LAMMPS*);
    ~PairLamfoamVdw() override;
    void compute(int,int) override;
    void settings(int,char**) override;
    void coeff(int,char**) override;
    void init_style() override;
    double init_one(int,int) override;
    double single(int,int,int,int,double,double,double,double&) override;
private:
    double maximumRadius_=0;
    double **hamaker_=nullptr, **separation_=nullptr;
    void allocate();
};
}
