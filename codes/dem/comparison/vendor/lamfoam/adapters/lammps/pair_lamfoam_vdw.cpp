// SPDX-License-Identifier: GPL-3.0-or-later
#include "lamfoam/core/MechanicalUnits.hpp"
#include "pair_lamfoam_vdw.h"
#include "lamfoam/core/VanDerWaals.hpp"
#include "atom.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "utils.h"
#include "update.h"
#include <algorithm>
#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;
PairLamfoamVdw::PairLamfoamVdw(LAMMPS* lmp):Pair(lmp)
{
    single_enable=1;
    restartinfo=0; // Reissue pair_style and pair_coeff after read_restart.
}
PairLamfoamVdw::~PairLamfoamVdw()
{
    if (copymode || !allocated) return;
    memory->destroy(setflag);memory->destroy(cutsq);
    memory->destroy(hamaker_);memory->destroy(separation_);
}
void PairLamfoamVdw::allocate()
{
    allocated=1;const int n=atom->ntypes+1;
    memory->create(setflag,n,n,"lamfoam/vdw:setflag");
    memory->create(cutsq,n,n,"lamfoam/vdw:cutsq");
    memory->create(hamaker_,n,n,"lamfoam/vdw:hamaker");
    memory->create(separation_,n,n,"lamfoam/vdw:separation");
    for(int i=0;i<n;++i)for(int j=0;j<n;++j)setflag[i][j]=0;
}
void PairLamfoamVdw::settings(int narg,char** arg)
{
    if(narg!=1)error->all(FLERR,"lamfoam/vdw requires maximum particle radius");
    maximumRadius_=utils::numeric(FLERR,arg[0],false,lmp);
    if(!std::isfinite(maximumRadius_) || maximumRadius_<=0)
        error->all(FLERR,"lamfoam/vdw maximum radius must be finite and positive");
}
void PairLamfoamVdw::coeff(int narg,char** arg)
{
    if(narg!=4)error->all(FLERR,"Expected pair_coeff I J lamfoam/vdw A h0");
    if(!allocated)allocate();
    int ilo,ihi,jlo,jhi;
    utils::bounds(FLERR,arg[0],1,atom->ntypes,ilo,ihi,error);
    utils::bounds(FLERR,arg[1],1,atom->ntypes,jlo,jhi,error);
    const double a=utils::numeric(FLERR,arg[2],false,lmp);
    const double h=utils::numeric(FLERR,arg[3],false,lmp);
    if(!std::isfinite(a)||a<0||!std::isfinite(h)||h<=0)
        error->all(FLERR,"lamfoam/vdw requires finite A>=0 and h0>0");
    int count=0;
    for(int i=ilo;i<=ihi;++i)for(int j=std::max(i,jlo);j<=jhi;++j){
        hamaker_[i][j]=a;separation_[i][j]=h;setflag[i][j]=1;++count;
    }
    if(!count)error->all(FLERR,"Empty lamfoam/vdw coefficient range");
}
void PairLamfoamVdw::init_style()
{
    if(!atom->radius_flag)error->all(FLERR,"lamfoam/vdw requires particle radii");
    if(!lamfoam::core::supportedMechanicalUnits(update->unit_style))
        error->all(FLERR,"lamfoam/vdw currently requires units si, cgs or micro");
    neighbor->add_request(this);
}
double PairLamfoamVdw::init_one(int i,int j)
{
    if(!setflag[i][j])error->all(FLERR,"Set all lamfoam/vdw type pairs explicitly; no mixing rule");
    hamaker_[j][i]=hamaker_[i][j];separation_[j][i]=separation_[i][j];
    return std::nextafter(2.1*maximumRadius_, HUGE_VAL);
}
void PairLamfoamVdw::compute(int eflag,int vflag)
{
    ev_init(eflag,vflag);
    const int nlocal=atom->nlocal,newton=force->newton_pair;
    for(int i=0;i<nlocal+atom->nghost;++i)
        if(!std::isfinite(atom->radius[i])||atom->radius[i]<=0||atom->radius[i]>maximumRadius_)
            error->one(FLERR,"lamfoam/vdw radius exceeds its declared neighbor bound or is invalid");
    for(int ii=0;ii<list->inum;++ii){
        const int i=list->ilist[ii],itype=atom->type[i];
        for(int jj=0;jj<list->numneigh[i];++jj){
            int j=list->firstneigh[i][jj];
            const double scale=force->special_lj[sbmask(j)];j&=NEIGHMASK;
            if(scale==0)continue;
            const int jtype=atom->type[j];
            const double dx=atom->x[i][0]-atom->x[j][0],dy=atom->x[i][1]-atom->x[j][1],dz=atom->x[i][2]-atom->x[j][2];
            const double rsq=dx*dx+dy*dy+dz*dz;
            if(rsq>=cutsq[itype][jtype])continue;
            if(rsq==0)error->one(FLERR,"lamfoam/vdw cannot define a normal for coincident centers");
            const double distance=std::sqrt(rsq),ri=atom->radius[i],rj=atom->radius[j];
            const auto value=lamfoam::core::vanDerWaals(distance-ri-rj,ri*rj/(ri+rj),hamaker_[itype][jtype],separation_[itype][jtype]);
            const double fpair=scale*value.normalForce/distance;
            const double delta[3]={dx,dy,dz};
            for(int k=0;k<3;++k){atom->f[i][k]+=delta[k]*fpair;
                if(newton||j<nlocal)atom->f[j][k]-=delta[k]*fpair;}
            if(evflag)ev_tally(i,j,nlocal,newton,scale*value.potential,0,fpair,dx,dy,dz);
        }
    }
    if(vflag_fdotr)virial_fdotr_compute();
}

double PairLamfoamVdw::single(int i,int j,int itype,int jtype,double rsq,
                            double /*factorCoul*/,double factorLJ,double& forceOverR)
{
    if(rsq<=0)error->one(FLERR,"lamfoam/vdw cannot define a normal for coincident centers");
    const double distance=std::sqrt(rsq),ri=atom->radius[i],rj=atom->radius[j];
    const auto value=lamfoam::core::vanDerWaals(distance-ri-rj,ri*rj/(ri+rj),
                                              hamaker_[itype][jtype],separation_[itype][jtype]);
    forceOverR=factorLJ*value.normalForce/distance;
    return factorLJ*value.potential;
}
