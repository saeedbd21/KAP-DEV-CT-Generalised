#ifndef EMM_K_H_
#define EMM_K_H_

#include "EMM_index_macros.H"
#include <AMReX_FArrayBox.H>
#include <limits>
#include <cmath>

#include "emm_prob.H"
#include "EMM_parm.H"

AMREX_GPU_HOST_DEVICE
inline
amrex::Real
EMM_estdt (amrex::Box const& bx, amrex::Array4<Real const> const& state,
           amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dx,
           Parm const& parm) noexcept
{
    using namespace amrex::literals;
    
    const auto lo = amrex::lbound(bx);
    const auto hi = amrex::ubound(bx);
#if !defined(__CUDACC__) || (__CUDACC_VER_MAJOR__ != 9) || (__CUDACC_VER_MINOR__ != 2)
    amrex::Real dt = std::numeric_limits<amrex::Real>::max();
#else
    amrex::Real dt = 1.e37_rt;
#endif

    for         (int k = lo.z; k <= hi.z; ++k) {
        for     (int j = lo.y; j <= hi.y; ++j) {
            for (int i = lo.x; i <= hi.x; ++i) {
                amrex::Real alpha1 = state(i,j,k,GALPHA);
                amrex::Real alpha2 = 1.0_rt - alpha1;
                amrex::Real arho1 = state(i,j,k,UARHO1);
                amrex::Real arho2 = state(i,j,k,UARHO2);
                amrex::Real mrho = arho1 + arho2;
                amrex::Real rho1 = state(i,j,k,UARHO1)/alpha1;
                amrex::Real rho2 = state(i,j,k,UARHO2)/alpha2;
                amrex::Real mx  = state(i,j,k,UMX);
                amrex::Real my  = state(i,j,k,UMY);
                amrex::Real mz  = state(i,j,k,UMZ);
                amrex::Real mrhoinv = 1.0_rt/mrho;
                amrex::Real rhoinv1 = 1.0_rt/rho1;
                amrex::Real rhoinv2 = 1.0_rt/rho2;
                amrex::Real vx = mx*mrhoinv;
                amrex::Real vy = my*mrhoinv;
                amrex::Real vz = mz*mrhoinv;
                amrex::Real ei = state(i,j,k,URHOE)/mrho - 0.5_rt*(vx*vx + vy*vy + vz*vz);
                amrex::Real p = 0.0;

                if(parm.tabulated1 == 0 && parm.tabulated2 == 0){
                    p = ( mrho*ei - ( alpha1*parm.eos_gamma1*parm.eos_pinf1/(parm.eos_gamma1-1.0_rt) 
                    + alpha2*(1.0-rho2*parm.eos_b0_1)*parm.eos_gamma2*parm.eos_pinf2/(parm.eos_gamma2-1.0_rt)) - alpha2*rho2*parm.q0)/( alpha1/(parm.eos_gamma1-1.0_rt)+alpha2*(1.0-rho2*parm.eos_b0_1)/(parm.eos_gamma2-1.0_rt));
                    // amrex::Print().SetPrecision(18) << "pSG:                  " << pSG << "\n";
                }else{
                    p = p_finder(alpha1, alpha2, state(i,j,k,UARHO1)/alpha1, state(i,j,k,UARHO2)/alpha2, state(i,j,k,GPRESS), state(i,j,k,GT1), state(i,j,k,GT2), mrho*ei, parm);
                //     // amrex::Print().SetPrecision(18) << "p:                    " << p << "\n";
                }
                //amrex::Real c1sq = amrex::max(parm.eos_gamma1*(p+parm.eos_pinf1)*rhoinv1,parm.smallr);
                //amrex::Real c2sq = amrex::max(parm.eos_gamma2*(p+parm.eos_pinf2)*rhoinv2,parm.smallr);
                //amrex::Real rhocsq = 1.0_rt/(alpha1/(rho1*c1sq)+alpha2/(rho2*c2sq));
                //amrex::Real mcs = std::sqrt(rhocsq/mrho);
                amrex::Real Y1 = amrex::max(alpha1*rho1/mrho,parm.smallr);
                amrex::Real Y2 = amrex::max(alpha2*rho2/mrho,parm.smallr);

                amrex::Real c1sq = 0.0;
                amrex::Real T1 = 0.0;
                if(parm.tabulated1 == 0){
                    c1sq = amrex::max(parm.eos_gamma1*(p+parm.eos_pinf1)/rho1,parm.smallr);
                }else{
                    T1 = T_finder(alpha1, rho1, p, state(i,j,k,GT1),parm, 1);
                    c1sq = amrex::max(std::pow(TPF(OSOS, T1, std::log10(p), parm, 1),2.0),parm.smallr);
                }
                amrex::Real c2sq = 0.0;
                amrex::Real T2 = 0.0;
                if(parm.tabulated2 == 0){
                    //c2sq = amrex::max(parm.eos_gamma2*(p+parm.eos_pinf2)/rho2,parm.smallr);
                    c2sq = amrex::max(parm.eos_gamma2*(p+parm.eos_pinf2)/rho2/rho2/(1.0/rho2-parm.eos_b0_1),parm.smallr);
                }else{
                    T2 = T_finder(alpha1, rho2, p, state(i,j,k,GT2),parm, 2);
                    c2sq = amrex::max(std::pow(TPF(OSOS, T2, std::log10(p), parm, 2),2.0),parm.smallr);
                }
                // amrex::Real c1sq = amrex::max(parm.eos_gamma1*(p+parm.eos_pinf1)/rho1,parm.smallr);
                // c1sq = amrex::max(parm.eos_gamma1*(p+parm.eos_pinf1)/rho1,parm.smallr);
                // amrex::Real c2sq = amrex::max(parm.eos_gamma2*(p+parm.eos_pinf2)/rho2,parm.smallr);
                amrex::Real mcs = std::sqrt(Y1*c1sq + Y2*c2sq);
                amrex::Real dtx = dx[0]/(std::abs(vx)+mcs);
                amrex::Real dty = dx[1]/(std::abs(vy)+mcs);
#if (AMREX_SPACEDIM == 2)
                dt = amrex::min(dt,amrex::min(dtx,dty));
                //Print() <<  "dt " << dt << "\n";
#endif
#if (AMREX_SPACEDIM == 3)
                amrex::Real dtz = dx[2]/(std::abs(vz)+mcs);
                dt = amrex::min(dt,amrex::min(dtx,amrex::min(dty,dtz)));
#endif
            }
        }
    }
    return dt;
}

// AMREX_GPU_DEVICE
// inline
// void
// EMM_compute_temperature (int i, int j, int k, amrex::Array4<amrex::Real> const& u,
//                          Parm const& parm) noexcept
// {
//     using namespace amrex::literals;

//     amrex::Real rhoinv = 1.0_rt/u(i,j,k,URHO);
//     amrex::Real mx = u(i,j,k,UMX);
//     amrex::Real my = u(i,j,k,UMY);
//     amrex::Real mz = u(i,j,k,UMZ);
//     u(i,j,k,UEINT) = u(i,j,k,UEDEN) - 0.5_rt * rhoinv * (mx*mx+my*my+mz*mz);
//     u(i,j,k,UTEMP) = rhoinv * u(i,j,k,UEINT) * (1.0_rt/parm.cv);
// }

#endif
