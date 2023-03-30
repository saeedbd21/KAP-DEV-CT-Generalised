#ifndef EMM_HYDRO_K_H_
#define EMM_HYDRO_K_H_

#include "EMM_index_macros.H"
#include "EMM_parm.H"
#include <AMReX_FArrayBox.H>
#include <cmath>
#include "EMM_eos.H"

AMREX_GPU_DEVICE
inline
void
EMM_ctoprim (int i, int j, int k,
             amrex::Array4<amrex::Real> const& u,
             amrex::Array4<amrex::Real> const& q,
             amrex::GeometryData const& geomdata,
             Parm const& parm, amrex::Real time) noexcept
{
    BL_PROFILE("EMM_ctoprim()");
    using namespace amrex::literals;

    amrex::Real mrho = u(i,j,k,UARHO1) + u(i,j,k,UARHO2);
    amrex::Real ux = u(i,j,k,UMX)/mrho;
    amrex::Real uy = u(i,j,k,UMY)/mrho;
    amrex::Real uz = u(i,j,k,UMZ)/mrho;
    amrex::Real ei = u(i,j,k,URHOE)/mrho - 0.5_rt*(ux*ux + uy*uy + uz*uz);
    amrex::Real alpha1 = u(i,j,k,GALPHA);
    amrex::Real alpha2 = 1.0_rt - alpha1;


    if(alpha1 > 1.0-parm.alpha_min/100.0){alpha1 = 1.0-parm.alpha_min/100.0;}
    if(alpha1 < parm.alpha_min/100.0){alpha1 = parm.alpha_min/100.0;}
    alpha2 = 1.0_rt - alpha1;
    u(i,j,k,GALPHA) = alpha1;

    amrex::Real rho1 = u(i,j,k,UARHO1)/alpha1;
    amrex::Real rho2 = u(i,j,k,UARHO2)/alpha2;

    // Print() << "u(i,j,k,GPRESS):          " << u(i,j,k,GPRESS) << "\n";
    // Print() << "u(i,j,k,GT1):             " << u(i,j,k,GT1) << "\n";
    // Print() << "u(i,j,k,GT2):             " << u(i,j,k,GT2) << "\n";
    // Print() << "i:                        " << i << "\n";
    // Print() << "j:                        " << j << "\n";
    // Print() << "k:                        " << k << "\n";

    amrex::Real p = 0.0;
    if(parm.tabulated1 == 0 && parm.tabulated2 == 0){
        p = ( mrho*ei - ( alpha1*parm.eos_gamma1*parm.eos_pinf1/(parm.eos_gamma1-1.0_rt) 
             + alpha2*(1.0-rho2*parm.eos_b0_1)*parm.eos_gamma2*parm.eos_pinf2/(parm.eos_gamma2-1.0_rt)) - alpha2*rho2*parm.q0 )/( alpha1/(parm.eos_gamma1-1.0_rt)+alpha2*(1.0-rho2*parm.eos_b0_1)/(parm.eos_gamma2-1.0_rt));
    }else{
        p = p_finder(alpha1, alpha2, u(i,j,k,UARHO1)/alpha1, u(i,j,k,UARHO2)/alpha2, u(i,j,k,GPRESS), u(i,j,k,GT1), u(i,j,k,GT2), mrho*ei, parm);
        // p = p_finder(u(i,j,k,GALPHA), 1.0-u(i,j,k,GALPHA), u(i,j,k,UARHO1)/alpha1, u(i,j,k,UARHO2)/alpha2, u(i,j,k,GPRESS), u(i,j,k,GT1), u(i,j,k,GT2), mrho*ei, parm);
        // p = p_finder(alpha1, alpha2, u(i,j,k,UARHO1)/alpha1, u(i,j,k,UARHO2)/alpha2, 1E5, u(i,j,k,GT1), u(i,j,k,GT2), mrho*ei, parm);
    }
    // Print() << "p:          " << p << "\n";
    // Print() << "alpha1:          " << alpha1 << "\n";
    // Print() << "rho1:          " << rho1 << "\n";

    // Print() << "parm.eos_pinf1:          " << parm.eos_pinf1 << "\n";
    // Print() << "parm.eos_gamma1:          " << parm.eos_gamma1 << "\n";
    // Print() << "alpha2:          " << alpha2<< "\n";
    // Print() << "rho2:          " << rho2<< "\n";
    // Print() << "parm.eos_pinf2:          " << parm.eos_pinf2 << "\n";

    // Print() << "parm.eos_b0_1:          " << parm.eos_b0_1<< "\n";
    // Print() << "parm.eos_gamma2:          " << parm.eos_gamma2<< "\n";
    // Print() << "parm.q0:          " << parm.q0<< "\n";

    

    // Print() << "pparametric:          " << ( mrho*ei - ( alpha1*parm.eos_gamma1*parm.eos_pinf1/(parm.eos_gamma1-1.0_rt) 
    //          + alpha2*(1.0-rho2*parm.eos_b0_1)*parm.eos_gamma2*parm.eos_pinf2/(parm.eos_gamma2-1.0_rt)) - alpha2*rho2*parm.q0 )/( alpha1/(parm.eos_gamma1-1.0_rt)+alpha2*(1.0-rho2*parm.eos_b0_1)/(parm.eos_gamma2-1.0_rt));

    // // if (p < 2300.0){p =2300.0;}
    // amrex::Real s = 0.0;

    // std::cin >> s;

    amrex::Real T1 = 0.0;
    amrex::Real T2 = 0.0;
    amrex::Real A = 0.0;
    amrex::Real B = 0.0;
    amrex::Real C = 0.0;

    
    if(parm.tabulated1 == 1){
        T1 = T_finder(alpha1, u(i,j,k,UARHO1)/alpha1, p, u(i,j,k,GT1), parm, 1);
        // T1 = p/(u(i,j,k,UARHO1)/alpha1*287.08);
    }else{
        if(rho1 < 0.0) { amrex::Print() << "rho1:          " << rho1 << "\n";}
        T1 = p/(rho1*287.08);
    }
    if(parm.tabulated2 == 1){
        T2 = T_finder(alpha1, u(i,j,k,UARHO2)/alpha2, p, u(i,j,k,GT2), parm, 2);
        // T2 = ((1.0/u(i,j,k,UARHO2)/alpha2)-6.8428e-4)*(p+664961465.0)/(1.1807-1.0)/3630.0;
    }else{
        A = parm.gamma0*parm.pinf0*(1.0/rho2-(parm.b1/rho2+parm.b0_1))/parm.cv0/(parm.gamma0-parm.b1);
        B = 1.0/rho2-(parm.b1/rho2+parm.b0_1);
        C = (p+parm.gamma0*parm.pinf0)*B/(parm.gamma0-1.0);
        T2 = (C/parm.cv0-A)/(1.0-parm.gamma0*parm.pinf0_1*B/(parm.gamma0-1.0)/parm.cv0);
        T2 = (p+parm.eos_pinf2)/rho2/parm.cv0/(parm.eos_gamma2 - 1.0);
    }

    q(i,j,k,QALPHA) = alpha1;
    q(i,j,k,QRHO1) = u(i,j,k,UARHO1)/alpha1;
    q(i,j,k,QRHO2) = u(i,j,k,UARHO2)/alpha2;
    q(i,j,k,QU) = ux;
    q(i,j,k,QV) = uy;
    q(i,j,k,QW) = uz;
    q(i,j,k,QPRES) = p;

    u(i,j,k,GPRESS) = p;
    u(i,j,k,GT1) = T1;
    u(i,j,k,GT2) = T2;
}

AMREX_GPU_DEVICE
inline
void
EMM_thermo (int i, int j, int k,
             amrex::Array4<amrex::Real> const& u_new,
             amrex::Array4<amrex::Real> const& u_old,
             Parm const& parm) noexcept
{
    BL_PROFILE("EMM_thermo()");
    using namespace amrex::literals;

    amrex::Real mrho = u_new(i,j,k,UARHO1) + u_new(i,j,k,UARHO2);
    amrex::Real ux = u_new(i,j,k,UMX)/mrho;
    amrex::Real uy = u_new(i,j,k,UMY)/mrho;
    amrex::Real uz = u_new(i,j,k,UMZ)/mrho;
    amrex::Real ei = u_new(i,j,k,URHOE)/mrho - 0.5_rt*(ux*ux + uy*uy + uz*uz);
    amrex::Real alpha1 = u_new(i,j,k,GALPHA);
    amrex::Real alpha2 = 1.0_rt - alpha1;

    if(alpha1 > 1.0-parm.alpha_min/100.0){alpha1 = 1.0-parm.alpha_min/100.0;}
    if(alpha1 < parm.alpha_min/100.0){alpha1 = parm.alpha_min/100.0;}
    alpha2 = 1.0_rt - alpha1;
    u_new(i,j,k,GALPHA) = alpha1;

    amrex::Real rho1 = u_new(i,j,k,UARHO1)/alpha1;
    amrex::Real rho2 = u_new(i,j,k,UARHO2)/alpha2;

    amrex::Real p = 0.0;
    if(parm.tabulated1 == 0 && parm.tabulated2 == 0){
        p = ( mrho*ei - ( alpha1*parm.eos_gamma1*parm.eos_pinf1/(parm.eos_gamma1-1.0_rt) 
             + alpha2*(1.0-rho2*parm.eos_b0_1)*parm.eos_gamma2*parm.eos_pinf2/(parm.eos_gamma2-1.0_rt)) - alpha2*rho2*parm.q0)/( alpha1/(parm.eos_gamma1-1.0_rt)+alpha2*(1.0-rho2*parm.eos_b0_1)/(parm.eos_gamma2-1.0_rt));
    }else{
        p = p_finder(alpha1, alpha2, u_new(i,j,k,UARHO1)/alpha1, u_new(i,j,k,UARHO2)/alpha2, u_old(i,j,k,GPRESS), u_old(i,j,k,GT1), u_old(i,j,k,GT2), mrho*ei, parm);
    }

    amrex::Real T1 = 0.0;
    amrex::Real T2 = 0.0;
    amrex::Real A = 0.0;
    amrex::Real B = 0.0;
    amrex::Real C = 0.0;
    if(parm.tabulated1 == 1){
        T1 = T_finder(alpha1, u_new(i,j,k,UARHO1)/alpha1, p, u_old(i,j,k,GT1), parm, 1);
    }else{
        if(rho1 < 0.0) { amrex::Print() << "rho1:          " << rho1 << "\n";}
        T1 = p/(rho1*287.08);
    }
    if(parm.tabulated2 == 1){
        T2 = T_finder(alpha1, u_new(i,j,k,UARHO2)/alpha2, p, u_old(i,j,k,GT2), parm, 2);
    }else{
        A = parm.gamma0*parm.pinf0*(1.0/rho2-(parm.b1/rho2+parm.b0_1))/parm.cv0/(parm.gamma0-parm.b1);
        B = 1.0/rho2-(parm.b1/rho2+parm.b0_1);
        C = (p+parm.gamma0*parm.pinf0)*B/(parm.gamma0-1.0);
        T2 = (C/parm.cv0-A)/(1.0-parm.gamma0*parm.pinf0_1*B/(parm.gamma0-1.0)/parm.cv0);
    }
    
    u_new(i,j,k,GPRESS) = p;
    u_new(i,j,k,GT1) = T1;
    u_new(i,j,k,GT2) = T2;
}

AMREX_GPU_DEVICE
inline
void
EMM_flux_to_dudt (int i, int j, int k, int n,
                  amrex::Array4<amrex::Real> const& dudt,
                  AMREX_D_DECL(amrex::Array4<amrex::Real const> const& fx,
                  amrex::Array4<amrex::Real const> const& fy,
                  amrex::Array4<amrex::Real const> const& fz),
                  AMREX_D_DECL(amrex::Array4<amrex::Real const> const& Hx,
                  amrex::Array4<amrex::Real const> const& Hy,
                  amrex::Array4<amrex::Real const> const& Hz),
                  AMREX_D_DECL(amrex::Array4<amrex::Real const> const& Kx,
                  amrex::Array4<amrex::Real const> const& Ky,
                  amrex::Array4<amrex::Real const> const& Kz),
                  AMREX_D_DECL(amrex::Array4<amrex::Real const> const& Mx,
                  amrex::Array4<amrex::Real const> const& My,
                  amrex::Array4<amrex::Real const> const& Mz),
                  AMREX_D_DECL(amrex::Array4<amrex::Real const> const& USx,
                  amrex::Array4<amrex::Real const> const& USy,
                  amrex::Array4<amrex::Real const> const& USz),
                  AMREX_D_DECL(amrex::Array4<amrex::Real const> const& VSx,
                  amrex::Array4<amrex::Real const> const& VSy,
                  amrex::Array4<amrex::Real const> const& VSz),
                  AMREX_D_DECL(amrex::Array4<amrex::Real const> const& WSx,
                  amrex::Array4<amrex::Real const> const& WSy,
                  amrex::Array4<amrex::Real const> const& WSz),
                  amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dxinv) noexcept
{
    //const amrex::Real* prob_lo = geomdata.ProbLo();
    //const amrex::Real* dx      = geomdata.CellSize();

    dudt(i,j,k,n) = dxinv[0]*(fx(i,j,k,n) - fx(i+1,j,k,n)
        +           Hx(i,j,k,n)*( USx(i,j,k)-USx(i+1,j,k)))
        +           dxinv[1]*(fy(i,j,k,n) - fy(i,j+1,k,n)
        +           Ky(i,j,k,n)*( VSy(i,j,k) - VSy(i,j+1,k) ))
#if (AMREX_SPACEDIM == 3)
        +           dxinv[2]*(fz(i,j,k,n) - fz(i,j,k+1,n)
        +           Mz(i,j,k,n)*( WSz(i,j,k) - WSz(i,j,k+1) ))
#endif
    ;
    dudt(i,j,k,GPRESS) = 0.0;
    dudt(i,j,k,GT1) = 0.0;
    dudt(i,j,k,GT2) = 0.0;
}

AMREX_GPU_DEVICE
inline
void
EMM_axisymmetricAdd (int i, int j, int k, amrex::Real dt,
                    amrex::Array4<amrex::Real> const& dudt,
                    amrex::Array4<amrex::Real const> const& q,
                    amrex::Array4<amrex::Real const> const& u,
                    amrex::GeometryData const& geomdata,
                    Parm const& parm,
                    amrex::Real time) noexcept
{
    const amrex::Real* prob_lo = geomdata.ProbLo();
    const amrex::Real* dx      = geomdata.CellSize();
    amrex::Real x = prob_lo[0] + (i+0.5_rt)*dx[0];
    amrex::Real y = prob_lo[1] + (j+0.5_rt)*dx[1];
    amrex::Real r = x;
    amrex::Real cc = parm.cordP;

    amrex::Real mrho = q(i, j, k, QALPHA)*q(i, j, k, QRHO1) + (1.0_rt-q(i, j, k, QALPHA))*q(i, j, k, QRHO2);
    amrex::Real alpha1 = q(i,j,k,QALPHA); amrex::Real alpha2 = 1.0 - q(i,j,k,QALPHA);
    amrex::Real rho1 = q(i,j,k,QRHO1); amrex::Real rho2 = q(i,j,k,QRHO2);
    amrex::Real T1 = 0;
    amrex::Real eint1 = 0;
    amrex::Real T2 = 0;
    amrex::Real eint2 = 0;
    amrex::Real pprime = 0.0;
    //     amrex::Print() << "cccccc:          " << cc << "\n";
    // amrex::Real s = 0.0;

    // std::cin >> s;
    if(parm.tabulated1 == 0){
        eint1 = (q(i, j, k, QPRES)+parm.eos_gamma1*parm.eos_pinf1)/(q(i, j, k, QRHO1)*(parm.eos_gamma1-1.0_rt));
    }else{
        T1 = T_finder(alpha1, q(i,j,k,QRHO1), q(i,j,k,QPRES), u(i,j,k,GT1), parm, 1);
        eint1 = TPF(OE, T1, std::log10(q(i,j,k,QPRES)), parm, 1);
    }
    if(parm.tabulated2 == 0){
        pprime = parm.gamma0*parm.pinf0_1*u(i,j,k,GT2)+parm.gamma0*parm.pinf0*(1.0-parm.b1)/(parm.gamma0-parm.b1);
        eint2 = (q(i,j,k,QPRES)+parm.gamma0*parm.pinf0)/(q(i,j,k,QPRES)+pprime)*parm.cv0*u(i,j,k,GT2)+parm.q0;
        // eint2 = (q(i, j, k, QPRES)+parm.eos_gamma2*parm.eos_pinf2)/(q(i, j, k, QRHO2)*(parm.eos_gamma2-1.0_rt));
    }else{
        T2 = T_finder(alpha1, q(i,j,k,QRHO2), q(i,j,k,QPRES), u(i,j,k,GT2), parm, 2);
        eint2 = TPF(OE, T2, std::log10(q(i,j,k,QPRES)), parm, 2);
    }
    // amrex::Real eint2 = (q(i, j, k, QPRES)+parm.eos_gamma2*parm.eos_pinf2)/(q(i, j, k, QRHO2)*(parm.eos_gamma2-1.0_rt));
    amrex::Real Y1 = q(i, j, k, QALPHA)*q(i, j, k, QRHO1)/mrho;
    amrex::Real Y2 = (1.0_rt-q(i, j, k, QALPHA))*q(i, j, k, QRHO2)/mrho;
    amrex::Real E = Y1*eint1 + Y2*eint2 + 0.5_rt*q(i, j, k, QU)*q(i, j, k, QU) + 0.5_rt*q(i, j, k, QV)*q(i, j, k, QV);

    amrex::Real c1sq = 0;
    amrex::Real c2sq = 0;
    if(parm.tabulated1 == 0){
        c1sq = parm.eos_gamma1*(q(i,j,k,QPRES)+parm.eos_pinf1)/q(i,j,k,QRHO1);
    }else{
        c1sq = std::pow(TPF(OSOS, T1, std::log10(q(i,j,k,QPRES)), parm, 1), 2.0);
    }
    if(parm.tabulated2 == 0){
        c2sq = -1.0/std::pow(rho2, 2.0)*parm.gamma0*parm.pinf0_1*(q(i,j,k,QPRES)+parm.gamma0*parm.pinf0*(1.0-parm.b1)/(parm.gamma0-parm.b1))*
        ((parm.gamma0-1.0)/((parm.gamma0-1.0)*parm.cv0-parm.gamma0*parm.pinf0_1*(1.0/rho2-((parm.b1/rho2+parm.b0_1))))+1.0/parm.cv0)
        +((q(i,j,k,QPRES)+(parm.gamma0*parm.pinf0*(1.0-parm.b1)/(parm.gamma0-parm.b1)))/((parm.gamma0-1.0)*parm.cv0-parm.gamma0*parm.pinf0_1*(1.0/rho2-(parm.b1/rho2+parm.b0_1))))
        *(1.0/std::pow(rho2, 2.0)*(parm.gamma0-parm.b1)*(parm.gamma0-1.0)*parm.cv0/(1.0/rho2-((parm.b1/rho2+parm.b0_1))));
        // c2sq = parm.eos_gamma2*(q(i,j,k,QPRES)+parm.eos_pinf2)/q(i,j,k,QRHO2);
    }else{
        c2sq = std::pow(TPF(OSOS, T2, std::log10(q(i,j,k,QPRES)), parm, 2), 2.0);
    }
    // amrex::Real c1sq = amrex::max(parm.eos_gamma1*(q(i,j,k,QPRES)+parm.eos_pinf1)/q(i,j,k,QRHO1),parm.smallr);
    // amrex::Real c2sq = amrex::max(parm.eos_gamma2*(q(i,j,k,QPRES)+parm.eos_pinf2)/q(i,j,k,QRHO2),parm.smallr);
    // amrex::Real K = (q(i,j,k,QRHO2)*c2sq - q(i,j,k,QRHO1)*c1sq)/( (q(i,j,k,QRHO2)*c2sq)/(1.0_rt - q(i,j,k,QALPHA)) + (q(i,j,k,QRHO1)*c1sq)/q(i,j,k,QALPHA));
    // amrex::Real K = (q(i,j,k,QALPHA)*(1.0-q(i,j,k,QALPHA))*(q(i,j,k,QRHO2)*c2sq-q(i,j,k,QRHO1)*c2sq) )/(q(i,j,k,QALPHA)*q(i,j,k,QRHO2)*c2sq+(1.0-q(i,j,k,QALPHA))*q(i,j,k,QRHO1)*c1sq);
    amrex::Real cpsq = (1.0)/(mrho*( alpha1/(rho1*c1sq) + alpha2/(rho2*c2sq) ));
    amrex::Real K1 = 0.0;
    if(time > parm.ktime){
        K1 = alpha1*( (mrho*cpsq)/(rho1*c1sq) - 1.0);
        //K1 = 0.0;
    }

    dudt(i, j, k, UARHO1)     -= (cc/r)*( q(i, j, k, QALPHA)*q(i, j, k, QRHO1)*q(i, j, k, QU) );
    dudt(i, j, k, UARHO2)     -= (cc/r)*( (1.0_rt-q(i, j, k, QALPHA))*q(i, j, k, QRHO2)*q(i, j, k, QU) );
    dudt(i, j, k, UMX)        -= (cc/r)*( mrho*q(i, j, k, QU)*q(i, j, k, QU));
    dudt(i, j, k, UMY)        -= (cc/r)*( mrho*q(i, j, k, QU)*q(i, j, k, QV));
    dudt(i, j, k, URHOE)      -= (cc/r)*( q(i, j, k, QU)*(mrho*E + q(i, j, k, QPRES)));
    dudt(i, j, k, GALPHA)     -= (cc/r)*( -K1*q(i, j, k, QU));
}

namespace {

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real xlimiter (int i, int j, int k, amrex::Array4<amrex::Real const> const& q,
                int comp) noexcept
{
    using namespace amrex::literals;

    amrex::Real slopex;
    amrex::Real qp1 = q(i+1,j,k,comp)-q(i,j,k,comp);
    amrex::Real qm1 = q(i,j,k,comp) - q(i-1,j,k,comp);
    if (std::fabs(qp1) <= 1E-30_rt){
        slopex = 0.0_rt;
    }else{
        amrex::Real r = qm1/qp1;
        amrex::Real lr;
        lr = amrex::max(0.0_rt,amrex::min(1.0_rt,r)); // Minmod
        // lr = amrex::max(0.0_rt,amrex::max(amrex::min(2.0_rt*r,1.0_rt),amrex::min(r,2.0_rt))); //Superbee
        // lr = amrex::max(0.0_rt,amrex::min(amrex::min(2.0_rt*r,0.5_rt*(1.0_rt+r)),2.0_rt)); //MC
        // lr = (r + std::fabs(r))/(1 + std::fabs(r));
        slopex = lr*qp1;
    }

    return slopex;
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real ylimiter (int i, int j, int k, amrex::Array4<amrex::Real const> const& q,
                int comp) noexcept
{
    using namespace amrex::literals;

    amrex::Real slopey;
    amrex::Real qp1 = q(i,j+1,k,comp)-q(i,j,k,comp);
    amrex::Real qm1 = q(i,j,k,comp) - q(i,j-1,k,comp);
    if (std::fabs(qp1) <= 1E-30_rt){
        slopey = 0.0;
    }else{
        amrex::Real r = qm1/qp1;
        amrex::Real lr;
        lr = amrex::max(0.0_rt,amrex::min(1.0_rt,r)); // Minmod
        // lr = amrex::max(0.0_rt,amrex::max(amrex::min(2.0_rt*r,1.0_rt),amrex::min(r,2.0_rt)));
        // lr = amrex::max(0.0_rt,amrex::min(amrex::min(2.0_rt*r,0.5_rt*(1.0_rt+r)),2.0_rt)); //MC
        // lr = (r + std::fabs(r))/(1 + std::fabs(r));
        slopey = lr*qp1;
    }

    return slopey;
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real zlimiter (int i, int j, int k, amrex::Array4<amrex::Real const> const& q,
                int comp) noexcept
{
    using namespace amrex::literals;

    amrex::Real slopez;
    amrex::Real qp1 = q(i,j,k+1,comp)-q(i,j,k,comp);
    amrex::Real qm1 = q(i,j,k,comp) - q(i,j,k-1,comp);
    if (std::fabs(qp1) <= 1E-30_rt){
        slopez = 0.0_rt;
    }else{
        amrex::Real r = qm1/qp1;
        amrex::Real lr;
        lr = amrex::max(0.0_rt,amrex::min(1.0_rt,r)); // Minmod
        //lr = amrex::max(0.0_rt,amrex::max(amrex::min(2.0_rt*r,1.0_rt),amrex::min(r,2.0_rt))); // Superbee
        // lr = amrex::max(0.0_rt,amrex::min(amrex::min(2.0_rt*r,0.5_rt*(1.0_rt+r)),2.0_rt)); // MC
        // lr = (r + std::fabs(r))/(1 + std::fabs(r));
        slopez = lr*qp1;
    }

    return slopez;
}
}

AMREX_GPU_DEVICE
inline
void
EMM_PMUSCL_reconstruct_x (int i, int j, int k, // Loop running on the cell centers, However qL,qR are face centered
             amrex::Array4<amrex::Real> const& qL,
             amrex::Array4<amrex::Real> const& qR,
             amrex::Array4<amrex::Real const> const& q, amrex::Real dt,
             amrex::GeometryData const& geomdata,
             amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dxinv,
             Parm const& parm) noexcept
{
    using namespace amrex::literals;

    qL(i,j,k,QALPHA) = q(i,j,k,QALPHA) - 0.5_rt*xlimiter(i, j, k, q, QALPHA);
    qL(i,j,k,QRHO1) = q(i,j,k,QRHO1) - 0.5_rt*xlimiter(i, j, k, q,QRHO1);
    qL(i,j,k,QRHO2) = q(i,j,k,QRHO2) - 0.5_rt*xlimiter(i, j, k, q,QRHO2);
    qL(i,j,k,QU) = q(i,j,k,QU)  - 0.5_rt*xlimiter(i, j, k, q,QU);
    qL(i,j,k,QV) = q(i,j,k,QV) - 0.5_rt*xlimiter(i, j, k, q,QV);
    qL(i,j,k,QW) = q(i,j,k,QW) - 0.5_rt*xlimiter(i, j, k, q,QW);
    qL(i,j,k,QPRES) = q(i,j,k,QPRES) - 0.5_rt*xlimiter(i, j, k, q,QPRES);

    // qR(i+1,j,k)
    qR(i+1,j,k,QALPHA) = q(i,j,k,QALPHA) + 0.5_rt*xlimiter(i, j, k, q,QALPHA);
    qR(i+1,j,k,QRHO1) = q(i,j,k,QRHO1) + 0.5_rt*xlimiter(i, j, k, q,QRHO1);
    qR(i+1,j,k,QRHO2) = q(i,j,k,QRHO2) + 0.5_rt*xlimiter(i, j, k, q,QRHO2);
    qR(i+1,j,k,QU) = q(i,j,k,QU) + 0.5_rt*xlimiter(i, j, k, q,QU);
    qR(i+1,j,k,QV) = q(i,j,k,QV) + 0.5_rt*xlimiter(i, j, k, q,QV);
    qR(i+1,j,k,QW) = q(i,j,k,QW) + 0.5_rt*xlimiter(i, j, k, q,QW);
    qR(i+1,j,k,QPRES) = q(i,j,k,QPRES) + 0.5_rt*xlimiter(i, j, k, q,QPRES);

    // qL(i,j,k,QALPHA) = q(i,j,k,QALPHA);
    // qL(i,j,k,QRHO1) = q(i,j,k,QRHO1);
    // qL(i,j,k,QRHO2) = q(i,j,k,QRHO2);
    // qL(i,j,k,QU) = q(i,j,k,QU);
    // qL(i,j,k,QV) = q(i,j,k,QV);
    // qL(i,j,k,QW) = q(i,j,k,QW);
    // qL(i,j,k,QPRES) = q(i,j,k,QPRES);

    // // qR(i+1,j,k)
    // qR(i+1,j,k,QALPHA) = q(i,j,k,QALPHA);
    // qR(i+1,j,k,QRHO1) = q(i,j,k,QRHO1);
    // qR(i+1,j,k,QRHO2) = q(i,j,k,QRHO2);
    // qR(i+1,j,k,QU) = q(i,j,k,QU);
    // qR(i+1,j,k,QV) = q(i,j,k,QV);
    // qR(i+1,j,k,QW) = q(i,j,k,QW);
    // qR(i+1,j,k,QPRES) = q(i,j,k,QPRES);
}

AMREX_GPU_DEVICE
inline
void
EMM_PMUSCL_reconstruct_y (int i, int j, int k,
             amrex::Array4<amrex::Real> const& qL,
             amrex::Array4<amrex::Real> const& qR,
             amrex::Array4<amrex::Real const> const& q, amrex::Real dt,
             amrex::GeometryData const& geomdata,
             amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dxinv,
             Parm const& parm) noexcept
{
    using namespace amrex::literals;

    qL(i,j,k,QALPHA) = q(i,j,k,QALPHA) - 0.5_rt*ylimiter(i, j, k, q, QALPHA);
    qL(i,j,k,QRHO1) = q(i,j,k,QRHO1) - 0.5_rt*ylimiter(i, j, k, q,QRHO1);
    qL(i,j,k,QRHO2) = q(i,j,k,QRHO2) - 0.5_rt*ylimiter(i, j, k, q,QRHO2);
    qL(i,j,k,QU) = q(i,j,k,QU) - 0.5_rt*ylimiter(i, j, k, q,QU);
    qL(i,j,k,QV) = q(i,j,k,QV) - 0.5_rt*ylimiter(i, j, k, q,QV);
    qL(i,j,k,QW) = q(i,j,k,QW) - 0.5_rt*ylimiter(i, j, k, q,QW);
    qL(i,j,k,QPRES) = q(i,j,k,QPRES) - 0.5_rt*ylimiter(i, j, k, q,QPRES);

    // qR(i+1,j,k
    qR(i,j+1,k,QALPHA) = q(i,j,k,QALPHA) + 0.5_rt*ylimiter(i, j, k, q,QALPHA);
    qR(i,j+1,k,QRHO1) = q(i,j,k,QRHO1) + 0.5_rt*ylimiter(i, j, k, q,QRHO1);
    qR(i,j+1,k,QRHO2) = q(i,j,k,QRHO2) + 0.5_rt*ylimiter(i, j, k, q,QRHO2);
    qR(i,j+1,k,QU) = q(i,j,k,QU) + 0.5_rt*ylimiter(i, j, k, q,QU);
    qR(i,j+1,k,QV) = q(i,j,k,QV) + 0.5_rt*ylimiter(i, j, k, q,QV);
    qR(i,j+1,k,QW) = q(i,j,k,QW) + 0.5_rt*ylimiter(i, j, k, q,QW);
    qR(i,j+1,k,QPRES) = q(i,j,k,QPRES) + 0.5_rt*ylimiter(i, j, k, q,QPRES);
}

AMREX_GPU_DEVICE
inline
void
EMM_PMUSCL_reconstruct_z (int i, int j, int k,
             amrex::Array4<amrex::Real> const& qL,
             amrex::Array4<amrex::Real> const& qR,
             amrex::Array4<amrex::Real const> const& q, amrex::Real dt,
             amrex::GeometryData const& geomdata,
             amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dxinv,
             Parm const& parm) noexcept
{
    using namespace amrex::literals;

    qL(i,j,k,QALPHA) = q(i,j,k,QALPHA) - 0.5_rt*zlimiter(i, j, k, q, QALPHA);
    qL(i,j,k,QRHO1) = q(i,j,k,QRHO1) - 0.5_rt*zlimiter(i, j, k, q,QRHO1);
    qL(i,j,k,QRHO2) = q(i,j,k,QRHO2) - 0.5_rt*zlimiter(i, j, k, q,QRHO2);
    qL(i,j,k,QU) = q(i,j,k,QU)  - 0.5_rt*zlimiter(i, j, k, q,QU);
    qL(i,j,k,QV) = q(i,j,k,QV) - 0.5_rt*zlimiter(i, j, k, q,QV);
    qL(i,j,k,QW) = q(i,j,k,QW) - 0.5_rt*zlimiter(i, j, k, q,QW);
    qL(i,j,k,QPRES) = q(i,j,k,QPRES) - 0.5_rt*zlimiter(i, j, k, q,QPRES);

    // qR(i+1,j,k
    qR(i,j,k+1,QALPHA) = q(i,j,k,QALPHA) + 0.5_rt*zlimiter(i, j, k, q,QALPHA);
    qR(i,j,k+1,QRHO1) = q(i,j,k,QRHO1) + 0.5_rt*zlimiter(i, j, k, q,QRHO1);
    qR(i,j,k+1,QRHO2) = q(i,j,k,QRHO2) + 0.5_rt*zlimiter(i, j, k, q,QRHO2);
    qR(i,j,k+1,QU) = q(i,j,k,QU) + 0.5_rt*zlimiter(i, j, k, q,QU);
    qR(i,j,k+1,QV) = q(i,j,k,QV) + 0.5_rt*zlimiter(i, j, k, q,QV);
    qR(i,j,k+1,QW) = q(i,j,k,QW) + 0.5_rt*zlimiter(i, j, k, q,QW);
    qR(i,j,k+1,QPRES) = q(i,j,k,QPRES) + 0.5_rt*zlimiter(i, j, k, q,QPRES);
}

namespace {

AMREX_GPU_DEVICE
inline
void
riemann (const amrex::Real gamma1, const amrex::Real pinf1, const amrex::Real gamma2, const amrex::Real pinf2,
         const amrex::Real rho1L, const amrex::Real rho2L, const amrex::Real alpha1L, const amrex::Real alpha2L,
         const amrex::Real mrhoL, const amrex::Real uL, const amrex::Real ut1L, const amrex::Real ut2L,
         const amrex::Real pL, const amrex::Real eint1L, const amrex::Real eint2L,
         const amrex::Real Y1L, const amrex::Real Y2L, const amrex::Real EL, const amrex::Real cL,
         const amrex::Real rho1R, const amrex::Real rho2R, const amrex::Real alpha1R, const amrex::Real alpha2R,
         const amrex::Real mrhoR, const amrex::Real uR, const amrex::Real ut1R, const amrex::Real ut2R,
         const amrex::Real pR, const amrex::Real eint1R, const amrex::Real eint2R,
         const amrex::Real Y1R, const amrex::Real Y2R, const amrex::Real ER, const amrex::Real cR,
         amrex::Real& flxarho1, amrex::Real& flxarho2, amrex::Real& flxrhou,
         amrex::Real& flxrhout, amrex::Real& flxrhoutt, amrex::Real& flxrhoE,
         amrex::Real& flxalpha,
         amrex::Real& US, amrex::Real& VS, amrex::Real& WS, int cdir) noexcept
{
    BL_PROFILE("riemann()");
    using namespace amrex::literals;

    amrex::Real SL = amrex::min(uL-cL,uR-cR); //o
    amrex::Real SR = amrex::max(uL+cL,uR+cR); //o

    // amrex::Real uStar = (mrhoL*uL*uL + pL - mrhoR*uR*uR - pR - SL*mrhoL*uL
    //              + SR*mrhoR*uR)/(mrhoL*uL - mrhoR*uR - SL*mrhoL + SR*mrhoR);
    amrex::Real uStar = (pR - pL + mrhoL*uL*(SL-uL) - mrhoR*uR*(SR-uR))/(mrhoL*(SL-uL)-mrhoR*(SR-uR));

    //Star variables
    amrex::Real alpha1S_L = alpha1L; amrex::Real alpha1S_R = alpha1R; //o
    amrex::Real alpha2S_L = alpha2L; amrex::Real alpha2S_R = alpha2R; //o

    amrex::Real al1rho1S_L = alpha1L*rho1L*(SL-uL)/(SL-uStar); amrex::Real al1rho1S_R = alpha1R*rho1R*(SR-uR)/(SR-uStar); //o
    amrex::Real al2rho2S_L = alpha2L*rho2L*(SL-uL)/(SL-uStar); amrex::Real al2rho2S_R = alpha2R*rho2R*(SR-uR)/(SR-uStar); //o

    amrex::Real mrhoS_L = al1rho1S_L + al2rho2S_L; //o
    amrex::Real mrhoS_R = al1rho1S_R + al2rho2S_R; //o

    amrex::Real pS = pR + mrhoR*uR*(uR-SR) - mrhoS_R*uStar*(uStar-SR);

    amrex::Real ES_R = (mrhoR*ER*(uR-SR) + pR*uR - pS*uStar)/(mrhoS_R*(uStar-SR)); //o
    amrex::Real ES_L = (mrhoL*EL*(uL-SL) + pL*uL - pS*uStar)/(mrhoS_L*(uStar-SL)); //o

    // amrex::Real ES_R = mrhoR*((SR-uR)/(SR-uStar))*(ER/mrhoR)+(uStar-uR)*(uStar+pR/(mrhoR*(SR-uR)));
    // amrex::Real ES_L = mrhoL*((SL-uL)/(SL-uStar))*(EL/mrhoL)+(uStar-uL)*(uStar+pL/(mrhoL*(SL-uL))); 

    if (0._rt <= SL){
        //Left State
        flxarho1 = alpha1L*rho1L*uL;
        flxarho2 = alpha2L*rho2L*uL;
        flxrhou = mrhoL*uL*uL+pL;
        flxrhout = mrhoL*uL*ut1L;
        flxrhoutt = mrhoL*uL*ut2L;
        flxrhoE = uL*(mrhoL*EL+pL);
        flxalpha = alpha1L*uL;
        // Face velocities for non-conservative update
        US = uL;
        VS = ut1L;
        WS = ut2L;
    }

    if (SL <= 0._rt && 0._rt <= uStar ){
        // Left Star State
        flxarho1 = alpha1L*rho1L*uL + SL*(al1rho1S_L - alpha1L*rho1L);
        flxarho2 = alpha2L*rho2L*uL + SL*(al2rho2S_L - alpha2L*rho2L);
        flxrhou = mrhoL*uL*uL+pL + SL*(mrhoS_L*uStar - mrhoL*uL);
        flxrhout = mrhoL*uL*ut1L + SL*(mrhoS_L*ut1L - mrhoL*ut1L);
        flxrhoutt = mrhoL*uL*ut2L + SL*(mrhoS_L*ut2L - mrhoL*ut2L);
        flxrhoE = uL*(mrhoL*EL+pL) + SL*(mrhoS_L*ES_L - mrhoL*EL);
        flxalpha = alpha1S_L*uStar;
        // Face velocities for non-conservative update
        US = uStar;
        VS = ut1L;
        WS = ut2L;
    }

    if (uStar <= 0._rt && 0._rt <= SR ){
        //Right Star State
        flxarho1 = alpha1R*rho1R*uR + SR*(al1rho1S_R - alpha1R*rho1R);
        flxarho2 = alpha2R*rho2R*uR + SR*(al2rho2S_R - alpha2R*rho2R);
        flxrhou = mrhoR*uR*uR+pR + SR*(mrhoS_R*uStar - mrhoR*uR);
        flxrhout = mrhoR*uR*ut1R + SR*(mrhoS_R*ut1R - mrhoR*ut1R);
        flxrhoutt = mrhoR*uR*ut2R + SR*(mrhoS_R*ut2R - mrhoR*ut2R);
        flxrhoE = uR*(mrhoR*ER+pR) + SR*(mrhoS_R*ES_R - mrhoR*ER);
        flxalpha = alpha1S_R*uStar;
        // Face velocities for non-conservative update
        US = uStar;
        VS = ut1R;
        WS = ut2R;
    }

    if ( 0._rt >= SR ){
        //Right State
        flxarho1 = alpha1R*rho1R*uR;
        flxarho2 = alpha2R*rho2R*uR;
        flxrhou = mrhoR*uR*uR+pR;
        flxrhout = mrhoR*uR*ut1R;
        flxrhoutt = mrhoR*uR*ut2R;
        flxrhoE = uR*(mrhoR*ER+pR);
        flxalpha = alpha1R*uR;
        // Face velocities for non-conservative update
        US = uR;
        VS = ut1R;
        WS = ut2R;
    }
}
}

AMREX_GPU_DEVICE
inline
void
EMM_riemann_x (int i, int j, int k,
               amrex::Array4<amrex::Real> const& fx,
               amrex::Array4<amrex::Real> const& USx,
               amrex::Array4<amrex::Real> const& VSx,
               amrex::Array4<amrex::Real> const& WSx,
               amrex::Array4<amrex::Real> const& qL,
               amrex::Array4<amrex::Real> const& qR,
               amrex::Array4<amrex::Real const> const& q,
               amrex::Array4<amrex::Real const> const& u,
               Parm const& parm) noexcept
{
    BL_PROFILE("EMM_riemann_x()");
    using namespace amrex::literals;

    // Left
    amrex::Real rho1R = qL(i,j,k,QRHO1);
    amrex::Real rho2R = qL(i,j,k,QRHO2);
    amrex::Real alpha1R = qL(i,j,k,QALPHA);
    amrex::Real alpha2R = 1.0_rt - qL(i,j,k,QALPHA);
    amrex::Real uR = qL(i,j,k,QU);
    amrex::Real ut1R = qL(i,j,k,QV);
    amrex::Real ut2R = qL(i,j,k,QW);
    amrex::Real pR = qL(i,j,k,QPRES);

    // if(alpha1R > 1.0-parm.alpha_min/100.0){alpha1R = 1.0-parm.alpha_min/100.0;}
    // if(alpha1R < parm.alpha_min/100.0){alpha1R = parm.alpha_min/100.0;}
    // alpha2R = 1.0_rt - alpha1R;

    amrex::Real c1sqR = 0.0;
    amrex::Real T1R = 0.0;
    amrex::Real eint1R = 0.0;
    if(parm.tabulated1 == 0){
        c1sqR = amrex::max(parm.eos_gamma1*(pR+parm.eos_pinf1)/rho1R,parm.smallr);
        eint1R = (pR+parm.eos_gamma1*parm.eos_pinf1)/(rho1R*(parm.eos_gamma1-1.0_rt));
    }else{
        T1R = T_finder(alpha1R, rho1R, pR, u(i+1,j,k,GT1), parm, 1);
        c1sqR = amrex::max(std::pow(TPF(OSOS, T1R, std::log10(pR), parm, 1), 2.0),parm.smallr);
        eint1R = TPF(OE, T1R, std::log10(pR), parm, 1);
    }

    amrex::Real c2sqR = 0.0;
    amrex::Real T2R = 0.0;
    amrex::Real eint2R = 0.0;
    amrex::Real AR = 0.0;
    amrex::Real BR = 0.0;
    amrex::Real CR = 0.0;
    amrex::Real pprimeR = 0.0;
    if(parm.tabulated2 == 0){
        AR = parm.gamma0*parm.pinf0*(1.0/rho2R-(parm.b1/rho2R+parm.b0_1))/parm.cv0/(parm.gamma0-parm.b1);
        BR = 1.0/rho2R-(parm.b1/rho2R+parm.b0_1);
        CR = (pR+parm.gamma0*parm.pinf0)*BR/(parm.gamma0-1.0);
        T2R = (CR/parm.cv0-AR)/(1.0-parm.gamma0*parm.pinf0_1*BR/(parm.gamma0-1.0)/parm.cv0);

        c2sqR = -1.0/std::pow(rho2R, 2.0)*parm.gamma0*parm.pinf0_1*(pR+parm.gamma0*parm.pinf0*(1.0-parm.b1)/(parm.gamma0-parm.b1))*
        ((parm.gamma0-1.0)/((parm.gamma0-1.0)*parm.cv0-parm.gamma0*parm.pinf0_1*(1.0/rho2R-((parm.b1/rho2R+parm.b0_1))))+1.0/parm.cv0)
        +((pR+(parm.gamma0*parm.pinf0*(1.0-parm.b1)/(parm.gamma0-parm.b1)))/((parm.gamma0-1.0)*parm.cv0-parm.gamma0*parm.pinf0_1*(1.0/rho2R-(parm.b1/rho2R+parm.b0_1))))
        *(1.0/std::pow(rho2R, 2.0)*(parm.gamma0-parm.b1)*(parm.gamma0-1.0)*parm.cv0/(1.0/rho2R-((parm.b1/rho2R+parm.b0_1))));
        pprimeR = parm.gamma0*parm.pinf0_1*T2R+parm.gamma0*parm.pinf0*(1.0-parm.b1)/(parm.gamma0-parm.b1);
        eint2R = (pR+parm.gamma0*parm.pinf0)/(pR+pprimeR)*parm.cv0*T2R+parm.q0;

        // c2sqR = amrex::max(parm.eos_gamma2*(pR+parm.eos_pinf2)/rho2R,parm.smallr);
        // eint2R = (pR+parm.eos_gamma2*parm.eos_pinf2)/(rho2R*(parm.eos_gamma2-1.0_rt));
    }else{
        T2R = T_finder(alpha1R, rho2R, pR, u(i+1,j,k,GT2),parm, 2);
        c2sqR = amrex::max(std::pow(TPF(OSOS, T2R, std::log10(pR), parm, 2), 2.0),parm.smallr);
        eint2R = TPF(OE, T2R, std::log10(pR), parm, 2);
    }

    // Print() << "RIGHT=============" << "\n";
    // Print() << "eint2R:           " << eint2R << "\n";
    // Print() << "c2R:              " << std::pow(c2sqR, 0.5) << "\n";
    // Print() << "T2R:              " << T2R << "\n";
    // Print() << "pR:               " << pR << "\n";

    amrex::Real mrhoR = alpha1R*rho1R + alpha2R*rho2R;
    // amrex::Real eint1R = (pR+parm.eos_gamma1*parm.eos_pinf1)/(rho1R*(parm.eos_gamma1-1.0_rt));
    // amrex::Real eint2R = (pR+parm.eos_gamma2*parm.eos_pinf2)/(rho2R*(parm.eos_gamma2-1.0_rt));
    amrex::Real Y1R = alpha1R*rho1R/mrhoR;
    amrex::Real Y2R = alpha2R*rho2R/mrhoR;
    amrex::Real ER = Y1R*eint1R + Y2R*eint2R + 0.5_rt*uR*uR + 0.5_rt*ut1R*ut1R + 0.5_rt*ut2R*ut2R;

    // amrex::Real c1sqR = amrex::max(parm.eos_gamma1*(pR+parm.eos_pinf1)/rho1R,parm.smallr);
    // amrex::Real c2sqR = amrex::max(parm.eos_gamma2*(pR+parm.eos_pinf2)/rho2R,parm.smallr);
    // amrex::Real rhocsqR = 1.0_rt/(alpha1R/(rho1R*c1sqR)+alpha2R/(rho2R*c2sqR));
    // amrex::Real cR = std::sqrt(rhocsqR/mrhoR);
    //amrex::Real aaL = 
    //if( (rho1R*c1sqR*rho2R*c2sqR)/(mrhoR*(alpha1R*rho2R*c2sqR+alpha2R*rho1R*c1sqR)) < 0.0){ Print() << "Negative speed of sound" << "\n";}
    // Wood Sound Speed
    // amrex::Real cR = std::sqrt((rho1R*c1sqR*rho2R*c2sqR)/(mrhoR*(alpha1R*rho2R*c2sqR+alpha2R*rho1R*c1sqR)));
    // Frozen Sound Speed
    // amrex::Real cR = std::sqrt(Y1R*parm.eos_gamma1*(pR+parm.eos_pinf1)/rho1R
                    // + Y2R*parm.eos_gamma2*(pR+parm.eos_pinf2)/rho2R);
    amrex::Real cR = std::sqrt(Y1R*c1sqR + Y2R*c2sqR);

    // Right
    amrex::Real rho1L = qR(i,j,k,QRHO1);
    amrex::Real rho2L = qR(i,j,k,QRHO2);
    amrex::Real alpha1L = qR(i,j,k,QALPHA);
    amrex::Real alpha2L = 1.0_rt - qR(i,j,k,QALPHA);
    amrex::Real uL = qR(i,j,k,QU);
    amrex::Real ut1L = qR(i,j,k,QV);
    amrex::Real ut2L = qR(i,j,k,QW);
    amrex::Real pL = qR(i,j,k,QPRES);

    // if(alpha1L > 1.0-parm.alpha_min/100.0){alpha1L = 1.0-parm.alpha_min/100.0;}
    // if(alpha1L < parm.alpha_min/100.0){alpha1L = parm.alpha_min/100.0;}
    // alpha2L = 1.0_rt - alpha1L;

    amrex::Real c1sqL = 0;
    amrex::Real T1L = 0;
    amrex::Real eint1L = 0;
    if(parm.tabulated1 == 0){
        c1sqL = amrex::max(parm.eos_gamma1*(pL+parm.eos_pinf1)/rho1L,parm.smallr);
        eint1L = (pL+parm.eos_gamma1*parm.eos_pinf1)/(rho1L*(parm.eos_gamma1-1.0_rt));
    }else{
        T1L = T_finder(alpha1L, rho1L, pL, u(i,j,k,GT1) ,parm, 1);
        c1sqL = std::pow(TPF(OSOS, T1L, std::log10(pL), parm, 1), 2.0);
        eint1L = TPF(OE, T1L, std::log10(pL), parm, 1);
    }

    amrex::Real c2sqL = 0;
    amrex::Real T2L = 0;
    amrex::Real eint2L = 0;
    amrex::Real AL = 0.0;
    amrex::Real BL = 0.0;
    amrex::Real CL = 0.0;
    amrex::Real pprimeL = 0.0;
    if(parm.tabulated2 == 0){
        // c2sqL = amrex::max(parm.eos_gamma2*(pL+parm.eos_pinf2)/rho2L,parm.smallr);
        // eint2L = (pL+parm.eos_gamma2*parm.eos_pinf2)/(rho2L*(parm.eos_gamma2-1.0_rt));
        AL = parm.gamma0*parm.pinf0*(1.0/rho2L-(parm.b1/rho2L+parm.b0_1))/parm.cv0/(parm.gamma0-parm.b1);
        BL = 1.0/rho2L-(parm.b1/rho2L+parm.b0_1);
        CL = (pL+parm.gamma0*parm.pinf0)*BL/(parm.gamma0-1.0);
        T2L = (CL/parm.cv0-AL)/(1.0-parm.gamma0*parm.pinf0_1*BL/(parm.gamma0-1.0)/parm.cv0);

        c2sqL = -1.0/std::pow(rho2L, 2.0)*parm.gamma0*parm.pinf0_1*(pL+parm.gamma0*parm.pinf0*(1.0-parm.b1)/(parm.gamma0-parm.b1))*
        ((parm.gamma0-1.0)/((parm.gamma0-1.0)*parm.cv0-parm.gamma0*parm.pinf0_1*(1.0/rho2L-((parm.b1/rho2L+parm.b0_1))))+1.0/parm.cv0)
        +((pL+(parm.gamma0*parm.pinf0*(1.0-parm.b1)/(parm.gamma0-parm.b1)))/((parm.gamma0-1.0)*parm.cv0-parm.gamma0*parm.pinf0_1*(1.0/rho2L-(parm.b1/rho2L+parm.b0_1))))
        *(1.0/std::pow(rho2L, 2.0)*(parm.gamma0-parm.b1)*(parm.gamma0-1.0)*parm.cv0/(1.0/rho2L-((parm.b1/rho2L+parm.b0_1))));
        pprimeL = parm.gamma0*parm.pinf0_1*T2L+parm.gamma0*parm.pinf0*(1.0-parm.b1)/(parm.gamma0-parm.b1);
        eint2L = (pL+parm.gamma0*parm.pinf0)/(pL+pprimeL)*parm.cv0*T2L+parm.q0;
    }else{
        T2L = T_finder(alpha1L, rho2L, pL, u(i,j,k,GT2),parm, 2);
        c2sqL = std::pow(TPF(OSOS, T2L, std::log10(pL), parm, 2), 2.0);
        eint2L = TPF(OE, T2L, std::log10(pL), parm, 2);
    }
    // Print() << "LEFT=============" << "\n";
    // Print() << "eint2L:           " << eint2L << "\n";
    // Print() << "c2sqL:            " << std::pow(c2sqL,0.5) << "\n";
    // Print() << "T2L:              " << T2L << "\n";
    // Print() << "pL:               " << pL << "\n";

    amrex::Real mrhoL = alpha1L*rho1L + alpha2L*rho2L;
    // amrex::Real eint1L = (pL+parm.eos_gamma1*parm.eos_pinf1)/(rho1L*(parm.eos_gamma1-1.0_rt));
    // amrex::Real eint2L = (pL+parm.eos_gamma2*parm.eos_pinf2)/(rho2L*(parm.eos_gamma2-1.0_rt));
    amrex::Real Y1L = alpha1L*rho1L/mrhoL;
    amrex::Real Y2L = alpha2L*rho2L/mrhoL;
    amrex::Real EL = Y1L*eint1L + Y2L*eint2L + 0.5_rt*uL*uL + 0.5_rt*ut1L*ut1L + 0.5_rt*ut2L*ut2L;
    // amrex::Real c1sqL = amrex::max(parm.eos_gamma1*(pL+parm.eos_pinf1)/rho1L,parm.smallr);

    // amrex::Real c2sqL = amrex::max(parm.eos_gamma2*(pL+parm.eos_pinf2)/rho2L,parm.smallr);
    // amrex::Real rhocsqL = 1.0_rt/(alpha1L/(rho1L*c1sqL)+alpha2L/(rho2L*c2sqL));
    // amrex::Real cL = std::sqrt(rhocsqL/mrhoL);
    //if( (rho1L*c1sqL*rho2L*c2sqL)/(mrhoL*(alpha1L*rho2L*c2sqL+alpha2L*rho1L*c1sqL)) < 0.0){ Print() << "Negative speed of sound" << "\n";}
    // amrex::Real cL = std::sqrt((rho1L*c1sqL*rho2L*c2sqL)/(mrhoL*(alpha1L*rho2L*c2sqL+alpha2L*rho1L*c1sqL)));
    // Frozen sound speed
    // amrex::Real cL = std::sqrt(Y1L*parm.eos_gamma1*(pL+parm.eos_pinf1)/rho1L
                    // + Y2L*parm.eos_gamma2*(pL+parm.eos_pinf2)/rho2L);
    // Print() << "alpha2L:              " << alpha2L << "\n";
    // Print() << "rho2L:                " << rho2L << "\n";
    // Print() << "mrhoL:                " << mrhoL << "\n";
    // Print() << "Y1L:                  " << Y1L << "\n";
    // Print() << "Y2L:                  " << Y2L << "\n";
    // Print() << "c1sqL:                " << c1sqL << "\n";
    // Print() << "c2sqL:                " << c2sqL << "\n";
    amrex::Real cL = std::sqrt(Y1L*c1sqL + Y2L*c2sqL);

    int cdir = 1;
    riemann(parm.eos_gamma1, parm.eos_pinf1, parm.eos_gamma2, parm.eos_pinf2,
            rho1L, rho2L, alpha1L, alpha2L, mrhoL, uL, ut1L, ut2L, pL, eint1L, eint2L, Y1L, Y2L, EL, cL,
            rho1R, rho2R, alpha1R, alpha2R, mrhoR, uR, ut1R, ut2R, pR, eint1R, eint2R, Y1R, Y2R, ER, cR,
            fx(i,j,k,UARHO1), fx(i,j,k,UARHO2), fx(i,j,k,UMX), fx(i,j,k,UMY), fx(i,j,k,UMZ), fx(i,j,k,URHOE),
            fx(i,j,k,GALPHA),
            //Non-conservative Face Velocities given by Riemann Solver
            USx(i,j,k), VSx(i,j,k), WSx(i,j,k), cdir);
}

AMREX_GPU_DEVICE
inline
void
EMM_riemann_y (int i, int j, int k,
               amrex::Array4<amrex::Real> const& fy,
               amrex::Array4<amrex::Real> const& USy,
               amrex::Array4<amrex::Real> const& VSy,
               amrex::Array4<amrex::Real> const& WSy,
               amrex::Array4<amrex::Real> const& qL,
               amrex::Array4<amrex::Real> const& qR,
               amrex::Array4<amrex::Real const> const& q,
               amrex::Array4<amrex::Real const> const& u,
               Parm const& parm) noexcept
{
    using namespace amrex::literals;

    // Left
    amrex::Real rho1R = qL(i,j,k,QRHO1);
    amrex::Real rho2R = qL(i,j,k,QRHO2);
    amrex::Real alpha1R = qL(i,j,k,QALPHA);
    amrex::Real alpha2R = 1.0_rt - qL(i,j,k,QALPHA);
    amrex::Real uR = qL(i,j,k,QV);
    amrex::Real ut1R = qL(i,j,k,QU);
    amrex::Real ut2R = qL(i,j,k,QW);
    amrex::Real pR = qL(i,j,k,QPRES);

    // if(alpha1R > 1.0-parm.alpha_min/100.0){alpha1R = 1.0-parm.alpha_min/100.0;}
    // if(alpha1R < parm.alpha_min/100.0){alpha1R = parm.alpha_min/100.0;}
    // alpha2R = 1.0_rt - alpha1R;

    amrex::Real c1sqR = 0.0;
    amrex::Real T1R = 0.0;
    amrex::Real eint1R = 0.0;
    if(parm.tabulated1 == 0){
        c1sqR = amrex::max(parm.eos_gamma1*(pR+parm.eos_pinf1)/rho1R,parm.smallr);
        eint1R = (pR+parm.eos_gamma1*parm.eos_pinf1)/(rho1R*(parm.eos_gamma1-1.0_rt));
    }else{
        T1R = T_finder(alpha1R, rho1R, pR, u(i,j+1,k,GT1),parm, 1);
        c1sqR = amrex::max(std::pow(TPF(OSOS, T1R, std::log10(pR), parm, 1), 2.0),parm.smallr);
        eint1R = TPF(OE, T1R, std::log10(pR), parm, 1);
    }

    amrex::Real c2sqR = 0.0;
    amrex::Real T2R = 0.0;
    amrex::Real eint2R = 0.0;
    amrex::Real AR = 0.0;
    amrex::Real BR = 0.0;
    amrex::Real CR = 0.0;
    amrex::Real pprimeR = 0.0;
    if(parm.tabulated2 == 0){
        // c2sqR = amrex::max(parm.eos_gamma2*(pR+parm.eos_pinf2)/rho2R,parm.smallr);
        // eint2R = (pR+parm.eos_gamma2*parm.eos_pinf2)/(rho2R*(parm.eos_gamma2-1.0_rt));
        AR = parm.gamma0*parm.pinf0*(1.0/rho2R-(parm.b1/rho2R+parm.b0_1))/parm.cv0/(parm.gamma0-parm.b1);
        BR = 1.0/rho2R-(parm.b1/rho2R+parm.b0_1);
        CR = (pR+parm.gamma0*parm.pinf0)*BR/(parm.gamma0-1.0);
        T2R = (CR/parm.cv0-AR)/(1.0-parm.gamma0*parm.pinf0_1*BR/(parm.gamma0-1.0)/parm.cv0);

        c2sqR = -1.0/std::pow(rho2R, 2.0)*parm.gamma0*parm.pinf0_1*(pR+parm.gamma0*parm.pinf0*(1.0-parm.b1)/(parm.gamma0-parm.b1))*
        ((parm.gamma0-1.0)/((parm.gamma0-1.0)*parm.cv0-parm.gamma0*parm.pinf0_1*(1.0/rho2R-((parm.b1/rho2R+parm.b0_1))))+1.0/parm.cv0)
        +((pR+(parm.gamma0*parm.pinf0*(1.0-parm.b1)/(parm.gamma0-parm.b1)))/((parm.gamma0-1.0)*parm.cv0-parm.gamma0*parm.pinf0_1*(1.0/rho2R-(parm.b1/rho2R+parm.b0_1))))
        *(1.0/std::pow(rho2R, 2.0)*(parm.gamma0-parm.b1)*(parm.gamma0-1.0)*parm.cv0/(1.0/rho2R-((parm.b1/rho2R+parm.b0_1))));
        pprimeR = parm.gamma0*parm.pinf0_1*T2R+parm.gamma0*parm.pinf0*(1.0-parm.b1)/(parm.gamma0-parm.b1);
        eint2R = (pR+parm.gamma0*parm.pinf0)/(pR+pprimeR)*parm.cv0*T2R+parm.q0;
    }else{
        T2R = T_finder(alpha1R, rho2R, pR, u(i,j+1,k,GT2),parm, 2);
        c2sqR = amrex::max(std::pow(TPF(OSOS, T2R, std::log10(pR), parm, 2), 2.0),parm.smallr);
        eint2R = TPF(OE, T2R, std::log10(pR), parm, 2);
    }

    amrex::Real mrhoR = alpha1R*rho1R + alpha2R*rho2R;
    // amrex::Real eint1R = (pR+parm.eos_gamma1*parm.eos_pinf1)/(rho1R*(parm.eos_gamma1-1.0_rt));
    // amrex::Real eint2R = (pR+parm.eos_gamma2*parm.eos_pinf2)/(rho2R*(parm.eos_gamma2-1.0_rt));
    amrex::Real Y1R = alpha1R*rho1R/mrhoR;
    amrex::Real Y2R = alpha2R*rho2R/mrhoR;
    amrex::Real ER = Y1R*eint1R + Y2R*eint2R + 0.5_rt*uR*uR + 0.5_rt*ut1R*ut1R + 0.5_rt*ut2R*ut2R;
    // amrex::Real c1sqR = parm.eos_gamma1*(pR+parm.eos_pinf1)/rho1R;
    // amrex::Real c2sqR = parm.eos_gamma2*(pR+parm.eos_pinf2)/rho2R;
    // amrex::Real rhocsqR = 1.0_rt/(alpha1R/(rho1R*c1sqR)+alpha2R/(rho2R*c2sqR));
    // amrex::Real cR = std::sqrt(rhocsqR/mrhoR);
    // amrex::Real cR = std::sqrt((rho1R*c1sqR*rho2R*c2sqR)/(mrhoR*(alpha1R*rho2R*c2sqR+alpha2R*rho1R*c1sqR)));
    // amrex::Real cR = std::sqrt(Y1R*parm.eos_gamma1*(pR+parm.eos_pinf1)/rho1R
                    // + Y2R*parm.eos_gamma2*(pR+parm.eos_pinf2)/rho2R);
    amrex::Real cR = std::sqrt(Y1R*c1sqR + Y2R*c2sqR);

    // Right
    amrex::Real rho1L = qR(i,j,k,QRHO1);
    amrex::Real rho2L = qR(i,j,k,QRHO2);
    amrex::Real alpha1L = qR(i,j,k,QALPHA);
    amrex::Real alpha2L = 1.0_rt - qR(i,j,k,QALPHA);
    amrex::Real uL = qR(i,j,k,QV);
    amrex::Real ut1L = qR(i,j,k,QU);
    amrex::Real ut2L = qR(i,j,k,QW);
    amrex::Real pL = qR(i,j,k,QPRES);

    // if(alpha1L > 1.0-parm.alpha_min/100.0){alpha1L = 1.0-parm.alpha_min/100.0;}
    // if(alpha1L < parm.alpha_min/100.0){alpha1L = parm.alpha_min/100.0;}
    // alpha2L = 1.0_rt - alpha1L;

    amrex::Real c1sqL = 0;
    amrex::Real T1L = 0;
    amrex::Real eint1L = 0;
    if(parm.tabulated1 == 0){
        c1sqL = amrex::max(parm.eos_gamma1*(pL+parm.eos_pinf1)/rho1L,parm.smallr);
        eint1L = (pL+parm.eos_gamma1*parm.eos_pinf1)/(rho1L*(parm.eos_gamma1-1.0_rt));
    }else{
        T1L = T_finder(alpha1L, rho1L, pL, u(i,j,k,GT1), parm, 1);
        c1sqL = std::pow(TPF(OSOS, T1L, std::log10(pL), parm, 1), 2.0);
        eint1L = TPF(OE, T1L, std::log10(pL), parm, 1);
    }

    amrex::Real c2sqL = 0;
    amrex::Real T2L = 0;
    amrex::Real eint2L = 0;
    amrex::Real AL = 0.0;
    amrex::Real BL = 0.0;
    amrex::Real CL = 0.0;
    amrex::Real pprimeL = 0.0;
    if(parm.tabulated2 == 0){
        // c2sqL = amrex::max(parm.eos_gamma2*(pL+parm.eos_pinf2)/rho2L,parm.smallr);
        // eint2L = (pL+parm.eos_gamma2*parm.eos_pinf2)/(rho2L*(parm.eos_gamma2-1.0_rt));
        AL = parm.gamma0*parm.pinf0*(1.0/rho2L-(parm.b1/rho2L+parm.b0_1))/parm.cv0/(parm.gamma0-parm.b1);
        BL = 1.0/rho2L-(parm.b1/rho2L+parm.b0_1);
        CL = (pL+parm.gamma0*parm.pinf0)*BL/(parm.gamma0-1.0);
        T2L = (CL/parm.cv0-AL)/(1.0-parm.gamma0*parm.pinf0_1*BL/(parm.gamma0-1.0)/parm.cv0);

        c2sqL = -1.0/std::pow(rho2L, 2.0)*parm.gamma0*parm.pinf0_1*(pL+parm.gamma0*parm.pinf0*(1.0-parm.b1)/(parm.gamma0-parm.b1))*
        ((parm.gamma0-1.0)/((parm.gamma0-1.0)*parm.cv0-parm.gamma0*parm.pinf0_1*(1.0/rho2L-((parm.b1/rho2L+parm.b0_1))))+1.0/parm.cv0)
        +((pL+(parm.gamma0*parm.pinf0*(1.0-parm.b1)/(parm.gamma0-parm.b1)))/((parm.gamma0-1.0)*parm.cv0-parm.gamma0*parm.pinf0_1*(1.0/rho2L-(parm.b1/rho2L+parm.b0_1))))
        *(1.0/std::pow(rho2L, 2.0)*(parm.gamma0-parm.b1)*(parm.gamma0-1.0)*parm.cv0/(1.0/rho2L-((parm.b1/rho2L+parm.b0_1))));
        pprimeL = parm.gamma0*parm.pinf0_1*T2L+parm.gamma0*parm.pinf0*(1.0-parm.b1)/(parm.gamma0-parm.b1);
        eint2L = (pL+parm.gamma0*parm.pinf0)/(pL+pprimeL)*parm.cv0*T2L+parm.q0;
    }else{
        T2L = T_finder(alpha1L, rho2L, pL, u(i,j,k,GT2),parm, 2);
        c2sqL = std::pow(TPF(OSOS, T2L, std::log10(pL), parm, 2), 2.0);
        eint2L = TPF(OE, T2L, std::log10(pL), parm, 2);
    }

    amrex::Real mrhoL = alpha1L*rho1L + alpha2L*rho2L;
    // amrex::Real eint1L = (pL+parm.eos_gamma1*parm.eos_pinf1)/(rho1L*(parm.eos_gamma1-1.0_rt));
    // amrex::Real eint2L = (pL+parm.eos_gamma2*parm.eos_pinf2)/(rho2L*(parm.eos_gamma2-1.0_rt));
    amrex::Real Y1L = alpha1L*rho1L/mrhoL;
    amrex::Real Y2L = alpha2L*rho2L/mrhoL;
    amrex::Real EL = Y1L*eint1L + Y2L*eint2L + 0.5_rt*uL*uL + 0.5_rt*ut1L*ut1L + 0.5_rt*ut2L*ut2L;
    // amrex::Real c1sqL = parm.eos_gamma1*(pL+parm.eos_pinf1)/rho1L;
    // amrex::Real c2sqL = parm.eos_gamma2*(pL+parm.eos_pinf2)/rho2L;
    // amrex::Real rhocsqL = 1.0_rt/(alpha1L/(rho1L*c1sqL)+alpha2L/(rho2L*c2sqL));
    // amrex::Real cL = std::sqrt(rhocsqL/mrhoL);
    // amrex::Real cL = std::sqrt((rho1L*c1sqL*rho2L*c2sqL)/(mrhoL*(alpha1L*rho2L*c2sqL+alpha2L*rho1L*c1sqL)));
    // amrex::Real cL = std::sqrt(Y1L*parm.eos_gamma1*(pL+parm.eos_pinf1)/rho1L
                    // + Y2L*parm.eos_gamma2*(pL+parm.eos_pinf2)/rho2L);
    amrex::Real cL = std::sqrt(Y1L*c1sqL + Y2L*c2sqL);

    int cdir = 1;
    riemann(parm.eos_gamma1, parm.eos_pinf1, parm.eos_gamma2, parm.eos_pinf2,
            rho1L, rho2L, alpha1L, alpha2L, mrhoL, uL, ut1L, ut2L, pL, eint1L, eint2L, Y1L, Y2L, EL, cL,
            rho1R, rho2R, alpha1R, alpha2R, mrhoR, uR, ut1R, ut2R, pR, eint1R, eint2R, Y1R, Y2R, ER, cR,
            fy(i,j,k,UARHO1), fy(i,j,k,UARHO2), fy(i,j,k,UMY), fy(i,j,k,UMX), fy(i,j,k,UMZ), fy(i,j,k,URHOE),
            fy(i,j,k,GALPHA),
            //Non-conservative Face Velocities given by Riemann Solver
            VSy(i,j,k), USy(i,j,k), WSy(i,j,k), cdir);
}

AMREX_GPU_DEVICE
inline
void
EMM_riemann_z (int i, int j, int k,
               amrex::Array4<amrex::Real> const& fz,
               amrex::Array4<amrex::Real> const& USz,
               amrex::Array4<amrex::Real> const& VSz,
               amrex::Array4<amrex::Real> const& WSz,
               amrex::Array4<amrex::Real> const& qL,
               amrex::Array4<amrex::Real> const& qR,
               amrex::Array4<amrex::Real const> const& q,
               amrex::Array4<amrex::Real const> const& u,
               Parm const& parm) noexcept
{
    using namespace amrex::literals;

    // Left
    amrex::Real rho1R = qL(i,j,k,QRHO1);
    amrex::Real rho2R = qL(i,j,k,QRHO2);
    amrex::Real alpha1R = qL(i,j,k,QALPHA);
    amrex::Real alpha2R = 1.0_rt - qL(i,j,k,QALPHA);
    amrex::Real uR = qL(i,j,k,QW);
    amrex::Real ut1R = qL(i,j,k,QU);
    amrex::Real ut2R = qL(i,j,k,QV);
    amrex::Real pR = qL(i,j,k,QPRES);

    // if(alpha1R > 1.0-parm.alpha_min/100.0){alpha1R = 1.0-parm.alpha_min/100.0;}
    // if(alpha1R < parm.alpha_min/100.0){alpha1R = parm.alpha_min/100.0;}
    // alpha2R = 1.0_rt - alpha1R;

    amrex::Real c1sqR = 0.0;
    amrex::Real T1R = 0.0;
    amrex::Real eint1R = 0.0;
    if(parm.tabulated1 == 0){
        c1sqR = amrex::max(parm.eos_gamma1*(pR+parm.eos_pinf1)/rho1R,parm.smallr);
        eint1R = (pR+parm.eos_gamma1*parm.eos_pinf1)/(rho1R*(parm.eos_gamma1-1.0_rt));
    }else{
        T1R = T_finder(alpha1R, rho1R, pR, u(i,j,k+1,GT1),parm, 1);
        c1sqR = amrex::max(std::pow(TPF(OSOS, T1R, std::log10(pR), parm, 1), 2.0),parm.smallr);
        eint1R = TPF(OE, T1R, std::log10(pR), parm, 1);
    }

    amrex::Real c2sqR = 0.0;
    amrex::Real T2R = 0.0;
    amrex::Real eint2R = 0.0;
    amrex::Real AR = 0.0;
    amrex::Real BR = 0.0;
    amrex::Real CR = 0.0;
    amrex::Real pprimeR = 0.0;
    if(parm.tabulated2 == 0){
        // c2sqR = amrex::max(parm.eos_gamma2*(pR+parm.eos_pinf2)/rho2R,parm.smallr);
        // eint2R = (pR+parm.eos_gamma2*parm.eos_pinf2)/(rho2R*(parm.eos_gamma2-1.0_rt));
        AR = parm.gamma0*parm.pinf0*(1.0/rho2R-(parm.b1/rho2R+parm.b0_1))/parm.cv0/(parm.gamma0-parm.b1);
        BR = 1.0/rho2R-(parm.b1/rho2R+parm.b0_1);
        CR = (pR+parm.gamma0*parm.pinf0)*BR/(parm.gamma0-1.0);
        T2R = (CR/parm.cv0-AR)/(1.0-parm.gamma0*parm.pinf0_1*BR/(parm.gamma0-1.0)/parm.cv0);

        c2sqR = -1.0/std::pow(rho2R, 2.0)*parm.gamma0*parm.pinf0_1*(pR+parm.gamma0*parm.pinf0*(1.0-parm.b1)/(parm.gamma0-parm.b1))*
        ((parm.gamma0-1.0)/((parm.gamma0-1.0)*parm.cv0-parm.gamma0*parm.pinf0_1*(1.0/rho2R-((parm.b1/rho2R+parm.b0_1))))+1.0/parm.cv0)
        +((pR+(parm.gamma0*parm.pinf0*(1.0-parm.b1)/(parm.gamma0-parm.b1)))/((parm.gamma0-1.0)*parm.cv0-parm.gamma0*parm.pinf0_1*(1.0/rho2R-(parm.b1/rho2R+parm.b0_1))))
        *(1.0/std::pow(rho2R, 2.0)*(parm.gamma0-parm.b1)*(parm.gamma0-1.0)*parm.cv0/(1.0/rho2R-((parm.b1/rho2R+parm.b0_1))));
        pprimeR = parm.gamma0*parm.pinf0_1*T2R+parm.gamma0*parm.pinf0*(1.0-parm.b1)/(parm.gamma0-parm.b1);
        eint2R = (pR+parm.gamma0*parm.pinf0)/(pR+pprimeR)*parm.cv0*T2R+parm.q0;

    }else{
        T2R = T_finder(alpha1R, rho2R, pR, u(i,j,k+1,GT2),parm, 2);
        c2sqR = amrex::max(std::pow(TPF(OSOS, T2R, std::log10(pR), parm, 2), 2.0),parm.smallr);
        eint2R = TPF(OE, T2R, std::log10(pR), parm, 2);
    }

    amrex::Real mrhoR = alpha1R*rho1R + alpha2R*rho2R;
    // amrex::Real eint1R = (pR+parm.eos_gamma1*parm.eos_pinf1)/(rho1R*(parm.eos_gamma1-1.0_rt));
    // amrex::Real eint2R = (pR+parm.eos_gamma2*parm.eos_pinf2)/(rho2R*(parm.eos_gamma2-1.0_rt));
    amrex::Real Y1R = alpha1R*rho1R/mrhoR;
    amrex::Real Y2R = alpha2R*rho2R/mrhoR;
    amrex::Real ER = Y1R*eint1R + Y2R*eint2R + 0.5_rt*uR*uR + 0.5_rt*ut1R*ut1R + 0.5_rt*ut2R*ut2R;
    //amrex::Real c1sqR = parm.eos_gamma1*(pR+parm.eos_pinf1)/rho1R;
    //amrex::Real c2sqR = parm.eos_gamma2*(pR+parm.eos_pinf2)/rho2R;
    //amrex::Real rhocsqR = 1.0_rt/(alpha1R/(rho1R*c1sqR)+alpha2R/(rho2R*c2sqR));
    //amrex::Real cR = std::sqrt(rhocsqR/mrhoR);
    // amrex::Real cR = std::sqrt(Y1R*parm.eos_gamma1*(pR+parm.eos_pinf1)/rho1R
                    // + Y2R*parm.eos_gamma2*(pR+parm.eos_pinf2)/rho2R);
    amrex::Real cR = std::sqrt(Y1R*c1sqR + Y2R*c2sqR);

    // Right
    amrex::Real rho1L = qR(i,j,k,QRHO1);
    amrex::Real rho2L = qR(i,j,k,QRHO2);
    amrex::Real alpha1L = qR(i,j,k,QALPHA);
    amrex::Real alpha2L = 1.0_rt - qR(i,j,k,QALPHA);
    amrex::Real uL = qR(i,j,k,QW);
    amrex::Real ut1L = qR(i,j,k,QU);
    amrex::Real ut2L = qR(i,j,k,QV);
    amrex::Real pL = qR(i,j,k,QPRES);

    // if(alpha1L > 1.0-parm.alpha_min/100.0){alpha1L = 1.0-parm.alpha_min/100.0;}
    // if(alpha1L < parm.alpha_min/100.0){alpha1L = parm.alpha_min/100.0;}
    // alpha2L = 1.0_rt - alpha1L;

    amrex::Real c1sqL = 0;
    amrex::Real T1L = 0;
    amrex::Real eint1L = 0;
    if(parm.tabulated1 == 0){
        c1sqL = amrex::max(parm.eos_gamma1*(pL+parm.eos_pinf1)/rho1L,parm.smallr);
        eint1L = (pL+parm.eos_gamma1*parm.eos_pinf1)/(rho1L*(parm.eos_gamma1-1.0_rt));
    }else{
        T1L = T_finder(alpha1L, rho1L, pL, u(i,j,k,GT1), parm, 1);
        c1sqL = std::pow(TPF(OSOS, T1L, std::log10(pL), parm, 1), 2.0);
        eint1L = TPF(OE, T1L, std::log10(pL), parm, 1);
    }

    amrex::Real c2sqL = 0;
    amrex::Real T2L = 0;
    amrex::Real eint2L = 0;
    amrex::Real AL = 0.0;
    amrex::Real BL = 0.0;
    amrex::Real CL = 0.0;
    amrex::Real pprimeL = 0.0;
    if(parm.tabulated2 == 0){
        // c2sqL = amrex::max(parm.eos_gamma2*(pL+parm.eos_pinf2)/rho2L,parm.smallr);
        // eint2L = (pL+parm.eos_gamma2*parm.eos_pinf2)/(rho2L*(parm.eos_gamma2-1.0_rt));
        AL = parm.gamma0*parm.pinf0*(1.0/rho2L-(parm.b1/rho2L+parm.b0_1))/parm.cv0/(parm.gamma0-parm.b1);
        BL = 1.0/rho2L-(parm.b1/rho2L+parm.b0_1);
        CL = (pL+parm.gamma0*parm.pinf0)*BL/(parm.gamma0-1.0);
        T2L = (CL/parm.cv0-AL)/(1.0-parm.gamma0*parm.pinf0_1*BL/(parm.gamma0-1.0)/parm.cv0);

        c2sqL = -1.0/std::pow(rho2L, 2.0)*parm.gamma0*parm.pinf0_1*(pL+parm.gamma0*parm.pinf0*(1.0-parm.b1)/(parm.gamma0-parm.b1))*
        ((parm.gamma0-1.0)/((parm.gamma0-1.0)*parm.cv0-parm.gamma0*parm.pinf0_1*(1.0/rho2L-((parm.b1/rho2L+parm.b0_1))))+1.0/parm.cv0)
        +((pL+(parm.gamma0*parm.pinf0*(1.0-parm.b1)/(parm.gamma0-parm.b1)))/((parm.gamma0-1.0)*parm.cv0-parm.gamma0*parm.pinf0_1*(1.0/rho2L-(parm.b1/rho2L+parm.b0_1))))
        *(1.0/std::pow(rho2L, 2.0)*(parm.gamma0-parm.b1)*(parm.gamma0-1.0)*parm.cv0/(1.0/rho2L-((parm.b1/rho2L+parm.b0_1))));
        pprimeL = parm.gamma0*parm.pinf0_1*T2L+parm.gamma0*parm.pinf0*(1.0-parm.b1)/(parm.gamma0-parm.b1);
        eint2L = (pL+parm.gamma0*parm.pinf0)/(pL+pprimeL)*parm.cv0*T2L+parm.q0;
    }else{
        T2L = T_finder(alpha1L, rho2L, pL, u(i,j,k,GT2),parm, 2);
        c2sqL = std::pow(TPF(OSOS, T2L, std::log10(pL), parm, 2), 2.0);
        eint2L = TPF(OE, T2L, std::log10(pL), parm, 2);
    }

    amrex::Real mrhoL = alpha1L*rho1L + alpha2L*rho2L;
    // amrex::Real eint1L = (pL+parm.eos_gamma1*parm.eos_pinf1)/(rho1L*(parm.eos_gamma1-1.0_rt));
    // amrex::Real eint2L = (pL+parm.eos_gamma2*parm.eos_pinf2)/(rho2L*(parm.eos_gamma2-1.0_rt));
    amrex::Real Y1L = alpha1L*rho1L/mrhoL;
    amrex::Real Y2L = alpha2L*rho2L/mrhoL;
    amrex::Real EL = Y1L*eint1L + Y2L*eint2L + 0.5_rt*uL*uL + 0.5_rt*ut1L*ut1L + 0.5_rt*ut2L*ut2L;
    //amrex::Real c1sqL = parm.eos_gamma1*(pL+parm.eos_pinf1)/rho1L;
    //amrex::Real c2sqL = parm.eos_gamma2*(pL+parm.eos_pinf2)/rho2L;
    //amrex::Real rhocsqL = 1.0_rt/(alpha1L/(rho1L*c1sqL)+alpha2L/(rho2L*c2sqL));
    //amrex::Real cL = std::sqrt(rhocsqL/mrhoL);
    // amrex::Real cL = std::sqrt(Y1L*parm.eos_gamma1*(pL+parm.eos_pinf1)/rho1L
                    // + Y2L*parm.eos_gamma2*(pL+parm.eos_pinf2)/rho2L);
    amrex::Real cL = std::sqrt(Y1L*c1sqL + Y2L*c2sqL);

    int cdir = 1;
    riemann(parm.eos_gamma1, parm.eos_pinf1, parm.eos_gamma2, parm.eos_pinf2,
            rho1L, rho2L, alpha1L, alpha2L, mrhoL, uL, ut1L, ut2L, pL, eint1L, eint2L, Y1L, Y2L, EL, cL,
            rho1R, rho2R, alpha1R, alpha2R, mrhoR, uR, ut1R, ut2R, pR, eint1R, eint2R, Y1R, Y2R, ER, cR,
            fz(i,j,k,UARHO1), fz(i,j,k,UARHO2), fz(i,j,k,UMZ), fz(i,j,k,UMX), fz(i,j,k,UMY), fz(i,j,k,URHOE),
            fz(i,j,k,GALPHA),
            //Non-conservative Face Velocities given by Riemann Solver
            WSz(i,j,k), USz(i,j,k), VSz(i,j,k), cdir);
}

// AMREX_GPU_DEVICE
// inline
// void
// EMM_THINC_reconstruct_x (int i, int j, int k,
//              amrex::Array4<amrex::Real> const& qL,
//              amrex::Array4<amrex::Real> const& qR,
//              amrex::Array4<amrex::Real const> const& q,  const int nprim) noexcept
// {
//     amrex::Real Beta=1.6_rt;
//     amrex::Real epsilon=1E-8_rt;
//     amrex::Real delta=1E-4_rt;

//     for(int comp=0; comp<nprim; comp++){
//         //Coeficients
//         amrex::Real qmin = amrex::min(q(i-1,j,k,comp),q(i+1,j,k,comp));
//         amrex::Real qmax = amrex::max(q(i-1,j,k,comp),q(i+1,j,k,comp))-qmin;
//         amrex::Real theta = 0.0_rt;
//         if(q(i+1,j,k,comp)-q(i-1,j,k,comp) < 0.0_rt){
//             theta = -1.0_rt;
//         }else if (q(i+1,j,k,comp)-q(i-1,j,k,comp) > 0.0_rt){
//             theta = 1.0_rt;
//         }

//         amrex::Real C = (q(i,j,k,comp)-qmin+epsilon)/(qmax+epsilon);
//         amrex::Real B = exp(Beta*theta*(2.0_rt*C-1.0_rt));
//         amrex::Real A = (B/cosh(Beta)-1.0_rt)/tanh(Beta);

//         //Print() << "A--- : " << A << "\n";
//         //Print() << "B--- : " << B << "\n";
//         //Print() << "C--- : " << C << "\n";
//         //Print() << "1.0_rt+A*tanh(Beta) : " <<  1.0_rt+A*tanh(Beta) << "\n";
//         // qL(i,j,k,comp) = qmin + 0.5_rt*qmax*(1.0_rt+theta*A);
//         // qR(i+1,j,k,comp) = qmin + 0.5_rt*qmax*(1.0_rt+theta*(tanh(Beta)+A)/(1.0_rt+A*tanh(Beta)));
//         qL(i,j,k,comp) = qmin + 0.5_rt*qmax*(1.0_rt+theta*(tanh(Beta)+A)/(1.0_rt+A*tanh(Beta)));
//         qR(i+1,j,k,comp) = qmin + 0.5_rt*qmax*(1.0_rt+theta*A);
//     }
// }

// AMREX_GPU_DEVICE
// inline
// void
// EMM_THINC_reconstruct_y (int i, int j, int k,
//              amrex::Array4<amrex::Real> const& qL,
//              amrex::Array4<amrex::Real> const& qR,
//              amrex::Array4<amrex::Real const> const& q,  const int nprim) noexcept
// {
//     amrex::Real Beta=1.6_rt;
//     amrex::Real epsilon=1E-8_rt;
//     amrex::Real delta=1E-4_rt;

//     for(int comp=0; comp<nprim; comp++){
//         //Coeficients
//         amrex::Real qmin = amrex::min(q(i,j-1,k,comp),q(i,j+1,k,comp));
//         amrex::Real qmax = amrex::max(q(i,j-1,k,comp),q(i,j+1,k,comp))-qmin;
//         amrex::Real theta;
//         if(q(i,j+1,k,comp)-q(i,j-1,k,comp) < 0.0_rt){
//             theta =-1.0_rt;
//         }else if (q(i,j+1,k,comp)-q(i,j-1,k,comp) > 0.0_rt){
//             theta = 1.0_rt;
//         }else{
//             theta = 0.0_rt;
//         }
//         amrex::Real C = (q(i,j,k,comp)-qmin+epsilon)/(qmax+epsilon);
//         amrex::Real B = exp(Beta*theta*(2.0_rt*C-1.0_rt));
//         amrex::Real A = (B/cosh(Beta)-1.0_rt)/tanh(Beta);

//         qL(i,j,k,comp) = qmin + 0.5_rt*qmax*(1.0_rt+theta*A);
//         qR(i,j+1,k,comp) = qmin + 0.5_rt*qmax*(1.0_rt+theta*(tanh(Beta)+A)/(1.0_rt+A*tanh(Beta)));
//     }
// }

// AMREX_GPU_DEVICE
// inline
// void
// EMM_THINC_reconstruct_z (int i, int j, int k,
//              amrex::Array4<amrex::Real> const& qL,
//              amrex::Array4<amrex::Real> const& qR,
//              amrex::Array4<amrex::Real const> const& q,  const int nprim) noexcept
// {
//     amrex::Real Beta=1.6_rt;
//     amrex::Real epsilon=1E-8_rt;
//     amrex::Real delta=1E-4_rt;

//     for(int comp=0; comp<nprim; comp++){
//         //Coeficients
//         amrex::Real qmin = amrex::min(q(i,j,k-1,comp),q(i,j,k+1,comp));
//         amrex::Real qmax = amrex::max(q(i,j,k-1,comp),q(i,j,k+1,comp))-qmin;
//         amrex::Real theta;
//         if(q(i,j,k+1,comp)-q(i,j,k-1,comp) < 0.0_rt){
//             theta =-1.0_rt;
//         }else if (q(i,j,k+1,comp)-q(i,j,k-1,comp) > 0.0_rt){
//             theta = 1.0_rt;
//         }else{
//             theta = 0.0_rt;
//         }
//         amrex::Real C = (q(i,j,k,comp)-qmin+epsilon)/(qmax+epsilon);
//         amrex::Real B = exp(Beta*theta*(2.0_rt*C-1.0_rt));
//         amrex::Real A = (B/cosh(Beta)-1.0_rt)/tanh(Beta);

//         qL(i,j,k,comp) = qmin + 0.5_rt*qmax*(1.0_rt+theta*A);
//         qR(i,j,k+1,comp) = qmin + 0.5_rt*qmax*(1.0_rt+theta*(tanh(Beta)+A)/(1.0_rt+A*tanh(Beta)));
//     }
// }

// AMREX_GPU_DEVICE
// inline
// void
// EMM_TBV_x(int i, int j, int k,
//              amrex::Array4<amrex::Real> const& qR_M, 
//              amrex::Array4<amrex::Real> const& qL_M, 
//              amrex::Array4<amrex::Real> const& qL_T,
//              amrex::Array4<amrex::Real> const& qR_T,
//              amrex::Array4<amrex::Real> const& q, const int nprim) noexcept
// {
//     amrex::Real Beta=1.6_rt;
//     amrex::Real epsilon=1E-8_rt;
//     amrex::Real delta=1E-4_rt;
//     amrex::Real TBV_T;
//     amrex::Real TBV_M;

//     for(int comp=0; comp<nprim; comp++){
//         amrex::Real qmin = amrex::min(q(i-1,j,k,comp),q(i+1,j,k,comp));
//         amrex::Real qmax = amrex::max(q(i-1,j,k,comp),q(i+1,j,k,comp))-qmin;
//         amrex::Real C = (q(i,j,k,comp)-qmin+epsilon)/(qmax+epsilon);

//         //Compute total boundary variations TBV for each cell
//         TBV_M =amrex::min(amrex::min(std::abs(qR_M(i,j,k,comp)-qL_M(i,j,k,comp)) + std::abs(qR_M(i+1,j,k,comp)-qL_M(i+1,j,k,comp)),
//                                      std::abs(qR_T(i,j,k,comp)-qL_M(i,j,k,comp)) + std::abs(qR_M(i+1,j,k,comp)-qL_T(i+1,j,k,comp))),
//                           amrex::min(std::abs(qR_M(i,j,k,comp)-qL_M(i,j,k,comp)) + std::abs(qR_M(i+1,j,k,comp)-qL_T(i+1,j,k,comp)),
//                                      std::abs(qR_T(i,j,k,comp)-qL_M(i,j,k,comp)) + std::abs(qR_M(i+1,j,k,comp)-qL_M(i+1,j,k,comp))));

//         TBV_T =amrex::min(amrex::min(std::abs(qR_M(i,j,k,comp)-qL_T(i,j,k,comp)) + std::abs(qR_T(i+1,j,k,comp)-qL_M(i+1,j,k,comp)),
//                                      std::abs(qR_T(i,j,k,comp)-qL_T(i,j,k,comp)) + std::abs(qR_T(i+1,j,k,comp)-qL_T(i+1,j,k,comp))),
//                           amrex::min(std::abs(qR_M(i,j,k,comp)-qL_T(i,j,k,comp)) + std::abs(qR_T(i+1,j,k,comp)-qL_T(i+1,j,k,comp)),
//                                      std::abs(qR_T(i,j,k,comp)-qL_T(i,j,k,comp)) + std::abs(qR_T(i+1,j,k,comp)-qL_M(i+1,j,k,comp))));

//         if(delta < C < 1.0_rt-delta && (q(i+1,j,k,comp)-q(i,j,k,comp))*(q(i,j,k,comp)-q(i-1,j,k,comp))>0.0_rt 
//                 && TBV_T < TBV_M){
//             qL_M(i,j,k,comp) = qL_T(i,j,k,comp);
//             qR_M(i+1,j,k,comp) = qR_T(i+1,j,k,comp);
//         }
//     }
// }

// AMREX_GPU_DEVICE
// inline
// void
// EMM_TBV_y(int i, int j, int k,
//              amrex::Array4<amrex::Real> const& qR_M, 
//              amrex::Array4<amrex::Real> const& qL_M, 
//              amrex::Array4<amrex::Real> const& qL_T,
//              amrex::Array4<amrex::Real> const& qR_T,
//              amrex::Array4<amrex::Real> const& q, const int nprim) noexcept
// {
//     amrex::Real Beta=1.6_rt;
//     amrex::Real epsilon=1E-8_rt;
//     amrex::Real delta=1E-4_rt;
//     amrex::Real TBV_T;
//     amrex::Real TBV_M;
//     for(int comp=0; comp<nprim; comp++){
//         amrex::Real qmin = amrex::min(q(i,j-1,k,comp),q(i,j+1,k,comp));
//         amrex::Real qmax = amrex::max(q(i,j-1,k,comp),q(i,j+1,k,comp))-qmin;
//         amrex::Real C = (q(i,j,k,comp)-qmin+epsilon)/(qmax+epsilon);
        
//         //Compute total boundary variations TBV for each cell
//         TBV_M =amrex::min(amrex::min(std::abs(qR_M(i,j,k,comp)-qL_M(i,j,k,comp)) + std::abs(qR_M(i,j+1,k,comp)-qL_M(i,j+1,k,comp)),
//                                      std::abs(qR_T(i,j,k,comp)-qL_M(i,j,k,comp)) + std::abs(qR_M(i,j+1,k,comp)-qL_T(i,j+1,k,comp))),
//                           amrex::min(std::abs(qR_M(i,j,k,comp)-qL_M(i,j,k,comp)) + std::abs(qR_M(i,j+1,k,comp)-qL_T(i,j+1,k,comp)),
//                                      std::abs(qR_T(i,j,k,comp)-qL_M(i,j,k,comp)) + std::abs(qR_M(i,j+1,k,comp)-qL_M(i,j+1,k,comp))));
        
//         TBV_T =amrex::min(amrex::min(std::abs(qR_M(i,j,k,comp)-qL_T(i,j,k,comp)) + std::abs(qR_T(i,j+1,k,comp)-qL_M(i,j+1,k,comp)),
//                                      std::abs(qR_T(i,j,k,comp)-qL_T(i,j,k,comp)) + std::abs(qR_T(i,j+1,k,comp)-qL_T(i,j+1,k,comp))),
//                           amrex::min(std::abs(qR_M(i,j,k,comp)-qL_T(i,j,k,comp)) + std::abs(qR_T(i,j+1,k,comp)-qL_T(i,j+1,k,comp)),
//                                      std::abs(qR_T(i,j,k,comp)-qL_T(i,j,k,comp)) + std::abs(qR_T(i,j+1,k,comp)-qL_M(i,j+1,k,comp))));
//         if(delta < C < 1.0_rt-delta && (q(i,j+1,k,comp)-q(i,j,k,comp))*(q(i,j,k,comp)-q(i,j-1,k,comp))>0.0_rt 
//                 && TBV_T < TBV_M){
//             qL_M(i,j,k,comp) = qL_T(i,j,k,comp);
//             qR_M(i,j+1,k,comp) = qR_T(i,j+1,k,comp);
//         }
//     }
// }

// AMREX_GPU_DEVICE
// inline
// void
// EMM_TBV_z(int i, int j, int k,
//              amrex::Array4<amrex::Real> const& qR_M, 
//              amrex::Array4<amrex::Real> const& qL_M, 
//              amrex::Array4<amrex::Real> const& qL_T,
//              amrex::Array4<amrex::Real> const& qR_T,
//              amrex::Array4<amrex::Real> const& q, const int nprim) noexcept
// {
//     amrex::Real Beta=1.6_rt;
//     amrex::Real epsilon=1E-8_rt;
//     amrex::Real delta=1E-4_rt;
//     amrex::Real TBV_T;
//     amrex::Real TBV_M;
//     for(int comp=0; comp<nprim; comp++){
//         amrex::Real qmin = amrex::min(q(i,j,k-1,comp),q(i,j,k+1,comp));
//         amrex::Real qmax = amrex::max(q(i,j,k-1,comp),q(i,j,k+1,comp))-qmin;
//         amrex::Real C = (q(i,j,k,comp)-qmin+epsilon)/(qmax+epsilon);
        
//         //Compute total boundary variations TBV for each cell
//         TBV_M =amrex::min(amrex::min(std::abs(qR_M(i,j,k,comp)-qL_M(i,j,k,comp)) + std::abs(qR_M(i,j,k+1,comp)-qL_M(i,j,k+1,comp)),
//                                      std::abs(qR_T(i,j,k,comp)-qL_M(i,j,k,comp)) + std::abs(qR_M(i,j,k+1,comp)-qL_T(i,j,k+1,comp))),
//                           amrex::min(std::abs(qR_M(i,j,k,comp)-qL_M(i,j,k,comp)) + std::abs(qR_M(i,j,k+1,comp)-qL_T(i,j,k+1,comp)),
//                                      std::abs(qR_T(i,j,k,comp)-qL_M(i,j,k,comp)) + std::abs(qR_M(i,j,k+1,comp)-qL_M(i,j,k+1,comp))));
        
//         TBV_T =amrex::min(amrex::min(std::abs(qR_M(i,j,k,comp)-qL_T(i,j,k,comp)) + std::abs(qR_T(i,j,k+1,comp)-qL_M(i,j,k+1,comp)),
//                                      std::abs(qR_T(i,j,k,comp)-qL_T(i,j,k,comp)) + std::abs(qR_T(i,j,k+1,comp)-qL_T(i,j,k+1,comp))),
//                           amrex::min(std::abs(qR_M(i,j,k,comp)-qL_T(i,j,k,comp)) + std::abs(qR_T(i,j,k+1,comp)-qL_T(i,j,k+1,comp)),
//                                      std::abs(qR_T(i,j,k,comp)-qL_T(i,j,k,comp)) + std::abs(qR_T(i,j,k+1,comp)-qL_M(i,j,k+1,comp))));
//         if(delta < C < 1.0_rt-delta && (q(i,j,k+1,comp)-q(i,j,k,comp))*(q(i,j,k,comp)-q(i,j,k-1,comp))>0.0_rt 
//                 && TBV_T < TBV_M){
//             qL_M(i,j,k,comp) = qL_T(i,j,k,comp);
//             qR_M(i,j,k+1,comp) = qR_T(i,j,k+1,comp);
//         }
//     }
// }
#endif
