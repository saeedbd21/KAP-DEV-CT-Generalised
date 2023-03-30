#include "EMM_derive.H"
#include "EMM.H"
#include "EMM_parm.H"

using namespace amrex;

void EMM_derpres (const Box& bx, FArrayBox& pfab, int dcomp, int /*ncomp*/,
                  const FArrayBox& datfab, const Geometry& /*geomdata*/,
                  Real /*time*/, const int* /*bcrec*/, int /*level*/)
{
    auto const dat = datfab.array();
    auto       p    = pfab.array();
    Parm const* parm = EMM::parm.get();
    
    amrex::ParallelFor(bx,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        amrex::Real mrho = dat(i,j,k,1) + dat(i,j,k,2);
        amrex::Real vx = dat(i,j,k,3)/mrho;
        amrex::Real vy = dat(i,j,k,4)/mrho;
        amrex::Real vz = dat(i,j,k,5)/mrho;
        amrex::Real ei = dat(i,j,k,0)/mrho - 0.5_rt*(vx*vx + vy*vy + vz*vz);
        amrex::Real alpha1 = dat(i,j,k,6); amrex::Real alpha2 = 1.0 - dat(i,j,k,6);
        // amrex::Real peos = ( mrho*ei - ( alpha1*parm->eos_gamma1*parm->eos_pinf1/(parm->eos_gamma1-1.0_rt) 
        //     + alpha2*parm->eos_gamma2*parm->eos_pinf2/(parm->eos_gamma2-1.0_rt)) )/( alpha1/(parm->eos_gamma1-1.0_rt)+alpha2/(parm->eos_gamma2-1.0_rt));
        amrex::Real peos = ( mrho*ei - ( alpha1*parm->eos_gamma1*parm->eos_pinf1/(parm->eos_gamma1-1.0_rt) 
            + alpha2*(1.0-dat(i,j,k,2)/(1.0-dat(i,j,k,6))*parm->eos_b0_1)*parm->eos_gamma2*parm->eos_pinf2/(parm->eos_gamma2-1.0_rt)) )/( alpha1/(parm->eos_gamma1-1.0_rt)+alpha2*(1.0-dat(i,j,k,2)/(1.0-dat(i,j,k,6))*parm->eos_b0_1)/(parm->eos_gamma2-1.0_rt));
        p(i,j,k,dcomp) = peos;
    });
}

void EMM_dervel (const Box& bx, FArrayBox& velfab, int dcomp, int /*ncomp*/,
                  const FArrayBox& datfab, const Geometry& /*geomdata*/,
                  Real /*time*/, const int* /*bcrec*/, int /*level*/)
{
    auto const dat = datfab.array();
    auto       vel = velfab.array();
    amrex::ParallelFor(bx,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        vel(i,j,k,dcomp) = dat(i,j,k,0)/(dat(i,j,k,1)+dat(i,j,k,2));
    });
}

void EMM_derden (const Box& bx, FArrayBox& denfab, int dcomp, int /*ncomp*/,
                  const FArrayBox& datfab, const Geometry& /*geomdata*/,
                  Real /*time*/, const int* /*bcrec*/, int /*level*/)
{
    auto const dat = datfab.array();
    auto       den = denfab.array();
    amrex::ParallelFor(bx,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        den(i,j,k,dcomp) = dat(i,j,k,0) + dat(i,j,k,1);
    });
}

void EMM_deralpha2 (const Box& bx, FArrayBox& alphafab, int dcomp, int /*ncomp*/,
                  const FArrayBox& datfab, const Geometry& /*geomdata*/,
                  Real /*time*/, const int* /*bcrec*/, int /*level*/)
{
    auto const dat = datfab.array();
    auto       alpha = alphafab.array();
    amrex::ParallelFor(bx,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        alpha(i,j,k,dcomp) = 1.0_rt - dat(i,j,k,0);
    });
}