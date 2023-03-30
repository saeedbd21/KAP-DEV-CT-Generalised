#ifndef EMM_PARM_H_
#define EMM_PARM_H_

#include <AMReX_REAL.H>
#include <AMReX_GpuMemory.H>

using namespace amrex::literals;

struct Parm
    : public amrex::Gpu::Managed
{
    int coord_type = 0;
    amrex::Real eos_gamma1 = 4.4_rt;
    amrex::Real eos_gamma2 = 1.4_rt;
    amrex::Real eos_pinf1 = 6.0e8_rt;
    amrex::Real eos_pinf2 = 0.0_rt;
    amrex::Real eos_mu = 28.97_rt;  // mean molecular weight
    amrex::Real alpha_min = 1.0e-6_rt;

    amrex::Real smallr = 1.e-19_rt;
    amrex::Real smallp = 1.e-10_rt;

    // Complex Thermodynamics Phase 1
    amrex::Real TMINT1 = 0.0;
    amrex::Real PMINT1 = 0.0;
    amrex::Real TMAXT1 = 0.0;
    amrex::Real PMAXT1 = 0.0;
    amrex::Real DT1 = 0.0;
    amrex::Real DP1 = 0.0;

    int mmTP1 = 121;
    int nnTP1 = 376;
    int tableRows1 = 45496;
    int tableColumns1 = 10;

    // Complex Thermodynamics Phase 2
    amrex::Real TMINT2 = 0.0;
    amrex::Real PMINT2 = 0.0;
    amrex::Real TMAXT2 = 0.0;
    amrex::Real PMAXT2 = 0.0;
    amrex::Real DT2 = 0.0;
    amrex::Real DP2 = 0.0;

    int mmTP2 = 853;
    int nnTP2 = 373;
    int tableRows2 = 318169;
    int tableColumns2 = 10;

    int tabulated1 = 0;
    int tabulated2 = 0;
    

    // GAMMA0 = 1.0147
    // CV0 = 4014d0
    // q00 = -1112426d0
    // pinf0_1 = -471025d0
    // pinf0 = 307078403d0
    // b1 = -0.6050d0
    // b0_1 =  1.5196d-3


    // NASG The only thing needs to be change (if) / can be transferred to the input file
    amrex::Real gamma0 = 1.19;
    amrex::Real cv0 = 3610.0;
    amrex::Real q0 = -1177788;
    amrex::Real pinf0_1 = 0.0;
    amrex::Real pinf0 = 6.2178e8; //!307078403d0;
    amrex::Real b1 = 0.0;
    amrex::Real b0_1 =  6.7212e-4;
    amrex::Real eos_b0_1 =  6.7212e-4;
    amrex::Real eos_q0 =  -1177788;

    // SG in ENASG FORM
    // amrex::Real gamma0 = 4.4;
    // amrex::Real cv0 = 612.326824824294;
    // amrex::Real q0 = 0.0;
    // amrex::Real pinf0_1 = 0.0;
    // amrex::Real pinf0 = 6E8;
    // amrex::Real b1 = 0.0;
    // amrex::Real b0_1 = 0E0;
    // amrex::Real eos_b0_1 =  0E0;
    // amrex::Real eos_q0 =  0.0;


    // NASG  Orig
    // amrex::Real gamma0 = 1.19;
    // amrex::Real cv0 = 3610.0;
    // amrex::Real q0 = -1177788;
    // amrex::Real pinf0_1 = 0.0;
    // amrex::Real pinf0 = 7.028e8; //!307078403d0;
    // amrex::Real b1 = 0.0;
    // amrex::Real b0_1 =  6.61e-4;
    // amrex::Real eos_b0_1 =  6.61e-4;
    // amrex::Real eos_q0 =  -1177788;


    std::vector<amrex::Real> t_pulse;
    std::vector<amrex::Real> p_pulse;

    amrex::Real ktime = 0.0;
    amrex::Real cordP = 1;

    void Initialize ();
};

#endif
