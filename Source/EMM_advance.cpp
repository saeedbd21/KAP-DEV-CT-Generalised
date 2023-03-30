
#include "EMM.H"
#include "EMM_hydro_K.H"
#include "EMM_K.H"
#include "EMM_eos.H"

using namespace amrex;

Real
EMM::advance (Real time, Real dt, int iteration, int ncycle)
{
    BL_PROFILE("EMM::advance()");

    for (int i = 0; i < num_state_data_types; ++i) {
        state[i].allocOldData();
        state[i].swapTimeLevels(dt);
    }

    MultiFab& S_new = get_new_data(State_Type);
    MultiFab& S_old = get_old_data(State_Type);
    MultiFab dSdt(grids,dmap,NUM_STATE,0,MFInfo(),Factory());
    MultiFab Sborder(grids,dmap,NUM_STATE,NUM_GROW,MFInfo(),Factory());

    FluxRegister* fr_as_crse = nullptr;
    if (do_reflux && level < parent->finestLevel()) {
        EMM& fine_level = getLevel(level+1);
        fr_as_crse = fine_level.flux_reg.get();
    }

    FluxRegister* fr_as_fine = nullptr;
    if (do_reflux && level > 0) {
        fr_as_fine = flux_reg.get();
    }

    if (fr_as_crse) {
        fr_as_crse->setVal(0.0_rt);
    }

    //1st order Time Integration
    // FillPatch(*this, Sborder, NUM_GROW, time, State_Type, 0, NUM_STATE);
    // compute_dSdt(Sborder, dSdt, dt, fr_as_crse, fr_as_fine, time);
    // MultiFab::LinComb(S_new, 1.0_rt, Sborder, 0, dt, dSdt, 0, 0, NUM_STATE, 0);
    // compute_thermodynamics(S_new, Sborder);

    // Time Integration with Runge-Kutta or Euler Half Time-Stepping
    // Half Time Stepping:
    FillPatch(*this, Sborder, NUM_GROW, time, State_Type, 0, NUM_STATE);
    compute_dSdt(Sborder, dSdt, 0.5_rt*dt, fr_as_crse, fr_as_fine, time);
    MultiFab::LinComb(S_new, 1.0_rt, Sborder, 0, 0.5_rt*dt, dSdt, 0, 0, NUM_STATE, 0);
    compute_thermodynamics(S_new, Sborder);

    FillPatch(*this, Sborder, NUM_GROW, time+0.5_rt*dt, State_Type, 0, NUM_STATE);
    compute_dSdt(Sborder, dSdt, dt, fr_as_crse, fr_as_fine, time);
    MultiFab::LinComb(S_new, 1.0_rt, S_old, 0, dt, dSdt, 0, 0, NUM_STATE, 0);
    compute_thermodynamics(S_new, Sborder);
    
    //Runge Kutta 2:
    // RK2 stage 1
    // FillPatch(*this, Sborder, NUM_GROW, time, State_Type, 0, NUM_STATE);
    // compute_dSdt(Sborder, dSdt, dt, fr_as_crse, fr_as_fine, time);
    // // U^* = U^n + dt*dUdt^n
    // MultiFab::LinComb(S_new, 1.0_rt, Sborder, 0, dt, dSdt, 0, 0, NUM_STATE, 0);

    // RK2 stage 2
    // After fillpatch Sborder = U^n+dt*dUdt^n
    // FillPatch(*this, Sborder, NUM_GROW, time+dt, State_Type, 0, NUM_STATE);
    // compute_dSdt(Sborder, dSdt, 0.5_rt*dt, fr_as_crse, fr_as_fine, time);
    // // S_new = 0.5*(Sborder+S_old) = U^n + 0.5*dt*dUdt^n
    // MultiFab::LinComb(S_new, 0.5_rt, Sborder, 0, 0.5_rt, S_old, 0, 0, NUM_STATE, 0);
    // // S_new += 0.5*dt*dSdt
    // MultiFab::Saxpy(S_new, 0.5_rt*dt, dSdt, 0, 0, NUM_STATE, 0);
    // We now have S_new = U^{n+1} = (U^n+0.5*dt*dUdt^n) + 0.5*dt*dUdt^*

    //SSP RK3 stage 1
    // FillPatch(*this, Sborder, NUM_GROW, time, State_Type, 0, NUM_STATE);
    // compute_dSdt(Sborder, dSdt, dt, fr_as_crse, fr_as_fine, time);
    // MultiFab::LinComb(S_new, 1.0_rt, Sborder, 0, dt, dSdt, 0, 0, NUM_STATE, 0);

    // //SSP RK3 stage 2
    // // After fillpatch Sborder = U^n+dt*dUdt^n
    // FillPatch(*this, Sborder, NUM_GROW, time+dt, State_Type, 0, NUM_STATE);
    // compute_dSdt(Sborder, dSdt, dt, fr_as_crse, fr_as_fine, time);
    // MultiFab::LinComb(S_new, 0.25_rt, Sborder, 0, 0.75_rt, S_old, 0, 0, NUM_STATE, 0);
    // MultiFab::Saxpy(S_new, 0.25_rt*dt, dSdt, 0, 0, NUM_STATE, 0);

    // //SSP RK3 stage 3
    // FillPatch(*this, Sborder, NUM_GROW, time+0.5_rt*dt, State_Type, 0, NUM_STATE);
    // compute_dSdt(Sborder, dSdt, dt, fr_as_crse, fr_as_fine, time);
    // MultiFab::LinComb(S_new, 2.0_rt/3.0_rt, Sborder, 0, 1.0_rt/3.0_rt, S_old, 0, 0, NUM_STATE, 0);
    // MultiFab::Saxpy(S_new, (2.0_rt/3.0_rt)*dt, dSdt, 0, 0, NUM_STATE, 0);

    return dt;
}

void
EMM::compute_dSdt (MultiFab& S, MultiFab& dSdt, Real dt,
                   FluxRegister* fr_as_crse, FluxRegister* fr_as_fine, Real time)
{
    BL_PROFILE("EMM::compute_dSdt()");

    const auto dx = geom.CellSizeArray();
    const auto dxinv = geom.InvCellSizeArray();
    const auto geomdata = geom.data();
    //auto dtdxinv = dt*dxinv;
    const int ncomp = NUM_STATE; //const int ncomp = NUM_STATE - 1;
    const int nprim = NPRIM;

    //Defined on the faces thanks to amrex::convert and IntVect::TheDimensionVector(idim)
    Array<MultiFab,AMREX_SPACEDIM> fluxes;
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        fluxes[idim].define(amrex::convert(S.boxArray(),IntVect::TheDimensionVector(idim)),
                            S.DistributionMap(), ncomp, 0);
    }

    Array<MultiFab,AMREX_SPACEDIM> qL; // May need to expand to include ghost cells
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        qL[idim].define(amrex::convert(S.boxArray(),IntVect::TheDimensionVector(idim)),
                            S.DistributionMap(), ncomp, 1);
    }

    Array<MultiFab,AMREX_SPACEDIM> qR; // May need to expand to include ghost cells
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        qR[idim].define(amrex::convert(S.boxArray(),IntVect::TheDimensionVector(idim)),
                            S.DistributionMap(), ncomp, 1);
    }

    // Array<MultiFab,AMREX_SPACEDIM> qL_THINC; // May need to expand to include ghost cells
    // for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
    //     qL_THINC[idim].define(amrex::convert(S.boxArray(),IntVect::TheDimensionVector(idim)),
    //                         S.DistributionMap(), ncomp, 1);
    // }

    // Array<MultiFab,AMREX_SPACEDIM> qR_THINC; // May need to expand to include ghost cells
    // for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
    //     qR_THINC[idim].define(amrex::convert(S.boxArray(),IntVect::TheDimensionVector(idim)),
    //                         S.DistributionMap(), ncomp, 1);
    // }

    Array<MultiFab,AMREX_SPACEDIM> US;
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        US[idim].define(amrex::convert(S.boxArray(),IntVect::TheDimensionVector(idim)),
                            S.DistributionMap(), 1, 0);
    }

    Array<MultiFab,AMREX_SPACEDIM> VS;
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        VS[idim].define(amrex::convert(S.boxArray(),IntVect::TheDimensionVector(idim)),
                            S.DistributionMap(), 1, 0);
    }

    Array<MultiFab,AMREX_SPACEDIM> WS;
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        WS[idim].define(amrex::convert(S.boxArray(),IntVect::TheDimensionVector(idim)),
                            S.DistributionMap(), 1, 0);
    }

    // Cell-centered
    Array<MultiFab,AMREX_SPACEDIM> H;
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        H[idim].define(S.boxArray(), S.DistributionMap(), ncomp, 0);
    }
    Array<MultiFab,AMREX_SPACEDIM> K;
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        K[idim].define(S.boxArray(), S.DistributionMap(), ncomp, 0);
    }
    Array<MultiFab,AMREX_SPACEDIM> M;
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        M[idim].define(S.boxArray(), S.DistributionMap(), ncomp, 0);
    }

    MultiFab qmf(S.boxArray(), S.DistributionMap(), nprim, NUM_GROW);
    // MultiFab& thermo = get_new_data(Thermo_Type);
    Parm const* lparm = parm.get();

    FArrayBox qtmp;
    for (MFIter mfi(S); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.tilebox();

        // auto const& sfab = S.array(mfi);
        auto sfab = S.array(mfi);
        // auto thermofab = thermo.array(mfi);
        auto const& dsdtfab = dSdt.array(mfi);

        AMREX_D_TERM(auto const& fxfab = fluxes[0].array(mfi);,
                     auto const& fyfab = fluxes[1].array(mfi);,
                     auto const& fzfab = fluxes[2].array(mfi););

        // Reconstructed States
        AMREX_D_TERM(auto const& qLxfab = qL[0].array(mfi);, // May need to expand to include ghost cells
                     auto const& qLyfab = qL[1].array(mfi);,
                     auto const& qLzfab = qL[2].array(mfi););

        AMREX_D_TERM(auto const& qRxfab = qR[0].array(mfi);, // May need to expand to include ghost cells
                     auto const& qRyfab = qR[1].array(mfi);,
                     auto const& qRzfab = qR[2].array(mfi););

        // Reconstructed States
        // AMREX_D_TERM(auto const& qLxfab_THINC = qL_THINC[0].array(mfi);, // May need to expand to include ghost cells
        //              auto const& qLyfab_THINC = qL_THINC[1].array(mfi);,
        //              auto const& qLzfab_THINC = qL_THINC[2].array(mfi););

        // AMREX_D_TERM(auto const& qRxfab_THINC = qR_THINC[0].array(mfi);, // May need to expand to include ghost cells
        //              auto const& qRyfab_THINC = qR_THINC[1].array(mfi);,
        //              auto const& qRzfab_THINC = qR_THINC[2].array(mfi););

        // Face velocities provied by the Riemann Solver
        AMREX_D_TERM(auto const& USxfab = US[0].array(mfi);,
                     auto const& USyfab = US[1].array(mfi);,
                     auto const& USzfab = US[2].array(mfi););

        AMREX_D_TERM(auto const& VSxfab = VS[0].array(mfi);,
                     auto const& VSyfab = VS[1].array(mfi);,
                     auto const& VSzfab = VS[2].array(mfi););

        AMREX_D_TERM(auto const& WSxfab = WS[0].array(mfi);,
                     auto const& WSyfab = WS[1].array(mfi);,
                     auto const& WSzfab = WS[2].array(mfi););

        //
        AMREX_D_TERM(auto const& Hxfab = H[0].array(mfi);,
                     auto const& Hyfab = H[1].array(mfi);,
                     auto const& Hzfab = H[2].array(mfi););

        AMREX_D_TERM(auto const& Kxfab = K[0].array(mfi);,
                     auto const& Kyfab = K[1].array(mfi);,
                     auto const& Kzfab = K[2].array(mfi););

        AMREX_D_TERM(auto const& Mxfab = M[0].array(mfi);,
                     auto const& Myfab = M[1].array(mfi);,
                     auto const& Mzfab = M[2].array(mfi););

        // Primitive Variable MultiFab
        auto const& q = qmf.array(mfi);

        const Box& bxg2 = amrex::grow(bx,2);
        // qtmp.resize(bxg2, nprim);
        // Elixir qeli = qtmp.elixir();
        // auto const& q = qtmp.array();

        // Print() << "Computing Primitive Variables" << "\n";
        amrex::ParallelFor(bxg2, 
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            EMM_ctoprim(i, j, k, sfab, q, geomdata, *lparm, time);
        });

        // Print() << "Computing non-conservative terms" << "\n";
        amrex::ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            amrex::Real alpha1 = q(i,j,k,QALPHA); amrex::Real alpha2 = 1.0 - q(i,j,k,QALPHA);
            amrex::Real c1sq = 0;
            amrex::Real c2sq = 0;
            amrex::Real T1 = 0;
            amrex::Real T2 = 0;
            if(lparm->tabulated1 == 0){
                c1sq = parm->eos_gamma1*(q(i,j,k,QPRES)+parm->eos_pinf1)/q(i,j,k,QRHO1);
            }else{
                T1 = T_finder(alpha1, q(i,j,k,QRHO1), q(i,j,k,QPRES), sfab(i,j,k,GT1),*lparm, 1);
                c1sq = std::pow(TPF(OSOS, T1, std::log10(q(i,j,k,QPRES)), *lparm, 1),2.0);
            }
            if(lparm->tabulated2 == 0){
                c2sq = -1.0/std::pow(q(i,j,k,QRHO2), 2.0)*parm->gamma0*parm->pinf0_1*(q(i,j,k,QPRES)+parm->gamma0*parm->pinf0*(1.0-parm->b1)/(parm->gamma0-parm->b1))*
                ((parm->gamma0-1.0)/((parm->gamma0-1.0)*parm->cv0-parm->gamma0*parm->pinf0_1*(1.0/q(i,j,k,QRHO2)-((parm->b1/q(i,j,k,QRHO2)+parm->b0_1))))+1.0/parm->cv0)
                +((q(i,j,k,QPRES)+(parm->gamma0*parm->pinf0*(1.0-parm->b1)/(parm->gamma0-parm->b1)))/((parm->gamma0-1.0)*parm->cv0-parm->gamma0*parm->pinf0_1*(1.0/q(i,j,k,QRHO2)-(parm->b1/q(i,j,k,QRHO2)+parm->b0_1))))
                *(1.0/std::pow(q(i,j,k,QRHO2), 2.0)*(parm->gamma0-parm->b1)*(parm->gamma0-1.0)*parm->cv0/(1.0/q(i,j,k,QRHO2)-((parm->b1/q(i,j,k,QRHO2)+parm->b0_1))));
                // c2sq = parm->eos_gamma2*(q(i,j,k,QPRES)+parm->eos_pinf2)/q(i,j,k,QRHO2);
            }else{


                T2 = T_finder(alpha1, q(i,j,k,QRHO2), q(i,j,k,QPRES), sfab(i,j,k,GT2),*lparm, 2);

                c2sq = std::pow(TPF(OSOS, T2, std::log10(q(i,j,k,QPRES)), *lparm, 2),2.0);


            }
            // amrex::Real c1sq = parm->eos_gamma1*(q(i,j,k,QPRES)+parm->eos_pinf1)/q(i,j,k,QRHO1);
            // amrex::Real c2sq = parm->eos_gamma2*(q(i,j,k,QPRES)+parm->eos_pinf2)/q(i,j,k,QRHO2);
            amrex::Real K = (q(i,j,k,QRHO2)*c2sq - q(i,j,k,QRHO1)*c1sq)/( (q(i,j,k,QRHO2)*c2sq)/(1.0_rt - q(i,j,k,QALPHA)) + (q(i,j,k,QRHO1)*c1sq)/q(i,j,k,QALPHA));
            // amrex::Real K = (q(i,j,k,QALPHA)*(1.0-q(i,j,k,QALPHA))*(q(i,j,k,QRHO2)*c2sq-q(i,j,k,QRHO1)*c2sq) )/(q(i,j,k,QALPHA)*q(i,j,k,QRHO2)*c2sq+(1.0-q(i,j,k,QALPHA))*q(i,j,k,QRHO1)*c1sq);
            
            amrex::Real rho1 = q(i,j,k,QRHO1); amrex::Real rho2 = q(i,j,k,QRHO2);
            amrex::Real mrho = alpha1*rho1+alpha2*rho2;
            amrex::Real cpsq = (1.0)/(mrho*( alpha1/(rho1*c1sq) + alpha2/(rho2*c2sq) ));
            amrex::Real K1 = 0.0;
            if(time > parm->ktime){
                K1 = alpha1*( (mrho*cpsq)/(rho1*c1sq) - 1.0);
                //K1 = 0.0;
            }
            // if(alpha1 < 0.0){K1 = 0.0;}

            Hxfab(i, j, k,UARHO1) = 0.0_rt;
            Hxfab(i, j, k,UARHO2) = 0.0_rt;
            Hxfab(i, j, k,UMX) = 0.0_rt;
            Hxfab(i, j, k,UMY) = 0.0_rt;
            Hxfab(i, j, k,UMZ) = 0.0_rt;
            Hxfab(i, j, k,URHOE) = 0.0_rt;
            Hxfab(i, j, k,GALPHA) = - q(i, j, k,QALPHA) - K1;

            Kyfab(i, j, k,UARHO1) = 0.0_rt;
            Kyfab(i, j, k,UARHO2) = 0.0_rt;
            Kyfab(i, j, k,UMX) = 0.0_rt;
            Kyfab(i, j, k,UMY) = 0.0_rt;
            Kyfab(i, j, k,UMZ) = 0.0_rt;
            Kyfab(i, j, k,URHOE) = 0.0_rt;
            Kyfab(i, j, k,GALPHA) = - q(i, j, k,QALPHA) - K1;

#if (AMREX_SPACEDIM == 3)
            Mzfab(i, j, k,UARHO1) = 0.0_rt;
            Mzfab(i, j, k,UARHO2) = 0.0_rt;
            Mzfab(i, j, k,UMX) = 0.0_rt;
            Mzfab(i, j, k,UMY) = 0.0_rt;
            Mzfab(i, j, k,UMZ) = 0.0_rt;
            Mzfab(i, j, k,URHOE) = 0.0_rt;
            Mzfab(i, j, k,GALPHA) = - q(i, j, k,QALPHA) - K1;
#endif
        });
        // const Box& bxg1 = amrex::grow(bx,1);
        
        // qLtmp.resize(bxg1,nprim);
        // Elixir qLeli = qLtmp.elixir();
        // auto const& qL = qLtmp.array();
        
        // qRtmp.resize(bxg1,nprim);
        // Elixir qReli = qRtmp.elixir();
        // auto const& qR = qRtmp.array();

        // x-direction
        int cdir = 0;
        const Box& xslpbx = amrex::grow(bx, cdir, 1); // i j k running on the cell centers here
        // Print() << "Computing PMUSCL and THINC X" << "\n";
        amrex::ParallelFor(xslpbx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            EMM_PMUSCL_reconstruct_x(i, j, k, qLxfab, qRxfab, q, dt, geomdata, dxinv, *lparm);
            //EMM_THINC_reconstruct_x(i, j, k, qLxfab_THINC, qRxfab_THINC, q, nprim);
        });
        // amrex::ParallelFor(xslpbx,
        // [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        // {
        //     //EMM_PMUSCL_reconstruct_x(i, j, k, qLxfab, qRxfab, q, dt, dxinv, *lparm);
        //     EMM_THINC_reconstruct_x(i, j, k, qLxfab_THINC, qRxfab_THINC, q, nprim);
        // });
        // Print() << "Computing TBV X" << "\n";
        // amrex::ParallelFor(bx,
        // [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        // {
        //     EMM_TBV_x(i, j, k, qLxfab, qRxfab, qLxfab_THINC, qRxfab_THINC, q, nprim);
        // });
        // Print() << "Computing Riemann X" << "\n";
        const Box& xflxbx = amrex::surroundingNodes(bx,cdir); // i j k running on the face centers here
        amrex::ParallelFor(xflxbx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            EMM_riemann_x(i, j, k, fxfab, USxfab, VSxfab, WSxfab, qLxfab, qRxfab, q, sfab, *lparm);
        });

        // y-direction
        cdir = 1;
        const Box& yslpbx = amrex::grow(bx, cdir, 1);
        //Print() << "Computing PMUSCL and THINC Y" << "\n";
        amrex::ParallelFor(yslpbx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            EMM_PMUSCL_reconstruct_y(i, j, k, qLyfab, qRyfab, q, dt, geomdata, dxinv, *lparm);
            //EMM_THINC_reconstruct_y(i, j, k, qLyfab_THINC, qRyfab_THINC, q, nprim);
        });

        // Print() << "Computing TBV Y" << "\n";
        // amrex::ParallelFor(bx,
        // [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        // {
        //     EMM_TBV_y(i, j, k, qLyfab, qRyfab, qLyfab_THINC, qRyfab_THINC, q, nprim);
        // });

        //Print() << "Computing Riemann Y" << "\n";
        const Box& yflxbx = amrex::surroundingNodes(bx,cdir);
        amrex::ParallelFor(yflxbx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            EMM_riemann_y(i, j, k, fyfab, USyfab, VSyfab, WSyfab, qLyfab, qRyfab, q, sfab, *lparm);
        });

        // z-direction
#if (AMREX_SPACEDIM == 3)
        cdir = 2;
        const Box& zslpbx = amrex::grow(bx, cdir, 1);
        //Print() << "Computing PMUSCL and THINC Z" << "\n";
        amrex::ParallelFor(zslpbx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            EMM_PMUSCL_reconstruct_z(i, j, k, qLzfab, qRzfab, q, dt, geomdata, dxinv, *lparm);
            //EMM_THINC_reconstruct_z(i, j, k, qLzfab_THINC, qRzfab_THINC, q, nprim);
        });
        // Print() << "Computing TBV Z" << "\n";
        // amrex::ParallelFor(bx,
        // [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        // {
        //     EMM_TBV_z(i, j, k, qLzfab, qRzfab, qLzfab_THINC, qRzfab_THINC, q, nprim);
        // });
        const Box& zflxbx = amrex::surroundingNodes(bx,cdir);
        //Print() << "Computing Riemann Z" << "\n";
        amrex::ParallelFor(zflxbx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            EMM_riemann_z(i, j, k, fzfab, USzfab, VSzfab, WSzfab, qLzfab, qRzfab, q, sfab, *lparm);
        });
#endif

        amrex::ParallelFor(bx, NV,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {
            EMM_flux_to_dudt(i, j, k, n, dsdtfab,
                            AMREX_D_DECL(fxfab,fyfab,fzfab),
                            AMREX_D_DECL(Hxfab,Hyfab,Hzfab),
                            AMREX_D_DECL(Kxfab,Kyfab,Kzfab), 
                            AMREX_D_DECL(Mxfab,Myfab,Mzfab),
                            AMREX_D_DECL(USxfab,USyfab,USzfab),
                            AMREX_D_DECL(VSxfab,VSyfab,VSzfab),
                            AMREX_D_DECL(WSxfab,WSyfab,WSzfab), dxinv);
        });

        // Print() << "Axisymmetric Terms" << "\n";
       if (parm->cordP > 0) {
    //             amrex::Print() << "cccccc:          " << parm->cordP << "\n";
    // amrex::Real s = 0.0;

    // std::cin >> s;
        amrex::ParallelFor(bx,
       [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
       {
            EMM_axisymmetricAdd(i, j, k, dt, dsdtfab, q, sfab, geomdata, *lparm, time);
       });
       }

        // don't have to do this, but we could
        // qeli.clear(); // don't need them anymore
        // qLeli.clear();
        // qReli.clear();
    }

    if (fr_as_crse) {
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
            const Real dA = (idim == 0) ? dx[1]*dx[2] : ((idim == 1) ? dx[0]*dx[2] : dx[0]*dx[1]);
            const Real scale = -dt*dA;
            fr_as_crse->CrseInit(fluxes[idim], idim, 0, 0, NCONS, scale, FluxRegister::ADD);
        }
    }

    if (fr_as_fine) {
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
            const Real dA = (idim == 0) ? dx[1]*dx[2] : ((idim == 1) ? dx[0]*dx[2] : dx[0]*dx[1]);
            const Real scale = dt*dA;
            fr_as_fine->FineAdd(fluxes[idim], idim, 0, 0, NCONS, scale);
        }
    }
}

void
EMM::compute_thermodynamics (MultiFab& Snew, MultiFab& Sborder)
{
    BL_PROFILE("EMM::thermodynamics()");
    //using namespace amrex::literals;

    Parm const* lparm = parm.get();
    const auto geomdata = geom.data();
    // MultiFab& cost = get_new_data(Cost_Type);

#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(Snew,TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        // amrex::Real wt = amrex::second();
        //const Box& bx = mfi.growntilebox(ng);
        const Box& bx = mfi.tilebox();
        auto snewfab = Snew.array(mfi);
        auto soldfab = Sborder.array(mfi);

        amrex::ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
                EMM_thermo(i, j, k, snewfab, soldfab, *lparm);
        });
        // wt = (amrex::second() - wt) / bx.d_numPts();
        // cost[mfi].plus<RunOn::Host>(wt, bx);
    }
}