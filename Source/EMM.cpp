
#include <EMM.H>
#include <EMM_K.H>
#include <EMM_tagging.H>
#include <EMM_parm.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_ParmParse.H>

#include <climits>

using namespace amrex;

constexpr int EMM::NUM_GROW;

BCRec     EMM::phys_bc;

int       EMM::verbose = 0;
IntVect   EMM::hydro_tile_size {AMREX_D_DECL(1024,16,16)};
Real      EMM::cfl       = 0.3_rt;
int       EMM::do_reflux = 1;
int       EMM::refine_max_dengrad_lev   = -1;
Real      EMM::refine_dengrad           = 1.0e10_rt;
Real      EMM::refine_vofgrad           = 1.0e10_rt;

Real      EMM::gravity = 0.0_rt;

// Gas
std::vector<std::vector<double>> EMM::TVEC1T(376, std::vector<double>(121));
std::vector<std::vector<double>> EMM::PVEC1T(376, std::vector<double>(121));
std::vector<std::vector<double>> EMM::RHOVEC1T(376, std::vector<double>(121));
std::vector<std::vector<double>> EMM::EVEC1T(376, std::vector<double>(121));
std::vector<std::vector<double>> EMM::SOSVEC1T(376, std::vector<double>(121));

// std::vector<std::vector<double>> EMM::TVEC1T(374, std::vector<double>(100));
// std::vector<std::vector<double>> EMM::PVEC1T(374, std::vector<double>(100));
// std::vector<std::vector<double>> EMM::RHOVEC1T(374, std::vector<double>(100));
// std::vector<std::vector<double>> EMM::EVEC1T(374, std::vector<double>(100));
// std::vector<std::vector<double>> EMM::SOSVEC1T(374, std::vector<double>(100));


// Liquid
// std::vector<std::vector<double>> EMM::TVEC2T(2181, std::vector<double>(745));
// std::vector<std::vector<double>> EMM::PVEC2T(2181, std::vector<double>(745));
// std::vector<std::vector<double>> EMM::RHOVEC2T(2181, std::vector<double>(745));
// std::vector<std::vector<double>> EMM::EVEC2T(2181, std::vector<double>(745));
// std::vector<std::vector<double>> EMM::SOSVEC2T(2181, std::vector<double>(745));

// std::vector<std::vector<double>> EMM::TVEC2T(2181, std::vector<double>(961));
// std::vector<std::vector<double>> EMM::PVEC2T(2181, std::vector<double>(961));
// std::vector<std::vector<double>> EMM::RHOVEC2T(2181, std::vector<double>(961));
// std::vector<std::vector<double>> EMM::EVEC2T(2181, std::vector<double>(961));
// std::vector<std::vector<double>> EMM::SOSVEC2T(2181, std::vector<double>(961));

std::vector<std::vector<double>> EMM::TVEC2T(853, std::vector<double>(373));
std::vector<std::vector<double>> EMM::PVEC2T(853, std::vector<double>(373));
std::vector<std::vector<double>> EMM::RHOVEC2T(853, std::vector<double>(373));
std::vector<std::vector<double>> EMM::EVEC2T(853, std::vector<double>(373));
std::vector<std::vector<double>> EMM::SOSVEC2T(853, std::vector<double>(373));

EMM::EMM ()
{}

EMM::EMM (Amr&            papa,
          int             lev,
          const Geometry& level_geom,
          const BoxArray& bl,
          const DistributionMapping& dm,
          Real            time)
    : AmrLevel(papa,lev,level_geom,bl,dm,time)
{
    if (do_reflux && level > 0) {
        flux_reg.reset(new FluxRegister(grids,dmap,crse_ratio,level,NCONS));
    }

    buildMetrics();
}

EMM::~EMM ()
{}

void
EMM::init (AmrLevel& old)
{
    auto& oldlev = dynamic_cast<EMM&>(old);

    Real dt_new    = parent->dtLevel(level);
    Real cur_time  = oldlev.state[State_Type].curTime();
    Real prev_time = oldlev.state[State_Type].prevTime();
    Real dt_old    = cur_time - prev_time;
    setTimeLevel(cur_time,dt_old,dt_new);

    MultiFab& S_new = get_new_data(State_Type);
    FillPatch(old,S_new,0,cur_time,State_Type,0,NUM_STATE);
}

void
EMM::init ()
{
    Real dt        = parent->dtLevel(level);
    Real cur_time  = getLevel(level-1).state[State_Type].curTime();
    Real prev_time = getLevel(level-1).state[State_Type].prevTime();
    Real dt_old = (cur_time - prev_time)/static_cast<Real>(parent->MaxRefRatio(level-1));
    setTimeLevel(cur_time,dt_old,dt);

    MultiFab& S_new = get_new_data(State_Type);
    FillCoarsePatch(S_new, 0, cur_time, State_Type, 0, NUM_STATE);
}

void
EMM::initData ()
{
    BL_PROFILE("EMM::initData()");

    const auto geomdata = geom.data();
    MultiFab& S_new = get_new_data(State_Type);
    // MultiFab& Thermo_new = get_new_data(Thermo_Type);

    Parm const* lparm = parm.get();
    //ProbParm const* lprobparm = prob_parm.get();

#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(S_new); mfi.isValid(); ++mfi)
    {
        const Box& box = mfi.validbox();
        auto sfab = S_new.array(mfi);
        // auto thermofab = Thermo_new.array(mfi);

        amrex::ParallelFor(box,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            emm_initdata(i, j, k, sfab, geomdata, *lparm);//, *lprobparm);
        });
    }
}

void
EMM::computeInitialDt (int                    finest_level,
                       int                    sub_cycle,
                       Vector<int>&           n_cycle,
                       const Vector<IntVect>& ref_ratio,
                       Vector<Real>&          dt_level,
                       Real                   stop_time)
{
    //
    // Grids have been constructed, compute dt for all levels.
    //
    if (level > 0) {
        return;
    }
    
    Real dt_0 = std::numeric_limits<Real>::max();
    int n_factor = 1;
    for (int i = 0; i <= finest_level; i++)
    {
        dt_level[i] = getLevel(i).initialTimeStep();
        n_factor   *= n_cycle[i];
        dt_0 = std::min(dt_0,n_factor*dt_level[i]);
    }
    
    //
    // Limit dt's by the value of stop_time.
    //
    const Real eps = 0.001_rt*dt_0;
    Real cur_time  = state[State_Type].curTime();
    if (stop_time >= 0.0_rt) {
        if ((cur_time + dt_0) > (stop_time - eps))
            dt_0 = stop_time - cur_time;
    }
    
    n_factor = 1;
    for (int i = 0; i <= finest_level; i++)
    {
        n_factor *= n_cycle[i];
        dt_level[i] = dt_0/n_factor;
    }
}

void
EMM::computeNewDt (int                    finest_level,
                   int                    sub_cycle,
                   Vector<int>&           n_cycle,
                   const Vector<IntVect>& ref_ratio,
                   Vector<Real>&          dt_min,
                   Vector<Real>&          dt_level,
                   Real                   stop_time,
                   int                    post_regrid_flag)
{
    //
    // We are at the end of a coarse grid timecycle.
    // Compute the timesteps for the next iteration.
    //
    if (level > 0) {
        return;
    }

    for (int i = 0; i <= finest_level; i++)
    {
        dt_min[i] = getLevel(i).estTimeStep();
    }

    if (post_regrid_flag == 1) 
    {
	//
	// Limit dt's by pre-regrid dt
	//
	for (int i = 0; i <= finest_level; i++)
	{
	    dt_min[i] = std::min(dt_min[i],dt_level[i]);
	}
    }
    else 
    {
	//
	// Limit dt's by change_max * old dt
	//
	static Real change_max = 1.1;
	for (int i = 0; i <= finest_level; i++)
	{
	    dt_min[i] = std::min(dt_min[i],change_max*dt_level[i]);
	}
    }
    
    //
    // Find the minimum over all levels
    //
    Real dt_0 = std::numeric_limits<Real>::max();
    int n_factor = 1;
    for (int i = 0; i <= finest_level; i++)
    {
        n_factor *= n_cycle[i];
        dt_0 = std::min(dt_0,n_factor*dt_min[i]);
    }

    //
    // Limit dt's by the value of stop_time.
    //
    const Real eps = 0.001_rt*dt_0;
    Real cur_time  = state[State_Type].curTime();
    if (stop_time >= 0.0_rt) {
        if ((cur_time + dt_0) > (stop_time - eps)) {
            dt_0 = stop_time - cur_time;
        }
    }

    n_factor = 1;
    for (int i = 0; i <= finest_level; i++)
    {
        n_factor *= n_cycle[i];
        dt_level[i] = dt_0/n_factor;
    }
}

void
EMM::post_regrid (int lbase, int new_finest)
{
}

void
EMM::post_timestep (int iteration)
{
    BL_PROFILE("post_timestep");

    if (do_reflux && level < parent->finestLevel()) {
        MultiFab& S = get_new_data(State_Type);
        EMM& fine_level = getLevel(level+1);
        fine_level.flux_reg->Reflux(S, 1.0_rt, 0, 0, NCONS, geom);
    }

    if (level < parent->finestLevel()) {
        avgDown();
    }
}

void
EMM::postCoarseTimeStep (Real time)
{
    BL_PROFILE("postCoarseTimeStep()");

    // This only computes sum on level 0
    if (verbose >= 2) {
        //printTotal();

        const MultiFab& S_new = get_new_data(State_Type);
        const Real cur_time = state[State_Type].curTime();

        // MultiFab alpha(S_new.boxArray(), S_new.DistributionMap(), 1, 1);
        // FillPatch(*this, alpha, alpha.nGrow(), cur_time, State_Type, Alpha, 1, 0);

        Real bubble_radius = 0.0;
        Real Vb = S_new.sum(GALPHA,false);

        // Vb *= 8.0_rt*geom.CellSize()[0]*geom.CellSize()[1]*geom.CellSize()[2];
        //bubble_radius = Vb*geom.CellSize()[0]*geom.CellSize()[1];
        // bubble_radius = 2.0_rt*std::pow(bubble_radius/(M_PI),0.5_rt);
        //bubble_radius = 2.0_rt*std::pow(bubble_radius/(2.0_rt*M_PI),0.5_rt);
        // bubble_radius = std::pow(3.0_rt*Vb/(4.0_rt*M_PI),1.0/3.0);
        // 1D computations:
        // bubble_radius = 0.5_rt*Vb*geom.CellSize()[0];
        
        
        
        bubble_radius = 0.5_rt*Vb*geom.CellSize()[1];


        // bubble_radius = Vb*geom.CellSize()[0]*geom.CellSize()[1];
        // // bubble_radius = 2.0_rt*std::pow(bubble_radius/(M_PI),0.5_rt);
        // bubble_radius = 2.0_rt*std::pow(bubble_radius/(2.0_rt*M_PI),0.5_rt);

        amrex::Print().SetPrecision(18) << "\n[EMM] " << time/1E-6 << " " << bubble_radius/1E-6 << " " << Vb*geom.CellSize()[0]*geom.CellSize()[1] << "\n";
    }
}

void
EMM::printTotal () const
{
    const MultiFab& S_new = get_new_data(State_Type);
    std::array<Real,5> tot;
    for (int comp = 0; comp < 5; ++comp) {
        tot[comp] = S_new.sum(comp,true) * geom.ProbSize();
    }
#ifdef BL_LAZY
    Lazy::QueueReduction( [=] () mutable {
#endif
            // ParallelDescriptor::ReduceRealSum(tot.data(), 5, ParallelDescriptor::IOProcessorNumber());
            // amrex::Print().SetPrecision(17) << "\n[EMM] Total mass       is " << tot[0] << "\n"
            //                                 <<   "      Total x-momentum is " << tot[1] << "\n"
            //                                 <<   "      Total y-momentum is " << tot[2] << "\n"
            //                                 <<   "      Total z-momentum is " << tot[3] << "\n"
            //                                 <<   "      Total energy     is " << tot[4] << "\n";
#ifdef BL_LAZY
        });
#endif
}

void
EMM::post_init (Real)
{
    if (level > 0) return;
    for (int k = parent->finestLevel()-1; k >= 0; --k) {
        getLevel(k).avgDown();
    }

    if (verbose >= 2) {
        printTotal();
    }
}

void
EMM::post_restart ()
{
}

void
EMM::errorEst (TagBoxArray& tags, int, int, Real time, int, int)
{
    BL_PROFILE("EMM::errorEst()");

    if (level < refine_max_dengrad_lev)
    {
        MultiFab S_new(get_new_data(State_Type).boxArray(),get_new_data(State_Type).DistributionMap(), NUM_STATE, 1);
        const Real cur_time = state[State_Type].curTime();
        FillPatch(*this, S_new, S_new.nGrow(), cur_time, State_Type, Arho1, NUM_STATE, 0);

        const char   tagval = TagBox::SET;
        const Real dengrad_threshold = refine_dengrad;

#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(S_new,TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const Box& bx = mfi.tilebox();
            const auto Sfab = S_new.array(mfi);
            auto tag = tags.array(mfi);

            amrex::ParallelFor(bx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                EMM_tag_denerror(i, j, k, tag, Sfab, dengrad_threshold, tagval);
            });
        }
    }
}

void
EMM::read_params ()
{
    ParmParse pp("emm");

    pp.query("v", verbose);
 
    Vector<int> tilesize(AMREX_SPACEDIM);
    if (pp.queryarr("hydro_tile_size", tilesize, 0, AMREX_SPACEDIM))
    {
	for (int i=0; i<AMREX_SPACEDIM; i++) hydro_tile_size[i] = tilesize[i];
    }
   
    pp.query("cfl", cfl);

    Vector<int> lo_bc(AMREX_SPACEDIM), hi_bc(AMREX_SPACEDIM);
    pp.getarr("lo_bc", lo_bc, 0, AMREX_SPACEDIM);
    pp.getarr("hi_bc", hi_bc, 0, AMREX_SPACEDIM);
    for (int i = 0; i < AMREX_SPACEDIM; ++i) {
        phys_bc.setLo(i,lo_bc[i]);
        phys_bc.setHi(i,hi_bc[i]);
    }

    pp.query("do_reflux", do_reflux);

    pp.query("refine_max_dengrad_lev", refine_max_dengrad_lev);
    pp.query("refine_dengrad", refine_dengrad);
    pp.query("refine_vofgrad", refine_vofgrad);

    pp.query("eos_gamma1", parm->eos_gamma1);
    pp.query("eos_gamma2", parm->eos_gamma2);
    pp.query("eos_pinf1", parm->eos_pinf1);
    pp.query("eos_pinf2", parm->eos_pinf2);
    pp.query("alpha_min", parm->alpha_min);

    pp.query("tableRows1", parm->tableRows1);
    pp.query("tableColumns1", parm->tableColumns1);
    pp.query("nnTP1", parm->nnTP1);
    pp.query("mmTP1", parm->mmTP1);

    pp.query("tableRows2", parm->tableRows2);
    pp.query("tableColumns2", parm->tableColumns2);
    pp.query("nnTP2", parm->nnTP2);
    pp.query("mmTP2", parm->mmTP2);

    pp.query("tabulated1", parm->tabulated1);
    pp.query("tabulated2", parm->tabulated2);

    pp.query("ktime", parm->ktime);


    pp.query("gamma0", parm->gamma0);
    pp.query("cv0", parm->cv0);
    pp.query("q0", parm->q0);
    pp.query("pinf0_1", parm->pinf0_1);
    pp.query("pinf0", parm->pinf0);
    pp.query("b1", parm->b1);
    pp.query("b0_1", parm->b0_1);
    pp.query("cordP", parm->cordP);



    // ------------------------------------------------------------------
    // TABLES FOR THE 1st PHASE
    std::ifstream file1("NEWAir_EoS_RKPRe.dat");
    // std::ifstream file1("IdealGas.dat");
    // std::ifstream file1("NEWAir_EoS_Helmholtze.dat");
    
    int rc;
    rc = parm->tableRows1 * parm->tableColumns1;
    // double A1[parm->tableRows1][parm->tableColumns1];
    std::vector<std::vector<double>> A1(parm->tableRows1, std::vector<double>(parm->tableColumns1));
    // double B1[rc];
    std::vector<double> B1(rc);
    int k = 0;
    for (int i = 0; i < parm->tableRows1; i++)
    {
        for (int j = 0; j < parm->tableColumns1; j++)
        {
            file1 >> A1[i][j];
            B1[k] = A1[i][j];
            k += 1;
        }
    }
    file1.close();

    const int m1 = 45496, n1 = 10;
    // const int m1 = 37400, n1 = 10;
    // auto Vee1 = new double[m1][n1];
    std::vector<std::vector<double>> Vee1(m1, std::vector<double>(n1));
    int i = 0;
    for (int j = 0; j < parm->tableColumns1; j++)
    {
        i = 0;
        for (int k = j; k < rc; k += 10)
        {
            Vee1[i][j] = B1[k];
            i += 1;
        }
    }

    k = 0;
    for (int i = 0; i < parm->nnTP1; i++)
    {
        for (int j = 0; j < parm->mmTP1; j++)
        {
            EMM::PVEC1T[i][j] = Vee1[k][0];
            EMM::TVEC1T[i][j] = Vee1[k][1];
            EMM::RHOVEC1T[i][j] = Vee1[k][2];
            EMM::EVEC1T[i][j] = Vee1[k][3];
            EMM::SOSVEC1T[i][j] = Vee1[k][7];
            k += 1;
        }
    }

    parm->TMINT1 = EMM::TVEC1T[0][0];
    parm->PMINT1 = EMM::PVEC1T[0][0];
    parm->TMAXT1 = EMM::TVEC1T[parm->nnTP1-1][parm->mmTP1-1];
    parm->PMAXT1 = EMM::PVEC1T[parm->nnTP1-1][parm->mmTP1-1];
    parm->DT1 = std::abs(EMM::TVEC1T[0][0] - EMM::TVEC1T[1][0]);
    parm->DP1 = std::abs(EMM::PVEC1T[0][0] - EMM::PVEC1T[0][1]);
    amrex::Print().SetPrecision(18) << "TMINT111111:            " << parm->TMINT1 << "\n";
    amrex::Print().SetPrecision(18) << "PMINT1:            " << parm->PMINT1 << "\n";
    amrex::Print().SetPrecision(18) << "TMAXT1:            " << parm->TMAXT1 << "\n";
    amrex::Print().SetPrecision(18) << "PMAXT1:            " << parm->PMAXT1 << "\n";
    amrex::Print().SetPrecision(18) << "DT1:               " << parm->DT1 << "\n";
    amrex::Print().SetPrecision(18) << "DP1:               " << parm->DP1 << "\n";
    // ------------------------------------------------------------------
    // TABLES FOR 2ND PHASE
    if(parm->tabulated2 == 1){
        // std::ifstream file2("2Dthermotable_H2O.dat");
        // std::ifstream file2("thermoSortedHR.dat");
        //std::ifstream file2("2Dthermotable_Tait.dat");
        //std::ifstream file2("MNASG_DEC.dat");
        std::ifstream file2("2Dthermotable_Tait_HighRes.dat");

        
        // std::ifstream file2("ENASG_Wider_Range.dat");
        // std::ifstream file2("ENASG1.dat");
        // std::ifstream file2("StiffenedGas.dat");
        // const int m2 = 1624845; const int n2 = 10;
        // const int m2 = 2095941; const int n2 = 8;
     //   const int m2 = 34689; const int n2 = 10;
        const int m2 = 318169; const int n2 = 10;

        // rc = 1624845 * 10;
        rc = 318169 * 10;
        std::vector<std::vector<double>> A2(m2, std::vector<double>(n2));
        // double B2[rc];
        std::vector<double> B2(rc);
        k = 0;

        for (int i = 0; i < parm->tableRows2; i++)
        {
            for (int j = 0; j < parm->tableColumns2; j++)
            {
                file2 >> A2[i][j];
                B2[k] = A2[i][j];
                k += 1;
            }
        }
        file2.close();

        std::vector<std::vector<double>> Vee2(m2, std::vector<double>(n2));
        // auto Vee2 = new double[m2][n2];
        i = 0;
        for (int j = 0; j < parm->tableColumns2; j++)
        {
            i = 0;
            for (k = j; k < rc; k += n2)
            {
                Vee2[i][j] = B2[k];
                i += 1;
            }
        }

        k = 0;
        for (int i = 0; i < parm->nnTP2; i++)
        {
            for (int j = 0; j < parm->mmTP2; j++)
            {
                // EMM::PVEC2T[i][j] = Vee2[k][0];
                // EMM::TVEC2T[i][j] = Vee2[k][1];
                // EMM::RHOVEC2T[i][j] = Vee2[k][2];
                // EMM::EVEC2T[i][j] = Vee2[k][3];
                // EMM::SOSVEC2T[i][j] = Vee2[k][7];
                
                // EMM::PVEC2T[i][j] = Vee2[k][1];
                // EMM::TVEC2T[i][j] = Vee2[k][0];
                // EMM::RHOVEC2T[i][j] = Vee2[k][2];
                // EMM::EVEC2T[i][j] = Vee2[k][3];
                // EMM::SOSVEC2T[i][j] = Vee2[k][7];
                
                EMM::PVEC2T[i][j] = Vee2[k][0];
                EMM::TVEC2T[i][j] = Vee2[k][1];
                EMM::RHOVEC2T[i][j] = Vee2[k][2];
                EMM::EVEC2T[i][j] = Vee2[k][3];
                EMM::SOSVEC2T[i][j] = Vee2[k][7];
                k += 1;
            }
        }

        
        //amrex::Print()<< "Saeed6 "<< EMM::TVEC2T[1][0]<<"\n";
        parm->TMINT2 = EMM::TVEC2T[0][0];
        parm->PMINT2 = EMM::PVEC2T[0][0];
        parm->TMAXT2 = EMM::TVEC2T[parm->nnTP2-1][parm->mmTP2-1];
        parm->PMAXT2 = EMM::PVEC2T[parm->nnTP2-1][parm->mmTP2-1];
        parm->DT2 = std::abs(EMM::TVEC2T[0][0] - EMM::TVEC2T[1][0]);
        parm->DP2 = std::abs(EMM::PVEC2T[0][0] - EMM::PVEC2T[0][1]);
    }
    amrex::Print().SetPrecision(18) << "TMINT2:            " << parm->TMINT2 << "\n";
    amrex::Print().SetPrecision(18) << "PMINT2:            " << parm->PMINT2 << "\n";
    amrex::Print().SetPrecision(18) << "TMAXT2:            " << parm->TMAXT2 << "\n";
    amrex::Print().SetPrecision(18) << "PMAXT2:            " << parm->PMAXT2 << "\n";
    amrex::Print().SetPrecision(18) << "DT2:               " << parm->DT2 << "\n";
    amrex::Print().SetPrecision(18) << "DP2:               " << parm->DP2 << "\n";

    amrex::Real line1;
    amrex::Real line2;
    std::ifstream fileBC("pressureBC.txt");
    while (fileBC >> line1 >> line2) {
        parm->t_pulse.push_back(line1);
        parm->p_pulse.push_back(line2);
    }
    fileBC.close();

    parm->coord_type = amrex::DefaultGeometry().Coord();
    parm->Initialize();
}

void
EMM::avgDown ()
{
    BL_PROFILE("EMM::avgDown()");

    if (level == parent->finestLevel()) return;

    auto& fine_lev = getLevel(level+1);

    MultiFab& S_crse =          get_new_data(State_Type);
    MultiFab& S_fine = fine_lev.get_new_data(State_Type);

    amrex::average_down(S_fine, S_crse, fine_lev.geom, geom,
                        0, S_fine.nComp(), parent->refRatio(level));

    const int nghost = 0;
}

void
EMM::buildMetrics ()
{
    // make sure dx == dy == dz
    const Real* dx = geom.CellSize();
    if (std::abs(dx[0]-dx[1]) > 1.e-12_rt*dx[0] 
#if (AMREX_SPACEDIM == 3)
    || std::abs(dx[0]-dx[2]) > 1.e-12_rt*dx[0]
#endif
    ) {
        amrex::Abort("EMM: must have dx == dy == dz\n");
    }
}

Real
EMM::estTimeStep ()
{
    BL_PROFILE("EMM::estTimeStep()");

    const auto dx = geom.CellSizeArray();
    const MultiFab& S = get_new_data(State_Type);
    Parm const* lparm = parm.get();

    Real estdt = amrex::ReduceMin(S, 0,
    [=] AMREX_GPU_HOST_DEVICE (Box const& bx, Array4<Real const> const& fab) noexcept -> Real
    {
        return EMM_estdt(bx, fab, dx, *lparm);
    });

    estdt *= cfl;
    ParallelDescriptor::ReduceRealMin(estdt);

    return estdt;
}

Real
EMM::initialTimeStep ()
{
    return estTimeStep();
}
