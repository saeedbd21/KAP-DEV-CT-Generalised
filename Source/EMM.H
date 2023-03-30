#ifndef EMM_H_
#define EMM_H_

#include <EMM_parm.H>
#include <emm_prob_parm.H>
#include <AMReX_AmrLevel.H>
#include <AMReX_FluxRegister.H>
#include <AMReX_LayoutData.H>

using namespace amrex;
class EMM
    :
    public amrex::AmrLevel
{
public:

    EMM ();
    EMM (amrex::Amr&            papa,
         int                    lev,
         const amrex::Geometry& level_geom,
         const amrex::BoxArray& bl,
         const amrex::DistributionMapping& dm,
         amrex::Real            time);
    virtual ~EMM ();

    EMM (const EMM& rhs) = delete;
    EMM& operator= (const EMM& rhs) = delete;
    
    // Restart from a checkpoint file.
    virtual void restart (amrex::Amr&     papa,
                          std::istream&   is,
                          bool            bReadSpecial = false) override;

    // Write checkpoint
    virtual void checkPoint(const std::string& dir,
                            std::ostream&      os,
                            amrex::VisMF::How  how = amrex::VisMF::NFiles,
                            bool               dump_old = true) override;

    virtual std::string thePlotFileType () const override {
        return {"HyperCLaw-V1.1"};
    }

    // Write a plotfile to specified directory.
    virtual void writePlotFile (const std::string& dir,
                                std::ostream&      os,
                                amrex::VisMF::How  how) override;

    // Initialize data on this level from another EMM (during regrid).
    virtual void init (amrex::AmrLevel& old) override;

    // Initialize data on this level after regridding if old level did not previously exist
    virtual void init () override;
    
    // Initialize grid data at problem start-up.
    virtual void initData () override;

    // Advance grids at this level in time.
    virtual amrex::Real advance (amrex::Real time,
                                 amrex::Real dt,
                                 int  iteration,
                                 int  ncycle) override;

    virtual void computeInitialDt (int                                  finest_level,
                                   int                                  sub_cycle,
                                   amrex::Vector<int>&                  n_cycle,
                                   const amrex::Vector<amrex::IntVect>& ref_ratio,
                                   amrex::Vector<amrex::Real>&          dt_level,
                                   amrex::Real                          stop_time) override;

    virtual void computeNewDt (int                                  finest_level,
                               int                                  sub_cycle,
                               amrex::Vector<int>&                  n_cycle,
                               const amrex::Vector<amrex::IntVect>& ref_ratio,
                               amrex::Vector<amrex::Real>&          dt_min,
                               amrex::Vector<amrex::Real>&          dt_level,
                               amrex::Real                          stop_time,
                               int                                  post_regrid_flag) override;

    virtual void post_regrid (int lbase, int new_finest) override;

    // Do work after timestep().
    virtual void post_timestep (int iteration) override;

    // After a full time step
    virtual void postCoarseTimeStep (amrex::Real time) override;

    // Do work after init().
    virtual void post_init (amrex::Real stop_time) override;

    virtual void post_restart () override;

    // Error estimation for regridding.
    virtual void errorEst (amrex::TagBoxArray& tb,
                           int                 clearval,
                           int                 tagval,
                           amrex::Real         time,
                           int                 n_error_buf = 0,
                           int                 ngrow = 0) override;

//    virtual int WorkEstType () override { return Cost_Type; }

    // Define data descriptors.
    static void variableSetUp ();

    // Cleanup data descriptors at end of run.
    static void variableCleanUp ();

    static int numGrow() { return NUM_GROW; };

#if !defined(AMREX_USE_CUDA)
protected:
#endif

    static void read_params ();

    EMM& getLevel (int lev) { return dynamic_cast<EMM&>(parent->getLevel(lev)); }

    void avgDown ();

    void buildMetrics ();

    amrex::Real estTimeStep ();

    // Compute initial time step.
    amrex::Real initialTimeStep ();

    //void computeTemp (amrex::MultiFab& State, int ng);
    void compute_thermodynamics (amrex::MultiFab& Snew, amrex::MultiFab& Sborder);

    void compute_dSdt (amrex::MultiFab& S, amrex::MultiFab& dSdt, amrex::Real dt,
                       amrex::FluxRegister* fr_as_crse, amrex::FluxRegister* fr_as_fine, amrex::Real time);

    void sourceTerm (amrex::MultiFab& SrcTerms, amrex::MultiFab& Sborder, amrex::Real dt);

    //void emm_computeBubRadius (int i, int j ,int k, amrex::Array4<amrex::Real const> sfab, amrex::Real time);

    void printTotal () const;

    std::unique_ptr<amrex::FluxRegister> flux_reg;

    static constexpr int NUM_GROW = 2;
    // enum StateVariable {
    //     Arho1, Arho2, XMOM, YMOM, ZMOM, RhoE, Alpha, Pressure,
    //     NUM_STATE
    // };
    enum StateVariable {
        Arho1, Arho2, XMOM, YMOM, ZMOM, RhoE, Alpha, Pressure, T1, T2,
        NUM_STATE
    };
    enum StateDataType {
        State_Type = 0,
        // Thermo_Type,
//        Cost_Type
        NUM_STATE_DATA_TYPE
    };
    static int num_state_data_types;

    static amrex::BCRec phys_bc;

    // Parameters
    static int verbose;
    static amrex::IntVect hydro_tile_size;
    static amrex::Real cfl;

    static int do_reflux;

    static int refine_max_dengrad_lev;
    static amrex::Real refine_dengrad;
    static amrex::Real refine_vofgrad;

    static amrex::Real gravity;

public:
    static std::unique_ptr<Parm> parm;
    static int coord_type;
    //static std::unique_ptr<ProbParm> prob_parm;
    static std::vector<std::vector<double>> TVEC1T;
    static std::vector<std::vector<double>> PVEC1T;
    static std::vector<std::vector<double>> RHOVEC1T;
    static std::vector<std::vector<double>> EVEC1T;
    static std::vector<std::vector<double>> SOSVEC1T;

    static std::vector<std::vector<double>> TVEC2T;
    static std::vector<std::vector<double>> PVEC2T;
    static std::vector<std::vector<double>> RHOVEC2T;
    static std::vector<std::vector<double>> EVEC2T;
    static std::vector<std::vector<double>> SOSVEC2T;
};

void EMM_bcfill (amrex::Box const& bx, amrex::FArrayBox& data,
                 const int dcomp, const int numcomp,
                 amrex::Geometry const& geom, const amrex::Real time,
                 const amrex::Vector<amrex::BCRec>& bcr, const int bcomp,
                 const int scomp);

void EMM_nullfill(amrex::Box const& bx, amrex::FArrayBox& data,
                const int dcomp, const int numcomp,
                amrex::Geometry const& geom, const amrex::Real time,
                const amrex::Vector<amrex::BCRec>& bcr, const int bcomp,
                const int scomp);
#endif
