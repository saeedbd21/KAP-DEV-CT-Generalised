
#include <EMM.H>
#include <EMM_index_macros.H>

using namespace amrex;

void
EMM::restart (Amr& papa, std::istream& is, bool bReadSpecial)
{
    AmrLevel::restart(papa,is,bReadSpecial);

    if (do_reflux && level > 0) {
        flux_reg.reset(new FluxRegister(grids,dmap,crse_ratio,level,NCONS));
    }

    buildMetrics();
}

void 
EMM::checkPoint (const std::string& dir, std::ostream& os, VisMF::How how, bool dump_old) 
{
    AmrLevel::checkPoint(dir, os, how, dump_old);
}

void
EMM::writePlotFile (const std::string& dir, std::ostream& os, VisMF::How how)
{
    BL_PROFILE("EMM::writePlotFile()");
    AmrLevel::writePlotFile(dir, os, how);
}
