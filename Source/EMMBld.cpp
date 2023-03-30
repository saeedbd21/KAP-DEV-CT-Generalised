
#include <AMReX_LevelBld.H>
#include <EMM.H>

using namespace amrex;

class EMMBld
    :
    public LevelBld
{
    virtual void variableSetUp () override;
    virtual void variableCleanUp () override;
    virtual AmrLevel *operator() () override;
    virtual AmrLevel *operator() (Amr&            papa,
                                  int             lev,
                                  const Geometry& level_geom,
                                  const BoxArray& ba,
                                  const DistributionMapping& dm,
                                  Real            time) override;
};

EMMBld EMM_bld;

LevelBld*
getLevelBld ()
{
    return &EMM_bld;
}

void
EMMBld::variableSetUp ()
{
    EMM::variableSetUp();
}

void
EMMBld::variableCleanUp ()
{
    EMM::variableCleanUp();
}

AmrLevel*
EMMBld::operator() ()
{
    return new EMM;
}

AmrLevel*
EMMBld::operator() (Amr&            papa,
                    int             lev,
                    const Geometry& level_geom,
                    const BoxArray& ba,
                    const DistributionMapping& dm,
                    Real            time)
{
    return new EMM(papa, lev, level_geom, ba, dm, time);
}
