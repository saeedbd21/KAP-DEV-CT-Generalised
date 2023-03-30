#ifndef EMM_TAGGING_H_
#define EMM_TAGGING_H_

#include <AMReX_FArrayBox.H>
#include <AMReX_TagBox.H>
#include <cmath>

AMREX_GPU_HOST_DEVICE
inline
void
EMM_tag_denerror (int i, int j, int k,
                  amrex::Array4<char> const& tag,
                  amrex::Array4<amrex::Real const> const& state,
                  amrex::Real dengrad_threshold, char tagval) noexcept
{
    
    amrex::Real mdenc   = state(i,j,k,UARHO1)   + state(i,j,k,UARHO2);
    amrex::Real mdenip1 = state(i+1,j,k,UARHO1) + state(i+1,j,k,UARHO2);
    amrex::Real mdenim1 = state(i-1,j,k,UARHO1) + state(i-1,j,k,UARHO2);
    amrex::Real mdenjp1 = state(i,j+1,k,UARHO1) + state(i,j+1,k,UARHO2);
    amrex::Real mdenjm1 = state(i,j-1,k,UARHO1) + state(i,j-1,k,UARHO2);

    amrex::Real ax = amrex::Math::abs( mdenip1 - mdenc );
    amrex::Real ay = amrex::Math::abs( mdenjp1 - mdenc );

#if (AMREX_SPACEDIM == 3)
    amrex::Real mdenkp1 = state(i,j,k+1,UARHO1) + state(i,j,k+1,UARHO2);
    amrex::Real mdenkm1 = state(i,j,k-1,UARHO1) + state(i,j,k-1,UARHO2);
    amrex::Real az = amrex::Math::abs( mdenkp1 - mdenc );
#endif

    ax = amrex::max(ax,amrex::Math::abs( mdenc - mdenim1 ));
    ay = amrex::max(ay,amrex::Math::abs( mdenc - mdenjm1 ));

#if (AMREX_SPACEDIM == 3)
    az = amrex::max(az,amrex::Math::abs( mdenc - mdenkm1 ));
#endif

    if (amrex::max(ax,ay
#if (AMREX_SPACEDIM == 3)
    ,az
#endif
    ) >= dengrad_threshold) {
        tag(i,j,k) = tagval;
    }
    //if(state(i,j,k,GALPHA) > 0.5_rt){tag(i,j,k) = tagval;}
}
#endif
