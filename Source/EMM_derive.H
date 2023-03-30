#ifndef EMM_DERIVE_H_
#define EMM_DERIVE_H_

#include <AMReX_FArrayBox.H>
#include <AMReX_Geometry.H>

void EMM_derpres (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int ncomp,
                  const amrex::FArrayBox& datafab, const amrex::Geometry& geomdata,
                  amrex::Real time, const int* bcrec, int level);

void EMM_dervel (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int ncomp,
                 const amrex::FArrayBox& datafab, const amrex::Geometry& geomdata,
                 amrex::Real time, const int* bcrec, int level);

void EMM_derden (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int ncomp,
                 const amrex::FArrayBox& datafab, const amrex::Geometry& geomdata,
                 amrex::Real time, const int* bcrec, int level);

void EMM_deralpha2 (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int ncomp,
                 const amrex::FArrayBox& datafab, const amrex::Geometry& geomdata,
                 amrex::Real time, const int* bcrec, int level);

#endif
