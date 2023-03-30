#ifndef EMM_EOS_H_
#define EMM_EOS_H_

#include <cmath>
#include "EMM_parm.H"
#include "EMM.H"
#include "EMM_index_macros.H"

namespace
{
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
        amrex::Real
        TPF(int tableToInterpolate,
            // std::vector<std::vector<double>> OVEC,
            amrex::Real T,
            amrex::Real p,
            Parm const &parm,
            int phaseNum) noexcept
    {
        using namespace amrex::literals;

        amrex::Real T1 = 0.0_rt;
        amrex::Real P1 = 0.0_rt;
        // amrex::Print().SetPrecision(18) << "T1:          " << T << "\n";
        // amrex::Print().SetPrecision(18) << "P1:          " << p << "\n";
        if(phaseNum == 1){
            T1 = std::min(std::max(T, parm.TMINT1), parm.TMAXT1 - parm.DT1 / 2.0_rt);
            P1 = std::min(std::max(p, parm.PMINT1), parm.PMAXT1 - parm.DP1 / 2.0_rt);
        }else{
            T1 = std::min(std::max(T, parm.TMINT2), parm.TMAXT2 - parm.DT2 / 2.0_rt);
            P1 = std::min(std::max(p, parm.PMINT2), parm.PMAXT2 - parm.DP2 / 2.0_rt);
        }
        // amrex::Print().SetPrecision(18) << "T1:          " << T1 << "\n";
        // amrex::Print().SetPrecision(18) << "P1:          " << P1 << "\n";
        // amrex::Print().SetPrecision(18) << "phaseNum:    " << phaseNum << "\n";
        // amrex::Print().SetPrecision(18) << "parm.TMINT1: " << parm.TMINT1 << "\n";
        // amrex::Print().SetPrecision(18) << "parm.TMINT2: " << parm.TMINT2 << "\n";
        // amrex::Print().SetPrecision(18) << "parm.DT1: " << parm.DT1 << "\n";
        // amrex::Print().SetPrecision(18) << "parm.DT2: " << parm.DT2 << "\n";
        // amrex::Print().SetPrecision(18) << "parm.DP1: " << parm.DP1 << "\n";
        // amrex::Print().SetPrecision(18) << "parm.DP2: " << parm.DP2 << "\n";
        int II = 0;
        II = phaseNum == 1 ? int((T1 - parm.TMINT1) / parm.DT1) : int((T1 - parm.TMINT2) / parm.DT2);
        if (II > parm.nnTP1 - 2 && phaseNum == 1)
        {
            II = II - 1;
        }else if(II > parm.nnTP2 - 2 && phaseNum == 2){
            II = II - 1;
        }

        int JJ = 0;
        JJ = phaseNum == 1 ? int((P1 - parm.PMINT1) / parm.DP1) : int((P1 - parm.PMINT2) / parm.DP2);
        if (JJ > parm.mmTP1 - 2 && phaseNum == 1)
        {
            JJ = JJ - 1;
        }else if(JJ > parm.mmTP2 - 2 && phaseNum == 2){
            JJ = JJ - 1;
        }

        amrex::Real TMIN = 0.0_rt;
        amrex::Real TMAX = 0.0_rt;
        amrex::Real PMIN = 0.0_rt;
        amrex::Real PMAX = 0.0_rt;
        if(phaseNum == 1){
            TMIN = EMM::TVEC1T[II][JJ];
            TMAX = EMM::TVEC1T[std::min(II + 1, parm.nnTP1)][JJ + 1];
            PMIN = EMM::PVEC1T[II][JJ];
            PMAX = EMM::PVEC1T[std::min(II + 1, parm.nnTP1)][JJ + 1];
        }else{
            TMIN = EMM::TVEC2T[II][JJ];
            TMAX = EMM::TVEC2T[std::min(II + 1, parm.nnTP2)][JJ + 1];
            PMIN = EMM::PVEC2T[II][JJ];
            PMAX = EMM::PVEC2T[std::min(II + 1, parm.nnTP2)][JJ + 1];
        }
        amrex::Real CT11 = 0.0_rt;
        amrex::Real CT12 = 0.0_rt;
        amrex::Real CT21 = 0.0_rt;
        amrex::Real CT22 = 0.0_rt;

        if(tableToInterpolate == ORHO){
            if(phaseNum == 1){
                CT11 = EMM::RHOVEC1T[II][JJ];
                CT12 = EMM::RHOVEC1T[II][JJ + 1];
                CT21 = EMM::RHOVEC1T[std::min(II + 1, parm.nnTP1)][JJ];
                CT22 = EMM::RHOVEC1T[std::min(II + 1, parm.nnTP1)][JJ + 1];
            }else{
                CT11 = EMM::RHOVEC2T[II][JJ];
                CT12 = EMM::RHOVEC2T[II][JJ + 1];
                CT21 = EMM::RHOVEC2T[std::min(II + 1, parm.nnTP2)][JJ];
                CT22 = EMM::RHOVEC2T[std::min(II + 1, parm.nnTP2)][JJ + 1];
            }
        }else if (tableToInterpolate == OSOS){
            if(phaseNum == 1){
                CT11 = EMM::SOSVEC1T[II][JJ];
                CT12 = EMM::SOSVEC1T[II][JJ + 1];
                CT21 = EMM::SOSVEC1T[std::min(II + 1, parm.nnTP1)][JJ];
                CT22 = EMM::SOSVEC1T[std::min(II + 1, parm.nnTP1)][JJ + 1];
            }else{
                CT11 = EMM::SOSVEC2T[II][JJ];
                CT12 = EMM::SOSVEC2T[II][JJ + 1];
                CT21 = EMM::SOSVEC2T[std::min(II + 1, parm.nnTP2)][JJ];
                CT22 = EMM::SOSVEC2T[std::min(II + 1, parm.nnTP2)][JJ + 1];
            }
        }else if (tableToInterpolate == OE){
            if(phaseNum == 1){
                CT11 = EMM::EVEC1T[II][JJ];
                CT12 = EMM::EVEC1T[II][JJ + 1];
                CT21 = EMM::EVEC1T[std::min(II + 1, parm.nnTP1)][JJ];
                CT22 = EMM::EVEC1T[std::min(II + 1, parm.nnTP1)][JJ + 1];
            }else{
                CT11 = EMM::EVEC2T[II][JJ];
                CT12 = EMM::EVEC2T[II][JJ + 1];
                CT21 = EMM::EVEC2T[std::min(II + 1, parm.nnTP2)][JJ];
                CT22 = EMM::EVEC2T[std::min(II + 1, parm.nnTP2)][JJ + 1];
            }
        }
        // amrex::Print().SetPrecision(18) << "CT11:       " << CT11 << "\n";
        // amrex::Print().SetPrecision(18) << "CT12:       " << CT12 << "\n";
        // amrex::Print().SetPrecision(18) << "CT21:       " << CT21 << "\n";
        // amrex::Print().SetPrecision(18) << "CT22:       " << CT22 << "\n";

        // amrex::Print().SetPrecision(18) << "**************************" << "\n";
        amrex::Real f1 = (TMAX - T1) / (TMAX - TMIN) * CT11 + (T1 - TMIN) / (TMAX - TMIN) * CT21;
        amrex::Real f2 = (TMAX - T1) / (TMAX - TMIN) * CT12 + (T1 - TMIN) / (TMAX - TMIN) * CT22;
        amrex::Real output = (PMAX - P1) / (PMAX - PMIN) * f1 + (P1 - PMIN) / (PMAX - PMIN) * f2;
        // amrex::Print().SetPrecision(18) << "**************************" << "\n";
        // amrex::Print().SetPrecision(18) << "f1:                                        " << f1 << "\n";
        // amrex::Print().SetPrecision(18) << "f2:                                        " << f2 << "\n";
        // amrex::Print().SetPrecision(18) << "(TMAX - T1) / (TMAX - TMIN):               " << (TMAX - T1) / (TMAX - TMIN) << "\n";
        // amrex::Print().SetPrecision(18) << "(T1 - TMIN) / (TMAX - TMIN):               " << (T1 - TMIN) / (TMAX - TMIN) << "\n";
        // amrex::Print().SetPrecision(18) << "(TMAX - T1) / (TMAX - TMIN)* CT11:         " << (TMAX - T1) / (TMAX - TMIN) * CT11 << "\n";
        // amrex::Print().SetPrecision(18) << "(T1 - TMIN) / (TMAX - TMIN)* CT21:         " << (T1 - TMIN) / (TMAX - TMIN) * CT21 << "\n";
        // amrex::Print().SetPrecision(18) << "II:         " << II << "\n";
        // amrex::Print().SetPrecision(18) << "JJ:         " << JJ << "\n";
        // amrex::Print().SetPrecision(18) << "TMIN:       " << TMIN << "\n";
        // amrex::Print().SetPrecision(18) << "TMAX:       " << TMAX << "\n";
        // amrex::Print().SetPrecision(18) << "PMIN:       " << PMIN << "\n";
        // amrex::Print().SetPrecision(18) << "PMAX:       " << PMAX << "\n";
        // amrex::Print().SetPrecision(18) << "CT11:       " << CT11 << "\n";
        // amrex::Print().SetPrecision(18) << "CT12:       " << CT12 << "\n";
        // amrex::Print().SetPrecision(18) << "CT21:       " << CT21 << "\n";
        // amrex::Print().SetPrecision(18) << "CT22:       " << CT22 << "\n";
        // amrex::Print().SetPrecision(18) << "T:          " << T << "\n";
        // amrex::Print().SetPrecision(18) << "T1:         " << T1 << "\n";
        // amrex::Print().SetPrecision(18) << "p:          " << p << "\n";
        // amrex::Print() << (T1 - parm.TMINT) / parm.DT << "\n";
        // amrex::Print() << int((T1 - parm.TMINT) / parm.DT) << "\n";
        return output;
    }

    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
        amrex::Real
        T_finder(amrex::Real alpha,
                 amrex::Real rho,
                 amrex::Real PG,
                 amrex::Real TG,
                 Parm const &parm,
                 int phaseNum) noexcept
    {
        amrex::Real TT = TG; // q(i, j, k, QT1);
        if (TT < 1.0) TT = 300.0;
        // amrex::Real TT = PG/(rho*287.058_rt);
        // amrex::Print() << "TT:         " << TT << "\n";
        TT = phaseNum == 1 ? std::min(std::max(TT, parm.TMINT1 * 1.001), parm.TMAXT1 * 0.99) : std::min(std::max(TT, parm.TMINT2 * 1.001), parm.TMAXT2 * 0.99);
        int zmax = 2000;
        // amrex::Print() << "**************************" << "\n";
        // if(rho < 0.0){
        //     amrex::Print() << "rho:         " << rho << "\n";  
        // }
        
        // amrex::Print() << "PG:          " << PG << "\n";
        // amrex::Print() << "TG:          " << TG << "\n";

        amrex::Real epsT = 0.0;
        amrex::Real rhoTTp = 0.0;
        amrex::Real rhoT = 0.0;
        amrex::Real drhodT = 0.0;
        amrex::Real TTo = 0.0;
        // amrex::Real alphaptf = 0.25;
        // for (double alphaptf = 1.0; alphaptf >= 0.05; alphaptf -= 0.05)
        for (double alphaptf = 1.0; alphaptf >= 0.1; alphaptf -= 0.1)
        {
            // amrex::Print() << "+++++++++++++++++++++++++" << "\n";
            for (int z = 1; z <= zmax; z++)
            {
                // amrex::Print() << "------------------------" << "\n";
                epsT = TT * 1E-3;
                rhoTTp = TPF(ORHO, TT + epsT, std::log10(PG), parm, phaseNum);
                rhoT = TPF(ORHO, TT, std::log10(PG), parm, phaseNum);
                drhodT = ((rhoTTp - rho) - (rhoT - rho)) / epsT + 1E-16;
                TTo = TT;
                TT = TTo - (rhoT - rho) / drhodT;
                TT = TTo * (1.0 - alphaptf) + TT * alphaptf;

                TT = phaseNum == 1 ? std::min(std::max(TT, parm.TMINT1 * 1.01), parm.TMAXT1 * 0.99) : std::min(std::max(TT, parm.TMINT2 * 1.01), parm.TMAXT2 * 0.99);
                    // amrex::Print() << "alphaptf:   " << alphaptf << "\n";
                    // amrex::Print() << "z:          " << z << "\n";
                    // amrex::Print() << "PG:         " << PG << "\n";
                    // amrex::Print() << "rho:        " << rho << "\n";
                    // amrex::Print() << "rhoTTp:     " << rhoTTp << "\n";
                    // amrex::Print() << "rhoT:       " << rhoT << "\n";
                    // amrex::Print() << "epsT:       " << epsT << "\n";
                    // amrex::Print() << "TT:         " << TT << "\n";
                    // amrex::Print() << "TTo:        " << TTo << "\n";

                if (std::abs(TT - TTo)/std::abs(TTo) < 1E-10)
                // if (std::abs(rhoT-rho) < 0.001*rho)
                {
                    return TT;
                }

            }
        }
        // amrex::Print() << "rhoTTp:     " << rhoTTp << "\n";
        // amrex::Print() << "rhoT:       " << rhoT << "\n";
        // amrex::Print() << "epsT:       " << epsT << "\n";
        // amrex::Print() << "TT:         " << TT << "\n";
        // amrex::Print() << "TTo:        " << TTo << "\n";
        // amrex::Print() << "PG:         " << PG << "\n";
        // amrex::Print() << "rho:        " << rho << "\n";
        // amrex::Print() << "TEMPERATURE DID NOT CONVERGE" << "\n";
        return TT;
    }

    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
        amrex::Real
        p_finder(amrex::Real alpha1,
                 amrex::Real alpha2,
                 amrex::Real rho1,
                 amrex::Real rho2,
                 amrex::Real p,
                 amrex::Real T1G,
                 amrex::Real T2G,
                 amrex::Real eint,
                 Parm const &parm) noexcept
    {
        BL_PROFILE("p_finder()");
        // amrex::Print() << "PFINDER:----------" << "\n";
        // amrex::Real alpharint = 0.25;
        // if (T1G <= 1.0) T1G = 300.0;
        // if (T2G <= 1.0) T2G = 300.0;
        // T1G = std::min(std::max(T1G, parm.TMINT1 * 1.02), parm.TMAXT1 * 0.98);
        // T2G = std::min(std::max(T2G, parm.TMINT1 * 1.02), parm.TMAXT1 * 0.98);
        amrex::Real pg = p;
        // amrex::Print() << "pg:            " << pg << "\n";
        // amrex::Print() << "rhoeint:       " << eint << "\n";
        // amrex::Print() << "alpha1:        " << alpha1 << "\n";
        // amrex::Print() << "alpha2:        " << alpha2 << "\n";
        // amrex::Print() << "rho1:          " << rho1 << "\n";
        // amrex::Print() << "rho2:          " << rho2 << "\n";
        // if(rho1 < 0.0 || rho2 < 0.0 || alpha1 < 0.0 || alpha2 < 0.0){
        //     amrex::Print() << "alpha1:        " << alpha1 << "\n";
        //     amrex::Print() << "alpha2:        " << alpha2 << "\n";
        //     amrex::Print() << "rho1:          " << rho1 << "\n";
        //     amrex::Print() << "rho2:          " << rho2 << "\n";
        // }
        // if(alpha1 > 1.0-parm.alpha_min/100.0){alpha1 = 1.0-parm.alpha_min/100.0;}
        // if(alpha1 < parm.alpha_min/100.0){alpha1 = parm.alpha_min/100.0;}
        // alpha2 = 1.0_rt - alpha1;

        for (double alpharint = 1.0; alpharint >= 0.1; alpharint -= 0.1)
        {
            for (int Z = 0; Z < 2000 ; Z += 1)
            {
                // amrex::Print() << "**************" << "\n";
                amrex::Real eG2 = 0.0;
                amrex::Real T2 = 0.0;
                if(parm.tabulated2 == 0){
                    eG2 = (pg+parm.eos_gamma2*parm.eos_pinf2)/(parm.eos_gamma2-1.0_rt)*(1.0/rho2);
                }else{
                    T2 = T_finder(alpha2, rho2, pg, T2G, parm, 2);
                    T2 = std::min(std::max(T2, parm.TMINT2 * 1.02), parm.TMAXT2 * 0.98);
                    eG2 = TPF(OE, T2, std::log10(pg), parm, 2);
                }
                
                // amrex::Print() << "eG2:           " << eG2 << "\n";

                amrex::Real eG1 = 0.0;
                amrex::Real T1 = 0.0;
                if(parm.tabulated1 == 0){
                    eG1 = (pg+parm.eos_gamma1*parm.eos_pinf1)/(parm.eos_gamma1-1.0_rt)*(1.0/rho1);
                //     // amrex::Print() << "eG1:           " << eG1 << "\n";
                }else{
                    T1 = T_finder(alpha1, rho1, pg, T1G, parm, 1);
                    T1 = std::min(std::max(T1, parm.TMINT1 * 1.02), parm.TMAXT1 * 0.98);

                    eG1 = TPF(OE, T1, std::log10(pg), parm, 1);
                }
                amrex::Real eps=1E-3*pg;

                amrex::Real ERR1 = alpha1*rho1*eG1 + alpha2*rho2*eG2 - eint;
                amrex::Real ptemp = pg + eps;

                if(parm.tabulated2 == 0){
                    eG2 = (ptemp+parm.eos_gamma2*parm.eos_pinf2)/(parm.eos_gamma2-1.0_rt)*(1.0/rho2);
                }else{
                    T2 = T_finder(alpha2, rho2, ptemp, T2G, parm, 2);
                    T2 = std::min(std::max(T2, parm.TMINT2 * 1.02), parm.TMAXT2 * 0.98);
                    eG2 = TPF(OE, T2, std::log10(ptemp), parm, 2);
                }
                
                eG1 = 0.0;
                T1 = 0.0;
                if(parm.tabulated1 == 0){
                    eG1 = (ptemp+parm.eos_gamma1*parm.eos_pinf1)/(parm.eos_gamma1-1.0_rt)*(1.0/rho1);
                //     // amrex::Print() << "eG1temp:       " << eG1 << "\n";
                }else{
                    T1 = T_finder(alpha1, rho1, ptemp, T1G, parm, 1);
                    T1 = std::min(std::max(T1, parm.TMINT1 * 1.02), parm.TMAXT1 * 0.98);

                    eG1 = TPF(OE, T1, std::log10(ptemp), parm, 1);
                }

                amrex::Real ERR2 = alpha1*rho1*eG1 + alpha2*rho2*eG2 - eint;

                amrex::Real derr = (ERR2-ERR1)/eps;
                amrex::Real pg0 = pg;
                pg = alpharint*(pg - ERR1/derr)+(1.0-alpharint)*pg0;
                pg = std::min(std::max(pg, std::pow(10.0_rt,parm.PMINT1) * 1.01), std::pow(10.0_rt,parm.PMAXT1) * 0.99);
                // pg = std::min(std::max(pg, std::pow(10.0_rt,parm.PMINT1) * 1.01), std::pow(10.0_rt,parm.PMAXT1) * 0.99);

                // amrex::Print() << "**************" << "\n";
                // amrex::Print() << "Z:            " << Z << "\n";
                // amrex::Print() << "pgl:          " << pg << "\n";
                // amrex::Print() << "ERR1:         " << ERR1 << "\n";
                // amrex::Print() << "ERR2:         " << ERR2 << "\n";
                // amrex::Print() << "CVG:          " << std::abs(pg-pg0)/std::abs(pg0) << "\n";
                if(std::abs(pg-pg0)/std::abs(pg0) < 1E-5){
                // if( ERR2 < 1E-3 && ERR1 < 1E-3 ){
                    // amrex::Print() << "Z:            " << Z << "\n";
                    // amrex::Print() << "pg:            " << pg << "\n";
                    return pg;
                    // break;
                }
            }
        }
        // amrex::Print() << "PRESSURE DID NOT CONVERGE" << "\n";
        return pg;
    }
}

#endif