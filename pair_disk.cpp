// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Binghan Liu (Virginia Tech)
   Date: 2025-05-18
------------------------------------------------------------------------- */

#include "pair_disk.h"
#include <iostream>
#include "atom.h"
#include "comm.h"
#include "force.h"
#include "neigh_list.h"
#include "math_special.h"
#include "memory.h"
#include "error.h"
#include "update.h"
#include <cmath>

using namespace LAMMPS_NS;
using namespace MathSpecial;

/* ---------------------------------------------------------------------- */

PairDisk::PairDisk(LAMMPS *lmp) : Pair(lmp)
{
  writedata = 1;
}
/* ---------------------------------------------------------------------- */

PairDisk::~PairDisk()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(form);
    memory->destroy(a12);
    memory->destroy(sigma);
    memory->destroy(d1);
    memory->destroy(d2);
    memory->destroy(a1);
    memory->destroy(a2);
    memory->destroy(diameter);
    memory->destroy(cut);
    memory->destroy(offset);
    memory->destroy(sigma3);
    memory->destroy(sigma6);
    memory->destroy(lj1);
    memory->destroy(lj2);
    memory->destroy(lj3);
    memory->destroy(lj4);
  }
}

/* ---------------------------------------------------------------------- */

void PairDisk::compute(int eflag, int vflag)
{
  int i,j,ii,jj,inum,jnum,itype,jtype;
  double xtmp,ytmp,ztmp,delx,dely,delz,evdwl,fpair=0.0;
  double rsq,dd,forcelj,factor_lj; // 'r' and 'dd' are used in different scopes
  double r2inv,r6inv; // Used in SMALL_SMALL
  double ab_r,c12_plus_c22,L_ab,A2_param_eff;
  double rad[23],c1[13],c2[13],r2_minus_c2[12],term[4],denominator[4],c1c2[9],abr[5],c12_minus_c22[9];
  int *ilist,*jlist,*numneigh,**firstneigh;
  double Lambda_a = 1.0; // Consider if these should be parameters
  double Lambda_b = 1.0;
  const double pi = M_PI;
  evdwl = 0.0; 
  ev_init(eflag,vflag);
  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  double *special_lj = force->special_lj;
  int newton_pair = force->newton_pair;
  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  // loop over neighbors of my atoms
  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      factor_lj = special_lj[sbmask(j)];
      j &= NEIGHMASK;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;
      jtype = type[j];

      if (rsq >= cutsq[itype][jtype]) continue;

      if (eflag) evdwl = 0.0; // Reset for current pair

      switch (form[itype][jtype]) {
      case SMALL_SMALL:
        r2inv = 1.0/rsq;
        r6inv = r2inv*r2inv*r2inv;
        forcelj = r6inv * (lj1[itype][jtype]*r6inv - lj2[itype][jtype]);
        fpair = factor_lj*forcelj*r2inv; // fpair is F(r)/r * factor_lj
        if (eflag) evdwl = r6inv*(r6inv*lj3[itype][jtype]-lj4[itype][jtype]) -
                     offset[itype][jtype];
        break;

      case SMALL_LARGE:
        rad[1] = sqrt(rsq);
        rad[2] = rsq;
        rad[3] = rad[1] * rad[2];
        rad[4] = rad[2] * rad[2];
        rad[5] = rad[1] * rad[4];
        rad[6] = rad[3] * rad[3];
        rad[7] = rad[1] * rad[6];
        rad[8] = rad[1] * rad[7];
        c2[1] = a2[itype][jtype];
        c2[2] = c2[1] * c2[1];
        c2[4] = c2[2] * c2[2];
        c2[6] = c2[4] * c2[2];
        c2[8] = c2[4] * c2[4];
        r2_minus_c2[1] = rad[2]-c2[2];
        if (r2_minus_c2[1] <= 0) {
            error->one(FLERR,"SMALL_LARGE interaction r <= a for pair colloid");
        }
        r2_minus_c2[4] = pow(r2_minus_c2[1], 4);
        r2_minus_c2[5] =  r2_minus_c2[4] * r2_minus_c2[1];
        r2_minus_c2[10] = r2_minus_c2[5] * r2_minus_c2[5];
        r2_minus_c2[11] = r2_minus_c2[10] * r2_minus_c2[1];
        term[0] = (2 * rad[1]) / r2_minus_c2[4];
        term[1] = (4 * rad[1] * (c2[2] + 2 * rad[2])) / r2_minus_c2[5];
        term[2] = (sigma6[itype][jtype] * (40 * c2[6] * rad[1] + 240 * c2[4] * rad[3] + 240 * c2[2] * rad[5] + 40 * rad[7])) / (5 * r2_minus_c2[10]);
        term[3] = (4 * sigma6[itype][jtype] * rad[1] * (c2[8] + 20 * c2[6] * rad[2] + 60 * c2[4] * rad[4] + 40 * c2[2] * rad[6] + 5 * rad[8])) / r2_minus_c2[11];
        fpair = M_PI * c2[2] * a12[itype][jtype] * Lambda_a * (term[0] - term[1] - term[2] + term[3]) * factor_lj / rad[1];

        if (eflag) {

          double U_attractive_part = - (1.0 / 2.0) * (2 * c2[2] * rad[2] + c2[4]) / r2_minus_c2[4];
          double U_repulsive_polynomial = (5 * rad[8] + 40 * c2[2] * rad[6] + 60 * c2[4] * rad[4] + 20 * c2[6] * rad[2] + c2[8]);
          double U_repulsive_part_geom = (1.0 / 5.0) * c2[2] * U_repulsive_polynomial / r2_minus_c2[10];

          evdwl = M_PI * a12[itype][jtype] * Lambda_a * ( U_attractive_part + sigma6[itype][jtype] * U_repulsive_part_geom )
                  - offset[itype][jtype];
                  }

        if (rsq <= c2[2])
          error->one(FLERR,"Overlapping small/large in pair colloid");
        break;

      case LARGE_LARGE:
        rad[1] = sqrt(rsq); // r
        rad[2] = rsq;       // r^2
        rad[3] = rad[1] * rad[2]; rad[4] = rad[2] * rad[2]; rad[5] = rad[1] * rad[4]; rad[6] = rad[3] * rad[3]; rad[7] = rad[1] * rad[6]; rad[8] = rad[1] * rad[7]; rad[9] = rad[1] * rad[8]; rad[10] = rad[1] * rad[9]; rad[11] = rad[1] * rad[10]; rad[12] = rad[1] * rad[11]; rad[13] = rad[1] * rad[12]; rad[14] = rad[1] * rad[13]; rad[15] = rad[1] * rad[14]; rad[16] = rad[1] * rad[15]; rad[17] = rad[1] * rad[16]; rad[18] = rad[1] * rad[17]; rad[19] = rad[1] * rad[18]; rad[20] = rad[1] * rad[19]; rad[21] = rad[1] * rad[20]; rad[22] = rad[1] * rad[21];

        c1[1] = a1[itype][jtype]; // radius of disk 1
        c1[2] = c1[1] * c1[1]; c1[4] = c1[2] * c1[2]; c1[6] = c1[4] * c1[2]; c1[8] = c1[4] * c1[4]; c1[10] = c1[4] * c1[6]; c1[12] = c1[6] * c1[6];

        c2[1] = a2[itype][jtype]; // radius of disk 2
        c2[2] = c2[1] * c2[1]; c2[4] = c2[2] * c2[2]; c2[6] = c2[4] * c2[2]; c2[8] = c2[4] * c2[4]; c2[10] = c2[4] * c2[6]; c2[12] = c2[6] * c2[6];

        c1c2[0] = c1[2] * c2[2]; c1c2[1] = c1[4] * c2[4]; c1c2[2] = c1[6] * c2[6]; c1c2[3] = c1[2] * c2[6]; c1c2[4] = c1[6] * c2[2]; c1c2[5] = c1[2] * c2[10]; c1c2[6] = c1[10] * c2[2]; c1c2[7] = c1[4] * c2[8]; c1c2[8] = c1[8] * c2[4];

        abr[1] = -c1[1]-c2[1]+rad[1]; abr[2] =  c1[1]-c2[1]+rad[1]; abr[3] = -c1[1]+c2[1]+rad[1]; abr[4] =  c1[1]+c2[1]+rad[1];
        ab_r = abr[1] * abr[2] * abr[3] * abr[4];

        if (ab_r <= 0) {
             error->one(FLERR,"LARGE_LARGE interaction non-positive ab_r in pair colloid");
        }

        denominator[0] = pow(ab_r, 5.0 / 2.0);
        denominator[1] = pow(ab_r, 17.0 / 2.0);
        denominator[2] = pow(ab_r, 7.0 / 2.0);
        denominator[3] = pow(ab_r, 19.0 / 2.0);

        c12_plus_c22 = c1[2] + c2[2];
        c12_minus_c22[1] = c1[2] - c2[2];
        c12_minus_c22[2] = c12_minus_c22[1] * c12_minus_c22[1]; c12_minus_c22[4] = c12_minus_c22[2] * c12_minus_c22[2];  c12_minus_c22[6] = c12_minus_c22[2] * c12_minus_c22[4]; c12_minus_c22[8] = c12_minus_c22[4] * c12_minus_c22[4];

        A2_param_eff = a12[itype][jtype] * sigma6[itype][jtype]; // Effective A2 parameter
        L_ab = Lambda_a * Lambda_b;
        dd = c1[1] + c2[1]; // For overlap check: sum of radii

        term[0] = (a12[itype][jtype] * c1[2] * c2[2] * M_PI * M_PI * (-2 * c12_plus_c22 * rad[1] + 8 * rad[3]) * L_ab) / (2.0 * denominator[0]);

        term[1] = -(A2_param_eff * c1[2]*c2[2]  * pi *
        (-2 * c12_minus_c22[6] * (13 * c1[8] + 181 * c1c2[4] + 396 * c1c2[1] + 181 * c1c2[3] + 13 * c2[8]) * rad[1] +
          4 * c12_minus_c22[4] * c12_plus_c22 * (59 * c1[8] - 228 * c1c2[4] - 2602 * c1c2[1] - 228 * c1c2[3] + 59 * c2[8]) * rad[3] -
          30 * c12_minus_c22[2] * (c1[12] - 438 * c1c2[6] + 3 * c1c2[8] + 2436 * c1c2[2] + 3 * c1c2[7] - 438 * c1c2[5] + c2[12]) * rad[5] -
          80 * c12_plus_c22 * (32 * c1[12] + 150 * c1c2[6] - 1698 * c1c2[8] + 3081 * c1c2[2] - 1698 * c1c2[7] + 150 * c1c2[5] + 32 * c2[12]) * rad[7] +
          20 * (308 * c1[12] - 1386 * c1c2[6] - 4482 * c1c2[8] + 12165 * c1c2[2] - 4482 * c1c2[7] - 1386 * c1c2[5] + 308 * c2[12]) * rad[9] -
          48 * c12_plus_c22 * (98 * c1[8] - 1365 * c1c2[4] + 2780 * c1c2[1] - 1365 * c1c2[3] + 98 * c2[8]) * rad[11] -
          56 * (26 * c1[8] + 515 * c1c2[4] - 1350 * c1c2[1] + 515 * c1c2[3] + 26 * c2[8]) * rad[13] +
          80 * c12_plus_c22 * (53 * c1[4] - 143 * c1c2[0] + 53 * c2[4]) * rad[15] -
          90 * (23 * c1[4] - 67 * c1[2]*c2[2] + 23 * c2[4]) * rad[17] +
          100 * c12_plus_c22 * rad[19] + 110 * rad[21]) * L_ab) / (5.0 * denominator[1]);

        term[2] = -(5 * c1[2] * a12[itype][jtype] * c2[2] * pi * pi * (-c12_minus_c22[2] - c12_plus_c22 * rad[2] + 2 * rad[4]) * (abr[1] * abr[2] * abr[3] + abr[1] * abr[2] * abr[4] + abr[1] * abr[3] * abr[4] + abr[2] * abr[3] * abr[4]) * L_ab) / (4.0 * denominator[2]);

        term[3] = (17 * c1[2] * A2_param_eff * c2[2] * pi * (-c12_minus_c22[8] * c12_plus_c22 * (c1[4] + 5 * c1[2]*c2[2] + c2[4]) -
         c12_minus_c22[6] * (13 * c1[8] + 181 * c1c2[4] + 396 * c1c2[1] + 181 * c1c2[3] + 13 * c2[8]) * rad[2] +
         c12_minus_c22[4] * c12_plus_c22 * (59 * c1[8] - 228 * c1c2[4] - 2602 * c1c2[1] - 228 * c1c2[3] + 59 * c2[8]) * rad[4] -
         5 * c12_minus_c22[2] * (c1[12] - 438 * c1c2[6] + 3 * c1c2[8] + 2436 * c1c2[2] + 3 * c1c2[7] - 438 * c1c2[5] + c2[12]) * rad[6] -
         10 * c12_plus_c22 * (32 * c1[12] + 150 * c1c2[6] - 1698 * c1c2[8] + 3081 * c1c2[2] - 1698 * c1c2[7] + 150 * c1c2[5] + 32 * c2[12]) * rad[8] +
         2 * (308 * c1[12] - 1386 * c1c2[6] - 4482 * c1c2[8] + 12165 * c1c2[2] - 4482 * c1c2[7] - 1386 * c1c2[5] + 308 * c2[12]) * rad[10] -
         4 * c12_plus_c22 * (98 * c1[8] - 1365 * c1c2[4] + 2780 * c1c2[1] - 1365 * c1c2[3] + 98 * c2[8]) * rad[12] -
         4 * (26 * c1[8] + 515 * c1c2[4] - 1350 * c1c2[1] + 515 * c1c2[3] + 26 * c2[8]) * rad[14] +
         5 * c12_plus_c22 * (53 * c1[4] - 143 * c1[2]*c2[2] + 53 * c2[4]) * rad[16] -
         5 * (23 * c1[4] - 67 * c1[2]*c2[2] + 23 * c2[4]) * rad[18] +
         5 * c12_plus_c22 * rad[20] + 5 * rad[22]) * (abr[1] * abr[2] * abr[3] + abr[1] * abr[2] * abr[4] + abr[1] * abr[3] * abr[4] + abr[2] * abr[3] * abr[4]) * L_ab) / (10.0 * denominator[3]);

        fpair = (1.0 / (rad[1])) * (term[0] + term[1] + term[2] + term[3]) * factor_lj;

        if (eflag) {
            double UddR_local, UddA_local;
            UddR_local = (1.0/5.0) * (pi * c1c2[0]) * A2_param_eff * L_ab * (1 / denominator[1]) *
                   (5 * rad[22] + 5 * rad[20] * c12_plus_c22 -
                    5 * rad[18] * (23 * c1[4] - 67 * c1c2[0] + 23 * c2[4]) +
                    5 * rad[16] * c12_plus_c22 * (53 * c1[4] - 143 * c1c2[0] + 53 * c2[4]) -
                    4 * rad[14] * (26 * c1[8] + 515 * c1c2[4] - 1350 * c1c2[1] + 515 * c1c2[3] + 26 * c2[8]) -
                    4 * rad[12] * (c1[2] + c2[2]) * (98 * c1[8] - 1365 * c1c2[4] + 2780 * c1c2[1] - 1365 * c1c2[3] + 98 * c2[8]) +
                    2 * rad[10] * (308 * c1[12] - 1386 * c1c2[6] - 4482 * c1c2[8] + 12165 * c1c2[2] - 4482 * c1c2[7] -
                                   1386 * c1c2[5] + 308 * c2[12]) -
                    10 * rad[8] * c12_plus_c22 * (32 * c1[12] + 150 * c1c2[6] - 1698 * c1c2[8] +
                                                 3081 * c1c2[2] - 1698 * c1c2[7] + 150 * c1c2[5] + 32 * c2[12]) -
                    5 * rad[6] * c12_minus_c22[2] * (c1[12] - 438 * c1c2[6] + 3 * c1c2[8] +
                                                    2436 * c1c2[2] + 3 * c1c2[7] - 438 * c1c2[5] + c2[12]) +
                    rad[4] * c12_minus_c22[4] * (c1[2] + c2[2]) * (59 * c1[8] - 228 * c1c2[4] -
                                                                2602 * c1c2[1] - 228 * c1c2[3] + 59 * c2[8]) -
                    rad[2] * c12_minus_c22[6] * (13 * c1[8] + 181 * c1c2[4] + 396* c1c2[1] +
                                                181 * c1c2[3] + 13 * c2[8]) -
                    c12_minus_c22[8] * (c1[2] + c2[2]) * (c1[4] + 5 * c1c2[0] + c2[4]));

             UddA_local = -pow(pi, 2.0) * a12[itype][jtype] * c1c2[0] * L_ab * (2.0 * rad[4] - rad[2] * c12_plus_c22 - c12_minus_c22[2]) / (2.0 * denominator[0]);
             evdwl = UddR_local + UddA_local - offset[itype][jtype];
        }
        if (rad[1] <= dd) { // dd is sum of radii c1[1]+c2[1]
          error->one(FLERR, "Overlapping large/large in pair colloid");
        }
        break;
       } // End of switch statement

      if (eflag) evdwl *= factor_lj;

      f[i][0] += delx*fpair;
      f[i][1] += dely*fpair;
      f[i][2] += delz*fpair;

      if (newton_pair || j < nlocal) {
        f[j][0] -= delx*fpair;
        f[j][1] -= dely*fpair;
        f[j][2] -= delz*fpair;
      }

      if (evflag) ev_tally(i,j,nlocal,newton_pair,
                           evdwl,0.0,fpair,delx,dely,delz);
    } // End of jj loop (inner neighbor loop)
  } // End of ii loop (outer atom loop)

  if (vflag_fdotr) virial_fdotr_compute();
} 

void PairDisk::allocate()
{
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag,n+1,n+1,"pair:setflag");
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      setflag[i][j] = 0;

  memory->create(cutsq,n+1,n+1,"pair:cutsq");
  memory->create(form,n+1,n+1,"pair:form");
  memory->create(a12,n+1,n+1,"pair:a12");
  memory->create(sigma,n+1,n+1,"pair:sigma");
  memory->create(d1,n+1,n+1,"pair:d1");
  memory->create(d2,n+1,n+1,"pair:d2");
  memory->create(a1,n+1,n+1,"pair:a1");
  memory->create(a2,n+1,n+1,"pair:a2");
  memory->create(diameter,n+1,n+1,"pair:diameter");
  memory->create(cut,n+1,n+1,"pair:cut");
  memory->create(offset,n+1,n+1,"pair:offset");
  memory->create(sigma3,n+1,n+1,"pair:sigma3");
  memory->create(sigma6,n+1,n+1,"pair:sigma6");
  memory->create(lj1,n+1,n+1,"pair:lj1");
  memory->create(lj2,n+1,n+1,"pair:lj2");
  memory->create(lj3,n+1,n+1,"pair:lj3");
  memory->create(lj4,n+1,n+1,"pair:lj4");
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairDisk::settings(int narg, char **arg)
{  
  if (narg != 1) error->all(FLERR,"Illegal pair_style command");

  cut_global = utils::numeric(FLERR,arg[0],false,lmp);

  // reset cutoffs that have been explicitly set

  if (allocated) {
    int i,j;
    for (i = 1; i <= atom->ntypes; i++)
      for (j = i; j <= atom->ntypes; j++)
        if (setflag[i][j]) cut[i][j] = cut_global;
  }
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairDisk::coeff(int narg, char **arg)
{
  if (narg < 6 || narg > 7)
    error->all(FLERR,"Incorrect args for pair coefficients");
  if (!allocated) allocate();

  int ilo,ihi,jlo,jhi;
  utils::bounds(FLERR,arg[0],1,atom->ntypes,ilo,ihi,error);
  utils::bounds(FLERR,arg[1],1,atom->ntypes,jlo,jhi,error);

  double a12_one = utils::numeric(FLERR,arg[2],false,lmp);
  double sigma_one = utils::numeric(FLERR,arg[3],false,lmp);
  double d1_one = utils::numeric(FLERR,arg[4],false,lmp);
  double d2_one = utils::numeric(FLERR,arg[5],false,lmp);

  double cut_one = cut_global;
  if (narg == 7) cut_one = utils::numeric(FLERR,arg[6],false,lmp);

  if (d1_one < 0.0 || d2_one < 0.0)
    error->all(FLERR,"Invalid d1 or d2 value for pair colloid coeff");

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo,i); j <= jhi; j++) {
      a12[i][j] = a12_one;
      sigma[i][j] = sigma_one;
      if (i == j && d1_one != d2_one)
        error->all(FLERR,"Invalid d1 or d2 value for pair colloid coeff");
      d1[i][j] = d1_one;
      d2[i][j] = d2_one;
      diameter[i][j] = 0.5*(d1_one+d2_one);
      cut[i][j] = cut_one;
      setflag[i][j] = 1;
      count++;
    }
  }

  if (count == 0) error->all(FLERR,"Incorrect args for pair coefficients");
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairDisk::init_one(int i, int j)
{
  if (setflag[i][j] == 0) {
    a12[i][j] = mix_energy(a12[i][i],a12[j][j],sigma[i][i],sigma[j][j]);
    sigma[i][j] = mix_distance(sigma[i][i],sigma[j][j]);
    d1[i][j] = mix_distance(d1[i][i],d1[j][j]);
    d2[i][j] = mix_distance(d2[i][i],d2[j][j]);
    diameter[i][j] = 0.5 * (d1[i][j] + d2[i][j]);
    cut[i][j] = mix_distance(cut[i][i],cut[j][j]);
  }

  sigma3[i][j] = sigma[i][j]*sigma[i][j]*sigma[i][j];
  sigma6[i][j] = sigma3[i][j]*sigma3[i][j];

  if (d1[i][j] == 0.0 && d2[i][j] == 0.0) form[i][j] = SMALL_SMALL;
  else if (d1[i][j] == 0.0 || d2[i][j] == 0.0) form[i][j] = SMALL_LARGE;
  else form[i][j] = LARGE_LARGE;

  if (form[i][j] == SMALL_LARGE) {
    if (d1[i][j] > 0.0) a2[i][j] = 0.5*d1[i][j];
    else a2[i][j] = 0.5*d2[i][j];
    a2[j][i] = a2[i][j];
  } else if (form[i][j] == LARGE_LARGE) {
    a2[j][i] = a1[i][j] = 0.5*d1[i][j];
    a1[j][i] = a2[i][j] = 0.5*d2[i][j];
  }

  form[j][i] = form[i][j];
  a12[j][i] = a12[i][j];
  sigma[j][i] = sigma[i][j];
  sigma3[j][i] = sigma3[i][j];
  sigma6[j][i] = sigma6[i][j];
  diameter[j][i] = diameter[i][j];

  double epsilon = a12[i][j]/4.0;
  lj1[j][i] = lj1[i][j] = 48.0 * epsilon * sigma6[i][j] * sigma6[i][j];
  lj2[j][i] = lj2[i][j] = 24.0 * epsilon * sigma6[i][j];
  lj3[j][i] = lj3[i][j] = 4.0 * epsilon * sigma6[i][j] * sigma6[i][j];
  lj4[j][i] = lj4[i][j] = 4.0 * epsilon * sigma6[i][j];

  offset[j][i] = offset[i][j] = 0.0;
  if (offset_flag && (cut[i][j] > 0.0)) {
    double tmp;
    offset[j][i] = offset[i][j] =
      single(0,0,i,j,cut[i][j]*cut[i][j],0.0,1.0,tmp);
  }

  return cut[i][j];
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairDisk::write_restart(FILE *fp)
{
  write_restart_settings(fp);

  int i,j;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      fwrite(&setflag[i][j],sizeof(int),1,fp);
      if (setflag[i][j]) {
        fwrite(&a12[i][j],sizeof(double),1,fp);
        fwrite(&sigma[i][j],sizeof(double),1,fp);
        fwrite(&d1[i][j],sizeof(double),1,fp);
        fwrite(&d2[i][j],sizeof(double),1,fp);
        fwrite(&cut[i][j],sizeof(double),1,fp);
      }
    }
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairDisk::read_restart(FILE *fp)
{
  read_restart_settings(fp);
  allocate();

  int i,j;

  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      if (comm->me == 0) utils::sfread(FLERR,&setflag[i][j],sizeof(int),1,fp,nullptr,error);
      MPI_Bcast(&setflag[i][j],1,MPI_INT,0,world);
      if (setflag[i][j]) {
        if (comm->me == 0) {
          utils::sfread(FLERR,&a12[i][j],sizeof(double),1,fp,nullptr,error);
          utils::sfread(FLERR,&sigma[i][j],sizeof(double),1,fp,nullptr,error);
          utils::sfread(FLERR,&d1[i][j],sizeof(double),1,fp,nullptr,error);
          utils::sfread(FLERR,&d2[i][j],sizeof(double),1,fp,nullptr,error);
          utils::sfread(FLERR,&cut[i][j],sizeof(double),1,fp,nullptr,error);
        }
        MPI_Bcast(&a12[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&sigma[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&d1[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&d2[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&cut[i][j],1,MPI_DOUBLE,0,world);
      }
    }
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairDisk::write_restart_settings(FILE *fp)
{
  fwrite(&cut_global,sizeof(double),1,fp);
  fwrite(&offset_flag,sizeof(int),1,fp);
  fwrite(&mix_flag,sizeof(int),1,fp);
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairDisk::read_restart_settings(FILE *fp)
{
  int me = comm->me;
  if (me == 0) {
    utils::sfread(FLERR,&cut_global,sizeof(double),1,fp,nullptr,error);
    utils::sfread(FLERR,&offset_flag,sizeof(int),1,fp,nullptr,error);
    utils::sfread(FLERR,&mix_flag,sizeof(int),1,fp,nullptr,error);
  }
  MPI_Bcast(&cut_global,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&offset_flag,1,MPI_INT,0,world);
  MPI_Bcast(&mix_flag,1,MPI_INT,0,world);
}

/* ----------------------------------------------------------------------
   proc 0 writes to data file
------------------------------------------------------------------------- */

void PairDisk::write_data(FILE *fp)
{
  for (int i = 1; i <= atom->ntypes; i++)
    fprintf(fp,"%d %g %g %g %g\n",i,a12[i][i],sigma[i][i],d1[i][i],d2[i][i]);
}

/* ----------------------------------------------------------------------
   proc 0 writes all pairs to data file
------------------------------------------------------------------------- */

void PairDisk::write_data_all(FILE *fp)
{
  for (int i = 1; i <= atom->ntypes; i++)
    for (int j = i; j <= atom->ntypes; j++)
      fprintf(fp,"%d %g %g %g %g %g\n",i,
              a12[i][j],sigma[i][j],d1[i][j],d2[i][j],cut[i][j]);
}

double PairDisk::single(int /*i*/, int /*j*/,
                        int itype, int jtype,
                        double rsq,
                        double /*factor_coul*/, double factor_lj,
                        double &fforce)
{
  double rad[23]            = {0.0};  
  double c1[13]             = {0.0};  
  double c2[13]             = {0.0};
  double r2_minus_c2[12]    = {0.0};  
  double term[4]            = {0.0};
  double denominator[4]     = {0.0};
  double c1c2[9]            = {0.0};  
  double abr[5]             = {0.0};   
  double c12_minus_c22[9]   = {0.0};   
  const double pi = M_PI;
  const double Lambda_a = 1.0;
  const double Lambda_b = 1.0;
  double phi = 0.0;        
  fforce     = 0.0;

  switch (form[itype][jtype]) {

  /* -------- 1. SMALL–SMALL ----------------------------------------- */
  case SMALL_SMALL: {
    double r2inv = 1.0 / rsq;
    double r6inv = r2inv * r2inv * r2inv;
    double forcelj = r6inv * (lj1[itype][jtype]*r6inv - lj2[itype][jtype]);

    fforce = factor_lj * forcelj * r2inv;          // F/r * factor
    phi    = r6inv * (r6inv*lj3[itype][jtype] - lj4[itype][jtype])
             - offset[itype][jtype];
    break;
  }

  /* -------- 2. SMALL–LARGE  -------------------- */
  case SMALL_LARGE: {
    rad[1]=sqrt(rsq);  rad[2]=rsq;  rad[3]=rad[1]*rad[2];
    rad[4]=rad[2]*rad[2];  rad[5]=rad[1]*rad[4];  rad[6]=rad[3]*rad[3];
    rad[7]=rad[1]*rad[6];  rad[8]=rad[1]*rad[7];

    c2[1]=a2[itype][jtype];
    c2[2]=c2[1]*c2[1];  c2[4]=c2[2]*c2[2];  c2[6]=c2[4]*c2[2];  c2[8]=c2[4]*c2[4];

    r2_minus_c2[1]=rad[2]-c2[2];
    if (r2_minus_c2[1]<=0.0) error->one(FLERR,"SMALL_LARGE r<=a in single");

    r2_minus_c2[4]=pow(r2_minus_c2[1],4);
    r2_minus_c2[5]=r2_minus_c2[4]*r2_minus_c2[1];
    r2_minus_c2[10]=r2_minus_c2[5]*r2_minus_c2[5];
    r2_minus_c2[11]=r2_minus_c2[10]*r2_minus_c2[1];

    term[0]=(2.0*rad[1])/r2_minus_c2[4];
    term[1]=(4.0*rad[1]*(c2[2]+2.0*rad[2]))/r2_minus_c2[5];
    term[2]=(sigma6[itype][jtype]*
            (40*c2[6]*rad[1]+240*c2[4]*rad[3]+240*c2[2]*rad[5]+40*rad[7]))
            /(5.0*r2_minus_c2[10]);
    term[3]=(4.0*sigma6[itype][jtype]*rad[1]*
            (c2[8]+20*c2[6]*rad[2]+60*c2[4]*rad[4]+40*c2[2]*rad[6]+5*rad[8]))
            /r2_minus_c2[11];

    fforce = pi * c2[2] * a12[itype][jtype] * Lambda_a *
             (term[0]-term[1]-term[2]+term[3]) * factor_lj / rad[1];

    double U_A = -0.5 * (2*c2[2]*rad[2]+c2[4]) / r2_minus_c2[4];
    double poly = (5*rad[8]+40*c2[2]*rad[6]+60*c2[4]*rad[4]+20*c2[6]*rad[2]+c2[8]);
    double U_R = 0.2 * c2[2] * poly / r2_minus_c2[10];

    phi = pi * a12[itype][jtype] * Lambda_a *
          (U_A + sigma6[itype][jtype]*U_R) - offset[itype][jtype];
    break;
  }

  /* -------- 3. LARGE–LARGE ------- */
  case LARGE_LARGE: {
    rad[1]=sqrt(rsq);
    for(int k=2;k<=22;++k) rad[k]=rad[1]*rad[k-1];

    c1[1]=a1[itype][jtype];  c2[1]=a2[itype][jtype];
    c1[2]=c1[1]*c1[1];  c2[2]=c2[1]*c2[1];
    c1[4]=c1[2]*c1[2];  c2[4]=c2[2]*c2[2];
    c1[6]=c1[4]*c1[2];  c2[6]=c2[4]*c2[2];
    c1[8]=c1[4]*c1[4];  c2[8]=c2[4]*c2[4];
    c1[10]=c1[4]*c1[6]; c2[10]=c2[4]*c2[6];
    c1[12]=c1[6]*c1[6]; c2[12]=c2[6]*c2[6];
    c1c2[0]=c1[2]*c2[2]; c1c2[1]=c1[4]*c2[4]; c1c2[2]=c1[6]*c2[6];
    c1c2[3]=c1[2]*c2[6]; c1c2[4]=c1[6]*c2[2];
    c1c2[5]=c1[2]*c2[10]; c1c2[6]=c1[10]*c2[2];
    c1c2[7]=c1[4]*c2[8];  c1c2[8]=c1[8]*c2[4];
    abr[1]=-c1[1]-c2[1]+rad[1];
    abr[2]= c1[1]-c2[1]+rad[1];
    abr[3]=-c1[1]+c2[1]+rad[1];
    abr[4]= c1[1]+c2[1]+rad[1];

    double ab_r = abr[1]*abr[2]*abr[3]*abr[4];
    if (ab_r<=0.0)
      error->one(FLERR,"LARGE_LARGE non-positive ab_r in single");

    denominator[0]=pow(ab_r,  5.0/2.0);
    denominator[1]=pow(ab_r, 17.0/2.0);
    denominator[2]=pow(ab_r,  7.0/2.0);
    denominator[3]=pow(ab_r, 19.0/2.0);

    double c12_plus_c22      = c1[2]+c2[2];
    c12_minus_c22[1]=c1[2]-c2[2];
    c12_minus_c22[2]=c12_minus_c22[1]*c12_minus_c22[1];
    c12_minus_c22[4]=c12_minus_c22[2]*c12_minus_c22[2];
    c12_minus_c22[6]=c12_minus_c22[2]*c12_minus_c22[4];
    c12_minus_c22[8]=c12_minus_c22[4]*c12_minus_c22[4];

    double A2eff = a12[itype][jtype]*sigma6[itype][jtype];
    double L_ab  = Lambda_a*Lambda_b;

    term[0]=(a12[itype][jtype]*c1[2]*c2[2]*pi*pi*
             (-2*c12_plus_c22*rad[1]+8*rad[3])*L_ab)/(2.0*denominator[0]);
    term[1]=-(A2eff*c1[2]*c2[2]*pi*pi*
              (-2*c12_minus_c22[6]*(13*c1[8]+181*c1c2[4]+396*c1c2[1]
                                    +181*c1c2[3]+13*c2[8])*rad[1]
               +4*c12_minus_c22[4]*c12_plus_c22*
                 (59*c1[8]-228*c1c2[4]-2602*c1c2[1]-228*c1c2[3]+59*c2[8])*rad[3]
               -30*c12_minus_c22[2]*
                 (c1[12]-438*c1c2[6]+3*c1c2[8]+2436*c1c2[2]+3*c1c2[7]
                  -438*c1c2[5]+c2[12])*rad[5]
               -80*c12_plus_c22*
                 (32*c1[12]+150*c1c2[6]-1698*c1c2[8]+3081*c1c2[2]
                  -1698*c1c2[7]+150*c1c2[5]+32*c2[12])*rad[7]
               +20*(308*c1[12]-1386*c1c2[6]-4482*c1c2[8]+12165*c1c2[2]
                    -4482*c1c2[7]-1386*c1c2[5]+308*c2[12])*rad[9]
               -48*c12_plus_c22*
                 (98*c1[8]-1365*c1c2[4]+2780*c1c2[1]-1365*c1c2[3]+98*c2[8])*rad[11]
               -56*(26*c1[8]+515*c1c2[4]-1350*c1c2[1]+515*c1c2[3]+26*c2[8])*rad[13]
               +80*c12_plus_c22*
                 (53*c1[4]-143*c1c2[0]+53*c2[4])*rad[15]
               -90*(23*c1[4]-67*c1c2[0]+23*c2[4])*rad[17]
               +100*c12_plus_c22*rad[19]+110*rad[21])*L_ab)
               /(5.0*denominator[1]);
    term[2]=-(5*c1[2]*a12[itype][jtype]*c2[2]*pi*pi*
             (-c12_minus_c22[2]-c12_plus_c22*rad[2]+2*rad[4])*
             (abr[1]*abr[2]*abr[3]+abr[1]*abr[2]*abr[4]
              +abr[1]*abr[3]*abr[4]+abr[2]*abr[3]*abr[4])*L_ab)
             /(4.0*denominator[2]);
    term[3]=(17*c1[2]*A2eff*c2[2]*pi*pi*
            (-c12_minus_c22[8]*c12_plus_c22*(c1[4]+5*c1c2[0]+c2[4])
             -c12_minus_c22[6]*(13*c1[8]+181*c1c2[4]+396*c1c2[1]
                                +181*c1c2[3]+13*c2[8])*rad[2]
             +c12_minus_c22[4]*c12_plus_c22*
               (59*c1[8]-228*c1c2[4]-2602*c1c2[1]-228*c1c2[3]+59*c2[8])*rad[4]
             -5*c12_minus_c22[2]*
               (c1[12]-438*c1c2[6]+3*c1c2[8]+2436*c1c2[2]+3*c1c2[7]
                -438*c1c2[5]+c2[12])*rad[6]
             -10*c12_plus_c22*
               (32*c1[12]+150*c1c2[6]-1698*c1c2[8]+3081*c1c2[2]
                -1698*c1c2[7]+150*c1c2[5]+32*c2[12])*rad[8]
             +2*(308*c1[12]-1386*c1c2[6]-4482*c1c2[8]+12165*c1c2[2]
                  -4482*c1c2[7]-1386*c1c2[5]+308*c2[12])*rad[10]
             -4*c12_plus_c22*
               (98*c1[8]-1365*c1c2[4]+2780*c1c2[1]-1365*c1c2[3]+98*c2[8])*rad[12]
             -4*(26*c1[8]+515*c1c2[4]-1350*c1c2[1]+515*c1c2[3]+26*c2[8])*rad[14]
             +5*c12_plus_c22*
               (53*c1[4]-143*c1c2[0]+53*c2[4])*rad[16]
             -5*(23*c1[4]-67*c1c2[0]+23*c2[4])*rad[18]
             +5*c12_plus_c22*rad[20]+5*rad[22])*
            (abr[1]*abr[2]*abr[3]+abr[1]*abr[2]*abr[4]
             +abr[1]*abr[3]*abr[4]+abr[2]*abr[3]*abr[4])*L_ab)
            /(10.0*denominator[3]);

    fforce = (term[0]+term[1]+term[2]+term[3]) * factor_lj / rad[1];

    double UddR = (1.0/5.0) * (pi * c1c2[0]) *  sigma6[itype][jtype] * sigma6[itype][jtype] * L_ab * (1 / denominator[1]) *
                   (5 * rad[22] + 5 * rad[20] * c12_plus_c22 -
                    5 * rad[18] * (23 * c1[4] - 67 * c1c2[0] + 23 * c2[4]) +
                    5 * rad[16] * c12_plus_c22 * (53 * c1[4] - 143 * c1c2[0] + 53 * c2[4]) -
                    4 * rad[14] * (26 * c1[8] + 515 * c1c2[4] - 1350 * c1c2[1] + 515 * c1c2[3] + 26 * c2[8]) -
                    4 * rad[12] * (c1[2] + c2[2]) * (98 * c1[8] - 1365 * c1c2[4] + 2780 * c1c2[1] - 1365 * c1c2[3] + 98 * c2[8]) +
                    2 * rad[10] * (308 * c1[12] - 1386 * c1c2[6] - 4482 * c1c2[8] + 12165 * c1c2[2] - 4482 * c1c2[7] -
                                   1386 * c1c2[5] + 308 * c2[12]) -
                    10 * rad[8] * c12_plus_c22 * (32 * c1[12] + 150 * c1c2[6] - 1698 * c1c2[8] +
                                                 3081 * c1c2[2] - 1698 * c1c2[7] + 150 * c1c2[5] + 32 * c2[12]) -
                    5 * rad[6] * c12_minus_c22[2] * (c1[12] - 438 * c1c2[6] + 3 * c1c2[8] +
                                                    2436 * c1c2[2] + 3 * c1c2[7] - 438 * c1c2[5] + c2[12]) +
                    rad[4] * c12_minus_c22[4] * (c1[2] + c2[2]) * (59 * c1[8] - 228 * c1c2[4] -
                                                                2602 * c1c2[1] - 228 * c1c2[3] + 59 * c2[8]) -
                    rad[2] * c12_minus_c22[6] * (13 * c1[8] + 181 * c1c2[4] + 396* c1c2[1] +
                                                181 * c1c2[3] + 13 * c2[8]) -
                    c12_minus_c22[8] * (c1[2] + c2[2]) * (c1[4] + 5 * c1c2[0] + c2[4]));
    double UddA = -pow(pi, 2.0) * a12[itype][jtype] * c1c2[0] * L_ab * (2.0 * rad[4] - rad[2] * c12_plus_c22 - c12_minus_c22[2]) / (2.0 * denominator[0]);

    phi = UddR + UddA - offset[itype][jtype];
    break;
  }
  } 
  return factor_lj * phi;
}
