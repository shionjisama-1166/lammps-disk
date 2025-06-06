/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* -*- c++ -*- ------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.
   Distributed under the GNU General Public License.  See the README file.
------------------------------------------------------------------------- */

#ifndef LMP_PAIR_DISK_H
#define LMP_PAIR_DISK_H

#include "pair.h"

#if !defined(PAIR_CLASS)
namespace LAMMPS_NS {


class PairDisk : public Pair {
 public:
  PairDisk(class LAMMPS *);
  ~PairDisk() override;

  /* main hooks */
  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  double init_one(int, int) override;

  /* restart / I/O */
  void write_restart(FILE *) override;
  void read_restart(FILE *) override;
  void write_restart_settings(FILE *) override;
  void read_restart_settings(FILE *) override;
  void write_data(FILE *) override;
  void write_data_all(FILE *) override;
  double single(int, int, int, int,
                double, double, double, double &) override;

 protected:
  enum { SMALL_SMALL, SMALL_LARGE, LARGE_LARGE };

  double cut_global;
  double **cut;
  double **a12, **d1, **d2, **diameter, **a1, **a2, **offset;
  double **sigma, **sigma3, **sigma6;
  double **lj1, **lj2, **lj3, **lj4;
  int    **form;

  void allocate();
};

}
#endif 
#endif
/* ---- register keyword so it appears in style_pair.h ------------------ */
#ifdef PAIR_CLASS
// clang-format off
PairStyle(disk,PairDisk);
// clang-format on
#endif
