// SPDX-License-Identifier: GPL-3.0-or-later
// Registration only. The force implementation is compiled unchanged from pinned LAMFOAM.
#include "lammpsplugin.h"
#include "version.h"
#include "pair_lamfoam_vdw.h"
static LAMMPS_NS::Pair *creator(LAMMPS_NS::LAMMPS *lmp) {
  return new LAMMPS_NS::PairLamfoamVdw(lmp);
}
extern "C" void lammpsplugin_init(void *lmp, void *handle, void *regfunc) {
  lammpsplugin_t p{};
  p.version=LAMMPS_VERSION; p.style="pair"; p.name="lamfoam/vdw";
  p.info="Unmodified LAMFOAM vdW pair for controlled relaxation comparison";
  p.author="LAMFOAM contributors"; p.handle=handle;
  p.creator.v1=reinterpret_cast<lammpsplugin_factory1*>(&creator);
  reinterpret_cast<lammpsplugin_regfunc>(regfunc)(&p,lmp);
}
