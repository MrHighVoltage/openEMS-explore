"""
 Rectangular Waveguide with local absorber test

 (c) 20@3-2025 Gadi Lahav <gadi@rfwithcare.com>

"""

### Import Libraries
import os
import sys
import tempfile


def _bootstrap_local_openems_runtime():
    """Prefer freshly built openEMS libs from the local build directory.

    This must run before importing CSXCAD/openEMS C-extension modules.
    Set OPENEMS_USE_LOCAL_BUILD=0 to disable this behavior.
    """
    if os.environ.get("OPENEMS_USE_LOCAL_BUILD", "1") == "0":
        return

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    build_dir = os.path.join(repo_root, "build")
    libopenems = os.path.join(build_dir, "libopenEMS.so")

    if not os.path.exists(libopenems):
        return

    cur = os.environ.get("LD_LIBRARY_PATH", "")
    paths = [p for p in cur.split(":") if p]

    if paths and os.path.abspath(paths[0]) == os.path.abspath(build_dir):
        return

    paths = [p for p in paths if os.path.abspath(p) != os.path.abspath(build_dir)]
    os.environ["LD_LIBRARY_PATH"] = ":".join([build_dir] + paths)

    # Re-exec so the dynamic loader uses the updated library search path.
    os.execvpe(sys.executable, [sys.executable] + sys.argv, os.environ)


_bootstrap_local_openems_runtime()

from pylab import *

from CSXCAD  import ContinuousStructure
from openEMS import openEMS
from openEMS.physical_constants import *

from CSXCAD.CSProperties import ABCtype

### Setup results folder
script_name = os.path.splitext(os.path.basename(__file__))[0]
results_folder = os.path.join(tempfile.gettempdir(), script_name)
if not os.path.exists(results_folder):
    os.makedirs(results_folder)
print(f"Results will be saved to: {results_folder}")

### Setup the simulation
Sim_Path = os.path.join(tempfile.gettempdir(), 'Rect_WG')

post_proc_only = False
unit = 1e-6; #drawing unit in um

# waveguide dimensions
# WR42
a = 30700;   #waveguide width
b = 4300;    #waveguide height
length = 50000;

# frequency range of interest
f_start = 50e9;
f_0     = 100e9;
f_stop  = 150e9;
lambda0 = C0/f_0/unit;

#waveguide TE-mode definition
TE_mode = 'TE10';

#targeted mesh resolution
# mesh_res = lambda0/30
mesh_res = lambda0/50

### Setup FDTD parameter & excitation function
FDTD = openEMS(NrTS=1e4);
FDTD.SetGaussExcite(0.5*(f_start+f_stop),0.5*(f_stop-f_start));

# boundary conditions
FDTD.SetBoundaryCond([0, 0, 0, 0, 0, 0]);

### Setup geometry & mesh
CSX = ContinuousStructure()
FDTD.SetCSX(CSX)
mesh = CSX.GetGrid()
mesh.SetDeltaUnit(unit)

mesh.AddLine('x', [0, a])
mesh.AddLine('y', [0, b])
mesh.AddLine('z', [0, length])

## Apply the waveguide port
ports = []
start=[0, 0, 3*mesh_res];
stop =[a, b, 4*mesh_res];
mesh.AddLine('z', [start[2], stop[2]])
ports.append(FDTD.AddRectWaveGuidePort( 0, start, stop, 'z', a*unit, b*unit, TE_mode, 1))

start=[0, 0, length-3*mesh_res];
stop =[a, b, length-4*mesh_res];
mesh.AddLine('z', [start[2], stop[2]])
ports.append(FDTD.AddRectWaveGuidePort( 1, start, stop, 'z', a*unit, b*unit, TE_mode))

# Add PEC Boxes
pecBlocks = CSX.AddMetal('PEC')

start = [0, 0, 0.0]
stop  = [ a, b, 1*mesh_res]
pecBlocks.AddBox(priority=5, start=start, stop=stop) # add a box-primitive to the metal property 'patch'
start = [ 0, 0, length-1*mesh_res]
stop  = [a, b, length]
pecBlocks.AddBox(priority=5, start=start, stop=stop) # add a box-primitive to the metal property 'patch'

#start = [0, 0., 2*mesh_res]
#stop  = [a, b, 2*mesh_res]

#mesh.AddLine('z', [stop[2]])
#abs1 = CSX.AddAbsorbingBC('abs1',NormalSignPositive = True, AbsorbingBoundaryType = ABCtype.MUR_1ST_SA, PhaseVelocity = 3.6196e+08)
#abs1.AddBox(start, stop, priority=6)



#start=[0, 0, length-2*mesh_res];
#stop =[a, b, length-2*mesh_res];

#mesh.AddLine('z', [stop[2]])
#abs2 = CSX.AddAbsorbingBC('abs2',NormalSignPositive = False, AbsorbingBoundaryType = ABCtype.MUR_1ST_SA, PhaseVelocity = 3.6196e+08)
#abs2.AddBox(start, stop, priority=6)

mesh.SmoothMeshLines('all', mesh_res, ratio=1.4)

### Define dump box...
#Et = CSX.AddDump('Et', file_type=0, sub_sampling=[2,2,2])
#start = [0, 0, 0];
#stop  = [a, b, length];
#Et.AddBox(start, stop);

### Run the simulation
if os.environ.get("OPENEMS_SHOW_CSXCAD", "0") == "1":
    CSX_file = os.path.join(Sim_Path, 'rect_wg.xml')
    if not os.path.exists(Sim_Path):
        os.mkdir(Sim_Path)
    CSX.Write2XML(CSX_file)
    from CSXCAD import AppCSXCAD_BIN
    os.system(AppCSXCAD_BIN + ' "{}"'.format(CSX_file))

if not post_proc_only:
    gpu_profile = os.environ.get("OPENEMS_GPU_PROFILE", "0") == "1"
    print(f"GPU config: profiling={gpu_profile}")
    FDTD.Run(
        Sim_Path,
        cleanup=True,
        engine="gpu",
        gpu_profile=gpu_profile,
    )

### Postprocessing & plotting
freq = linspace(f_start,f_stop,201)
for port in ports:
    port.CalcPort(Sim_Path, freq)

s11 = ports[0].uf_ref / ports[0].uf_inc
s21 = ports[1].uf_ref / ports[0].uf_inc
ZL  = ports[0].uf_tot / ports[0].if_tot
ZL_a = ports[0].ZL # analytic waveguide impedance

## Plot s-parameter
fig1 = figure()
plot(freq*1e-6,20*log10(abs(s11)),'k-',linewidth=2, label='$S_{11}$')
grid()
plot(freq*1e-6,20*log10(abs(s21)),'r--',linewidth=2, label='$S_{21}$')
legend();
ylabel('S-Parameter (dB)')
xlabel(r'frequency (MHz) $\rightarrow$')
s_param_file = os.path.join(results_folder, 's_parameter.png')
fig1.savefig(s_param_file, dpi=150, bbox_inches='tight')
print(f"Saved S-Parameter plot to: {s_param_file}")
close(fig1)

## Compare analytic and numerical wave-impedance
fig2 = figure()
plot(freq*1e-6,real(ZL), linewidth=2, label='$\Re\{Z_L\}$')
grid()
plot(freq*1e-6,imag(ZL),'r--', linewidth=2, label='$\Im\{Z_L\}$')
plot(freq*1e-6,ZL_a,'g-.',linewidth=2, label='$Z_{L, analytic}$')
ylabel('ZL $(\Omega)$')
xlabel(r'frequency (MHz) $\rightarrow$')
legend()
zl_file = os.path.join(results_folder, 'wave_impedance.png')
fig2.savefig(zl_file, dpi=150, bbox_inches='tight')
print(f"Saved Wave Impedance plot to: {zl_file}")
close(fig2)

print(f"All results saved to: {results_folder}")
