import os
import sys


### Import Libraries
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

repo_python = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_python not in sys.path:
    sys.path.insert(0, repo_python)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'modules')))

import modules.util_stackup_reader as stackup_reader
import modules.util_gds_reader as gds_reader
import modules.util_utilities as utilities
import modules.util_simulation_setup as simulation_setup
import modules.util_meshlines as util_meshlines

from openEMS import openEMS
import numpy as np
import matplotlib.pyplot as plt

# Model comments
#
# Dual dipole 245 GHz, design by IHP Klaus Schmalz
# Port defined in GDSIIfile on layer 201
# nf2ff_box field sampling at boundary for pattern calculation


# ======================== workflow settings ================================

# preview model/mesh only?
# postprocess existing data without re-running simulation?
preview_only = os.environ.get("OPENEMS_PREVIEW_ONLY", "0") == "1"
postprocess_only = os.environ.get("OPENEMS_POSTPROCESS_ONLY", "0") == "1"

# ===================== input files and path settings =======================

gds_filename = "mWATTBAT_openEMS.gds"   # geometries
XML_filename = "SG13G2_200um.xml"       # stackup

# preprocess GDSII for safe handling of cutouts/holes?
preprocess_gds = True

# merge via polygons with distance less than .. microns, set to 0 to disable via merging.
merge_polygon_size = 2

# get path for this simulation file
script_path = utilities.get_script_path(__file__)

# use script filename as model basename; optional tags keep debug output isolated
model_basename = utilities.get_basename(__file__)
run_tag = os.environ.get("OPENEMS_RUN_TAG", "").strip()
if run_tag:
    model_basename += "_" + run_tag

# set and create directory for simulation output
sim_path = utilities.create_sim_path (script_path,model_basename)
print('Simulation data directory: ', sim_path)

# change current path to model script path
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ======================== simulation settings ================================

unit = 1e-6   # geometry dimensions and all lengths unit is µm (micrometer)
margin = float(os.environ.get("OPENEMS_MARGIN", "100"))
if margin < 0:
    raise ValueError("OPENEMS_MARGIN must be non-negative")

fstart = 100e9
fstop  = 200e9
num_freq = 1001

ftarget = 150e9  # frequency for antenna pattern calculation

refined_cellsize = 1  # mesh cell size in conductor region

# choices for boundary:
# 'PEC' : perfect electric conductor (default)
# 'PMC' : perfect magnetic conductor, useful for symmetries
# 'MUR' : simple MUR absorbing boundary conditions
# 'PML_8' : PML absorbing boundary conditions
Boundaries = ['PML_8', 'PML_8', 'PML_8', 'PML_8', 'PML_8', 'PML_8']
boundary_override = os.environ.get("OPENEMS_BOUNDARY_MODE", "").strip()
if boundary_override:
    Boundaries = [boundary_override] * 6
boundary_list = os.environ.get("OPENEMS_BOUNDARIES", "").strip()
if boundary_list:
    parsed_boundaries = [item.strip() for item in boundary_list.split(",")]
    if len(parsed_boundaries) != 6:
        raise ValueError("OPENEMS_BOUNDARIES must contain six comma-separated values")
    Boundaries = parsed_boundaries

cells_per_wavelength = float(os.environ.get("OPENEMS_CELLS_PER_WAVELENGTH", "12"))
if cells_per_wavelength < 10:
    raise ValueError("OPENEMS_CELLS_PER_WAVELENGTH must be at least 10")
energy_limit_db = float(os.environ.get("OPENEMS_ENERGY_LIMIT_DB", "-45"))
if not np.isfinite(energy_limit_db) or energy_limit_db >= 0:
    raise ValueError("OPENEMS_ENERGY_LIMIT_DB must be a finite negative dB value")
energy_limit = energy_limit_db  # end criteria for residual energy (dB)

# ports from GDSII Data, polygon geometry from specified special layer
# note that for multiport simulation, excitations are switched on/off in simulation_setup.createSimulation below
simulation_ports = simulation_setup.all_simulation_ports()
simulation_ports.add_port(simulation_setup.simulation_port(portnumber=1, voltage=1, port_Z0=85, source_layernum=201, target_layername='TopMetal2', direction='y'))


def add_compact_gap_feed_extensions(allpolygons, simulation_ports, metals_list):
    """Replace a long GDS port marker by two feed-stub extensions and a small gap.

    The source layer is a layout marker, not metal.  In compact_gap mode, add
    simulation-only TopMetal2 rectangles that extend its two terminals to a
    short central gap.  The port helper then places the lumped source across
    that same gap.  This leaves the input GDS untouched and makes the changed
    feed geometry explicit in the generated CSX model.
    """
    port_mode = os.environ.get("OPENEMS_PORT_MODE", "").strip().lower()
    if port_mode != "compact_gap":
        return

    gap_length = float(os.environ.get("OPENEMS_PORT_GAP_LENGTH", "1.0"))
    if not np.isfinite(gap_length) or gap_length <= 0:
        raise ValueError("OPENEMS_PORT_GAP_LENGTH must be a finite positive value")

    source_layer = simulation_ports.portlayers[0]
    marker_polygons = [
        poly for poly in allpolygons.polygons if poly.layernum == source_layer
    ]
    if len(marker_polygons) != 1:
        raise ValueError(
            "compact_gap mode requires exactly one GDS port-marker polygon on "
            f"layer {source_layer}, found {len(marker_polygons)}"
        )

    port = simulation_ports.get_port_by_layernumber(source_layer)
    if port is None or port.direction != "y" or port.target_layername is None:
        raise ValueError(
            "compact_gap mode currently requires one y-directed in-plane port"
        )

    marker = marker_polygons[0]
    marker_length = marker.ymax - marker.ymin
    if gap_length >= marker_length:
        raise ValueError(
            "OPENEMS_PORT_GAP_LENGTH must be shorter than the GDS port-marker "
            f"length ({marker_length:g} um)"
        )

    target_metal = metals_list.getbylayername(port.target_layername)
    if target_metal is None:
        raise ValueError(
            f"Cannot find port target layer {port.target_layername} in stackup"
        )

    gap_center = 0.5 * (marker.ymin + marker.ymax)
    gap_ymin = gap_center - 0.5 * gap_length
    gap_ymax = gap_center + 0.5 * gap_length
    target_layernum = int(target_metal.layernum)

    # These rectangles contact the original lower/upper TopMetal2 stubs at the
    # marker boundaries and terminate at the compact feed gap.
    allpolygons.add_rectangle(
        marker.xmin, marker.ymin, marker.xmax, gap_ymin, target_layernum
    )
    allpolygons.add_rectangle(
        marker.xmin, gap_ymax, marker.xmax, marker.ymax, target_layernum
    )
    print(
        "Compact feed geometry: extended TopMetal2 stubs to "
        f"y=[{gap_ymin:g}, {gap_ymax:g}] um; "
        f"lumped-port gap={gap_length:g} um"
    )


# ======================== simulation ================================

# get technology stackup data
materials_list, dielectrics_list, metals_list = stackup_reader.read_substrate (XML_filename)
# get list of layers from technology
layernumbers = metals_list.getlayernumbers()
layernumbers.extend(simulation_ports.portlayers)

# read geometries from GDSII, only purpose 0
allpolygons = gds_reader.read_gds(gds_filename, layernumbers, purposelist=[0], metals_list=metals_list, preprocess=preprocess_gds, merge_polygon_size=merge_polygon_size)
add_compact_gap_feed_extensions(allpolygons, simulation_ports, metals_list)

# calculate maximum cellsize from wavelength in dielectric
wavelength_air = 3e8/fstop / unit
max_cellsize = (wavelength_air)/(np.sqrt(materials_list.eps_max)*cells_per_wavelength)
air_around = float(os.environ.get("OPENEMS_AIR_AROUND", str(0.5*wavelength_air)))
if air_around < 0:
    raise ValueError("OPENEMS_AIR_AROUND must be non-negative")


########### create model, run and post-process ###########

# Prepare simulation for port 1 excitation
excite_ports = [1]  # list of ports that are excited for this simulation run
# Multi-face PML and the strongly graded mWATTBAT mesh need additional
# timestep margin; 0.4 is stable on both the GPU and AVX2 engines.
timestep_factor = float(os.environ.get("OPENEMS_TIMESTEP_FACTOR", "0.4"))
if not 0 < timestep_factor <= 1:
    raise ValueError("OPENEMS_TIMESTEP_FACTOR must be in the range (0, 1]")
nr_ts = int(os.environ.get("OPENEMS_NR_TS", "1000000000"))
if nr_ts < 1:
    raise ValueError("OPENEMS_NR_TS must be positive")
FDTD = openEMS(
    NrTS=nr_ts,
    EndCriteria=np.exp(energy_limit/10 * np.log(10)),
    TimeStepFactor=timestep_factor,
)
FDTD.SetGaussExcite( (fstart+fstop)/2, (fstop-fstart)/2 )
FDTD.SetBoundaryCond( Boundaries )

FDTD = simulation_setup.setupSimulation(
    excite_ports,
    simulation_ports,
    FDTD,
    materials_list,
    dielectrics_list,
    metals_list,
    allpolygons,
    max_cellsize,
    refined_cellsize,
    margin,
    unit,
    xy_mesh_function = util_meshlines.create_xy_mesh_from_polygons,
    air_around = air_around
    )

# add nf2ff box for antenna pattern calculation
nf2ff_box = FDTD.CreateNF2FFBox(opt_resolution = [max_cellsize]*3, frequency = [ftarget])

# run simulation
simulation_engine = os.environ.get("OPENEMS_ENGINE", "avx2_multithreaded")
run_options = {}
if simulation_engine == "gpu":
    run_options["gpu_profile"] = os.environ.get("OPENEMS_GPU_PROFILE", "1") != "0"
sub1_data_path = simulation_setup.runSimulation(
    excite_ports,
    FDTD,
    sim_path,
    model_basename,
    preview_only,
    postprocess_only,
    force_simulation=os.environ.get("OPENEMS_FORCE_SIMULATION", "0") == "1",
    engine=simulation_engine,
    run_options=run_options,
)

# simulation is finished, get results, CSX port definition is read from simulation ports object
CSX_port1 = simulation_ports.get_port_by_number(1).CSXport

# definition of some utility functions
def dB(value, factor=20):
    return factor*np.log10(np.abs(value))
def dBm(value, factor=20):
    return dB(value/1e-3, factor=factor)

# evaluate results for 1-port simulation
if not preview_only:
    f = np.linspace(fstart, fstop, num_freq)

    s11 = utilities.calculate_Sij(1, 1, f, sim_path, simulation_ports)
    s11_dB = dB(s11)

    # write Touchstone S1P file
    s1p_name = os.path.join(sim_path, model_basename + '.s1p')
    utilities.write_snp (np.array([s11]),f, s1p_name)

    # plot return loss
    fig, axis = plt.subplots(num="Return Loss", tight_layout=True)
    axis.plot(f/1e9, s11_dB, 'k-', linewidth=2, label='S11 (dB)')
    axis.grid()
    axis.set_xmargin(0)
    axis.set_xlabel('Frequency (GHz)')
    axis.set_title("Return Loss")
    axis.legend()

    print('Calculating antenna pattern, this will take a while!')

    theta = np.arange(-180.0, 180.0, 2.0)
    phi   = [0., 90.]
    nf2ff_res = nf2ff_box.CalcNF2FF(sub1_data_path, ftarget, theta, phi)

    # INPUT POWER values into feed port at evaluation frequency
    # P_acc = accepted power (incoming - reflected)
    # P_inc = incident power (incoming), regardless of reflection

    # get accepted power into port
    Pin_accepted = np.interp(ftarget, f, CSX_port1.P_acc)
    # get incident power
    P_incident = np.interp(ftarget, f, CSX_port1.P_inc)

    # get total radiated power at nf2ff frequency
    Prad_total = nf2ff_res.Prad[0]
    # get directivity (peak value) at nf2ff frequency
    Dmax_dB = dB(nf2ff_res.Dmax[0], 10)
    # calculate radiation efficiency (lossless antenna is 1, regardless of input matching)
    radiation_efficiency = Prad_total/Pin_accepted
    radiation_efficiency_dB = dB(radiation_efficiency, 10)  # negative for lossy antenna
    # loss from reflection due to mismatch
    mismatch_loss =  Pin_accepted/P_incident
    mismatch_loss_dB = dB(mismatch_loss, 10)

    # evaluate far field components
    # normalize E field components to peak value of absolute field, and multiply by directivity
    directivity_abs  = dB(nf2ff_res.E_norm[0]/np.max(nf2ff_res.E_norm[0])) + Dmax_dB
    directivity_abs_xz = directivity_abs[:,0] # get field in xz plane, that is phi=0, which is index 0
    directivity_abs_yz = directivity_abs[:,1] # get field in yz plane, that is phi=90, which is index 1

    gain_abs_xz = directivity_abs_xz + radiation_efficiency_dB
    gain_abs_yz = directivity_abs_yz + radiation_efficiency_dB

    # display radiated power and directivity
    print(f"Antenna parameters at {ftarget/1e9:g} GHz:")
    print(f"    Incident power P_incident  =  {P_incident:.3e} W  =  {dBm(P_incident, 10):g} dBm")
    print(f"    Accepted power Pin_accepted  =  {Pin_accepted:.3e} W  =  {dBm(Pin_accepted, 10):g} dBm")
    print(f"    Radiated power Prad_total  =  {Prad_total:.3e} W  =  {dBm(Prad_total, 10):g} dBm")
    print(f"    Directivity Dmax  =  {Dmax_dB:g} dBi")
    print(f"    efficiency nu_rad  =  {100*radiation_efficiency:g} %  =  {radiation_efficiency_dB:g} dB")
    print(f"    mismatch loss  =  {mismatch_loss:g} (linear)  =  {mismatch_loss_dB:g} dB")


    fig, axis = plt.subplots(num="Directivity", tight_layout=True)
    axis.plot(theta, np.squeeze(directivity_abs_xz), 'k-',  linewidth=2, label='xz-plane')
    axis.plot(theta, np.squeeze(directivity_abs_yz), 'r--', linewidth=2, label='yz-plane')
    axis.grid()
    axis.set_xmargin(0)
    axis.set_xlabel('Theta (deg)')
    axis.set_ylabel('Directivity (dBi)')
    axis.set_title(f'Directivity at Frequency: {ftarget/1e9:g} GHz')
    axis.legend()

    fig, axis = plt.subplots(num="Gain", tight_layout=True)
    axis.plot(theta, np.squeeze(gain_abs_xz), 'k-',  linewidth=2, label='xz-plane')
    axis.plot(theta, np.squeeze(gain_abs_yz), 'r--', linewidth=2, label='yz-plane')
    axis.grid()
    axis.set_xmargin(0)
    axis.set_xlabel('Theta (deg)')
    axis.set_ylabel('Gain (dBi)')
    axis.set_title(f'Gain at Frequency: {ftarget/1e9:g} GHz')
    axis.legend()

    # show all plots
    #plt.show()
    plt.savefig('results.png', bbox_inches='tight')
