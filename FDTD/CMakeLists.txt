
if (WITH_MPI)
  set(MPI_SOURCES
     ${CMAKE_CURRENT_SOURCE_DIR}/openems_fdtd_mpi.cpp
     ${CMAKE_CURRENT_SOURCE_DIR}/operator_mpi.cpp
     ${CMAKE_CURRENT_SOURCE_DIR}/engine_mpi.cpp
     )
endif()

if (WITH_GPU)
  set(GPU_SOURCES
     ${CMAKE_CURRENT_SOURCE_DIR}/engine_vulkan.cpp
     ${CMAKE_CURRENT_SOURCE_DIR}/operator_vulkan.cpp
     )
endif()

if(OPENEMS_ENABLE_AVX2)
  set(AVX2_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/operator_avx2.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/engine_avx2.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/operator_avx2_multithread.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/engine_avx2_multithread.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/engine_interface_avx2_fdtd.cpp
  )
endif()

set(SOURCES
  ${SOURCES}
  ${MPI_SOURCES}
  ${GPU_SOURCES}
  ${CMAKE_CURRENT_SOURCE_DIR}/engine.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/operator.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/engine_multithread.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/operator_cylinder.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/engine_cylinder.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/engine_sse.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/operator_sse.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/operator_sse_compressed.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/engine_sse_compressed.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/operator_multithread.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/excitation.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/operator_cylindermultigrid.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/engine_cylindermultigrid.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/engine_interface_fdtd.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/engine_interface_sse_fdtd.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/temporal_blocking.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/engine_benchmark.cpp
  ${AVX2_SOURCES}
  PARENT_SCOPE
)
