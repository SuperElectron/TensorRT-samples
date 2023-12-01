#
# To look at the shared object library, try this!
# nm -D /opt/nvidia/deepstream/deepstream-6.1/lib/libnvbufsurface.so

set(MODULE_NAME "nvds.cmake")
message(STATUS ${MODULE_NAME} [start] ----------------------)
SET(NVDS_INSTALL_DIR /opt/nvidia/deepstream/deepstream)

include_directories(
        ${NVDS_INSTALL_DIR}/sources/includes
        /usr/local/cuda/include
)


target_link_directories(${LIBRARY_NAME} PUBLIC
        ${NVDS_INSTALL_DIR}/lib
        /usr/local/cuda/lib64
        /usr/local/cuda/targets/x86_64-linux/lib
        )

message(STATUS "Using CUDA dir /usr/local/cuda/lib64 AND /usr/local/cuda/include")

add_library(nvds SHARED IMPORTED)
set_target_properties(nvds PROPERTIES
        IMPORTED_LOCATION ${NVDS_INSTALL_DIR}/lib/libcudart.so
        IMPORTED_LOCATION ${NVDS_INSTALL_DIR}/lib/libnvdsgst_meta.so
        IMPORTED_LOCATION ${NVDS_INSTALL_DIR}/lib/libnvds_meta.so
        IMPORTED_LOCATION ${NVDS_INSTALL_DIR}/lib/libnvds_yml_parser.so
        IMPORTED_LOCATION ${NVDS_INSTALL_DIR}/lib/libnvbufsurface.so
        IMPORTED_LOCATION ${NVDS_INSTALL_DIR}/lib/libnvbufsurftransform.so
        )

target_link_libraries(${LIBRARY_NAME} PUBLIC
        nvds
        cudart
        nvdsgst_meta
        nvds_meta
        nvds_yml_parser
        nvbufsurface
        nvbufsurftransform
        cuda
        )

message(STATUS ${MODULE_NAME} [finish] ----------------------)