
ADD_EXECUTABLE(matcher main.cpp)
TARGET_LINK_LIBRARIES(matcher rtabmap_gui)

SET_TARGET_PROPERTIES( matcher 
  PROPERTIES OUTPUT_NAME ${PROJECT_PREFIX}-matcher)

IF(TORCH_FOUND)
ADD_EXECUTABLE(native_superglue_selftest native_superglue_selftest.cpp)
TARGET_LINK_LIBRARIES(native_superglue_selftest rtabmap_core)
SET_TARGET_PROPERTIES( native_superglue_selftest
  PROPERTIES OUTPUT_NAME ${PROJECT_PREFIX}-native_superglue_selftest)

ADD_EXECUTABLE(native_lightglue_selftest native_lightglue_selftest.cpp)
TARGET_LINK_LIBRARIES(native_lightglue_selftest rtabmap_core)
SET_TARGET_PROPERTIES( native_lightglue_selftest
  PROPERTIES OUTPUT_NAME ${PROJECT_PREFIX}-native_lightglue_selftest)
ENDIF()
 
