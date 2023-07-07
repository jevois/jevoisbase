# Add install directive for our includes:
install(DIRECTORY jevoisbase
  COMPONENT include
  DESTINATION "${JEVOIS_MODULES_ROOT}/include"
  FILES_MATCHING PATTERN "*.[Hh]"
  PATTERN "*.hpp"
  PATTERN ".svn" EXCLUDE
  )

set(CPACK_COMPONENTS_ALL include)
