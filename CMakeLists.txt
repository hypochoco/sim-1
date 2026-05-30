cmake_minimum_required(VERSION 3.15)

project(sim-1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# use true file directory
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# app
if (NOT APPLE)
    add_executable(main)
else()
    # macOS app bundle
    add_executable(main MACOSX_BUNDLE)
    set_target_properties(main PROPERTIES
        MACOSX_BUNDLE TRUE
        MACOSX_BUNDLE_BUNDLE_NAME "Sim1Bundle"
    )
endif()

# engine sub directory
add_subdirectory(external/engine)

# glob headers and src
if (NOT APPLE)
    file(GLOB_RECURSE MY_SOURCES CONFIGURE_DEPENDS
        "src/*.cpp"
        "include/*.h"
    )
else()
    file(GLOB_RECURSE MY_SOURCES CONFIGURE_DEPENDS
        "src/*.cpp"
        "src/*.mm"
        "include/*.h"
    )
endif()

file(GLOB_RECURSE MY_HEADERS CONFIGURE_DEPENDS
    "include/**/*.h"
    "include/**/*.hpp"
    "include/**/*.inl"
)

file(GLOB_RECURSE EXTERNAL_HEADERS CONFIGURE_DEPENDS
    "external/**/*.h"
    "external/**/*.hpp"
    "external/**/*.inl"
)

target_sources(main PRIVATE ${MY_SOURCES})

target_sources(main PRIVATE ${EXTERNAL_HEADERS})

source_group(
    FILE_SET HEADERS
    BASE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include"
    FILES ${MY_SOURCES}
)

source_group(
    TREE "${CMAKE_CURRENT_SOURCE_DIR}"
    PREFIX "project files"
    FILES ${MY_SOURCES} ${MY_HEADERS}
)

source_group(
    TREE "${CMAKE_CURRENT_SOURCE_DIR}/external"
    PREFIX "external dependencies"
    FILES ${EXTERNAL_HEADERS}
)

target_include_directories(main
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

# engine
target_link_libraries(main PRIVATE engine::engine)
