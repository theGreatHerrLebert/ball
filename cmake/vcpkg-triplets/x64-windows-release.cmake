# Custom vcpkg triplet — x64 Windows, dynamic, RELEASE-ONLY.
#
# The stock `x64-windows` triplet builds BOTH debug and release variants of
# every dependency (Qt5, Boost, ...), which roughly doubles the cold-cache
# build time on CI. BALL's CI builds Release only, so the debug halves are
# dead weight. `VCPKG_BUILD_TYPE release` drops them.
#
# Selected via the `windows-vcpkg` preset in CMakePresets.json
# (VCPKG_TARGET_TRIPLET + VCPKG_OVERLAY_TRIPLETS).
set(VCPKG_TARGET_ARCHITECTURE x64)
set(VCPKG_CRT_LINKAGE dynamic)
set(VCPKG_LIBRARY_LINKAGE dynamic)
set(VCPKG_BUILD_TYPE release)
