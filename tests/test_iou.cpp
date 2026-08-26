#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "iou.hpp"

// This file checks the rotated overlap calculation against a handful of cases whose answer can be
// worked out on paper, which is the only way to know a clipping and rotation routine is actually
// right rather than merely self-consistent.

TEST_CASE("rotated 3d iou matches hand-computed cases") {
  SECTION("identical boxes overlap completely") {
    const Box box{0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0};
    REQUIRE(intersection_over_union_3d(box, box) == Catch::Approx(1.0));
  }

  SECTION("disjoint boxes do not overlap at all") {
    const Box first{0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0};
    const Box second{10.0, 10.0, 0.0, 1.0, 1.0, 1.0, 0.0};
    REQUIRE(intersection_over_union_3d(first, second) == Catch::Approx(0.0));
  }

  SECTION("two boxes offset by half their length") {
    const Box first{0.0, 0.0, 0.0, 2.0, 1.0, 1.0, 0.0};
    const Box second{1.0, 0.0, 0.0, 2.0, 1.0, 1.0, 0.0};
    REQUIRE(intersection_over_union_3d(first, second) == Catch::Approx(1.0 / 3.0));
  }

  SECTION("two unit cubes, one rotated 45 degrees") {
    const Box first{0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0};
    const Box second{0.0, 0.0, 0.0, 1.0, 1.0, 1.0, M_PI / 4.0};
    REQUIRE(intersection_over_union_3d(first, second) == Catch::Approx(0.7071).margin(1e-3));
  }

  SECTION("a small box fully inside a larger one") {
    const Box small_box{0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0};
    const Box large_box{0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0};
    REQUIRE(intersection_over_union_3d(small_box, large_box) == Catch::Approx(0.125));
  }
}
