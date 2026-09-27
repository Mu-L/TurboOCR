#include <catch_amalgamated.hpp>

#include <algorithm>
#include <vector>

#include "turbo_ocr/common/geometry/box.h"

using turbo_ocr::Box;

TEST_CASE("Box default construction is zeroed", "[box]") {
  Box b{};
  for (int i = 0; i < 4; ++i) {
    CHECK(b[i][0] == 0);
    CHECK(b[i][1] == 0);
  }
}

TEST_CASE("Box equality and comparison", "[box]") {
  Box a{{{{{10, 20}}, {{30, 20}}, {{30, 40}}, {{10, 40}}}}};
  Box b = a;
  CHECK(a == b);

  Box c{{{{{10, 20}}, {{30, 20}}, {{30, 40}}, {{10, 41}}}}};
  CHECK(a != c);
  CHECK(a < c); // a[3][1]=40 < c[3][1]=41
}

TEST_CASE("sorted_boxes orders top-to-bottom, left-to-right", "[box]") {
  // Box A at y=100, x=200
  Box a{{{{{200, 100}}, {{300, 100}}, {{300, 130}}, {{200, 130}}}}};
  // Box B at y=100, x=50 (same line, left of A)
  Box b{{{{{50, 105}}, {{150, 105}}, {{150, 130}}, {{50, 130}}}}};
  // Box C at y=300, x=10 (lower line)
  Box c{{{{{10, 300}}, {{110, 300}}, {{110, 330}}, {{10, 330}}}}};

  std::vector<Box> boxes = {a, c, b};
  turbo_ocr::sorted_boxes(boxes);

  // A and B are within kSameLineThreshold of each other, so B.x < A.x decides
  CHECK(boxes[0] == b);
  CHECK(boxes[1] == a);
  CHECK(boxes[2] == c);
}

TEST_CASE("sorted_boxes same-line tolerance does not depend on absolute Y",
          "[box]") {
  // REGRESSION: this used to quantize y/10 into FIXED bands, so whether two
  // boxes counted as one line depended on where they sat relative to a
  // multiple of 10 rather than on the gap between them. Tops of 29 and 34 (5px
  // apart) landed in different bands while 30 and 34 (4px apart) did not.
  // Every pair below is 5px apart and must therefore sort left-to-right,
  // wherever the band edge happens to fall.
  for (int top = 25; top <= 35; ++top) {
    INFO("right-hand box top = " << top);
    Box right{{{{{400, top}}, {{500, top}}, {{500, top + 26}}, {{400, top + 26}}}}};
    const int lt = top + 5;
    Box left{{{{{50, lt}}, {{150, lt}}, {{150, lt + 26}}, {{50, lt + 26}}}}};

    std::vector<Box> boxes = {right, left};
    turbo_ocr::sorted_boxes(boxes);
    CHECK(boxes[0] == left); // same line => leftmost first
    CHECK(boxes[1] == right);
  }
}

TEST_CASE("sorted_boxes does not chain a staircase into one line", "[box]") {
  // Each box sits 9px below the previous: within tolerance of its NEIGHBOUR,
  // but the last is 45px below the first. Measuring the gap from the current
  // line's own top stops the run from collapsing into one tall "line" that
  // would then sort purely by X. X DECREASES as Y increases, so a collapse
  // would return them reversed.
  std::vector<Box> boxes;
  for (int i = 0; i < 6; ++i) {
    const int y = 100 + i * 9;
    const int x = 500 - i * 50;
    boxes.push_back(Box{{{{{x, y}}, {{x + 40, y}}, {{x + 40, y + 20}}, {{x, y + 20}}}}});
  }
  auto shuffled = boxes;
  std::swap(shuffled[0], shuffled[4]);
  std::swap(shuffled[1], shuffled[3]);
  turbo_ocr::sorted_boxes(shuffled);

  // A box may only precede a higher one when the two are within the same-line
  // tolerance; consecutive 9px steps legitimately pair up, so no exact
  // permutation is asserted.
  for (std::size_t i = 0; i + 1 < shuffled.size(); ++i) {
    INFO("index " << i);
    CHECK(shuffled[i][0][1] <= shuffled[i + 1][0][1] + 10);
  }
  // The run must NOT have become a single line (fully reversed).
  CHECK(shuffled.front()[0][1] < shuffled.back()[0][1]);
}

TEST_CASE("sorted_boxes is deterministic regardless of input order", "[box]") {
  // The output may depend only on the set of boxes: the GPU detector emits
  // them in an order that varies run to run.
  std::vector<Box> boxes;
  for (int i = 0; i < 12; ++i) {
    const int y = 40 + (i % 4) * 7;  // clusters that straddle band edges
    const int x = 10 + ((i * 37) % 400);
    boxes.push_back(Box{{{{{x, y}}, {{x + 30, y}}, {{x + 30, y + 18}}, {{x, y + 18}}}}});
  }
  // Two lines starting at the same X, 9px apart (small type), and two boxes
  // sharing a top-left corner: the ties the order used to be left to.
  boxes.push_back(Box{{{{{739, 530}}, {{797, 530}}, {{797, 541}}, {{739, 541}}}}});
  boxes.push_back(Box{{{{{739, 539}}, {{788, 539}}, {{788, 549}}, {{739, 549}}}}});
  boxes.push_back(Box{{{{{600, 700}}, {{650, 700}}, {{650, 712}}, {{600, 712}}}}});
  boxes.push_back(Box{{{{{600, 700}}, {{680, 700}}, {{680, 714}}, {{600, 714}}}}});
  auto expected = boxes;
  turbo_ocr::sorted_boxes(expected);

  auto rotated = boxes;
  for (std::size_t r = 1; r < boxes.size(); ++r) {
    std::rotate(rotated.begin(), rotated.begin() + 1, rotated.end());
    auto got = rotated;
    turbo_ocr::sorted_boxes(got);
    INFO("rotation " << r);
    CHECK(got == expected);
  }
  auto reversed = boxes;
  std::reverse(reversed.begin(), reversed.end());
  turbo_ocr::sorted_boxes(reversed);
  CHECK(reversed == expected);
}

TEST_CASE("sorted_boxes puts the upper of two same-X lines first", "[box]") {
  // Seen on GPU: 9px-apart lines at the same X came back in either order
  // depending on detector output order.
  Box upper{{{{{739, 530}}, {{797, 530}}, {{797, 541}}, {{739, 541}}}}};
  Box lower{{{{{739, 539}}, {{788, 539}}, {{788, 549}}, {{739, 549}}}}};
  for (const bool lower_first : {false, true}) {
    INFO("lower first in input: " << lower_first);
    std::vector<Box> boxes = lower_first ? std::vector<Box>{lower, upper}
                                         : std::vector<Box>{upper, lower};
    turbo_ocr::sorted_boxes(boxes);
    CHECK(boxes[0] == upper);
    CHECK(boxes[1] == lower);
  }
}

TEST_CASE("sorted_boxes empty vector", "[box]") {
  std::vector<Box> boxes;
  turbo_ocr::sorted_boxes(boxes);
  CHECK(boxes.empty());
}

TEST_CASE("sorted_boxes single element", "[box]") {
  Box a{{{{{10, 20}}, {{30, 20}}, {{30, 40}}, {{10, 40}}}}};
  std::vector<Box> boxes = {a};
  turbo_ocr::sorted_boxes(boxes);
  CHECK(boxes.size() == 1);
  CHECK(boxes[0] == a);
}

TEST_CASE("is_vertical_box detects vertical text", "[box]") {
  // Horizontal box: width=100, height=30
  Box horiz{{{{{0, 0}}, {{100, 0}}, {{100, 30}}, {{0, 30}}}}};
  CHECK_FALSE(turbo_ocr::is_vertical_box(horiz));

  // Vertical box: width=30, height=100 (h >= w * 1.5)
  Box vert{{{{{0, 0}}, {{30, 0}}, {{30, 100}}, {{0, 100}}}}};
  CHECK(turbo_ocr::is_vertical_box(vert));

  // Square box: width=100, height=100 (NOT vertical, h < w*1.5)
  Box square{{{{{0, 0}}, {{100, 0}}, {{100, 100}}, {{0, 100}}}}};
  CHECK_FALSE(turbo_ocr::is_vertical_box(square));
}

TEST_CASE("is_vertical_box edge case at boundary ratio", "[box]") {
  // width=20, height=30 -> ratio = 1.5 exactly
  // is_vertical_box uses h*h >= w*w*225/100 -> 900 >= 400*2.25 = 900 -> true
  Box boundary{{{{{0, 0}}, {{20, 0}}, {{20, 30}}, {{0, 30}}}}};
  CHECK(turbo_ocr::is_vertical_box(boundary));

  // width=20, height=29 -> 841 >= 900 -> false
  Box just_below{{{{{0, 0}}, {{20, 0}}, {{20, 29}}, {{0, 29}}}}};
  CHECK_FALSE(turbo_ocr::is_vertical_box(just_below));
}
