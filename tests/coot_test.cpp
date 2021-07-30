/**
 * @file coot_test.cpp
 * @author Ryan Curtin
 *
 * Test ensmallen algorithms with bandicoot.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <bandicoot>
#define ENS_PRINT_INFO
#define ENS_PRINT_WARN
#define COOT_DEFAULT_BACKEND CUDA_BACKEND
#define COOT_USE_U64S64
#include <ensmallen.hpp>
#include "catch.hpp"
#include "test_function_tools.hpp"

using namespace ens;
using namespace ens::test;

/**
 * Run SGD on logistic regression and make sure the results are acceptable.
 */
TEST_CASE("SGDLogisticRegressionTest", "[CootTest]")
{
  coot::mat data, testData, shuffledData;
  coot::Row<size_t> responses, testResponses, shuffledResponses;

  LogisticRegressionTestData(data, testData, shuffledData,
      responses, testResponses, shuffledResponses);
  responses.print();
  shuffledResponses.print();
  LogisticRegressionFunction<coot::mat, coot::Row<size_t>> lr(shuffledData, shuffledResponses, 0.5);

  StandardSGD sgd;
  coot::mat coordinates = lr.GetInitialPoint();
  sgd.Optimize(lr, coordinates);

  // Ensure that the error is close to zero.
  const double acc = lr.ComputeAccuracy(data, responses, coordinates);
  REQUIRE(acc == Approx(100.0).epsilon(0.003)); // 0.3% error tolerance.

  const double testAcc = lr.ComputeAccuracy(testData, testResponses,
      coordinates);
  REQUIRE(testAcc == Approx(100.0).epsilon(0.006)); // 0.6% error tolerance.
}

/**
 * Run L-BFGS on logistic regression and make sure the results are acceptable.
 *
TEST_CASE("LBFGSLogisticRegressionTest", "[CootTest]")
{
  coot::mat data, testData, shuffledData;
  coot::Row<size_t> responses, testResponses, shuffledResponses;

  LogisticRegressionTestData(data, testData, shuffledData,
      responses, testResponses, shuffledResponses);
  LogisticRegressionFunction<coot::mat, coot::Row<size_t>> lr(shuffledData, shuffledResponses, 0.5);

  L_BFGS lbfgs;
  coot::mat coordinates = lr.GetInitialPoint();
  lbfgs.Optimize(lr, coordinates);

  // Ensure that the error is close to zero.
  const double acc = lr.ComputeAccuracy(data, responses, coordinates);
  REQUIRE(acc == Approx(100.0).epsilon(0.003)); // 0.3% error tolerance.

  const double testAcc = lr.ComputeAccuracy(testData, testResponses,
      coordinates);
  REQUIRE(testAcc == Approx(100.0).epsilon(0.006)); // 0.6% error tolerance.
}
*/
