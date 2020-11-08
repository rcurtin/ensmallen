/**
 * @file test_function_tools.hpp
 * @author Marcus Edel
 * @author Ryan Curtin
 * @author Conrad Sanderson
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef ENSMALLEN_TESTS_TEST_FUNCTION_TOOLS_HPP
#define ENSMALLEN_TESTS_TEST_FUNCTION_TOOLS_HPP

#include "catch.hpp"

/**
 * Create the data for the a logistic regression test.
 *
 * @param data Matrix object to store the data into.
 * @param testData Matrix object to store the test data into.
 * @param shuffledData Matrix object to store the shuffled data into.
 * @param responses Matrix object to store the overall responses into.
 * @param testResponses Matrix object to store the test responses into.
 * @param shuffledResponses Matrix object to store the shuffled responses into.
 */
template<typename MatType, typename LabelsType>
inline void LogisticRegressionTestData(MatType& data,
                                       MatType& testData,
                                       MatType& shuffledData,
                                       LabelsType& responses,
                                       LabelsType& testResponses,
                                       LabelsType& shuffledResponses)
{
  // Generate a two-Gaussian dataset.
  arma::mat armaData = arma::mat(3, 100000);
  arma::Row<size_t> armaResponses = arma::Row<size_t>(100000);
  for (size_t i = 0; i < 50000; ++i)
  {
    // The first Gaussian is centered at (1, 1, 1) and has covariance I.
    armaData.col(i) = arma::randn<arma::Col<typename MatType::elem_type>>(3) +
        arma::Col<typename MatType::elem_type>("1.0 1.0 1.0");
    armaResponses(i) = 0;
  }
  for (size_t i = 50000; i < 100000; ++i)
  {
    // The second Gaussian is centered at (9, 9, 9) and has covariance I.
    armaData.col(i) = arma::randn<arma::Col<typename MatType::elem_type>>(3) +
        arma::Col<typename MatType::elem_type>("9.0 9.0 9.0");
    armaResponses(i) = 1;
  }

  // Shuffle the dataset.
  arma::uvec indices = arma::shuffle(arma::linspace<arma::uvec>(0,
      armaData.n_cols - 1, armaData.n_cols));
  arma::mat armaShuffledData = arma::mat(3, 100000);
  arma::Row<size_t> armaShuffledResponses = arma::Row<size_t>(100000);
  for (size_t i = 0; i < armaData.n_cols; ++i)
  {
    armaShuffledData.col(i) = armaData.col(indices(i));
    armaShuffledResponses(i) = armaResponses[indices(i)];
  }

  // Create a test set.
  arma::mat armaTestData = arma::mat(3, 100000);
  arma::Row<size_t> armaTestResponses = arma::Row<size_t>(100000);
  for (size_t i = 0; i < 50000; ++i)
  {
    armaTestData.col(i) = arma::randn<arma::Col<typename MatType::elem_type>>(3) +
        arma::Col<typename MatType::elem_type>("1.0 1.0 1.0");
    armaTestResponses(i) = 0;
  }
  for (size_t i = 50000; i < 100000; ++i)
  {
    armaTestData.col(i) = arma::randn<arma::Col<typename MatType::elem_type>>(3) +
        arma::Col<typename MatType::elem_type>("9.0 9.0 9.0");
    armaTestResponses(i) = 1;
  }

  data = MatType(armaData);
  testData = MatType(armaTestData);
  shuffledData = MatType(armaShuffledData);
  responses = LabelsType(armaResponses);
  testResponses = LabelsType(armaTestResponses);
  shuffledResponses = LabelsType(armaShuffledResponses);
}

// Check the values of two matrices.
template<typename MatType>
inline void CheckMatrices(const MatType& a,
                          const MatType& b,
                          double tolerance = 1e-5)
{
  REQUIRE(a.n_rows == b.n_rows);
  REQUIRE(a.n_cols == b.n_cols);

  for (size_t i = 0; i < a.n_elem; ++i)
  {
    if (std::abs(a(i)) < tolerance / 2)
      REQUIRE(b(i) == Approx(0.0).margin(tolerance / 2.0));
    else
      REQUIRE(a(i) == Approx(b(i)).epsilon(tolerance));
  }
}

#endif
