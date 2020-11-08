/**
 * @file logistic_regression_function.cpp
 * @author Sumedh Ghaisas
 *
 * Implementation of the LogisticRegressionFunction class.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_PROBLEMS_LOGISTIC_REGRESSION_FUNCTION_IMPL_HPP
#define ENSMALLEN_PROBLEMS_LOGISTIC_REGRESSION_FUNCTION_IMPL_HPP

// In case it hasn't been included yet.
#include "logistic_regression_function.hpp"

namespace ens {
namespace test {

template<typename MatType, typename LabelsType>
typename MatType::elem_type conv_to_op(
    const MatType& sigmoid,
    const LabelsType& responses,
    typename std::enable_if<coot::is_coot_type<MatType>::value>::type* = 0)
{
  return accu(log(1.0 -
      coot::conv_to<MatType>::from(responses) + sigmoid %
      (2 * coot::conv_to<MatType>::from(responses) - 1.0)));
}

template<typename MatType, typename LabelsType>
typename MatType::elem_type conv_to_op(
    const MatType& sigmoid,
    const LabelsType& responses,
    typename std::enable_if<arma::is_arma_type<MatType>::value>::type* = 0)
{
  return accu(log(1.0 -
      arma::conv_to<MatType>::from(responses) + sigmoid %
      (2 * arma::conv_to<MatType>::from(responses) - 1.0)));
}

template<typename MatType, typename LabelsType>
typename MatType::elem_type conv_to_op_2(
    const LabelsType& responses,
    const MatType& sigmoids,
    typename std::enable_if<coot::is_coot_type<MatType>::value>::type* = 0)
{
  return -accu(coot::conv_to<MatType>::from(responses) - sigmoids);
}

template<typename MatType, typename LabelsType>
typename MatType::elem_type conv_to_op_2(
    const LabelsType& responses,
    const MatType& sigmoids,
    typename std::enable_if<arma::is_arma_type<MatType>::value>::type* = 0)
{
  return -accu(arma::conv_to<MatType>::from(responses) - sigmoids);
}

template<typename MatType, typename LabelsType, typename MatType2, typename
MatType3>
void conv_to_op_3(
    MatType& gradient,
    const size_t numCols,
    const MatType& sigmoids,
    const LabelsType& responses,
    const MatType2& predictors,
    const MatType3& regularization,
    typename std::enable_if<coot::is_coot_type<MatType>::value>::type* = 0)
{
  gradient.tail_cols(numCols) = (sigmoids -
      coot::conv_to<MatType3>::from(responses)) *
      predictors.t() + regularization;
}

template<typename MatType, typename LabelsType, typename MatType2, typename MatType3>
void conv_to_op_3(
    MatType& gradient,
    const size_t numCols,
    const MatType& sigmoids,
    const LabelsType& responses,
    const MatType2& predictors,
    const MatType3& regularization,
    typename std::enable_if<arma::is_arma_type<MatType>::value>::type* = 0)
{
  gradient.tail_cols(numCols) = (sigmoids -
      arma::conv_to<MatType3>::from(responses)) *
      predictors.t() + regularization;
}

template<typename MatType, typename LabelsType>
LogisticRegressionFunction<MatType, LabelsType>::LogisticRegressionFunction(
    MatType& predictors,
    LabelsType& responses,
    const double lambda) :
    // We promise to be well-behaved... the elements won't be modified.
    predictors(predictors),
    responses(responses),
    lambda(lambda)
{
  initialPoint = MatType(1, predictors.n_rows + 1);
  initialPoint.zeros();

  // Sanity check.
  if (responses.n_elem != predictors.n_cols)
  {
    std::ostringstream oss;
    oss << "LogisticRegressionFunction::LogisticRegressionFunction(): "
        << "predictors matrix has " << predictors.n_cols << " points, but "
        << "responses vector has " << responses.n_elem << " elements (should be"
        << " " << predictors.n_cols << ")!" << std::endl;
    throw std::logic_error(oss.str());
  }
}

template<typename MatType, typename LabelsType>
LogisticRegressionFunction<MatType, LabelsType>::LogisticRegressionFunction(
    MatType& predictors,
    LabelsType& responses,
    MatType& initialPoint,
    const double lambda) :
    initialPoint(initialPoint),
    predictors(predictors),
    responses(responses),
    lambda(lambda)
{
  // To check if initialPoint is compatible with predictors.
  if (initialPoint.n_rows != (predictors.n_rows + 1) ||
      initialPoint.n_cols != 1)
    this->initialPoint = arma::Row<typename MatType::elem_type>(
        predictors.n_rows + 1, arma::fill::zeros);
}

/**
 * Shuffle the datapoints.
 */
template<typename MatType, typename LabelsType>
void LogisticRegressionFunction<MatType, LabelsType>::Shuffle()
{
  MatType newPredictors;
  LabelsType newResponses;

  arma::uvec ordering = arma::shuffle(arma::linspace<arma::uvec>(0,
      predictors.n_cols - 1, predictors.n_cols));

  newPredictors.set_size(predictors.n_rows, predictors.n_cols);
  newResponses.set_size(responses.n_elem);
  for (size_t i = 0; i < predictors.n_cols; ++i)
  {
    newPredictors.col(i) = predictors.col(ordering[i]);
    newResponses[i] = (typename LabelsType::elem_type) responses[ordering[i]];
  }

  // Take ownership of the new data.
  predictors = std::move(newPredictors);
  responses = std::move(newResponses);
}

/**
 * Evaluate the logistic regression objective function given the estimated
 * parameters.
 */
template<typename MatType, typename LabelsType>
typename MatType::elem_type
LogisticRegressionFunction<MatType, LabelsType>::Evaluate(
    const MatType& parameters) const
{
  // The objective function is the log-likelihood function (w is the parameters
  // vector for the model; y is the responses; x is the predictors; sig() is the
  // sigmoid function):
  //   f(w) = sum(y log(sig(w'x)) + (1 - y) log(sig(1 - w'x))).
  // We want to minimize this function.  L2-regularization is just lambda
  // multiplied by the squared l2-norm of the parameters then divided by two.
  typedef typename MatType::elem_type ElemType;

  // For the regularization, we ignore the first term, which is the intercept
  // term and take every term except the last one in the decision variable.
  const ElemType regularization = 0.5 * lambda *
      dot(parameters.tail_cols(parameters.n_elem - 1),
          parameters.tail_cols(parameters.n_elem - 1));

  // Calculate vectors of sigmoids.  The intercept term is parameters(0, 0) and
  // does not need to be multiplied by any of the predictors.
  const MatType sigmoid = 1.0 / (1.0 +
      exp(-(parameters(0, 0) +
            parameters.tail_cols(parameters.n_elem - 1) * predictors)));

  // Assemble full objective function.  Often the objective function and the
  // regularization as given are divided by the number of features, but this
  // doesn't actually affect the optimization result, so we'll just ignore those
  // terms for computational efficiency.  Note that the conversion causes some
  // copy and slowdown, but this is so negligible compared to the rest of the
  // calculation it is not worth optimizing for.
  const ElemType result = conv_to_op(sigmoid, responses);

  // Invert the result, because it's a minimization.
  return regularization - result;
}

/**
 * Evaluate the logistic regression objective function given the estimated
 * parameters for a given batch from a given point.
 */
template<typename MatType, typename LabelsType>
typename MatType::elem_type
LogisticRegressionFunction<MatType, LabelsType>::Evaluate(
    const MatType& parameters,
    const size_t begin,
    const size_t batchSize) const
{
  typedef typename MatType::elem_type ElemType;

  // Calculate the regularization term.
  const ElemType regularization = lambda *
      (batchSize / (2.0 * predictors.n_cols)) *
      dot(parameters.tail_cols(parameters.n_elem - 1),
          parameters.tail_cols(parameters.n_elem - 1));

  // Calculate the sigmoid function values.
  const MatType sigmoid = 1.0 / (1.0 +
      exp(-(parameters(0, 0) +
            parameters.tail_cols(parameters.n_elem - 1) *
                predictors.cols(begin, begin + batchSize - 1))));

  // Compute the objective for the given batch size from a given point.
  const ElemType result = conv_to_op(sigmoid, responses.subvec(begin, begin + batchSize - 1));

  // Invert the result, because it's a minimization.
  return regularization - result;
}

//! Evaluate the gradient of the logistic regression objective function.
template<typename MatType, typename LabelsType>
template<typename GradType>
void LogisticRegressionFunction<MatType, LabelsType>::Gradient(
    const MatType& parameters,
    GradType& gradient) const
{
  typedef typename MatType::elem_type ElemType;
  // Regularization term.
  MatType regularization;
  regularization = lambda * parameters.tail_cols(parameters.n_elem - 1);

  const MatType sigmoids = (1 / (1 + exp(-parameters(0, 0)
      - parameters.tail_cols(parameters.n_elem - 1) * predictors)));

  gradient.set_size(parameters.n_rows, parameters.n_cols);
  gradient[0] = conv_to_op_2(responses, sigmoids);
  conv_to_op_3(gradient, parameters.n_elem - 1, sigmoids, responses, predictors,
      regularization);
}

//! Evaluate the gradient of the logistic regression objective function for a
//! given batch size.
template<typename MatType, typename LabelsType>
template<typename GradType>
void LogisticRegressionFunction<MatType, LabelsType>::Gradient(
                const MatType& parameters,
                const size_t begin,
                GradType& gradient,
                const size_t batchSize) const
{
  typedef typename MatType::elem_type ElemType;

  // Regularization term.
  MatType regularization;
  regularization = lambda * parameters.tail_cols(parameters.n_elem - 1)
      / predictors.n_cols * batchSize;

  const MatType exponents = parameters(0, 0) +
      parameters.tail_cols(parameters.n_elem - 1) *
      predictors.cols(begin, begin + batchSize - 1);
  // Calculating the sigmoid function values.
  const MatType sigmoids = 1.0 / (1.0 + arma::exp(-exponents));

  gradient.set_size(parameters.n_rows, parameters.n_cols);
  gradient[0] = conv_to_op_2(responses.subvec(begin, begin + batchSize - 1),
      sigmoids);
  conv_to_op_3(gradient, parameters.n_elem - 1, sigmoids,
      responses.subvec(begin, begin + batchSize - 1),
      predictors.cols(begin, begin + batchSize - 1),
      regularization);
}

/**
 * Evaluate the partial gradient of the logistic regression objective
 * function with respect to the individual features in the parameter.
 */
template <typename MatType, typename LabelsType>
void LogisticRegressionFunction<MatType, LabelsType>::PartialGradient(
    const MatType& parameters,
    const size_t j,
    arma::sp_mat& gradient) const
{
  const MatType diffs = responses -
      (1 / (1 + exp(-parameters(0, 0) -
                     parameters.tail_cols(parameters.n_elem - 1) *
                          predictors)));

  gradient.set_size(parameters.n_rows, parameters.n_cols);

  if (j == 0)
  {
    gradient[j] = -accu(diffs);
  }
  else
  {
    gradient[j] = dot(-predictors.row(j - 1), diffs) + lambda *
        parameters(0, j);
  }
}

template<typename MatType, typename LabelsType>
template<typename GradType>
typename MatType::elem_type
LogisticRegressionFunction<MatType, LabelsType>::EvaluateWithGradient(
    const MatType& parameters,
    GradType& gradient) const
{
  typedef typename MatType::elem_type ElemType;

  // Regularization term.
  MatType regularization = lambda *
      parameters.tail_cols(parameters.n_elem - 1);

  const ElemType objectiveRegularization = lambda / 2.0 *
      dot(parameters.tail_cols(parameters.n_elem - 1),
          parameters.tail_cols(parameters.n_elem - 1));

  // Calculate the sigmoid function values.
  const MatType sigmoids = 1.0 / (1.0 +
      exp(-(parameters(0, 0) +
            parameters.tail_cols(parameters.n_elem - 1) * predictors)));

  gradient.set_size(parameters.n_rows, parameters.n_cols);
  gradient[0] = conv_to_op_2(responses, sigmoids);
  conv_to_op_3(gradient, parameters.n_elem - 1, sigmoids, responses, predictors,
      regularization);

  // Now compute the objective function using the sigmoids.
  const ElemType result = conv_to_op(sigmoids, responses);

  // Invert the result, because it's a minimization.
  return objectiveRegularization - result;
}

template<typename MatType, typename LabelsType>
template<typename GradType>
typename MatType::elem_type
LogisticRegressionFunction<MatType, LabelsType>::EvaluateWithGradient(
    const MatType& parameters,
    const size_t begin,
    GradType& gradient,
    const size_t batchSize) const
{
//  arma::wall_clock c_overall;
//  arma::wall_clock c;
  typedef typename MatType::elem_type ElemType;

  // Regularization term.
//  c_overall.tic();
//  c.tic();
  MatType regularization =
      lambda * parameters.tail_cols(parameters.n_elem - 1) / predictors.n_cols *
      batchSize;
//  std::cout << " - regularization: " << c.toc() << "s\n";
//  regularization.print("regularization");

//  c.tic();
  const ElemType objectiveRegularization = lambda *
      (batchSize / (2.0 * predictors.n_cols)) *
      dot(parameters.tail_cols(parameters.n_elem - 1),
          parameters.tail_cols(parameters.n_elem - 1));
//  std::cout << " - objectiveRegularization: " << c.toc() << "s\n";
//  std::cout << "value: " << objectiveRegularization << "\n";

  // Calculate the sigmoid function values.
//  c.tic();
  const MatType sigmoids = 1.0 / (1.0 +
      exp(-(parameters(0, 0) +
            parameters.tail_cols(parameters.n_elem - 1) *
                predictors.cols(begin, begin + batchSize - 1))));
//  std::cout << " - sigmoids: " << c.toc() << "s\n";
//  sigmoids.print("sigmoids");

  gradient.set_size(parameters.n_rows, parameters.n_cols);
//  c.tic();
  gradient[0] = conv_to_op_2(responses.subvec(begin, begin + batchSize - 1),
      sigmoids);
//  std::cout << " - gradient[0]: " << c.toc() << "s\n";
//  std::cout << "value: " << gradient[0] << "\n";
//  c.tic();
  conv_to_op_3(gradient, parameters.n_elem - 1, sigmoids,
      responses.subvec(begin, begin + batchSize - 1),
      predictors.cols(begin, begin + batchSize - 1),
      regularization);
//  std::cout << " - gradient: " << c.toc() << "s\n";

  // Now compute the objective function using the sigmoids.
//  c.tic();
  const ElemType result = conv_to_op(sigmoids, responses.subvec(begin, begin + batchSize - 1));
//  std::cout << " - result: " << c.toc() << "s\n";
//  std::cout << "value: " << result << "\n";
//  std::cout << "EvaluateWithGradient() time: " << c_overall.toc() << "s\n";

  // Invert the result, because it's a minimization.
//  std::cout << "EvaluateWithGradient() objective result: " << (objectiveRegularization - result) << "\n";
//  std::cout << "EvaluateWithGradient() gradient result:\n";
//  gradient.print("gradient");
  return objectiveRegularization - result;
}

template<typename MatType, typename LabelsType>
void ClassifyImpl(
    const MatType& dataset,
    LabelsType& labels,
    const MatType& parameters,
    const double decisionBoundary,
    typename std::enable_if<coot::is_coot_type<MatType>::value>::type* = 0)
{
  // Calculate sigmoid function for each point.  The (1.0 - decisionBoundary)
  // term correctly sets an offset so that floor() returns 0 or 1 correctly.
  labels = coot::conv_to<LabelsType>::from((1.0 /
      (1.0 + exp(-parameters(0) -
      parameters.tail_cols(parameters.n_elem - 1) * dataset))) +
      (1.0 - decisionBoundary));
}

template<typename MatType, typename LabelsType>
void ClassifyImpl(
    const MatType& dataset,
    LabelsType& labels,
    const MatType& parameters,
    const double decisionBoundary,
    typename std::enable_if<arma::is_arma_type<MatType>::value>::type* = 0)
{
  // Calculate sigmoid function for each point.  The (1.0 - decisionBoundary)
  // term correctly sets an offset so that floor() returns 0 or 1 correctly.
  labels = arma::conv_to<LabelsType>::from((1.0 /
      (1.0 + exp(-parameters(0) -
      parameters.tail_cols(parameters.n_elem - 1) * dataset))) +
      (1.0 - decisionBoundary));
}

template<typename MatType, typename LabelsType>
void LogisticRegressionFunction<MatType, LabelsType>::Classify(
    const MatType& dataset,
    LabelsType& labels,
    const MatType& parameters,
    const double decisionBoundary) const
{
  ClassifyImpl(dataset, labels, parameters, decisionBoundary);
}

template<typename MatType, typename LabelsType>
double LogisticRegressionFunction<MatType, LabelsType>::ComputeAccuracy(
    const MatType& predictors,
    const LabelsType& responses,
    const MatType& parameters,
    const double decisionBoundary) const
{
  // Predict responses using the current model.
  LabelsType tempResponses;
  Classify(predictors, tempResponses, parameters, decisionBoundary);

  // Count the number of responses that were correct.
  size_t count = 0;
  for (size_t i = 0; i < responses.n_elem; i++)
  {
    if (responses(i) == tempResponses(i))
      count++;
  }

  return (double) (count * 100) / responses.n_elem;
}

} // namespace test
} // namespace ens

#endif
